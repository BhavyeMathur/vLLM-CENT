"""ParallelPlan: which part of each tensor every tensor-parallel rank owns.

The plan describes ownership only. It never mentions communication: the
all_reduce after a row-parallel linear is derived by ApplyTensorParallel from
the axes and the ops.

A plan is written as statements, rules first, exceptions after::

    plan = ParallelPlan(tp=8)
    plan.split("q_head", size=64)               # rule: split q_head everywhere
    plan.split("intermediate")                  # rule on a plain axis: no size
    plan.replicate("k_proj.weight")             # exception: keep it whole
    plan.split("hidden", only=("lm_head.weight",))  # exception: split one tensor
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from types import MappingProxyType

from ..ir.types import Replicate, Shard

__all__ = ["ParallelPlan"]


class ParallelPlan:
    """A mutable builder of tensor-parallel ownership rules.

    The builder is mutable on purpose: with an immutable plan,
    ``plan.replicate(...)`` would have to be written
    ``plan = plan.replicate(...)``, and forgetting the assignment would
    silently do nothing. ApplyTensorParallel copies the plan when it is
    created, so later edits never change an existing pass.

    Errors that need only the plan are raised by the method that causes
    them. Errors that need the IR (unknown tensor names, an axis missing from
    a tensor, a dimension that does not divide) are raised by the pass.
    """

    def __init__(self, tp: int) -> None:
        """Create a plan without rules.

        Args:
            tp: Number of tensor-parallel ranks.

        Raises:
            ValueError: If tp is less than 1.
        """

        if tp < 1:
            raise ValueError(f"tp must be at least 1, got {tp}")
        self._tp = tp
        # Global rules, in the order they were written.
        self._split_axes: list[str] = []
        # Sizes of axes that appear as the outer factor of a packed
        # dimension, e.g. q_head = 64 inside q_head*head_dim. A plain axis
        # needs no entry: its dimension length is its size.
        self._axis_sizes: dict[str, int] = {}
        # Per-tensor exceptions, keyed by argument name (a parameter FQN or
        # an input name).
        self._exceptions: dict[str, Replicate | Shard] = {}

    @property
    def tp(self) -> int:
        """Number of tensor-parallel ranks."""

        return self._tp

    @property
    def split_axes(self) -> tuple[str, ...]:
        """Axes split on every tensor that has them, in the order written."""

        return tuple(self._split_axes)

    @property
    def exceptions(self) -> Mapping[str, Replicate | Shard]:
        """Per-tensor placements that override the rules.

        The mapping is a read-only view; use split() and replicate() to add
        exceptions.
        """

        return MappingProxyType(self._exceptions)

    def axis_size(self, axis: str) -> int | None:
        """Return the size given for ``axis``, or None if none was given.

        Args:
            axis: Axis name.

        Returns:
            The axis size, e.g. the number of heads.
        """

        return self._axis_sizes.get(axis)

    def split(
        self,
        axis: str,
        *,
        size: int | None = None,
        only: Iterable[str] | None = None,
    ) -> None:
        """Split ``axis`` evenly over the tp ranks.

        Without ``only`` this is a rule: every tensor whose axes contain
        ``axis`` (as a plain or outermost packed axis) is split. With
        ``only`` it is an exception for the named tensors alone.

        Args:
            axis: Semantic axis to split, e.g. ``"q_head"``.
            size: Size of the axis. Required when the axis is the outer
                factor of a packed dimension, because the dimension length
                alone does not tell how many heads it holds.
            only: Names of the tensors this exception applies to.

        Raises:
            ValueError: If size cannot be split evenly over tp, the axis was
                given a different size before, the rule already exists, an
                exception names an axis that is already a rule, ``only`` is
                empty, or a named tensor already has an exception.
        """

        if size is not None:
            self._record_size(axis, size)

        if only is None:
            if axis in self._split_axes:
                raise ValueError(f"axis {axis!r} is already split")
            self._split_axes.append(axis)
            return

        if axis in self._split_axes:
            raise ValueError(
                f"axis {axis!r} is already split on every tensor; "
                "split(..., only=...) would be redundant"
            )
        # Materialize first: ``only`` may be a generator that can be iterated
        # once.
        names = tuple(only)
        if not names:
            raise ValueError(f"split({axis!r}, only=...) names no tensor")
        for name in names:
            self._add_exception(name, Shard(axis))

    def replicate(self, *names: str) -> None:
        """Keep the named tensors whole on every rank, despite the rules.

        Args:
            *names: Tensor names.

        Raises:
            ValueError: If no name is given, or a named tensor already has an
                exception.
        """

        if not names:
            raise ValueError("replicate() names no tensor")
        for name in names:
            self._add_exception(name, Replicate())

    def _record_size(self, axis: str, size: int) -> None:
        """Remember the size of ``axis``.

        Args:
            axis: Axis name.
            size: Axis size.

        Raises:
            ValueError: If the size is not a positive multiple of tp, or the
                axis already has a different size.
        """

        # Every rank must get the same number of whole heads: with 8 KV heads
        # and tp=16, each rank would get half a head.
        if size < 1 or size % self._tp != 0:
            raise ValueError(
                f"axis {axis!r} of size {size} cannot be split evenly over "
                f"tp={self._tp}"
            )
        known = self._axis_sizes.get(axis)
        if known is not None and known != size:
            raise ValueError(f"axis {axis!r} was given size {known}, now {size}")
        self._axis_sizes[axis] = size

    def _add_exception(self, name: str, placement: Replicate | Shard) -> None:
        """Record one per-tensor exception.

        Args:
            name: Tensor name.
            placement: Placement the tensor gets instead of the rules'.

        Raises:
            ValueError: If the tensor already has an exception; two
                exceptions for one tensor would silently conflict.
        """

        if name in self._exceptions:
            raise ValueError(
                f"tensor {name!r} already has exception {self._exceptions[name]}"
            )
        self._exceptions[name] = placement
