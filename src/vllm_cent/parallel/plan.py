"""ParallelPlan: which part of each tensor every tensor-parallel rank owns.

The plan describes ownership only. It never mentions communication: the
all_reduce after a row-parallel linear is derived by ApplyTensorParallel from
the axes and the ops.

A plan is written as statements, rules first, exceptions after::

    plan = ParallelPlan(tp=8)
    plan.split("q_head")                        # rule: split q_head everywhere
    plan.split("intermediate")
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

    def split(self, axis: str, *, only: Iterable[str] | None = None) -> None:
        """Split ``axis`` evenly over the tp ranks.

        Without ``only`` this is a rule: every tensor whose axes contain
        ``axis`` (as a plain or outermost packed axis) is split. With
        ``only`` it is an exception for the named tensors alone.

        Args:
            axis: Semantic axis to split, e.g. ``"q_head"``. How many heads it
                has comes from the model (its axis sizes), not from the plan.
            only: Names of the tensors this exception applies to.

        Raises:
            ValueError: If the rule already exists, an exception names an axis
                that is already a rule, ``only`` is empty, or a named tensor
                already has an exception.
        """

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
