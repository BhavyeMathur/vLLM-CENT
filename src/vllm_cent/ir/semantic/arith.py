"""Ops that reduce a tensor along its last dimension: sum and max.

A reduction combines every element of one row (the last dimension) into one
value, so the result drops that dimension: ``[*, n]`` becomes ``[*]``. The
result keeps the operand's other axes and any split of those axes.

Whether the reduced dimension may be split across tensor-parallel ranks
depends on the combining function. A sum of per-rank partial sums is the
whole sum, so ``reduce_sum`` over a split dimension gives a partial sum that
an all_reduce completes. A maximum has no such collective in the IR, so
``reduce_max`` needs the whole row on every rank.
"""

from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass

from ..base import Operation, Value
from ..types import Partial, Placement, Shard, TensorRole, TensorType

__all__ = [
    "MaxOp",
    "SumOp",
]


@dataclass(frozen = True, eq = False, slots = True)
class _LastDimReductionOp(Operation):
    """Shared shape and axes rules of reductions over the last dimension.

    Subclasses name the op and decide which placements they accept.
    """

    NUM_OPERANDS = 1

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result is x without its last dimension, as an activation.

        Raises:
            ValueError: If x is a scalar (nothing to reduce), or the
                subclass rejects x's placement.
        """

        x = self.x.type
        if x.rank < 1:
            raise ValueError(f"{self.NAME}: x needs a dimension to reduce over")
        placement = self._result_placement(x)
        # A shard of an earlier dimension stays a shard; the reduced, whole
        # last dimension disappears from the whole shape too.
        global_shape = None
        if isinstance(placement, Shard):
            assert x.global_shape is not None
            global_shape = x.global_shape[:-1]
        return (
            TensorType(
                shape = x.shape[:-1],
                dtype = x.dtype,
                role = TensorRole.ACTIVATION,
                axes = x.axes[:-1],
                global_shape = global_shape,
                placement = placement,
            ),
        )

    @abstractmethod
    def _result_placement(self, x: TensorType) -> Placement:
        """Return the result's placement, or raise if x's is not supported.

        Args:
            x: The operand type.

        Returns:
            Which part of the result each rank holds.

        Raises:
            ValueError: If the reduction cannot be computed locally.
        """


@dataclass(frozen = True, eq = False, slots = True)
class SumOp(_LastDimReductionOp):
    """Sum of each row: ``y[...] = sum(x[..., i] for i)``.

    SumOp(operands=(x,))

    Examples:
        ``x [2, 3]`` with rows ``(1, 2, 3)`` and ``(4, 5, 6)`` gives
        ``[2]`` holding ``(6, 15)``.
    """

    NAME = "aloi.reduce_sum"

    def _result_placement(self, x: TensorType) -> Placement:
        """Summing is linear, so split rows and partial sums stay correct.

        - x split along the reduced last dimension: each rank sums its slice
          of every row, a partial sum of the whole row's sum.
        - x a partial sum: summing each rank's part and adding the results
          equals summing the whole, so the result is a partial sum too.
        - otherwise x's placement (whole, or split along an earlier axis).

        Args:
            x: The operand type.

        Returns:
            The result's placement.
        """

        if x.sharded_dim == x.rank - 1:
            return Partial()
        return x.placement


@dataclass(frozen = True, eq = False, slots = True)
class MaxOp(_LastDimReductionOp):
    """Largest element of each row: ``y[...] = max(x[..., i] for i)``.

    MaxOp(operands=(x,))

    Examples:
        ``x [2, 3]`` with rows ``(1, 5, 3)`` and ``(-4, -2, -6)`` gives
        ``[2]`` holding ``(5, -2)``.
    """

    NAME = "aloi.reduce_max"

    def _result_placement(self, x: TensorType) -> Placement:
        """A maximum needs every element of the row on one rank.

        Args:
            x: The operand type.

        Returns:
            x's placement: whole, or split along an earlier axis.

        Raises:
            ValueError: If x is a partial sum (the maximum of the parts is not
                the maximum of their sum) or its last dimension is split (the
                IR has no all-reduce-max to combine per-rank maxima).
        """

        if isinstance(x.placement, Partial):
            raise ValueError(f"{self.NAME}: x is a partial sum; all_reduce it first")
        if x.sharded_dim == x.rank - 1:
            raise ValueError(
                f"{self.NAME}: x's last dimension is split across ranks; "
                "the maximum needs the whole row on every rank"
            )
        return x.placement
