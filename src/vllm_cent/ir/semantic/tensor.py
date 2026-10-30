"""Ops that move or reinterpret elements without computing: copy and reshape.

Neither op changes a value. A copy makes a new value with the same
type; a reshape lists the same row-major elements under a new shape.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

from ..base import Operation, Value
from ..types import Replicate, TensorRole, TensorType

__all__ = [
    "CopyOp",
    "ReshapeOp",
]


@dataclass(frozen = True, eq = False, slots=True)
class CopyOp(Operation):
    """A copy of x as a new value; every element stays where it is.

    CopyOp(operands=(x,))
    """

    NAME = "aloi.copy"
    NUM_OPERANDS = 1

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result has x's shape, dtype, axes and placement.

        A partial sum is accepted: copying is linear, so copying each rank's
        part and summing later gives the same tensor as summing first.
        """

        return (replace(self.x.type, role = TensorRole.ACTIVATION),)


@dataclass(frozen = True, eq = False, slots=True)
class ReshapeOp(Operation):
    """View x's elements, in row-major order, with a new shape.

    No element moves: this is not a transpose or a strided view. A (2, 3)
    tensor can become (6,) or (3, 2), and all three list the same six values
    in the same order.

    ReshapeOp(operands=(x,), shape=(6,))

    Attributes:
        shape: The result shape, fully known (no -1). It has as many elements
            as x.
    """

    NAME = "aloi.reshape"
    NUM_OPERANDS = 1

    shape: tuple[int, ...]

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result has the new shape and x's dtype, with no axes.

        Axes are dropped: a general reshape can merge or split axes in ways a
        name cannot describe. split_heads and merge_heads are the reshapes
        that keep axis names.

        Raises:
            ValueError: If a dimension is negative, the element counts
                differ, or x is split or a partial sum. Which elements a rank
                owns can change under a reshape: after (8 heads x 128) on one
                rank becomes another shape, the split dimension may be merged
                with others, and Shard(axis) can no longer describe it.
        """

        x = self.x.type
        if any(d < 0 for d in self.shape) or prod(self.shape) != prod(x.shape):
            raise ValueError(
                f"{self.NAME}: cannot reshape {x.shape} to {self.shape}"
            )
        if not isinstance(x.placement, Replicate):
            raise ValueError(
                f"{self.NAME}: x is {x.placement}; only whole tensors are reshaped"
            )
        return (TensorType(shape = tuple(self.shape), dtype = x.dtype),)
