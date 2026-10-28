"""Ops that move or reinterpret elements without computing.

None of these ops changes a value. A copy makes a new value with the same
type; a reshape lists the same row-major elements under a new shape; a
broadcast repeats each element along a new last dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import prod

from ..base import Operation, Value
from ..types import Replicate, TensorRole, TensorType

__all__ = [
    "BroadcastOp",
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


@dataclass(frozen=True, eq=False, slots=True)
class BroadcastOp(Operation):
    """Repeat every element of x ``size`` times along a new last dimension.

    ``y[..., i] = x[...]`` for every ``i < size``. Decompositions use it to
    combine a per-row value with its row, e.g. subtracting each row's maximum
    in softmax.

    BroadcastOp(operands=(x,), size=2048, axis="context")

    Attributes:
        size: Length of the new last dimension.
        axis: Semantic name of the new dimension. Required once x is
            annotated with axes; ignored while x has none.

    Examples:
        Row maxima ``[1, 1, 8]`` with axes ``[batch, seq_len, q_head]``,
        broadcast with ``size=2048, axis="context"``, give
        ``[1, 1, 8, 2048]`` with axes ``[batch, seq_len, q_head, context]``.
    """

    NAME = "aloi.broadcast"
    NUM_OPERANDS = 1

    size: int
    axis: str | None = None

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result is x with a new last dimension of ``size``, as an activation.

        Repeating values is linear and moves nothing between ranks, so x's
        placement is kept: a partial sum stays partial, and a split of an
        earlier axis stays split (the new dimension is whole on every rank).

        Raises:
            ValueError: If ``size`` is below 1, or x is annotated and ``axis``
                is missing or already names one of x's dimensions (a split of
                that name would then match two dimensions).
        """

        x = self.x.type
        if self.size < 1:
            raise ValueError(f"{self.NAME}: size must be at least 1, got {self.size}")
        axes: tuple[str, ...] = ()
        if x.axes:
            # If the axes have already been annotated, then the new axis must 
            # also be annotated, and the name of the new axis may not repeat
            # existing axes names. Otherwise, Shard(axis) will hit two axes
            if self.axis is None or self.axis in x.axes:
                raise ValueError(
                    f"{self.NAME}: x has axes {x.axes}, so the new dimension "
                    f"needs a new axis name, got {self.axis!r}"
                )
            axes = (*x.axes, self.axis)
        global_shape = None if x.global_shape is None else (*x.global_shape, self.size)
        return (
            replace(
                x, 
                shape=(*x.shape, self.size), 
                role=TensorRole.ACTIVATION,
                axes=axes, 
                global_shape=global_shape,
            ),
        )
