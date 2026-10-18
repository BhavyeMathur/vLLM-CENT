"""Ops that split a packed head dimension into heads and merge it back.

``[batch, seq_len, heads*head_dim]`` and ``[batch, seq_len, heads, head_dim]``
list the same elements in the same order; these ops keep the axis names.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...base import Operation, Value
from ...types import (
    Partial,
    Placement,
    Shard,
    TensorType,
    packed_axis,
    packed_axis_factors,
)

__all__ = [
    "MergeHeadsOp",
    "SplitHeadsOp",
]


def _reshaped_placement(
    tensor_type: TensorType, whole_result_shape: tuple[int, ...]
) -> tuple[tuple[int, ...] | None, Placement]:
    """Return the global shape and placement of a head reshape's result.

    Splitting or merging heads moves no data between ranks, so the result
    keeps the operand's placement. Only a shard records its whole shape.

    Args:
        tensor_type: The operand type.
        whole_result_shape: Result shape of the whole, unsplit tensor.

    Returns:
        ``(global_shape, placement)``.
    """

    if isinstance(tensor_type.placement, Shard):
        return whole_result_shape, tensor_type.placement
    return None, tensor_type.placement


@dataclass(frozen=True, eq=False, slots=True)
class SplitHeadsOp(Operation):
    """Split the last dimension into ``[heads, head_dim]``.

    ``[batch, seq_len, q_head*head_dim]`` becomes
    ``[batch, seq_len, q_head, head_dim]``. The packed axis name says what the
    two new dimensions mean, so the result's axes follow from the operand's.

    SplitHeadsOp(operands=(x,), head_dim=128)

    Attributes:
        head_dim: Length of one head; the new last dimension.
    """

    NAME = "aloi.split_heads"
    NUM_OPERANDS = 1

    head_dim: int

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Reshape the last dimension; keep shard placements on the heads.

        A shard of q_head*head_dim holds whole heads, so after the split it
        is a shard of the new q_head dimension.

        Raises:
            ValueError: If x is a scalar or a partial sum, head_dim does not
                divide the last dimension, or the annotated last axis is not
                a packed pair such as q_head*head_dim.
        """

        x = self.x.type
        if x.rank < 1:
            raise ValueError(f"{self.NAME}: x needs a dimension to split")
        if self.head_dim < 1 or x.shape[-1] % self.head_dim != 0:
            raise ValueError(
                f"{self.NAME}: head_dim {self.head_dim} does not divide "
                f"x's last dimension {x.shape[-1]}"
            )
        if isinstance(x.placement, Partial):
            raise ValueError(f"{self.NAME}: x is a partial sum; all_reduce it first")

        axes: tuple[str, ...] = ()
        if x.axes:
            factors = packed_axis_factors(x.axes[-1])
            if len(factors) != 2:
                raise ValueError(
                    f"{self.NAME}: x's last axis {x.axes[-1]!r} must pack exactly "
                    "two axes, e.g. q_head*head_dim"
                )
            axes = (*x.axes[:-1], *factors)

        shape = (*x.shape[:-1], x.shape[-1] // self.head_dim, self.head_dim)
        whole = (*x.full_shape[:-1], x.full_shape[-1] // self.head_dim, self.head_dim)
        global_shape, placement = _reshaped_placement(x, whole)
        return (
            TensorType(
                shape=shape,
                dtype=x.dtype,
                axes=axes,
                global_shape=global_shape,
                placement=placement,
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class MergeHeadsOp(Operation):
    """Merge the last two dimensions ``[heads, head_dim]`` into one.

    The inverse of SplitHeadsOp: ``[batch, seq_len, q_head, head_dim]``
    becomes ``[batch, seq_len, q_head*head_dim]``.

    MergeHeadsOp(operands=(x,))
    """

    NAME = "aloi.merge_heads"
    NUM_OPERANDS = 1

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Flatten the last two dimensions, heads outermost.

        Raises:
            ValueError: If x has fewer than two dimensions, is a partial sum,
                or is split along head_dim. A head_dim shard would become an
                inner factor of the packed axis, which cannot be split
                contiguously.
        """

        x = self.x.type
        if x.rank < 2:
            raise ValueError(f"{self.NAME}: x needs [heads, head_dim] dimensions")
        if isinstance(x.placement, Partial):
            raise ValueError(f"{self.NAME}: x is a partial sum; all_reduce it first")
        if x.sharded_dim == x.rank - 1:
            raise ValueError(
                f"{self.NAME}: x is split along its last dimension, which would "
                "become the inner factor of the merged axis"
            )

        axes: tuple[str, ...] = ()
        if x.axes:
            axes = (*x.axes[:-2], packed_axis(x.axes[-2], x.axes[-1]))

        shape = (*x.shape[:-2], x.shape[-2] * x.shape[-1])
        whole = (*x.full_shape[:-2], x.full_shape[-2] * x.full_shape[-1])
        global_shape, placement = _reshaped_placement(x, whole)
        return (
            TensorType(
                shape=shape,
                dtype=x.dtype,
                axes=axes,
                global_shape=global_shape,
                placement=placement,
            ),
        )
