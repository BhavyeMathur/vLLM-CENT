"""Rotary position embedding (RoPE) from a precomputed cos/sin table."""

from __future__ import annotations

from dataclasses import dataclass, replace

from ...base import Operation, Value
from ...types import Partial, Replicate, TensorRole, TensorType
from ._common import _POSITION_DTYPES

__all__ = [
    "RopeOp",
]


@dataclass(frozen=True, eq=False, slots=True)
class RopeOp(Operation):
    """Rotary position embedding of one head tensor, via a cos/sin table.

    ``cos_sin_cache[p]`` holds ``rotary_dim // 2`` cosines then as many sines
    for position p. Each head's first ``rotary_dim`` elements are rotated by
    the angles of its token's position; the rest pass through. The op is the
    same for every model: the RoPE variant (theta, llama3 or YaRN scaling)
    only changes the table's values, never this op.

    RopeOp(operands=(x, positions, cos_sin_cache), is_neox_style=True)

    Attributes:
        is_neox_style: True pairs element i with element i + rotary_dim/2
            (Hugging Face / vLLM ``rotate_half``); False pairs elements 2i
            and 2i+1 (Meta's original Llama code, GPT-J).
    """

    NAME = "aloi.rope"
    NUM_OPERANDS = 3

    is_neox_style: bool = True

    @property
    def x(self) -> Value:
        return self.operands[0]

    @property
    def positions(self) -> Value:
        return self.operands[1]

    @property
    def cos_sin_cache(self) -> Value:
        return self.operands[2]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result has x's shape, dtype, axes and placement.

        Raises:
            ValueError: If x is not ``[*, heads, head_dim]``, the table is not
                ``[positions, rotary_dim]`` with an even rotary_dim of at most
                head_dim, positions do not give one integer per token of x,
                the axes disagree, or the placements cannot be applied
                locally.
        """

        x = self.x.type
        positions = self.positions.type
        table = self.cos_sin_cache.type
        if x.rank < 2:
            raise ValueError(
                f"{self.NAME}: x must be [*, heads, head_dim], got {x.shape}"
            )
        # Compare with the whole head_dim: a split head_dim is a placement
        # problem, reported below with its own message.
        head_dim = x.full_shape[-1]
        if table.rank != 2 or table.shape[1] % 2 or table.shape[1] > head_dim:
            raise ValueError(
                f"{self.NAME}: cos_sin_cache {table.shape} must be [positions, "
                f"rotary_dim] with an even rotary_dim <= head_dim {head_dim}"
            )
        if positions.dtype not in _POSITION_DTYPES:
            raise ValueError(
                f"{self.NAME}: positions must be integers, got {positions.dtype}"
            )
        # One position per token: x's dimensions before [heads, head_dim].
        if positions.shape != x.shape[:-2]:
            raise ValueError(
                f"{self.NAME}: positions shape {positions.shape} must equal "
                f"x's token dimensions {x.shape[:-2]}"
            )
        if x.axes and positions.axes and positions.axes != x.axes[:-2]:
            raise ValueError(
                f"{self.NAME}: positions axes {positions.axes} must equal "
                f"x's token axes {x.axes[:-2]}"
            )

        # A rotation pairs elements within one head, so a rank needs whole
        # head vectors; every rank needs every position and the whole table.
        if not isinstance(positions.placement, Replicate):
            raise ValueError(f"{self.NAME}: positions must be replicated")
        if not isinstance(table.placement, Replicate):
            raise ValueError(f"{self.NAME}: cos_sin_cache must be replicated")
        if isinstance(x.placement, Partial):
            raise ValueError(f"{self.NAME}: x is a partial sum; all_reduce it first")
        if x.sharded_dim == x.rank - 1:
            raise ValueError(f"{self.NAME}: x is split along head_dim")
        return (replace(x, role=TensorRole.ACTIVATION),)
