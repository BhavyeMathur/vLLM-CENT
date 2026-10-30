"""Attention temperature tuning: scale queries by a function of their position.

Llama 4 applies it in the layers without RoPE ("NoPE" layers), so that
attention stays sharp over very long contexts.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from ...base import Operation, Value
from ...types import Replicate, TensorRole, TensorType
from ._common import _POSITION_DTYPES

__all__ = [
    "AttnTemperatureOp",
]


@dataclass(frozen=True, eq=False, slots=True)
class AttnTemperatureOp(Operation):
    """Multiply each token's head vectors by a scale that grows with position.

    For a token at position ``p`` the scale is
    ``log(floor((p + 1) / floor_scale) + 1) * attn_scale + 1``. Positions
    below ``floor_scale - 1`` get scale 1, so short contexts are unchanged.

    AttnTemperatureOp(operands=(q, positions), floor_scale=8192.0, attn_scale=0.1)

    Attributes:
        floor_scale: Positions per step of the logarithm's argument.
        attn_scale: Weight of the logarithm in the scale.

    Examples:
        With ``floor_scale=8192`` and ``attn_scale=0.1``, a token at
        position 8191 gets ``log(1 + 1) * 0.1 + 1 = 1.0693`` and one at 100
        gets exactly 1.
    """

    NAME = "aloi.attn_temperature"
    NUM_OPERANDS = 2

    floor_scale: float
    attn_scale: float

    @property
    def x(self) -> Value:
        return self.operands[0]

    @property
    def positions(self) -> Value:
        return self.operands[1]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result has x's shape, dtype, axes and placement.

        The scale multiplies whole tokens, so any split of x (heads, or even a
        partial sum, since scaling is linear) can be scaled locally.

        Raises:
            ValueError: If the constants are not finite or floor_scale is not
                positive, x is not ``[*, heads, head_dim]``, positions do not
                give one integer per token of x, the axes disagree, or the
                positions are split across ranks.
        """

        if not (isfinite(self.floor_scale) and self.floor_scale > 0):
            raise ValueError(f"{self.NAME}: floor_scale must be positive and finite")
        if not isfinite(self.attn_scale):
            raise ValueError(f"{self.NAME}: attn_scale must be finite")
        x = self.x.type
        positions = self.positions.type
        if x.rank < 2:
            raise ValueError(
                f"{self.NAME}: x must be [*, heads, head_dim], got {x.shape}"
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
        if not isinstance(positions.placement, Replicate):
            raise ValueError(f"{self.NAME}: positions must be replicated")
        return (replace(x, role=TensorRole.ACTIVATION),)
