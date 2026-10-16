"""Describe attention over explicitly supplied query, key, and value tensors."""

from dataclasses import dataclass
from math import isfinite

from .base import Operation, TensorType

__all__ = ["AttentionOperation"]


@dataclass(frozen=True, slots=True, kw_only=True)
class AttentionOperation(Operation):
    """Compute scaled dot-product attention, independently for each query head.

    Inputs use row-major shapes ``Q[B,Hq,L,D]``, ``K[B,Hkv,S,D]``, and
    ``V[B,Hkv,S,Dv]``. Each contiguous group of query heads shares one key/value head:
    query head ``h`` reads KV head ``h // (Hq // Hkv)``. The result has shape
    ``[B,Hq,L,Dv]``. An optional additive mask is shared ``[L,S]`` or complete
    ``[B,Hq,L,S]``; there is no general broadcasting or boolean mask convention.

    The mask can contain finite numbers or negative infinity. Each score row
    must retain at least one finite entry. No dropout, projections, rotary
    positions, cache updates, or head concatenation are implied.

    Attributes:
        name: Unique graph operation name.
        inputs: Query, key, value, and optionally additive-mask tensor names.
        output: Attention result name and four-dimensional type.
        block: Optional source model-block label.
        causal: Whether query row ``i`` excludes keys after its declared position.
        query_position_offset: Nonnegative position of query row zero; causal
            rows admit key indices through ``query_position_offset + i``.
        scale: Finite score multiplier; ``None`` selects ``1 / sqrt(D)``.

    Examples:
        With one query and five keys, ``causal=True`` and offset zero admit only
        key zero. Offset four admits all five keys. The caller supplies the
        offset explicitly; a longer key tensor does not imply cache history.
    """

    causal: bool = False
    query_position_offset: int = 0
    scale: float | None = None

    def __post_init__(self) -> None:
        """Require valid positional and numerical parameters.

        Raises:
            ValueError: If the position is negative or an explicit scale is nonfinite.
        """
        if self.query_position_offset < 0:
            raise ValueError("Attention query position offset must be nonnegative")
        if self.scale is not None and not isfinite(self.scale):
            raise ValueError("Attention scale must be finite")

    def verify(self, input_types: tuple[TensorType, ...]) -> None:
        """Check exact tensor dimensions, grouped heads, and optional mask shape.

        Args:
            input_types: Query, key, value, and optional additive-mask types.

        Raises:
            ValueError: If arity, scalar formats, dimensions, or head groups disagree.
        """
        if len(input_types) not in (3, 4):
            raise ValueError("Attention requires query, key, value, and optional mask")
        if any(t.dtype != self.output.type.dtype for t in input_types):
            raise ValueError("Attention requires one scalar format")
        query, key, value = input_types[:3]
        if any(len(t.shape) != 4 for t in (query, key, value)):
            raise ValueError("Attention query, key, and value must have rank four")
        batch, heads, queries, width = query.shape
        key_batch, kv_heads, keys, key_width = key.shape
        value_batch, value_heads, value_keys, value_width = value.shape
        if not (
            batch == key_batch == value_batch
            and kv_heads == value_heads
            and heads % kv_heads == 0
            and width == key_width
            and keys == value_keys
            and self.output.type.shape == (batch, heads, queries, value_width)
        ):
            raise ValueError("Incompatible attention tensor shapes or grouped heads")
        if len(input_types) == 4 and input_types[3].shape not in (
            (queries, keys),
            (batch, heads, queries, keys),
        ):
            raise ValueError("Attention mask must be [L,S] or [B,Hq,L,S]")
