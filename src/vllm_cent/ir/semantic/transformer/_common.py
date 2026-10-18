"""Constants shared by several transformer op modules."""

from __future__ import annotations

from ...types import DType

__all__: list[str] = []


# Rank of every attention tensor: [batch, tokens, heads, head_dim]
# activations, KV caches, and [batch, seq_len, q_heads, context] scores.
_ATTENTION_RANK = 4

# Integer dtypes accepted for token positions.
_POSITION_DTYPES = (DType.INT64, DType.INT32)
