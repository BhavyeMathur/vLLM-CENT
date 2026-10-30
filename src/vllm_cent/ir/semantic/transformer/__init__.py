"""Semantic ops of decoder-only transformers: heads, RoPE, KV cache, attention.

These ops describe one decode step of attention without a transpose or an
expand. All tensors keep the token-first layout:

- an activation of one head type is ``[batch, seq_len, heads, head_dim]``;
- a KV cache is ``[batch, context, kv_heads, head_dim]``;
- attention scores are ``[batch, seq_len, q_heads, context]``.

Grouped-query attention (GQA) is built into gqa_qk and gqa_sv: with
``group = q_heads // kv_heads``, query head ``h`` reads KV head
``h // group``. Multi-head attention is the special case ``group == 1``.

Every op checks shapes, then axes (once annotated), then placements (once
tensor parallelism is applied), like the generic semantic ops.
"""

from .attention import AttentionOp, GqaQKOp, GqaSVOp
from .heads import MergeHeadsOp, SplitHeadsOp
from .kv_cache import KvUpdateOp
from .rope import RopeOp

__all__ = [
    "AttentionOp",
    "GqaQKOp",
    "GqaSVOp",
    "KvUpdateOp",
    "MergeHeadsOp",
    "RopeOp",
    "SplitHeadsOp",
]
