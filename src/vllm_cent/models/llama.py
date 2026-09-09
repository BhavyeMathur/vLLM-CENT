"""Llama model descriptions."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True, kw_only=True)
class LlamaModelSpec:
    """Dimensions of one Llama transformer block.

    Attributes:
        hidden_size: Width of the residual stream. Must be at least 1 and
            divisible by ``num_attention_heads``.
        num_attention_heads: Number of query heads. Must be at least 1 and
            divisible by ``num_kv_heads``.
        num_kv_heads: Number of key/value heads. Must be at least 1 and no
            greater than ``num_attention_heads``.
        intermediate_size: Width of the feed-forward network. Must be at least
            1.
    """

    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int

    @property
    def head_size(self) -> int:
        """Return the width of one attention head.

        Returns:
            ``hidden_size`` divided by ``num_attention_heads``.
        """

        return self.hidden_size // self.num_attention_heads
