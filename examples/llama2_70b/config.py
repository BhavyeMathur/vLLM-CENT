"""Llama model dimensions, named as in the Hugging Face ``LlamaConfig``."""

from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["LLAMA2_70B", "LlamaConfig"]


@dataclass(frozen=True, slots=True)
class LlamaConfig:
    """Dimensions of one Llama decoder block.

    Field names follow Hugging Face ``LlamaConfig``, so values can be copied
    from a checkpoint's ``config.json`` without renaming. The class has no
    defaults: the Llama2-70B values live in ``LLAMA2_70B``, and small test
    configs are made with ``dataclasses.replace(LLAMA2_70B, ...)``.

    Attributes:
        hidden_size: Length of the activation vector of one token.
        num_attention_heads: Number of query heads.
        num_key_value_heads: Number of key/value heads. Fewer than query heads
            means grouped-query attention: each KV head serves
            ``num_attention_heads // num_key_value_heads`` query heads.
        head_dim: Length of one head's query, key, or value vector.
        intermediate_size: Width of the feed-forward hidden layer.
        rms_norm_eps: Constant added to the mean square in RMSNorm, so a zero
            vector does not divide by zero.
        rope_theta: Base frequency of the rotary position embedding.
        dtype: Element type of weights and activations.
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    dtype: torch.dtype

    def __post_init__(self) -> None:
        """Check the relationships between head counts and sizes.

        Raises:
            ValueError: If the query heads do not exactly cover
                ``hidden_size``, or if the query heads cannot be split evenly
                among the KV heads.
        """

        # Attention cuts the hidden vector into num_attention_heads pieces of
        # head_dim elements each, so the pieces must add up to the whole.
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError(
                f"hidden_size {self.hidden_size} != num_attention_heads "
                f"{self.num_attention_heads} * head_dim {self.head_dim}"
            )
        # Grouped-query attention gives every KV head the same number of
        # query heads.
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads {self.num_attention_heads} is not "
                f"divisible by num_key_value_heads {self.num_key_value_heads}"
            )


# Values from meta-llama/Llama-2-70b-hf config.json. head_dim is not stored
# there; it follows from 8192 / 64 = 128.
LLAMA2_70B = LlamaConfig(
    hidden_size=8192,
    num_attention_heads=64,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=28672,
    rms_norm_eps=1e-5,
    rope_theta=10000.0,
    dtype=torch.float16,
)
