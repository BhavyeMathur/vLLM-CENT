"""Typed Llama config, parsed from a Hugging Face config.json."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

__all__ = ["LlamaConfig"]


@dataclass(frozen=True, slots=True)
class LlamaConfig:
    """Dimensions of one Llama decoder layer.

    Field names follow Hugging Face ``LlamaConfig``. Build one from a
    checkpoint's config.json with ``LlamaConfig.from_hf``; small test configs
    are made with ``dataclasses.replace``.

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
        rope_theta: Base of the rotary position frequencies.
        rope_scaling: The ``rope_scaling`` entry of config.json, e.g.
            ``{"rope_type": "llama3", "factor": 8.0, ...}``, or None for plain
            RoPE. Passed unchanged to ``aloi.RotaryEmbedding``.
        max_position_embeddings: Longest context the model supports; the
            rotary cos/sin table has one row per position.
        dtype: Element type of weights and activations.
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    intermediate_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Mapping[str, float | int | str] | None
    max_position_embeddings: int
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

    @classmethod
    def from_hf(cls, hf: Mapping[str, Any]) -> LlamaConfig:
        """Parse a Hugging Face Llama config.json.

        Values the file may omit get Hugging Face's defaults:
        ``num_key_value_heads`` = ``num_attention_heads`` (no GQA),
        ``head_dim`` = ``hidden_size // num_attention_heads``, ``rope_theta``
        = 10000. The dtype comes from ``torch_dtype`` (or the newer
        ``dtype``).

        Args:
            hf: The parsed config.json.

        Returns:
            The typed config.

        Raises:
            KeyError: If a required entry is missing.
            ValueError: For an unsupported dtype or activation.
            NotImplementedError: For projection biases, which the layer does
                not model.
        """

        if hf.get("hidden_act", "silu") != "silu":
            raise ValueError(f"hidden_act {hf['hidden_act']!r}; the MLP uses silu")
        if hf.get("attention_bias", False) or hf.get("mlp_bias", False):
            raise NotImplementedError("projection biases are not modeled")
        dtype_name = hf.get("torch_dtype", hf.get("dtype", "float32"))
        dtype = _DTYPES.get(str(dtype_name))
        if dtype is None:
            raise ValueError(
                f"unsupported dtype {dtype_name!r}; use one of {sorted(_DTYPES)}"
            )

        heads = int(hf["num_attention_heads"])
        scaling = hf.get("rope_scaling")
        return cls(
            hidden_size=int(hf["hidden_size"]),
            num_attention_heads=heads,
            num_key_value_heads=int(hf.get("num_key_value_heads", heads)),
            head_dim=int(hf.get("head_dim") or hf["hidden_size"] // heads),
            intermediate_size=int(hf["intermediate_size"]),
            rms_norm_eps=float(hf["rms_norm_eps"]),
            rope_theta=float(hf.get("rope_theta", 10000.0)),
            # A read-only view keeps the frozen config from being edited
            # through the mapping.
            rope_scaling=None if scaling is None else MappingProxyType(dict(scaling)),
            max_position_embeddings=int(hf["max_position_embeddings"]),
            dtype=dtype,
        )


# config.json dtype names.
_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
