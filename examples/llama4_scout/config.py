"""Typed Llama 4 text config, parsed from a Hugging Face config.json."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

import torch

__all__ = ["Llama4Config"]

# Hugging Face's default spacing of NoPE layers: when config.json gives no
# ``no_rope_layers``, every fourth layer (3, 7, 11, ...) has no RoPE.
_NO_ROPE_LAYER_INTERVAL = 4


@dataclass(frozen=True, slots=True)
class Llama4Config:
    """Dimensions of the Llama 4 text decoder layers.

    Field names follow Hugging Face ``Llama4TextConfig``. Build one from a
    checkpoint's config.json with ``Llama4Config.from_hf``; small test configs
    are made with ``dataclasses.replace``.

    Attributes:
        hidden_size: Length of the activation vector of one token.
        num_attention_heads: Number of query heads.
        num_key_value_heads: Number of key/value heads (grouped-query
            attention: each serves ``num_attention_heads // num_key_value_heads``
            query heads).
        head_dim: Length of one head's query, key, or value vector.
        num_hidden_layers: Number of decoder layers.
        intermediate_size: Hidden width of each routed expert and of the shared
            expert.
        num_local_experts: Number of routed experts per MoE layer.
        num_experts_per_tok: Experts chosen per token (top-k).
        moe_layers: Indices of the layers whose feed-forward network is a
            mixture of experts.
        no_rope_layers: One entry per layer: 1 means the layer uses RoPE, 0
            means it does not (a NoPE layer).
        use_qk_norm: Whether RoPE layers RMS-normalize queries and keys after
            rotating them.
        attn_temperature_tuning: Whether NoPE layers scale their queries by a
            function of position.
        floor_scale: Positions per step of that scale's logarithm.
        attn_scale: Weight of that logarithm.
        attention_chunk_size: RoPE layers attend only within chunks of this
            many positions.
        rms_norm_eps: Constant added to the mean square in every RMSNorm,
            including the query/key norm.
        rope_theta: Base of the rotary position frequencies.
        rope_scaling: The ``rope_scaling`` entry of config.json, passed
            unchanged to ``aloi.RotaryEmbedding``.
        max_position_embeddings: Longest context the model supports; the
            rotary cos/sin table has one row per position.
        dtype: Element type of weights and activations.
    """

    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    num_hidden_layers: int
    intermediate_size: int
    num_local_experts: int
    num_experts_per_tok: int
    moe_layers: tuple[int, ...]
    no_rope_layers: tuple[int, ...]
    use_qk_norm: bool
    attn_temperature_tuning: bool
    floor_scale: float
    attn_scale: float
    attention_chunk_size: int
    rms_norm_eps: float
    rope_theta: float
    rope_scaling: Mapping[str, float | int | str] | None
    max_position_embeddings: int
    dtype: torch.dtype

    def __post_init__(self) -> None:
        """Check the relationships between head counts, experts and layers.

        Raises:
            ValueError: If the query heads cannot be split evenly among the KV
                heads, top-k exceeds the experts, or ``no_rope_layers`` does
                not have one entry per layer.
        """

        # Grouped-query attention gives every KV head the same number of
        # query heads. (Unlike Llama 2/3, hidden_size need not equal
        # heads * head_dim; for Scout both are 5120.)
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError(
                f"num_attention_heads {self.num_attention_heads} is not "
                f"divisible by num_key_value_heads {self.num_key_value_heads}"
            )
        if not 1 <= self.num_experts_per_tok <= self.num_local_experts:
            raise ValueError(
                f"num_experts_per_tok {self.num_experts_per_tok} must be between "
                f"1 and num_local_experts {self.num_local_experts}"
            )
        if len(self.no_rope_layers) != self.num_hidden_layers:
            raise ValueError(
                f"no_rope_layers has {len(self.no_rope_layers)} entries for "
                f"{self.num_hidden_layers} layers"
            )

    def uses_rope(self, layer: int) -> bool:
        """Tell whether a layer rotates queries and keys (False: a NoPE layer).

        Args:
            layer: Decoder layer index.

        Returns:
            Whether the layer applies RoPE.

        Raises:
            ValueError: If the layer index is outside the model.
        """

        if not 0 <= layer < self.num_hidden_layers:
            raise ValueError(
                f"layer {layer} is outside the {self.num_hidden_layers} layers"
            )
        return self.no_rope_layers[layer] == 1

    @classmethod
    def from_hf(cls, hf: Mapping[str, Any]) -> Llama4Config:
        """Parse a Hugging Face Llama 4 config.json.

        The text settings are read from ``text_config`` when present (the
        multimodal ``llama4`` checkpoint), else from the top level (a
        ``llama4_text`` config). Omitted values get Hugging Face's defaults:
        NoPE every fourth layer, temperature tuning on with
        ``floor_scale=8192`` and ``attn_scale=0.1``, chunks of 8192, and
        ``moe_layers`` every ``interleave_moe_layer_step`` layers.

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

        text: Mapping[str, Any] = hf.get("text_config", hf)
        if text.get("hidden_act", "silu") != "silu":
            raise ValueError(f"hidden_act {text['hidden_act']!r}; the experts use silu")
        if text.get("attention_bias", False):
            raise NotImplementedError("projection biases are not modeled")
        dtype_name = text.get("torch_dtype", hf.get("torch_dtype", "float32"))
        dtype = _DTYPES.get(str(dtype_name))
        if dtype is None:
            raise ValueError(
                f"unsupported dtype {dtype_name!r}; use one of {sorted(_DTYPES)}"
            )

        layers = int(text["num_hidden_layers"])
        heads = int(text["num_attention_heads"])
        # An empty or missing list means Hugging Face's default spacing.
        no_rope_layers = tuple(text.get("no_rope_layers") or ()) or tuple(
            int((layer + 1) % _NO_ROPE_LAYER_INTERVAL != 0) for layer in range(layers)
        )
        step = int(text.get("interleave_moe_layer_step", 1))
        moe_layers = tuple(text.get("moe_layers") or range(step - 1, layers, step))
        scaling = text.get("rope_scaling")
        return cls(
            hidden_size=int(text["hidden_size"]),
            num_attention_heads=heads,
            num_key_value_heads=int(text.get("num_key_value_heads", heads)),
            head_dim=int(text.get("head_dim") or text["hidden_size"] // heads),
            num_hidden_layers=layers,
            intermediate_size=int(text["intermediate_size"]),
            num_local_experts=int(text["num_local_experts"]),
            num_experts_per_tok=int(text.get("num_experts_per_tok", 1)),
            moe_layers=moe_layers,
            no_rope_layers=no_rope_layers,
            use_qk_norm=bool(text.get("use_qk_norm", True)),
            attn_temperature_tuning=bool(text.get("attn_temperature_tuning", True)),
            floor_scale=float(text.get("floor_scale", 8192)),
            attn_scale=float(text.get("attn_scale", 0.1)),
            attention_chunk_size=int(text.get("attention_chunk_size", 8192)),
            rms_norm_eps=float(text["rms_norm_eps"]),
            rope_theta=float(text.get("rope_theta", 500000.0)),
            # A read-only view keeps the frozen config from being edited
            # through the mapping.
            rope_scaling=None if scaling is None else MappingProxyType(dict(scaling)),
            max_position_embeddings=int(text["max_position_embeddings"]),
            dtype=dtype,
        )


# config.json dtype names.
_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
