"""One Llama decoder layer written with ALOI ops.

The layer follows Hugging Face's ``LlamaDecoderLayer``, so parameter names
are checkpoint keys (``self_attn.q_proj.weight``, ``mlp.down_proj.weight``)::

    x ─ input_layernorm ─ self_attn ─(+x)─ h ─ post_attention_layernorm ─ mlp ─(+h)─ out

    self_attn:  q/k/v_proj → split_heads → rope(q, k) → kv_update(k, v)
                → gqa_qk → softmax → gqa_sv → merge_heads → o_proj
    mlp:        down_proj(silu(gate_proj(x)) * up_proj(x))

Layers are ``aloi.Linear`` / ``aloi.RMSNorm``, used like ``nn.Linear`` /
``nn.RMSNorm``; head reshapes, RoPE, the KV cache and attention call ALOI ops
such as ``aloi.split_heads`` directly. Every ALOI layer and op ends in an
``aloi::*`` custom op, which torch.export keeps as one graph node, while the
built-in layers may be decomposed into ``mm``/``pow``/``mean``. SiLU, the gate product and
the residual adds are plain PyTorch (``F.silu``, ``*``, ``+``); the importer
reads them as aten ops.

The layer is one decode step: one new token per request, and the KV caches
are explicit inputs and outputs (functional, no internal state). The cache
holds ``context`` positions; the new token is written at ``positions`` and
attention reads the whole cache. That matches decoding when the new token is
the last cache position (``positions == context - 1``).
TODO(attention mask): A cache with unused tail positions needs a mask; the
toy layer does not model one.

``build_meta_decoder_layer`` creates the parameters on the meta device: they
carry a shape and dtype but no storage, so the real 70B dimensions export
without allocating memory.

Calling an ALOI op or an nn.Module is typed as returning Any; the ``Tensor``
annotations on local variables restore the type.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from vllm_cent.frontend import aloi

from .config import LlamaConfig

__all__ = [
    "LlamaAttention",
    "LlamaDecoderLayer",
    "LlamaMLP",
    "build_meta_decoder_layer",
    "example_inputs",
]


class LlamaAttention(nn.Module):
    """Grouped-query self-attention for one decode step.

    Attributes:
        config: Model dimensions and RoPE parameters.
        q_proj: Hidden to all query heads, weight ``[heads * head_dim, hidden]``.
        k_proj: Hidden to all key heads, weight ``[kv_heads * head_dim, hidden]``.
        v_proj: Hidden to all value heads, weight ``[kv_heads * head_dim, hidden]``.
        o_proj: All query heads back to hidden, weight ``[hidden, heads * head_dim]``.
        rotary_emb: RoPE for q and k, with a cos/sin table of every position.
        scale: Dot-product scale, ``head_dim ** -0.5``.
    """

    def __init__(self, config: LlamaConfig) -> None:
        """Create the projections.

        Args:
            config: Model dimensions.
        """

        super().__init__()
        self.config = config
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        self.q_proj = aloi.Linear(config.hidden_size, q_width)
        self.k_proj = aloi.Linear(config.hidden_size, kv_width)
        self.v_proj = aloi.Linear(config.hidden_size, kv_width)
        self.o_proj = aloi.Linear(q_width, config.hidden_size)
        self.rotary_emb = aloi.RotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            rope_scaling=config.rope_scaling,
        )
        self.scale = config.head_dim**-0.5

    def forward(
        self, x: Tensor, positions: Tensor, k_cache: Tensor, v_cache: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Attend from the new token to the cache, including itself.

        Args:
            x: Normalized hidden states, ``[batch, seq_len, hidden]``.
            positions: ``[batch, seq_len]`` cache positions of the new tokens.
            k_cache: ``[batch, context, kv_heads, head_dim]``.
            v_cache: ``[batch, context, kv_heads, head_dim]``.

        Returns:
            ``(output [batch, seq_len, hidden], new k_cache, new v_cache)``.
        """

        head_dim = self.config.head_dim
        q = aloi.split_heads(self.q_proj(x), head_dim)
        k = aloi.split_heads(self.k_proj(x), head_dim)
        v = aloi.split_heads(self.v_proj(x), head_dim)

        q, k = self.rotary_emb(positions, q, k)
        k_cache = aloi.kv_update(k_cache, k, positions)
        v_cache = aloi.kv_update(v_cache, v, positions)

        scores = aloi.gqa_qk(q, k_cache, self.scale)
        probs = aloi.softmax(scores)
        context = aloi.gqa_sv(probs, v_cache)
        output: Tensor = self.o_proj(aloi.merge_heads(context))
        return output, k_cache, v_cache


class LlamaMLP(nn.Module):
    """SwiGLU feed-forward network.

    Attributes:
        gate_proj: Hidden to intermediate; passed through SiLU.
        up_proj: Hidden to intermediate; multiplied by the gate.
        down_proj: Intermediate back to hidden.
    """

    def __init__(self, config: LlamaConfig) -> None:
        """Create the projections.

        Args:
            config: Model dimensions.
        """

        super().__init__()
        self.gate_proj = aloi.Linear(config.hidden_size, config.intermediate_size)
        self.up_proj = aloi.Linear(config.hidden_size, config.intermediate_size)
        self.down_proj = aloi.Linear(config.intermediate_size, config.hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

        Args:
            x: Normalized hidden states, ``[batch, seq_len, hidden]``.

        Returns:
            ``[batch, seq_len, hidden]``.
        """

        gate: Tensor = self.gate_proj(x)
        up: Tensor = self.up_proj(x)
        output: Tensor = self.down_proj(F.silu(gate) * up)
        return output


class LlamaDecoderLayer(nn.Module):
    """One pre-norm Llama decoder layer with residual connections.

    Attributes:
        config: Model dimensions.
        input_layernorm: RMSNorm before attention.
        self_attn: Grouped-query self-attention.
        post_attention_layernorm: RMSNorm before the feed-forward network.
        mlp: SwiGLU feed-forward network.
    """

    def __init__(self, config: LlamaConfig) -> None:
        """Create the sublayers.

        Args:
            config: Model dimensions.
        """

        super().__init__()
        self.config = config
        self.input_layernorm = aloi.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = LlamaAttention(config)
        self.post_attention_layernorm = aloi.RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.mlp = LlamaMLP(config)

    def forward(
        self, x: Tensor, positions: Tensor, k_cache: Tensor, v_cache: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Run one decode step of the layer.

        Args:
            x: Hidden states, ``[batch, seq_len, hidden]``.
            positions: ``[batch, seq_len]`` cache positions of the new tokens.
            k_cache: ``[batch, context, kv_heads, head_dim]``.
            v_cache: ``[batch, context, kv_heads, head_dim]``.

        Returns:
            ``(hidden states, new k_cache, new v_cache)``.
        """

        normalized: Tensor = self.input_layernorm(x)
        attention, k_cache, v_cache = self.self_attn(
            normalized, positions, k_cache, v_cache
        )
        hidden = x + attention
        ffn_input: Tensor = self.post_attention_layernorm(hidden)
        ffn_output: Tensor = self.mlp(ffn_input)
        return hidden + ffn_output, k_cache, v_cache


def build_meta_decoder_layer(config: LlamaConfig) -> LlamaDecoderLayer:
    """Create a decoder layer whose parameters live on the meta device.

    Args:
        config: Model dimensions.

    Returns:
        The layer, with parameters in ``config.dtype``.
    """

    # Every tensor created inside this context, including the parameters
    # made in __init__, is a meta tensor: shape and dtype, but no memory.
    with torch.device("meta"):
        layer = LlamaDecoderLayer(config)
    # Parameters are created in the default dtype (fp32); convert them to the
    # model dtype afterwards, the same way a checkpoint loader would.
    return layer.to(config.dtype)


def example_inputs(
    config: LlamaConfig,
    context_len: int = 2048,
    batch: int = 1,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Create meta sample inputs for one decode step.

    One new token per request; the KV caches hold ``context_len`` positions,
    the last of which receives the new token.

    Args:
        config: Model dimensions.
        context_len: Number of cache positions, including the new token.
        batch: Number of requests.

    Returns:
        The ``forward`` arguments ``(x, positions, k_cache, v_cache)``.
    """

    x = torch.empty(batch, 1, config.hidden_size, dtype=config.dtype, device="meta")
    positions = torch.empty(batch, 1, dtype=torch.int64, device="meta")
    cache_shape = (batch, context_len, config.num_key_value_heads, config.head_dim)
    k_cache = torch.empty(cache_shape, dtype=config.dtype, device="meta")
    v_cache = torch.empty(cache_shape, dtype=config.dtype, device="meta")
    return x, positions, k_cache, v_cache
