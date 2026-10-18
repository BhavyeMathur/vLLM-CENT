"""One Llama 4 (Scout) text decoder layer written with ALOI ops.

The layer follows Hugging Face's ``Llama4TextDecoderLayer``, so parameter
names are checkpoint keys of one layer (``self_attn.q_proj.weight``,
``feed_forward.router.weight``)::

    x ─ input_layernorm ─ self_attn ─(+x)─ h ─ post_attention_layernorm ─ feed_forward ─(+h)─ out

    self_attn, RoPE layer:  q/k/v_proj → split_heads → rope(q, k) → qk_norm(q, k)
                            → kv_update → gqa_qk → softmax → gqa_sv → merge_heads → o_proj
    self_attn, NoPE layer:  q/k/v_proj → split_heads → attn_temperature(q)
                            → kv_update → gqa_qk → softmax → gqa_sv → merge_heads → o_proj
    feed_forward (MoE):     router → topk_sigmoid → experts(x, scores) + shared_expert(x)

Llama 4 differs from Llama 2/3 in three places:

- Every fourth layer is a NoPE layer: it has no rotary embedding, and its
  queries are scaled by ``attn_temperature`` (larger at far positions) so
  that attention stays sharp over long contexts. RoPE layers instead
  RMS-normalize queries and keys after rotating them (``qk_norm``, no
  learned weight).
- The feed-forward network is a mixture of experts: a router scores 16
  experts per token, the best one (top-1) runs with its sigmoid score
  multiplied into its input, and a shared expert runs on every token. The
  experts are written densely (see ``aloi.moe_experts``).
- RoPE pairs adjacent elements (GPT-J style, ``is_neox_style=False``), as
  Hugging Face's checkpoint does. vLLM uses NeoX pairing and permutes the
  q/k projection rows when loading instead.

As in the Llama 2/3 example, one decode step: one new token per request, the
KV caches are explicit inputs and outputs, and attention reads the whole
cache (the new token is the last cache position).

TODO(chunked attention): RoPE layers attend only within chunks of
``attention_chunk_size`` (8192) positions. With at most 8192 cached positions
the chunk is the whole cache, so no mask is needed; longer contexts need a
chunk mask, which the toy layer does not model, so they are rejected.

Calling an ALOI op or an nn.Module is typed as returning Any; the ``Tensor``
annotations on local variables restore the type.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn

from vllm_cent.frontend import aloi

from .config import Llama4Config

__all__ = [
    "Llama4Attention",
    "Llama4DecoderLayer",
    "Llama4MLP",
    "Llama4MoE",
    "build_meta_decoder_layer",
    "example_inputs",
]


class Llama4Attention(nn.Module):
    """Grouped-query self-attention of one RoPE or NoPE layer, one decode step.

    Attributes:
        config: Model dimensions.
        use_rope: Whether this layer rotates q and k (False: a NoPE layer).
        q_proj: Hidden to all query heads, weight ``[heads * head_dim, hidden]``.
        k_proj: Hidden to all key heads, weight ``[kv_heads * head_dim, hidden]``.
        v_proj: Hidden to all value heads, weight ``[kv_heads * head_dim, hidden]``.
        o_proj: All query heads back to hidden, weight ``[hidden, heads * head_dim]``.
        rotary_emb: RoPE for q and k (RoPE layers only).
        qk_norm: Weightless RMSNorm over head_dim for q and k (RoPE layers
            with ``use_qk_norm`` only).
        scale: Dot-product scale, ``head_dim ** -0.5``.
    """

    rotary_emb: aloi.RotaryEmbedding | None
    qk_norm: aloi.RMSNorm | None

    def __init__(self, config: Llama4Config, layer: int) -> None:
        """Create the projections and the layer type's position handling.

        Args:
            config: Model dimensions.
            layer: Decoder layer index; selects RoPE or NoPE.
        """

        super().__init__()
        self.config = config
        self.use_rope = config.uses_rope(layer)
        q_width = config.num_attention_heads * config.head_dim
        kv_width = config.num_key_value_heads * config.head_dim
        self.q_proj = aloi.Linear(config.hidden_size, q_width)
        self.k_proj = aloi.Linear(config.hidden_size, kv_width)
        self.v_proj = aloi.Linear(config.hidden_size, kv_width)
        self.o_proj = aloi.Linear(q_width, config.hidden_size)
        self.rotary_emb = None
        self.qk_norm = None
        if self.use_rope:
            self.rotary_emb = aloi.RotaryEmbedding(
                config.head_dim,
                config.max_position_embeddings,
                config.rope_theta,
                rope_scaling=config.rope_scaling,
                is_neox_style=False,
            )
            if config.use_qk_norm:
                self.qk_norm = aloi.RMSNorm(
                    config.head_dim, config.rms_norm_eps, elementwise_affine=False
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

        Raises:
            NotImplementedError: For a RoPE layer whose cache is longer than
                one attention chunk (see the module TODO).
        """

        if self.use_rope and k_cache.shape[1] > self.config.attention_chunk_size:
            raise NotImplementedError(
                f"context {k_cache.shape[1]} exceeds attention_chunk_size "
                f"{self.config.attention_chunk_size}; chunk masks are not modeled"
            )
        head_dim = self.config.head_dim
        q = aloi.split_heads(self.q_proj(x), head_dim)
        k = aloi.split_heads(self.k_proj(x), head_dim)
        v = aloi.split_heads(self.v_proj(x), head_dim)

        if self.rotary_emb is not None:
            q, k = self.rotary_emb(positions, q, k)
            # Hugging Face normalizes after rotating, both q and k.
            if self.qk_norm is not None:
                q = self.qk_norm(q)
                k = self.qk_norm(k)
        elif self.config.attn_temperature_tuning:
            q = aloi.attn_temperature(
                q, positions, self.config.floor_scale, self.config.attn_scale
            )
        k_cache = aloi.kv_update(k_cache, k, positions)
        v_cache = aloi.kv_update(v_cache, v, positions)

        scores = aloi.gqa_qk(q, k_cache, self.scale)
        probs = aloi.softmax(scores)
        context = aloi.gqa_sv(probs, v_cache)
        output: Tensor = self.o_proj(aloi.merge_heads(context))
        return output, k_cache, v_cache


class Llama4MLP(nn.Module):
    """SwiGLU feed-forward network; Llama 4's shared expert.

    Attributes:
        gate_proj: Hidden to intermediate; passed through SiLU.
        up_proj: Hidden to intermediate; multiplied by the gate.
        down_proj: Intermediate back to hidden.
    """

    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        """Create the projections.

        Args:
            hidden_size: Length of each token's vector.
            intermediate_size: Width of the hidden layer.
        """

        super().__init__()
        self.gate_proj = aloi.Linear(hidden_size, intermediate_size)
        self.up_proj = aloi.Linear(hidden_size, intermediate_size)
        self.down_proj = aloi.Linear(intermediate_size, hidden_size)

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``down_proj(silu(gate_proj(x)) * up_proj(x))``.

        Args:
            x: ``[batch, seq_len, hidden]``.

        Returns:
            ``[batch, seq_len, hidden]``.
        """

        gate: Tensor = self.gate_proj(x)
        up: Tensor = self.up_proj(x)
        output: Tensor = self.down_proj(F.silu(gate) * up)
        return output


class Llama4MoE(nn.Module):
    """Mixture-of-experts feed-forward network with a shared expert.

    Attributes:
        top_k: Experts chosen per token.
        router: Hidden to one logit per expert, weight ``[experts, hidden]``.
        experts: The routed SwiGLU experts.
        shared_expert: A SwiGLU network every token goes through.
    """

    def __init__(self, config: Llama4Config) -> None:
        """Create the router and the experts.

        Args:
            config: Model dimensions.
        """

        super().__init__()
        self.top_k = config.num_experts_per_tok
        self.router = aloi.Linear(config.hidden_size, config.num_local_experts)
        # Llama 4 multiplies each expert's input, not its output, by the score.
        self.experts = aloi.MoEExperts(
            config.num_local_experts,
            config.hidden_size,
            config.intermediate_size,
            apply_router_weight_on_input=True,
        )
        self.shared_expert = Llama4MLP(config.hidden_size, config.intermediate_size)

    def forward(self, x: Tensor) -> Tensor:
        """Route each token to its top experts and add the shared expert.

        Args:
            x: ``[batch, seq_len, hidden]``.

        Returns:
            ``[batch, seq_len, hidden]``.
        """

        logits: Tensor = self.router(x)
        # [batch, seq_len, experts]: the sigmoid of the chosen experts' logits,
        # 0 for the others.
        scores = aloi.topk_sigmoid(logits, self.top_k)
        routed: Tensor = self.experts(x, scores)
        shared: Tensor = self.shared_expert(x)
        return routed + shared


class Llama4DecoderLayer(nn.Module):
    """One pre-norm Llama 4 decoder layer with residual connections.

    Attributes:
        input_layernorm: RMSNorm before attention.
        self_attn: Grouped-query self-attention (RoPE or NoPE).
        post_attention_layernorm: RMSNorm before the feed-forward network.
        feed_forward: Mixture-of-experts network.
    """

    def __init__(self, config: Llama4Config, layer: int) -> None:
        """Create the sublayers of decoder layer ``layer``.

        Args:
            config: Model dimensions.
            layer: Decoder layer index.

        Raises:
            NotImplementedError: For a dense (non-MoE) layer. Scout has none;
                Maverick alternates dense and MoE layers.
        """

        super().__init__()
        if layer not in config.moe_layers:
            raise NotImplementedError(f"layer {layer} is dense; only MoE layers are modeled")
        self.input_layernorm = aloi.RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = Llama4Attention(config, layer)
        self.post_attention_layernorm = aloi.RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )
        self.feed_forward = Llama4MoE(config)

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
        ffn_output: Tensor = self.feed_forward(ffn_input)
        return hidden + ffn_output, k_cache, v_cache


def build_meta_decoder_layer(config: Llama4Config, layer: int) -> Llama4DecoderLayer:
    """Create decoder layer ``layer`` with parameters on the meta device.

    Args:
        config: Model dimensions.
        layer: Decoder layer index; selects RoPE or NoPE.

    Returns:
        The layer, with parameters in ``config.dtype``.
    """

    # Every tensor created inside this context, including the parameters
    # made in __init__, is a meta tensor: shape and dtype, but no memory.
    with torch.device("meta"):
        decoder_layer = Llama4DecoderLayer(config, layer)
    # Parameters are created in the default dtype (fp32); convert them to the
    # model dtype afterwards, the same way a checkpoint loader would.
    return decoder_layer.to(config.dtype)


def example_inputs(
    config: Llama4Config,
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
