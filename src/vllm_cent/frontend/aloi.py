"""ALOI ops and layers for writing models that torch.export keeps whole.

Two levels, following PyTorch's ``F.linear`` / ``nn.Linear`` split:

- Ops (lowercase), e.g. ``aloi.linear(x, weight)``: ``aloi::*`` custom ops.
  The op namespace is also ``aloi``, so the exported graph shows
  ``aloi.linear.default``. torch.export records each call as one node, while
  built-in layers may be decomposed into ``mm``/``pow``/``mean``.
- Layers (capitalized), e.g. ``aloi.Linear(in_features, out_features)``:
  nn.Modules that own the weights and call one op in ``forward``. A model uses
  them like ``nn.Linear``; parameter names stay ``q_proj.weight``.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping

import torch
from torch import Tensor, nn
import torch.nn.functional as F


@torch.library.custom_op(
    "aloi::rms_norm",
    mutates_args=(),
)
def rms_norm(x: Tensor, weight: Tensor, eps: float | None = None) -> Tensor:
    return F.rms_norm(input=x, normalized_shape=(x.shape[-1],), weight=weight, eps=eps)


@rms_norm.register_fake
def _(x: Tensor, weight: Tensor, eps: float | None = None) -> Tensor:
    result = torch.empty_like(x)
    return result


@torch.library.custom_op(
    "aloi::linear",
    mutates_args=(),
)
def linear(x: Tensor, weight: Tensor) -> Tensor:
    return F.linear(x, weight)


@linear.register_fake
def _(x: Tensor, weight: Tensor) -> Tensor:
    if x.shape[-1] != weight.shape[-1]:
        raise ValueError("Weight should have shape [out_features, in_features]")
    return x.new_empty((*x.shape[:-1], weight.shape[0]))


# ---------------------------------------------------------------------------
# Transformer ops. Every real kernel computes in fp32 and casts the result back
# to the input dtype, so fp16/bf16 inputs lose precision only once.
#
# Layouts are token-first: activations [batch, tokens, heads, head_dim], KV
# caches [batch, context, kv_heads, head_dim], scores [batch, tokens, heads,
# context]. With group = heads // kv_heads, query head h reads KV head
# h // group (grouped-query attention).
# ---------------------------------------------------------------------------


@torch.library.custom_op("aloi::split_heads", mutates_args=())
def split_heads(x: Tensor, head_dim: int) -> Tensor:
    """Reshape ``[*, heads * head_dim]`` to ``[*, heads, head_dim]``."""

    # clone(): a custom op's output must not alias its input.
    return x.reshape(*x.shape[:-1], -1, head_dim).clone()


@split_heads.register_fake
def _(x: Tensor, head_dim: int) -> Tensor:
    """Return the split shape after checking head_dim divides it."""

    if x.shape[-1] % head_dim != 0:
        raise ValueError(f"head_dim {head_dim} does not divide {x.shape[-1]}")
    return x.new_empty((*x.shape[:-1], x.shape[-1] // head_dim, head_dim))


@torch.library.custom_op("aloi::merge_heads", mutates_args=())
def merge_heads(x: Tensor) -> Tensor:
    """Reshape ``[*, heads, head_dim]`` to ``[*, heads * head_dim]``."""

    return x.reshape(*x.shape[:-2], -1).clone()


@merge_heads.register_fake
def _(x: Tensor) -> Tensor:
    """Return the merged shape."""

    return x.new_empty((*x.shape[:-2], x.shape[-2] * x.shape[-1]))


# ---------------------------------------------------------------------------
# Rotary position embedding, vLLM style.
#
# Two parts, as in vLLM and Hugging Face:
# 1. Frequencies depend on the model's RoPE variant ("rope_type": default,
#    llama3, ...). RotaryEmbedding turns them into a cos/sin table once, at
#    construction: cos_sin_cache[p] = [cos(p * f), sin(p * f)].
# 2. The rotation is the same for every model; it only needs the table, the
#    positions, and the pairing style. That is the aloi::rope op.
# A table lookup suits CENT: rotating becomes elementwise multiply-adds with
# vectors read from DRAM, with no sin/cos evaluated at run time.
# ---------------------------------------------------------------------------

# Rope scaling settings, as in a Hugging Face config.json ``rope_scaling``:
# ``{"rope_type": "llama3", "factor": 8.0, ...}``. The keys depend on the type.
RopeScaling = Mapping[str, float | int | str]


def _default_frequencies(
    rotary_dim: int, base: float, scaling: RopeScaling
) -> tuple[Tensor, float]:
    """Plain RoPE: ``inv_freq[i] = base ** (-2i / rotary_dim)``.

    Args:
        rotary_dim: Number of rotated elements per head.
        base: Frequency base ("rope_theta").
        scaling: Unused.

    Returns:
        ``(inv_freq [rotary_dim // 2] fp32, attention scaling 1.0)``.
    """

    del scaling
    exponents = torch.arange(0, rotary_dim, 2, dtype=torch.float32) / rotary_dim
    return 1.0 / (base**exponents), 1.0


def _llama3_frequencies(
    rotary_dim: int, base: float, scaling: RopeScaling
) -> tuple[Tensor, float]:
    """Llama 3.1 "llama3" scaling of the default frequencies.

    Long wavelengths (low frequencies) are slowed down by ``factor``, short
    ones are kept, and the band in between is blended. Follows Hugging Face's
    ``_compute_llama3_parameters``.

    Args:
        rotary_dim: Number of rotated elements per head.
        base: Frequency base ("rope_theta").
        scaling: ``factor``, ``low_freq_factor``, ``high_freq_factor`` and
            ``original_max_position_embeddings``.

    Returns:
        ``(inv_freq [rotary_dim // 2] fp32, attention scaling 1.0)``.
    """

    inv_freq, _ = _default_frequencies(rotary_dim, base, scaling)
    factor = float(scaling["factor"])
    low_freq_factor = float(scaling["low_freq_factor"])
    high_freq_factor = float(scaling["high_freq_factor"])
    original_max_position = float(scaling["original_max_position_embeddings"])

    wavelength = 2 * math.pi / inv_freq
    low_freq_wavelength = original_max_position / low_freq_factor
    high_freq_wavelength = original_max_position / high_freq_factor
    scaled = torch.where(wavelength > low_freq_wavelength, inv_freq / factor, inv_freq)
    smooth = (original_max_position / wavelength - low_freq_factor) / (
        high_freq_factor - low_freq_factor
    )
    blended = (1 - smooth) * scaled / factor + smooth * scaled
    is_medium = (wavelength >= high_freq_wavelength) & (
        wavelength <= low_freq_wavelength
    )
    return torch.where(is_medium, blended, scaled), 1.0


# rope_type -> frequency function. Adding a RoPE variant (linear, yarn, ...)
# means adding one function here; the op and the IR do not change. The
# attention scaling is YaRN's "mscale" factor on cos/sin; 1.0 for the others.
_ROPE_FREQUENCY_FUNCTIONS: dict[
    str, Callable[[int, float, RopeScaling], tuple[Tensor, float]]
] = {
    "default": _default_frequencies,
    "llama3": _llama3_frequencies,
}


def rope_inverse_frequencies(
    rotary_dim: int, base: float, rope_scaling: RopeScaling | None = None
) -> tuple[Tensor, float]:
    """Return the rotary frequencies of a RoPE variant.

    Args:
        rotary_dim: Number of rotated elements per head.
        base: Frequency base ("rope_theta").
        rope_scaling: Hugging Face ``rope_scaling`` settings, or None for
            plain RoPE. The variant is ``rope_type`` (or the older ``type``).

    Returns:
        ``(inv_freq [rotary_dim // 2] fp32, attention scaling)``.

    Raises:
        NotImplementedError: For a RoPE variant without a function above.
    """

    scaling: RopeScaling = rope_scaling or {}
    rope_type = str(scaling.get("rope_type", scaling.get("type", "default")))
    function = _ROPE_FREQUENCY_FUNCTIONS.get(rope_type)
    if function is None:
        raise NotImplementedError(
            f"rope_type {rope_type!r}; supported: {sorted(_ROPE_FREQUENCY_FUNCTIONS)}"
        )
    return function(rotary_dim, base, scaling)


@torch.library.custom_op("aloi::rope", mutates_args=())
def rope(
    x: Tensor, positions: Tensor, cos_sin_cache: Tensor, is_neox_style: bool = True
) -> Tensor:
    """Rotate ``x [*tokens, heads, head_dim]`` by its tokens' positions.

    ``cos_sin_cache[p]`` holds ``rotary_dim // 2`` cosines then as many sines
    for position p. The first ``rotary_dim`` elements of each head are
    rotated; the rest pass through (partial rotary, e.g. GPT-NeoX). Pairs:

    - NeoX style (Llama, Qwen, Mistral in Hugging Face/vLLM): element i pairs
      with element i + rotary_dim/2 (``rotate_half``).
    - GPT-J style (Meta's original Llama code, GPT-J, ChatGLM): elements
      2i and 2i+1 pair up.
    """

    rotary_dim = cos_sin_cache.shape[-1]
    cos, sin = cos_sin_cache[positions].to(torch.float32).chunk(2, dim=-1)
    # [*tokens, rotary_dim // 2] -> broadcast over the heads dimension.
    cos, sin = cos[..., None, :], sin[..., None, :]
    x32 = x.to(torch.float32)
    rotary, passthrough = x32[..., :rotary_dim], x32[..., rotary_dim:]
    if is_neox_style:
        first, second = rotary.chunk(2, dim=-1)
    else:
        first, second = rotary[..., 0::2], rotary[..., 1::2]
    new_first = first * cos - second * sin
    new_second = second * cos + first * sin
    if is_neox_style:
        rotated = torch.cat((new_first, new_second), dim=-1)
    else:
        rotated = torch.stack((new_first, new_second), dim=-1).flatten(-2)
    return torch.cat((rotated, passthrough), dim=-1).to(x.dtype)


@rope.register_fake
def _(
    x: Tensor, positions: Tensor, cos_sin_cache: Tensor, is_neox_style: bool = True
) -> Tensor:
    """Return x's shape after checking positions and the table."""

    if tuple(positions.shape) != tuple(x.shape[:-2]):
        raise ValueError(f"positions {positions.shape} do not match x {x.shape}")
    rotary_dim = cos_sin_cache.shape[-1]
    if cos_sin_cache.dim() != 2 or rotary_dim % 2 or rotary_dim > x.shape[-1]:
        raise ValueError(
            f"cos_sin_cache {cos_sin_cache.shape} does not fit head_dim {x.shape[-1]}"
        )
    return torch.empty_like(x)


@torch.library.custom_op("aloi::kv_update", mutates_args=())
def kv_update(cache: Tensor, new: Tensor, positions: Tensor) -> Tensor:
    """Return a copy of ``cache`` with ``new[b, t]`` at ``positions[b, t]``."""

    updated = cache.clone()
    batch_index = torch.arange(cache.shape[0], device=cache.device)[:, None]
    updated[batch_index, positions] = new
    return updated


@kv_update.register_fake
def _(cache: Tensor, new: Tensor, positions: Tensor) -> Tensor:
    """Return the cache's shape after checking new fits it."""

    if (cache.shape[0], *cache.shape[2:]) != (new.shape[0], *new.shape[2:]):
        raise ValueError(f"new {new.shape} does not fit cache {cache.shape}")
    return torch.empty_like(cache)


def _repeat_kv_heads(cache: Tensor, query_heads: int) -> Tensor:
    """Repeat each KV head for its group of query heads, in fp32.

    Args:
        cache: ``[batch, context, kv_heads, head_dim]``.
        query_heads: Number of query heads.

    Returns:
        ``[batch, context, query_heads, head_dim]``.
    """

    group = query_heads // cache.shape[2]
    return cache.to(torch.float32).repeat_interleave(group, dim=2)


@torch.library.custom_op("aloi::gqa_qk", mutates_args=())
def gqa_qk(q: Tensor, k_cache: Tensor, scale: float) -> Tensor:
    """Scores ``[batch, tokens, heads, context]`` of q against the cache."""

    keys = _repeat_kv_heads(k_cache, q.shape[2])
    scores = torch.einsum("bthd,bchd->bthc", q.to(torch.float32), keys)
    return (scores * scale).to(q.dtype)


@gqa_qk.register_fake
def _(q: Tensor, k_cache: Tensor, scale: float) -> Tensor:
    """Return the scores shape after checking the heads group."""

    if q.shape[2] % k_cache.shape[2] != 0:
        raise ValueError(f"{q.shape[2]} heads do not group over {k_cache.shape[2]}")
    return q.new_empty((q.shape[0], q.shape[1], q.shape[2], k_cache.shape[1]))


@torch.library.custom_op("aloi::softmax", mutates_args=())
def softmax(x: Tensor) -> Tensor:
    """Softmax over the last dimension."""

    return torch.softmax(x.to(torch.float32), dim=-1).to(x.dtype)


@softmax.register_fake
def _(x: Tensor) -> Tensor:
    """Return x's shape."""

    return torch.empty_like(x)


@torch.library.custom_op("aloi::gqa_sv", mutates_args=())
def gqa_sv(probs: Tensor, v_cache: Tensor) -> Tensor:
    """Probability-weighted sum of cached values, ``[batch, tokens, heads, head_dim]``."""

    values = _repeat_kv_heads(v_cache, probs.shape[2])
    context = torch.einsum("bthc,bchd->bthd", probs.to(torch.float32), values)
    return context.to(probs.dtype)


@gqa_sv.register_fake
def _(probs: Tensor, v_cache: Tensor) -> Tensor:
    """Return the output shape after checking the heads group."""

    if probs.shape[2] % v_cache.shape[2] != 0:
        raise ValueError(f"{probs.shape[2]} heads do not group over {v_cache.shape[2]}")
    return probs.new_empty((*probs.shape[:3], v_cache.shape[3]))


# ---------------------------------------------------------------------------
# Layers: nn.Modules that own weights and call one ALOI op. Calling a custom op
# is typed as returning Any, so each forward annotates its result as Tensor.
# ---------------------------------------------------------------------------


class Linear(nn.Module):
    """Bias-free linear layer computing ``aloi.linear(x, weight)``.

    Llama projections have no bias, so none is modeled.

    Attributes:
        weight: ``[out_features, in_features]``, the ``nn.Linear`` layout; the
            layer computes ``x @ weight.T``.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Create the weight.

        Args:
            in_features: Length of each input vector.
            out_features: Length of each output vector.
        """

        super().__init__()
        # Uninitialized, like nn.Linear before reset_parameters. On the meta
        # device there are no values at all; a loader or test fills them.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``x @ weight.T``.

        Args:
            x: ``[*, in_features]``.

        Returns:
            ``[*, out_features]``.
        """

        projected: Tensor = linear(x, self.weight)
        return projected


class RMSNorm(nn.Module):
    """RMSNorm over the last dimension, computing ``aloi.rms_norm``.

    Attributes:
        weight: Scale for each feature, ``[hidden_size]``.
        eps: Constant added to the mean square before the square root.
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        """Create the weight.

        Args:
            hidden_size: Length of the normalized last dimension.
            eps: Constant added to the mean square before the square root.
        """

        super().__init__()
        # Same initial value as nn.RMSNorm: every feature is scaled by 1.
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` and scale it by ``weight``.

        Args:
            x: ``[*, hidden_size]``.

        Returns:
            Same shape and dtype as ``x``.
        """

        normalized: Tensor = rms_norm(x, self.weight, self.eps)
        return normalized


class RotaryEmbedding(nn.Module):
    """Rotary position embedding with a precomputed cos/sin table.

    Matches vLLM's ``get_rope``: the table covers every position up to
    ``max_position`` and is built once; ``forward`` rotates q and k by
    looking up their positions.

    Attributes:
        rotary_dim: Number of rotated elements per head.
        is_neox_style: Pairing style; see ``aloi.rope``.
        cos_sin_cache: ``[max_position, rotary_dim]`` buffer: cosines, then
            sines. Non-persistent, like vLLM's: it is derived from the
            config, not loaded from a checkpoint.
    """

    # Declared for type checkers: register_buffer sets the attribute at run
    # time, and nn.Module.__getattr__ would otherwise type it as Tensor | Module.
    cos_sin_cache: Tensor

    def __init__(
        self,
        head_dim: int,
        max_position: int,
        base: float,
        rope_scaling: RopeScaling | None = None,
        rotary_dim: int | None = None,
        is_neox_style: bool = True,
    ) -> None:
        """Build the table.

        Args:
            head_dim: Length of one head.
            max_position: Number of positions the table covers.
            base: Frequency base ("rope_theta").
            rope_scaling: Hugging Face ``rope_scaling`` settings, or None.
            rotary_dim: Rotated elements per head; all of them by default.
            is_neox_style: Pairing style; see ``aloi.rope``.

        Raises:
            ValueError: If rotary_dim is odd or larger than head_dim.
        """

        super().__init__()
        self.rotary_dim = head_dim if rotary_dim is None else rotary_dim
        if self.rotary_dim % 2 or self.rotary_dim > head_dim:
            raise ValueError(
                f"rotary_dim {self.rotary_dim} must be even and at most {head_dim}"
            )
        self.is_neox_style = is_neox_style
        inv_freq, attention_scaling = rope_inverse_frequencies(
            self.rotary_dim, base, rope_scaling
        )
        angles = torch.outer(torch.arange(max_position, dtype=torch.float32), inv_freq)
        table = torch.cat((angles.cos(), angles.sin()), dim=-1) * attention_scaling
        self.register_buffer("cos_sin_cache", table, persistent=False)

    def forward(self, positions: Tensor, q: Tensor, k: Tensor) -> tuple[Tensor, Tensor]:
        """Rotate queries and keys by their tokens' positions.

        Args:
            positions: ``[batch, seq_len]`` token positions.
            q: ``[batch, seq_len, heads, head_dim]``.
            k: ``[batch, seq_len, kv_heads, head_dim]``.

        Returns:
            The rotated ``(q, k)``.
        """

        table: Tensor = self.cos_sin_cache
        rotated_q: Tensor = rope(q, positions, table, self.is_neox_style)
        rotated_k: Tensor = rope(k, positions, table, self.is_neox_style)
        return rotated_q, rotated_k
