"""Semantic roles and axes for the Llama decoder layer's function arguments.

This is the model's half of the Semantic IR contract: it names what every
input dimension means. ``AnnotateRolesAndAxes`` attaches these annotations
and the ops propagate them, so op results need no entry here.

Keys are the argument names produced by the importer: parameter FQNs (the
Hugging Face checkpoint keys of one layer) and the ``forward`` argument
names.
"""

from __future__ import annotations

from vllm_cent.ir import TensorAnnotation, TensorRole, packed_axis

from .config import LlamaConfig

__all__ = [
    "BATCH",
    "CONTEXT",
    "HEAD_DIM",
    "HIDDEN",
    "INTERMEDIATE",
    "KV_HEAD",
    "KV_PROJ_OUT",
    "POSITION",
    "Q_HEAD",
    "Q_PROJ_OUT",
    "ROTARY_DIM",
    "SEQ_LEN",
    "axis_sizes",
    "semantic_annotations",
]

# Axis names. Constants instead of string literals, so a typo is a NameError.
BATCH = "batch"
# Tokens processed in this step (1 when decoding).
SEQ_LEN = "seq_len"
# Positions held by a KV cache. A different axis from seq_len: attention sums
# over the context, never over the new tokens.
CONTEXT = "context"
HIDDEN = "hidden"
Q_HEAD = "q_head"
KV_HEAD = "kv_head"
HEAD_DIM = "head_dim"
INTERMEDIATE = "intermediate"
# Rows and columns of the RoPE cos/sin table: one row per possible position,
# rotary_dim columns (cosines, then sines).
POSITION = "position"
ROTARY_DIM = "rotary_dim"

# A projection's output dimension holds every head, head after head:
# 64 query heads x 128 = 8192 and 8 KV heads x 128 = 1024 for the 70B models.
# Naming both factors lets TP split it into whole heads.
Q_PROJ_OUT = packed_axis(Q_HEAD, HEAD_DIM)
KV_PROJ_OUT = packed_axis(KV_HEAD, HEAD_DIM)


def semantic_annotations() -> dict[str, TensorAnnotation]:
    """Return the annotation of every argument of the decoder layer.

    Returns:
        A new mapping from argument name to annotation.
    """

    activation = TensorRole.ACTIVATION
    weight = TensorRole.WEIGHT
    cache = TensorRole.KV_CACHE
    return {
        # Inputs of one decode step.
        "x": TensorAnnotation(activation, (BATCH, SEQ_LEN, HIDDEN)),
        "positions": TensorAnnotation(activation, (BATCH, SEQ_LEN)),
        "k_cache": TensorAnnotation(cache, (BATCH, CONTEXT, KV_HEAD, HEAD_DIM)),
        "v_cache": TensorAnnotation(cache, (BATCH, CONTEXT, KV_HEAD, HEAD_DIM)),
        # RMSNorm scales each hidden element by its own weight.
        "input_layernorm.weight": TensorAnnotation(weight, (HIDDEN,)),
        "post_attention_layernorm.weight": TensorAnnotation(weight, (HIDDEN,)),
        # RoPE cos/sin table: a constant computed from the config, resident
        # like a weight and needed whole on every rank.
        "self_attn.rotary_emb.cos_sin_cache": TensorAnnotation(
            weight, (POSITION, ROTARY_DIM)
        ),
        # nn.Linear layout [out_features, in_features]. q/k/v read the hidden
        # vector and produce packed heads; o_proj reads the packed query heads
        # and produces the hidden vector.
        "self_attn.q_proj.weight": TensorAnnotation(weight, (Q_PROJ_OUT, HIDDEN)),
        "self_attn.k_proj.weight": TensorAnnotation(weight, (KV_PROJ_OUT, HIDDEN)),
        "self_attn.v_proj.weight": TensorAnnotation(weight, (KV_PROJ_OUT, HIDDEN)),
        "self_attn.o_proj.weight": TensorAnnotation(weight, (HIDDEN, Q_PROJ_OUT)),
        # SwiGLU: gate and up widen hidden to intermediate, down narrows back.
        "mlp.gate_proj.weight": TensorAnnotation(weight, (INTERMEDIATE, HIDDEN)),
        "mlp.up_proj.weight": TensorAnnotation(weight, (INTERMEDIATE, HIDDEN)),
        "mlp.down_proj.weight": TensorAnnotation(weight, (HIDDEN, INTERMEDIATE)),
    }


def axis_sizes(config: LlamaConfig) -> dict[str, int]:
    """Return the sizes TP needs for the packed head axes.

    q_head and kv_head are the outer factors of q_head*head_dim and
    kv_head*head_dim, so their sizes (head counts) cannot be read from the
    dimension length. Plain axes such as intermediate need no entry.

    Args:
        config: Model dimensions.

    Returns:
        Axis name to size.
    """

    return {
        Q_HEAD: config.num_attention_heads,
        KV_HEAD: config.num_key_value_heads,
    }
