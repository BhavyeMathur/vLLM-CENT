"""Semantic roles and axes for the Llama 4 decoder layer's function arguments.

This is the model's half of the Semantic IR contract: it names what every
input dimension means. ``AnnotateRolesAndAxes`` attaches these annotations
and the ops propagate them, so op results need no entry here.

Keys are the argument names produced by the importer: parameter FQNs (the
checkpoint keys of one layer, except the routed experts' weights, see
``aloi.MoEExperts``) and the ``forward`` argument names. RoPE and NoPE layers
differ only in the rotary table, which NoPE layers do not have.
"""

from __future__ import annotations

from vllm_cent.ir import TensorAnnotation, TensorRole, packed_axis

from .config import Llama4Config

__all__ = [
    "BATCH",
    "CONTEXT",
    "EXPERT",
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
# Positions held by a KV cache; attention sums over them.
CONTEXT = "context"
HIDDEN = "hidden"
Q_HEAD = "q_head"
KV_HEAD = "kv_head"
HEAD_DIM = "head_dim"
# Hidden width of the routed experts and the shared expert. One axis for
# both, so one TP rule splits every expert the same way.
INTERMEDIATE = "intermediate"
# One entry per routed expert: router logits and stacked expert weights.
EXPERT = "expert"
# Rows and columns of the RoPE cos/sin table.
POSITION = "position"
ROTARY_DIM = "rotary_dim"

# A projection's output dimension holds every head, head after head:
# 40 query heads x 128 = 5120 and 8 KV heads x 128 = 1024 for Scout.
Q_PROJ_OUT = packed_axis(Q_HEAD, HEAD_DIM)
KV_PROJ_OUT = packed_axis(KV_HEAD, HEAD_DIM)


def semantic_annotations(config: Llama4Config, layer: int) -> dict[str, TensorAnnotation]:
    """Return the annotation of every argument of decoder layer ``layer``.

    Args:
        config: Model dimensions; tells RoPE layers from NoPE layers.
        layer: Decoder layer index.

    Returns:
        A new mapping from argument name to annotation.
    """

    activation = TensorRole.ACTIVATION
    weight = TensorRole.WEIGHT
    cache = TensorRole.KV_CACHE
    annotations = {
        # Inputs of one decode step.
        "x": TensorAnnotation(activation, (BATCH, SEQ_LEN, HIDDEN)),
        "positions": TensorAnnotation(activation, (BATCH, SEQ_LEN)),
        "k_cache": TensorAnnotation(cache, (BATCH, CONTEXT, KV_HEAD, HEAD_DIM)),
        "v_cache": TensorAnnotation(cache, (BATCH, CONTEXT, KV_HEAD, HEAD_DIM)),
        # RMSNorm scales each hidden element by its own weight. The q/k norm
        # has no weight, so it has no entry.
        "input_layernorm.weight": TensorAnnotation(weight, (HIDDEN,)),
        "post_attention_layernorm.weight": TensorAnnotation(weight, (HIDDEN,)),
        # nn.Linear layout [out_features, in_features].
        "self_attn.q_proj.weight": TensorAnnotation(weight, (Q_PROJ_OUT, HIDDEN)),
        "self_attn.k_proj.weight": TensorAnnotation(weight, (KV_PROJ_OUT, HIDDEN)),
        "self_attn.v_proj.weight": TensorAnnotation(weight, (KV_PROJ_OUT, HIDDEN)),
        "self_attn.o_proj.weight": TensorAnnotation(weight, (HIDDEN, Q_PROJ_OUT)),
        # The router gives one logit per expert.
        "feed_forward.router.weight": TensorAnnotation(weight, (EXPERT, HIDDEN)),
        # Routed experts: one nn.Linear-layout matrix per expert, stacked.
        "feed_forward.experts.gate_proj": TensorAnnotation(
            weight, (EXPERT, INTERMEDIATE, HIDDEN)
        ),
        "feed_forward.experts.up_proj": TensorAnnotation(
            weight, (EXPERT, INTERMEDIATE, HIDDEN)
        ),
        "feed_forward.experts.down_proj": TensorAnnotation(
            weight, (EXPERT, HIDDEN, INTERMEDIATE)
        ),
        # The shared expert is an ordinary SwiGLU MLP.
        "feed_forward.shared_expert.gate_proj.weight": TensorAnnotation(
            weight, (INTERMEDIATE, HIDDEN)
        ),
        "feed_forward.shared_expert.up_proj.weight": TensorAnnotation(
            weight, (INTERMEDIATE, HIDDEN)
        ),
        "feed_forward.shared_expert.down_proj.weight": TensorAnnotation(
            weight, (HIDDEN, INTERMEDIATE)
        ),
    }
    if config.uses_rope(layer):
        # RoPE cos/sin table: a constant computed from the config, resident
        # like a weight and needed whole on every rank.
        annotations["self_attn.rotary_emb.cos_sin_cache"] = TensorAnnotation(
            weight, (POSITION, ROTARY_DIM)
        )
    return annotations


def axis_sizes(config: Llama4Config) -> dict[str, int]:
    """Return the sizes TP needs for the packed head axes.

    q_head and kv_head are the outer factors of q_head*head_dim and
    kv_head*head_dim, so their sizes (head counts) cannot be read from the
    dimension length.

    Args:
        config: Model dimensions.

    Returns:
        Axis name to size.
    """

    return {
        Q_HEAD: config.num_attention_heads,
        KV_HEAD: config.num_key_value_heads,
    }
