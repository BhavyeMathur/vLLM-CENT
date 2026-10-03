"""Semantic roles and axes for the toy Llama block's function arguments.

This is the model's half of the Semantic IR contract: it names what every
input dimension means. ``AnnotateRolesAndAxes`` attaches these annotations
and the ops propagate them, so op results need no entry here.

Keys are the argument names produced by the importer: parameter FQNs and the
``forward`` argument names.
"""

from __future__ import annotations

from vllm_cent.ir import TensorAnnotation, TensorRole, packed_axis

__all__ = [
    "BATCH",
    "HEAD_DIM",
    "HIDDEN",
    "Q_HEAD",
    "Q_PROJ_OUT",
    "SEQ_LEN",
    "semantic_annotations",
]

# Axis names. Constants instead of string literals, so a typo is a NameError.
BATCH = "batch"
SEQ_LEN = "seq_len"
HIDDEN = "hidden"
Q_HEAD = "q_head"
HEAD_DIM = "head_dim"

# q_proj's output dimension holds every query head, head after head:
# 64 heads x 128 elements = 8192 for Llama2-70B. Naming both factors lets TP
# shard it along q_head in whole heads.
Q_PROJ_OUT = packed_axis(Q_HEAD, HEAD_DIM)


def semantic_annotations() -> dict[str, TensorAnnotation]:
    """Return the annotation of every argument of the toy block.

    Returns:
        A new mapping from argument name to annotation.
    """

    return {
        # The model input: one hidden vector per token per request.
        "x": TensorAnnotation(TensorRole.ACTIVATION, (BATCH, SEQ_LEN, HIDDEN)),
        # RMSNorm scales each hidden element by its own weight.
        "input_layernorm.weight": TensorAnnotation(TensorRole.WEIGHT, (HIDDEN,)),
        # nn.Linear layout [out_features, in_features]: rows produce the
        # packed query heads, columns read the hidden vector.
        "q_proj.weight": TensorAnnotation(TensorRole.WEIGHT, (Q_PROJ_OUT, HIDDEN)),
    }
