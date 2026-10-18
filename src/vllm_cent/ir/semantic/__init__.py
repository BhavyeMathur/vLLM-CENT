"""Semantic IR ops, grouped by the rule that infers their result type.

- ``tensor``: copy and reshape, which move or reinterpret elements.
- ``elementwise``: activations, add and mul, computed element by element.
- ``linalg``: linear, a contraction over a named axis.
- ``normalization``: RMSNorm and softmax, which need a whole last dimension.
- ``collective``: all_reduce, which changes a tensor's placement.
- ``transformer``: ops that only transformer-family models use.
"""

from .collective import AllReduceOp
from .elementwise import ActivationOp, AddOp, MulOp
from .linalg import LinearOp
from .normalization import RMSNormOp, SoftmaxOp
from .tensor import CopyOp, ReshapeOp

__all__ = [
    "ActivationOp",
    "AddOp",
    "AllReduceOp",
    "CopyOp",
    "LinearOp",
    "MulOp",
    "RMSNormOp",
    "ReshapeOp",
    "SoftmaxOp",
]
