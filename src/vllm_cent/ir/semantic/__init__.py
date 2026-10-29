"""Semantic IR ops, grouped by the rule that infers their result type.

- ``tensor``: copy and reshape, which move or reinterpret elements.
- ``elementwise``: activations, exp, add, mul and div, element by element.
- ``arith``: sum and max of each row (the last dimension).
- ``linalg``: linear, a contraction over a named axis.
- ``normalization``: RMSNorm and softmax, which need a whole last dimension.
- ``collective``: all_reduce, which changes a tensor's placement.
- ``transformer``: ops that only transformer-family models use.
"""

from .arith import MaxOp, SumOp
from .collective import AllReduceOp
from .elementwise import (
    ActivationOp,
    ActivationRules,
    AddOp,
    DivOp,
    ExpOp,
    MulOp,
    SubOp,
)
from .linalg import LinearOp
from .normalization import RMSNormOp, RMSNormRules, SoftmaxOp, SoftmaxRules
from .tensor import BroadcastOp, CopyOp, ReshapeOp

__all__ = [
    "ActivationOp",
    "ActivationRules",
    "AddOp",
    "AllReduceOp",
    "BroadcastOp",
    "CopyOp",
    "DivOp",
    "ExpOp",
    "LinearOp",
    "MaxOp",
    "MulOp",
    "RMSNormOp",
    "RMSNormRules",
    "ReshapeOp",
    "SoftmaxOp",
    "SoftmaxRules",
    "SubOp",
    "SumOp",
]
