"""Custom operations for ALOI"""

from __future__ import annotations

import torch
from torch import Tensor
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
