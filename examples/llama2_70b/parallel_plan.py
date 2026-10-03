"""Tensor-parallel plan for the toy Llama block."""

from __future__ import annotations

from vllm_cent.parallel import ParallelPlan

from .annotations import Q_HEAD
from .config import LlamaConfig

__all__ = ["make_parallel_plan"]


def make_parallel_plan(config: LlamaConfig, tp: int) -> ParallelPlan:
    """Split the attention heads over ``tp`` ranks, Megatron style.

    One rule covers q_proj (its output rows) and, once the model has it,
    o_proj (its input columns); the all_reduce after o_proj is derived.

    The rules grow with the model: a rule whose axis no tensor has is an
    error, so kv_head and intermediate are added together with k_proj/v_proj
    and the feed-forward layers.

    Args:
        config: Model dimensions; supplies the head count.
        tp: Number of tensor-parallel ranks.

    Returns:
        The plan.
    """

    plan = ParallelPlan(tp=tp)
    # q_head is the outer factor of the packed q_head*head_dim dimension, so
    # its size cannot be read from the 8192-long dimension itself.
    plan.split(Q_HEAD, size=config.num_attention_heads)
    return plan
