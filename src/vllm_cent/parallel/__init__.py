"""Tensor parallelism: the plan of who owns what.

The pass that applies a plan, deriving the communication it implies, is
``vllm_cent.ir.passes.ApplyTensorParallel``.
"""

from .plan import ParallelPlan

__all__ = ["ParallelPlan"]
