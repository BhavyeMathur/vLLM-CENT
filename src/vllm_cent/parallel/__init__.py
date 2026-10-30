"""Tensor parallelism: who owns what, and the communication that implies."""

from .plan import ParallelPlan
from .tensor_parallel import ApplyTensorParallel

__all__ = ["ApplyTensorParallel", "ParallelPlan"]
