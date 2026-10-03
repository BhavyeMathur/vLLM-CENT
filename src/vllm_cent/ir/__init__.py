"""ALOI IR: generic SSA core, semantic dialect, printer, and pass infrastructure."""

from .annotate import AnnotateRolesAndAxes, TensorAnnotation
from .base import Function, Module, Operation, Value, clone_function, rebuild_function
from .passes import ClonePass, FunctionPass, Pass, PassManager
from .printer import format_type, print_function, print_module
from .semantic import AllReduceOp, LinearOp, RMSNormOp
from .types import (
    DType,
    Partial,
    Placement,
    Replicate,
    Shard,
    TensorRole,
    TensorType,
    outermost_axis,
    packed_axis,
)

__all__ = [
    "AllReduceOp",
    "AnnotateRolesAndAxes",
    "ClonePass",
    "DType",
    "Function",
    "FunctionPass",
    "LinearOp",
    "Module",
    "Operation",
    "Partial",
    "Pass",
    "PassManager",
    "Placement",
    "RMSNormOp",
    "Replicate",
    "Shard",
    "TensorAnnotation",
    "TensorRole",
    "TensorType",
    "Value",
    "clone_function",
    "format_type",
    "outermost_axis",
    "packed_axis",
    "print_function",
    "print_module",
    "rebuild_function",
]
