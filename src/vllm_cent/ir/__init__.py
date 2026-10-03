"""ALOI IR: generic SSA core, semantic dialect, printer, and pass infrastructure."""

from .base import Function, Module, Operation, Value, clone_function
from .passes import ClonePass, FunctionPass, Pass, PassManager
from .printer import format_type, print_function, print_module
from .semantic import LinearOp, RMSNormOp
from .types import DType, TensorRole, TensorType

__all__ = [
    "ClonePass",
    "DType",
    "Function",
    "FunctionPass",
    "LinearOp",
    "Module",
    "Operation",
    "Pass",
    "PassManager",
    "RMSNormOp",
    "TensorRole",
    "TensorType",
    "Value",
    "clone_function",
    "format_type",
    "print_function",
    "print_module",
]
