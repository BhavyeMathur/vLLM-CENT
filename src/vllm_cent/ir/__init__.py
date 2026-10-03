"""ALOI IR: generic SSA core, semantic dialect, printer, and pass infrastructure."""

from .annotate import AnnotateRolesAndAxes, TensorAnnotation
from .base import Function, Module, Operation, Value, clone_function, rebuild_function
from .passes import ClonePass, FunctionPass, Pass, PassManager
from .printer import format_type, print_function, print_module
from .semantic import LinearOp, RMSNormOp
from .types import DType, TensorRole, TensorType, packed_axis

__all__ = [
    "AnnotateRolesAndAxes",
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
    "TensorAnnotation",
    "TensorRole",
    "TensorType",
    "Value",
    "clone_function",
    "format_type",
    "packed_axis",
    "print_function",
    "print_module",
    "rebuild_function",
]
