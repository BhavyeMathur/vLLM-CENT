"""ALOI IR: generic SSA core, semantic dialect, printer, and pass infrastructure."""

from .annotate import AnnotateRolesAndAxes, TensorAnnotation
from .base import Function, Module, Operation, Value, clone_function, rebuild_function
from .passes import ClonePass, FunctionPass, Pass, PassManager
from .printer import format_type, print_function, print_module
from .semantic import (
    AddOp,
    AllReduceOp,
    LinearOp,
    MulOp,
    RMSNormOp,
    SiluOp,
    SoftmaxOp,
)
from .transformer import (
    GqaQKOp,
    GqaSVOp,
    KvUpdateOp,
    MergeHeadsOp,
    RopeOp,
    SplitHeadsOp,
)
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
    packed_axis_factors,
)

__all__ = [
    "AddOp",
    "AllReduceOp",
    "AnnotateRolesAndAxes",
    "ClonePass",
    "DType",
    "Function",
    "FunctionPass",
    "GqaQKOp",
    "GqaSVOp",
    "KvUpdateOp",
    "LinearOp",
    "MergeHeadsOp",
    "Module",
    "MulOp",
    "Operation",
    "Partial",
    "Pass",
    "PassManager",
    "Placement",
    "RMSNormOp",
    "Replicate",
    "RopeOp",
    "Shard",
    "SiluOp",
    "SoftmaxOp",
    "SplitHeadsOp",
    "TensorAnnotation",
    "TensorRole",
    "TensorType",
    "Value",
    "clone_function",
    "format_type",
    "outermost_axis",
    "packed_axis",
    "packed_axis_factors",
    "print_function",
    "print_module",
    "rebuild_function",
]
