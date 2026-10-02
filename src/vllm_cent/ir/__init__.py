"""IR interfaces."""

from .base import Value, Operation, Function, Module
from .types import DType, TensorRole, TensorType

__all__ = ["Value", "Operation", "Function", "Module"]
