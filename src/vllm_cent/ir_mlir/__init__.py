"""Expose optional MLIR import, export, and standard optimization passes.

The bridge handles tensor addition and multiplication using native ``func``
and ``arith`` operations.
Importing this package does not load IREE; calling a bridge function does.
"""

from .bridge import from_mlir, optimize_mlir, to_mlir

__all__ = ["from_mlir", "optimize_mlir", "to_mlir"]
