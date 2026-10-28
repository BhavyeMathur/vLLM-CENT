"""Device IR ops that communicate between devices over CXL.

CXL is a fabric, not a place where tensors are computed. A collective reads
each device's local tensor from PNM memory and writes the combined tensor
back there.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..semantic import AllReduceOp
from .domain import _PNMOp

__all__ = [
    "CXLAllReduceOp",
]


@dataclass(frozen=True, eq=False, slots=True)
class CXLAllReduceOp(_PNMOp, AllReduceOp):
    """``aloi.all_reduce`` over CXL; the partial sums must be in PNM memory."""

    NAME = "cxl.all_reduce"
