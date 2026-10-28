"""Device IR ops computed by processing-in-memory (PIM) units.

Their operands and results live in PIM memory (on CENT, the DRAM banks).
Each op has the type rules of the semantic op it implements.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..semantic import ActivationOp, LinearOp, MulOp
from .domain import _PIMOp

__all__ = [
    "PIMActivationOp",
    "PIMEwmulOp",
    "PIMGemvOp",
]


@dataclass(frozen=True, eq=False, slots=True)
class PIMGemvOp(_PIMOp, LinearOp):
    """``aloi.linear`` as matrix-vector products in the memory banks."""

    NAME = "pim.gemv"


@dataclass(frozen=True, eq=False, slots=True)
class PIMEwmulOp(_PIMOp, MulOp):
    """``aloi.mul`` element by element in the memory banks."""

    NAME = "pim.ewmul"


@dataclass(frozen=True, eq=False, slots=True)
class PIMActivationOp(_PIMOp, ActivationOp):
    """``aloi.activation`` applied by the PIM units.

    On CENT this is the AiM ``AF`` instruction, which reads the function from
    a lookup table written beforehand.
    """

    # TODO(PIM activation functions): AF evaluates whatever lookup table was
    # loaded, so which ActivationKinds it supports, and at what accuracy, is
    # a target property. It belongs in the DeviceSpec (lesson 1.3), and a
    # lowering recipe must check it before choosing this op over
    # pnm.activation.
    NAME = "pim.activation"
