"""Device IR ops computed by processing-in-memory (PIM) units.

Their operands and results live in PIM memory (on CENT, the DRAM banks).
Each op has the type rules of the semantic op it implements.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..semantic import ActivationRules, LinearOp, MulOp
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
class PIMActivationOp(_PIMOp, ActivationRules):
    """``aloi.activation`` applied by the PIM units.

    On CENT this is the AiM ``AF`` instruction, which reads the function from
    a lookup table written beforehand.
    """

    # AF evaluates whatever lookup table was loaded, so which functions it
    # supports is a target property: DeviceCapability.pim_activations lists
    # them with their AFid, and LowerToDevice checks it.
    NAME = "pim.activation"
