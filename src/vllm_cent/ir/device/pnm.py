"""Device IR ops computed by processing-near-memory (PNM) units.

Their operands and results live in PNM memory (on CENT, the Shared Buffer
read by the accumulators, the exponent unit and the RISC-V cores). Each op
has the type rules of the semantic op it implements.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..semantic import (
    ActivationRules,
    AddOp,
    DivOp,
    ExpOp,
    MaxOp,
    MulOp,
    RMSNormRules,
    SoftmaxRules,
    SubOp,
    SumOp,
)
from .domain import _PNMOp

__all__ = [
    "PNMActivationOp",
    "PNMAddOp",
    "PNMDivOp",
    "PNMExpOp",
    "PNMMaxOp",
    "PNMMulOp",
    "PNMRMSNormOp",
    "PNMSoftmaxOp",
    "PNMSubOp",
    "PNMSumOp",
]


@dataclass(frozen=True, eq=False, slots=True)
class PNMAddOp(_PNMOp, AddOp):
    """``aloi.add`` element by element, e.g. a residual connection."""

    NAME = "pnm.add"


@dataclass(frozen=True, eq=False, slots=True)
class PNMSubOp(_PNMOp, SubOp):
    """``aloi.sub`` element by element."""

    NAME = "pnm.sub"


@dataclass(frozen=True, eq=False, slots=True)
class PNMMulOp(_PNMOp, MulOp):
    """``aloi.mul`` element by element."""

    NAME = "pnm.mul"


@dataclass(frozen=True, eq=False, slots=True)
class PNMDivOp(_PNMOp, DivOp):
    """``aloi.div`` element by element."""

    NAME = "pnm.div"


@dataclass(frozen=True, eq=False, slots=True)
class PNMExpOp(_PNMOp, ExpOp):
    """``aloi.exp`` element by element."""

    NAME = "pnm.exp"


@dataclass(frozen=True, eq=False, slots=True)
class PNMMaxOp(_PNMOp, MaxOp):
    """``aloi.reduce_max``: the largest element of each row."""

    NAME = "pnm.reduce_max"


@dataclass(frozen=True, eq=False, slots=True)
class PNMSumOp(_PNMOp, SumOp):
    """``aloi.reduce_sum``: the sum of each row."""

    NAME = "pnm.reduce_sum"


@dataclass(frozen=True, eq=False, slots=True)
class PNMActivationOp(_PNMOp, ActivationRules):
    """``aloi.activation`` computed near memory, e.g. by the RISC-V cores."""

    NAME = "pnm.activation"


@dataclass(frozen=True, eq=False, slots=True)
class PNMSoftmaxOp(_PNMOp, SoftmaxRules):
    """``aloi.softmax`` as one PNM operation, for a target with a unit for it.

    A target without such a unit lowers ``aloi.softmax`` through its
    decomposition (reduce_max, sub, exp, reduce_sum, div) instead. This op is
    deliberately not ``Decomposable``: its parts would be semantic ops, and a
    device op must not turn back into Semantic IR.
    """

    # TODO(PNM softmax unit): The CENT paper describes PNM exponent units and
    # RISC-V cores; whether a target computes a whole softmax in one PNM
    # operation (a special function unit or a RISC-V routine) is a target
    # property for the DeviceSpec. Until it is recorded there, a recipe must
    # not assume this op is available.
    NAME = "pnm.softmax"


@dataclass(frozen=True, eq=False, slots=True)
class PNMRMSNormOp(_PNMOp, RMSNormRules):
    """``aloi.rms_norm`` as one PNM operation: x, and the weight if any, in PNM.

    Like ``pnm.softmax``, it shares the semantic op's rules without its
    future decomposition, so it never turns back into Semantic IR.
    """

    # TODO(RMSNorm on CENT): The existing CENT lowering computes the sum of
    # squares with MAC_ABK in the banks and only the reciprocal root near
    # memory, so a target may prefer a PIM/PNM split of this op over one PNM
    # operation. Which one it supports is a DeviceSpec property; the split
    # waits for the RMSNorm decomposition (see RMSNormOp).
    NAME = "pnm.rms_norm"
