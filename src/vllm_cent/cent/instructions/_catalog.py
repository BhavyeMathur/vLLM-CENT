"""List the concrete instruction types that form the typed CENT IR."""

from .arithmetic import (
    Accumulate,
    ApplyActivation,
    ElementwiseMultiply,
    Exponent,
    MacAllBanks,
    Reduction,
    RunRiscV,
)
from .base import CentInstruction
from .data_movement import (
    BroadcastCxl,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ReadActivation,
    ReadMac,
    ReadSingleBank,
    ReceiveCxl,
    SendCxl,
    WriteAllBanks,
    WriteBias,
    WriteGlobalBuffer,
    WriteSingleBank,
)

__all__: list[str] = []

# Dispatch consumers deliberately have different support matrices. This catalog
# supplies one authoritative inventory for completeness tests without making
# validation, rendering, AiM serialization, and functional execution share one
# behavior object.
_CENT_INSTRUCTION_TYPES: tuple[type[CentInstruction], ...] = (
    MacAllBanks,
    ElementwiseMultiply,
    ApplyActivation,
    Exponent,
    Reduction,
    Accumulate,
    RunRiscV,
    SendCxl,
    ReceiveCxl,
    BroadcastCxl,
    WriteSingleBank,
    ReadSingleBank,
    WriteAllBanks,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    WriteBias,
    ReadMac,
    ReadActivation,
    WriteGlobalBuffer,
)
