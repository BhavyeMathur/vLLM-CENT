"""Public typed instruction interface for CENT programs."""

from .address import (
    CentChannelSet,
    CentMemoryAddress,
    CentSharedBufferAddress,
)
from .base import CentInstruction, CentOpcode
from .arithmetic import (
    Accumulate,
    ApplyActivation,
    ElementwiseMultiply,
    Exponent,
    MacAllBanks,
    MacOperandSource,
    Reduction,
    RunRiscV,
)
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
from .validation import (
    validate_address,
    validate_channels,
    validate_instruction,
    validate_shared_buffer_address,
)

__all__ = [
    "Accumulate",
    "ApplyActivation",
    "BroadcastCxl",
    "CentChannelSet",
    "CentInstruction",
    "CentMemoryAddress",
    "CentOpcode",
    "CentSharedBufferAddress",
    "CopyBankToGlobalBuffer",
    "CopyGlobalBufferToBank",
    "ElementwiseMultiply",
    "Exponent",
    "MacAllBanks",
    "MacOperandSource",
    "ReadActivation",
    "ReadMac",
    "ReadSingleBank",
    "ReceiveCxl",
    "Reduction",
    "RunRiscV",
    "SendCxl",
    "WriteAllBanks",
    "WriteBias",
    "WriteGlobalBuffer",
    "WriteSingleBank",
    "validate_address",
    "validate_channels",
    "validate_instruction",
    "validate_shared_buffer_address",
]
