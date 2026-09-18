"""Public typed instruction interface for CENT programs."""

from .address import (
    CentBankRegisterAddress,
    CentChannelSet,
    CentGlobalBufferAddress,
    CentMemoryAddress,
    CentSharedBufferAddress,
)
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
from .base import CentInstruction, CentOpcode
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
    "CentBankRegisterAddress",
    "CentChannelSet",
    "CentGlobalBufferAddress",
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
