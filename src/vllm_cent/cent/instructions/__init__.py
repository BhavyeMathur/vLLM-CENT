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
    Reduction,
    RunRiscV,
)
from .data_movement import (
    BroadcastCxl,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
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
