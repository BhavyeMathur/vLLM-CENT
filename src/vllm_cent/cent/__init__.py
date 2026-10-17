"""Public hardware, instruction, program-building, and text-rendering APIs.

These types describe device storage and commands without choosing a model
family. Layout selection and operation lowering live outside this package.
"""

from .aim import (
    AIM_SIMULATOR_BANKS,
    AIM_SIMULATOR_BURST_LENGTH,
    AIM_SIMULATOR_CHANNELS,
    AimSimulatorCompatibilityError,
    render_aim_channel_mask,
    render_aim_instruction,
    render_aim_trace,
    validate_aim_hardware,
)
from .builder import CentProgramBuilder
from .hardware import BANKS_PER_GROUP, CentBlockPlacementSpec, CentHardwareSpec
from .instructions import (
    Accumulate,
    ApplyActivation,
    BroadcastCxl,
    CentBankRegisterAddress,
    CentChannelSet,
    CentGlobalBufferAddress,
    CentInstruction,
    CentMemoryAddress,
    CentOpcode,
    CentSharedBufferAddress,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    Exponent,
    MacAllBanks,
    MacOperandSource,
    ReadActivation,
    ReadMac,
    ReadSingleBank,
    ReceiveCxl,
    Reduction,
    RunRiscV,
    SendCxl,
    WriteAllBanks,
    WriteBias,
    WriteGlobalBuffer,
    WriteSingleBank,
)
from .program import CentProgram
from .render import render_instruction, render_text_program
from .utils import ceil_div

__all__ = [
    "AIM_SIMULATOR_BANKS",
    "AIM_SIMULATOR_BURST_LENGTH",
    "AIM_SIMULATOR_CHANNELS",
    "BANKS_PER_GROUP",
    "Accumulate",
    "AimSimulatorCompatibilityError",
    "ApplyActivation",
    "BroadcastCxl",
    "CentBankRegisterAddress",
    "CentBlockPlacementSpec",
    "CentChannelSet",
    "CentGlobalBufferAddress",
    "CentHardwareSpec",
    "CentInstruction",
    "CentMemoryAddress",
    "CentOpcode",
    "CentProgram",
    "CentProgramBuilder",
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
    "ceil_div",
    "render_aim_channel_mask",
    "render_aim_instruction",
    "render_aim_trace",
    "render_instruction",
    "render_text_program",
    "validate_aim_hardware",
]

from .reference import (
    InitializeAccumulator,
    ReadAccumulator,
    ReferenceCopy,
    ReferenceStoreSlice,
    ReferenceFill,
    ReferenceReduceSum,
    ReferenceRmsScale,
    ReferenceBroadcast,
    ReferenceInstruction,
)

__all__ += [
    "InitializeAccumulator",
    "ReadAccumulator",
    "ReferenceBroadcast",
    "ReferenceCopy",
    "ReferenceFill",
    "ReferenceInstruction",
    "ReferenceReduceSum",
    "ReferenceRmsScale",
    "ReferenceStoreSlice",
]
