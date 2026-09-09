"""Llama-to-CENT compiler interface."""

from dataclasses import dataclass

from .cent import CentHardwareSpec, CentProgram
from .models.llama import LlamaModelSpec


@dataclass(frozen=True, slots=True, kw_only=True)
class CentMappingSpec:
    """Placement choices for one transformer block.

    Attributes:
        channels_per_block: Channels assigned to the block. Must be at least 1
            and no greater than ``CentHardwareSpec.num_channels``.
        reuse_size: Number of general-buffer chunks reused during matrix-vector
            multiplication. Must be at least 1.
    """

    channels_per_block: int
    reuse_size: int


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodeStepSpec:
    """Runtime dimensions of one decode step.

    Attributes:
        sequence_length: Number of tokens in the attention context, including
            the token currently being decoded. Must be at least 1.
    """

    sequence_length: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CompileRequest:
    """Inputs required to compile one Llama transformer block.

    Attributes:
        model: Dimensions of the source Llama block.
        hardware: Geometry of the target CENT device.
        mapping: Placement choices used by the compiler.
        step: Runtime dimensions of the decode step.
    """

    model: LlamaModelSpec
    hardware: CentHardwareSpec
    mapping: CentMappingSpec
    step: DecodeStepSpec


def compile_transformer_block(request: CompileRequest) -> CentProgram:
    """Compile one Llama decode block into CENT commands.

    Args:
        request: Model, hardware, placement, and decode-step inputs.

    Returns:
        The ordered commands for executing the block on CENT.
    """

    raise NotImplementedError


def render_text_trace(program: CentProgram) -> str:
    """Serialize commands in the CENT simulator's trace format.

    Args:
        program: Commands to serialize in execution order.

    Returns:
        Newline-terminated trace text with one command per line.
    """

    raise NotImplementedError
