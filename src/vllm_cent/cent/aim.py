"""Serialize the DRAM portion of a CENT program for the AiM simulator."""

from .instructions import (
    ApplyActivation,
    CentChannelSet,
    CentInstruction,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    MacAllBanks,
    ReadActivation,
    ReadMac,
    ReadSingleBank,
    WriteAllBanks,
    WriteBias,
    WriteGlobalBuffer,
    WriteSingleBank,
)
from .program import CentProgram

__all__ = [
    "AIM_SIMULATOR_BANKS",
    "AIM_SIMULATOR_BURST_LENGTH",
    "AIM_SIMULATOR_CHANNELS",
    "AimSimulatorCompatibilityError",
    "render_aim_channel_mask",
    "render_aim_instruction",
    "render_aim_trace",
    "validate_aim_hardware",
]


AIM_SIMULATOR_CHANNELS = 32
"""Fixed channel count in the AiM simulator."""

AIM_SIMULATOR_BANKS = 16
"""Banks per GDDR6 channel modeled by the AiM simulator."""

AIM_SIMULATOR_BURST_LENGTH = 16
"""FP16/BF16 values in one 256-bit AiM transfer."""

_MAC_SOURCE_CFR = 0
_ACTIVATION_FUNCTION_CFR = 2
_SIMULATOR_REGISTER = 0


class AimSimulatorCompatibilityError(ValueError):
    """Report a CENT operation that the AiM trace ABI cannot represent."""


def validate_aim_hardware(program: CentProgram) -> None:
    """Check the fixed geometry required by the current AiM simulator.

    Args:
        program: Program whose hardware description will be serialized.

    Raises:
        AimSimulatorCompatibilityError: If channel count, bank count, or burst
            width differs from the simulator's fixed GDDR6 target.
    """

    hardware = program.hardware
    expected = (
        ("num_channels", hardware.num_channels, AIM_SIMULATOR_CHANNELS),
        ("num_banks", hardware.num_banks, AIM_SIMULATOR_BANKS),
        (
            "burst_length",
            hardware.burst_length,
            AIM_SIMULATOR_BURST_LENGTH,
        ),
    )
    for name, actual, required in expected:
        if actual != required:
            raise AimSimulatorCompatibilityError(
                f"AiM simulator requires {name}={required}, got {actual}"
            )


def render_aim_channel_mask(channels: CentChannelSet) -> str:
    """Pack channels using the AiM simulator's reversed 32-bit mask order.

    The simulator maps the least-significant mask bit to physical channel 31,
    so channel zero occupies bit 31. This differs from the paper-readable
    renderer, which intentionally uses the conventional bit-n mapping.

    Args:
        channels: Physical AiM channel numbers to select.

    Returns:
        Lowercase hexadecimal channel mask accepted by the trace parser.

    Raises:
        AimSimulatorCompatibilityError: If a channel is outside 0 through 31.
    """

    mask = 0
    for channel in channels.channels:
        if channel >= AIM_SIMULATOR_CHANNELS:
            raise AimSimulatorCompatibilityError(
                "AiM simulator channel is outside 0 through 31"
            )
        mask |= 1 << (AIM_SIMULATOR_CHANNELS - 1 - channel)
    return hex(mask)


def _require_zero_column(instruction: CentInstruction, column: int) -> None:
    """Reject a column that the AiM trace grammar would silently discard.

    Args:
        instruction: Instruction reported in an incompatibility message.
        column: Paper-level scalar column coordinate.

    Raises:
        AimSimulatorCompatibilityError: If ``column`` is not zero.
    """

    if column != 0:
        raise AimSimulatorCompatibilityError(
            f"AiM trace cannot encode {instruction.opcode.value} column {column}"
        )


def _require_simulator_register(
    instruction: CentInstruction, register: int
) -> None:
    """Reject a MAC register ID absent from the AiM trace grammar.

    Args:
        instruction: Instruction reported in an incompatibility message.
        register: Compiler-visible accumulation register.

    Raises:
        AimSimulatorCompatibilityError: If the register is not zero.
    """

    if register != _SIMULATOR_REGISTER:
        raise AimSimulatorCompatibilityError(
            f"AiM trace cannot encode {instruction.opcode.value} register "
            f"{register}"
        )


def _render_single_bank_transfer(
    instruction: WriteSingleBank | ReadSingleBank,
) -> tuple[str, ...]:
    """Expand a paper transfer into the simulator's one-burst records.

    The trace grammar omits ``OPsize`` and ``CO`` for these two operations.
    Repeating the row access preserves the timing model for consecutive bursts;
    the AiM simulator does not store or calculate transferred values.

    Args:
        instruction: Single-bank read or write to serialize.

    Returns:
        One trace record for each burst in the instruction.

    Raises:
        AimSimulatorCompatibilityError: If the transfer starts after column
            zero, which cannot be represented by this timing expansion.
    """

    _require_zero_column(instruction, instruction.address.column)
    channels = CentChannelSet(channels=(instruction.address.channel,))
    channel_mask = render_aim_channel_mask(channels)
    first_gpr = (
        instruction.source.slot
        if isinstance(instruction, WriteSingleBank)
        else instruction.destination.slot
    )
    return tuple(
        f"AiM {instruction.opcode.value} {first_gpr + offset} "
        f"{channel_mask} {instruction.address.bank} {instruction.address.row}"
        for offset in range(instruction.operation_size)
    )


def render_aim_instruction(instruction: CentInstruction) -> tuple[str, ...]:
    """Render one supported instruction as AiM trace records.

    Configuration-register writes are emitted immediately before operations
    whose meaning depends on that hidden state. The result may therefore hold
    more than one line.

    Args:
        instruction: Typed CENT instruction to serialize.

    Returns:
        Ordered trace records without trailing newlines.

    Raises:
        AimSimulatorCompatibilityError: If the simulator lacks the instruction
            or cannot encode one of its operands.
    """

    if isinstance(instruction, (WriteSingleBank, ReadSingleBank)):
        return _render_single_bank_transfer(instruction)

    if isinstance(instruction, WriteAllBanks):
        _require_zero_column(instruction, instruction.column)
        _require_simulator_register(
            instruction, instruction.accumulation_register
        )
        channels = CentChannelSet(channels=(instruction.channel,))
        return (
            f"AiM WR_ABK {instruction.source.slot} "
            f"{render_aim_channel_mask(channels)} {instruction.row}",
        )

    if isinstance(instruction, MacAllBanks):
        _require_zero_column(instruction, instruction.column)
        _require_simulator_register(
            instruction, instruction.accumulation_register
        )
        channel_mask = render_aim_channel_mask(instruction.channels)
        # CFR0 is persistent simulator state rather than a MAC_ABK operand.
        # Reassert it immediately before every MAC so the rendered meaning does
        # not depend on an earlier instruction or the simulator's default. A
        # stateful whole-program optimizer may safely coalesce equal writes.
        return (
            f"W CFR {_MAC_SOURCE_CFR} {instruction.operand_source.value}",
            f"AiM MAC_ABK {instruction.operation_size} {channel_mask} "
            f"{instruction.row}",
        )

    if isinstance(instruction, ElementwiseMultiply):
        _require_zero_column(instruction, instruction.column)
        return (
            f"AiM EWMUL {instruction.operation_size} "
            f"{render_aim_channel_mask(instruction.channels)} {instruction.row}",
        )

    if isinstance(instruction, ApplyActivation):
        _require_simulator_register(
            instruction, instruction.accumulation_register
        )
        channel_mask = render_aim_channel_mask(instruction.channels)
        return (
            f"W CFR {_ACTIVATION_FUNCTION_CFR} "
            f"{instruction.activation_function_id}",
            f"AiM AF {channel_mask}",
        )

    if isinstance(
        instruction, (CopyBankToGlobalBuffer, CopyGlobalBufferToBank)
    ):
        _require_zero_column(instruction, instruction.column)
        return (
            f"AiM {instruction.opcode.value} {instruction.operation_size} "
            f"{render_aim_channel_mask(instruction.channels)} "
            f"{instruction.bank} {instruction.row}",
        )

    if isinstance(instruction, WriteBias):
        return (
            f"AiM WR_BIAS {instruction.source.slot} "
            f"{render_aim_channel_mask(instruction.channels)}",
        )

    if isinstance(instruction, (ReadMac, ReadActivation)):
        _require_simulator_register(
            instruction, instruction.accumulation_register
        )
        return (
            f"AiM {instruction.opcode.value} {instruction.destination.slot} "
            f"{render_aim_channel_mask(instruction.channels)}",
        )

    if isinstance(instruction, WriteGlobalBuffer):
        _require_zero_column(instruction, instruction.column)
        return (
            f"AiM WR_GB {instruction.operation_size} "
            f"{instruction.source.slot} "
            f"{render_aim_channel_mask(instruction.channels)}",
        )

    raise AimSimulatorCompatibilityError(
        f"AiM simulator does not implement {instruction.opcode.value}"
    )


def render_aim_trace(program: CentProgram) -> str:
    """Render a complete trace accepted by the AiM simulator.

    The adapter covers AiM's DRAM-side instructions. It deliberately rejects
    CENT PNM and CXL instructions such as ``EXP``, ``RED``, ``RISCV``, and
    ``SEND_CXL`` because the referenced simulator does not implement them.
    ``EOC`` is appended because the trace frontend requires it at end of file.

    Args:
        program: Validated CENT program for the fixed AiM geometry.

    Returns:
        Trace text with one record per line and a final newline.

    Raises:
        AimSimulatorCompatibilityError: If the target geometry or an
            instruction cannot be represented by AiM's trace ABI.
    """

    validate_aim_hardware(program)
    lines = [
        line
        for instruction in program.instructions
        for line in render_aim_instruction(instruction)
    ]
    lines.append("AiM EOC")
    return "\n".join(lines) + "\n"
