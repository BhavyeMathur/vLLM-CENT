"""Serialize the DRAM portion of a CENT program for the AiM simulator."""

from functools import singledispatch

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


def _require_simulator_register(instruction: CentInstruction, register: int) -> None:
    """Reject a MAC register ID absent from the AiM trace grammar.

    Args:
        instruction: Instruction reported in an incompatibility message.
        register: Compiler-visible accumulation register.

    Raises:
        AimSimulatorCompatibilityError: If the register is not zero.
    """

    if register != _SIMULATOR_REGISTER:
        raise AimSimulatorCompatibilityError(
            f"AiM trace cannot encode {instruction.opcode.value} register {register}"
        )


def _render_single_bank_transfer(
    instruction: WriteSingleBank | ReadSingleBank,
        *,
        first_gpr: int,
) -> tuple[str, ...]:
    """Expand a paper transfer into the simulator's one-burst records.

    The trace grammar omits ``OPsize`` and ``CO`` for these two operations.
    Repeating the row access preserves the timing model for consecutive bursts;
    the AiM simulator does not store or calculate transferred values.

    Args:
        instruction: Single-bank read or write to serialize.
        first_gpr: Shared Buffer source or destination slot for the first burst.

    Returns:
        One trace record for each burst in the instruction.

    Raises:
        AimSimulatorCompatibilityError: If the transfer starts after column
            zero, which cannot be represented by this timing expansion.
    """

    _require_zero_column(instruction, instruction.address.column)
    channels = CentChannelSet(channels=(instruction.address.channel,))
    channel_mask = render_aim_channel_mask(channels)
    return tuple(
        f"AiM {instruction.opcode.value} {first_gpr + offset} "
        f"{channel_mask} {instruction.address.bank} {instruction.address.row}"
        for offset in range(instruction.operation_size)
    )


@singledispatch
def render_aim_instruction(instruction: CentInstruction) -> tuple[str, ...]:
    """Render one supported instruction as AiM trace records.

    Configuration-register writes are emitted immediately before operations
    whose meaning depends on that hidden state. The result may therefore hold
    more than one line. Callers rendering a standalone instruction are
    responsible for validating its target-dependent address bounds first;
    :func:`render_aim_trace` receives an already validated program.

    Args:
        instruction: Typed CENT instruction to serialize.

    Returns:
        Ordered trace records without trailing newlines.

    Raises:
        AimSimulatorCompatibilityError: If the simulator lacks the instruction
            or cannot encode one of its operands.
    """

    # Known CENT instructions report their assembly name. The base class and
    # unknown subclasses have no opcode, so report the Python type without
    # accidentally raising AttributeError while formatting this error.
    opcode = getattr(type(instruction), "OPCODE", None)
    instruction_name = (
        opcode.value if opcode is not None else type(instruction).__name__
    )
    raise AimSimulatorCompatibilityError(
        f"AiM simulator does not implement {instruction_name}"
    )


@render_aim_instruction.register(WriteSingleBank)
def _render_aim_write_single_bank(
        instruction: WriteSingleBank,
) -> tuple[str, ...]:
    """Encode one Shared-Buffer-to-bank transfer for AiM.

    Args:
        instruction: Single-bank write to encode.

    Returns:
        One AiM trace record per burst.

    Raises:
        AimSimulatorCompatibilityError: If the transfer column is not encodable.
    """

    return _render_single_bank_transfer(
        instruction,
        first_gpr=instruction.source.slot,
    )


@render_aim_instruction.register(ReadSingleBank)
def _render_aim_read_single_bank(instruction: ReadSingleBank) -> tuple[str, ...]:
    """Encode one bank-to-Shared-Buffer transfer for AiM.

    Args:
        instruction: Single-bank read to encode.

    Returns:
        One AiM trace record per burst.

    Raises:
        AimSimulatorCompatibilityError: If the transfer column is not encodable.
    """

    return _render_single_bank_transfer(
        instruction,
        first_gpr=instruction.destination.slot,
    )


@render_aim_instruction.register(WriteAllBanks)
def _render_aim_write_all_banks(instruction: WriteAllBanks) -> tuple[str, ...]:
    """Encode one all-bank write for AiM.

    Args:
        instruction: All-bank write to encode.

    Returns:
        One AiM trace record.

    Raises:
        AimSimulatorCompatibilityError: If its column or register is not
            encodable by AiM.
    """

    _require_zero_column(instruction, instruction.column)
    _require_simulator_register(instruction, instruction.accumulation_register)
    channels = CentChannelSet(channels=(instruction.channel,))
    return (
        f"AiM WR_ABK {instruction.source.slot} "
        f"{render_aim_channel_mask(channels)} {instruction.row}",
    )


@render_aim_instruction.register(MacAllBanks)
def _render_aim_mac_all_banks(instruction: MacAllBanks) -> tuple[str, ...]:
    """Encode one all-bank MAC and its explicit source configuration.

    Args:
        instruction: All-bank MAC to encode.

    Returns:
        CFR source selection followed by the AiM MAC record.

    Raises:
        AimSimulatorCompatibilityError: If its column or register is not
            encodable by AiM.
    """

    _require_zero_column(instruction, instruction.column)
    _require_simulator_register(instruction, instruction.accumulation_register)
    channel_mask = render_aim_channel_mask(instruction.channels)
    # CFR0 is persistent simulator state rather than a MAC_ABK operand. Reassert
    # it immediately before every MAC so meaning does not depend on an earlier
    # instruction or the simulator default.
    return (
        f"W CFR {_MAC_SOURCE_CFR} {instruction.operand_source.value}",
        f"AiM MAC_ABK {instruction.operation_size} {channel_mask} {instruction.row}",
    )


@render_aim_instruction.register(ElementwiseMultiply)
def _render_aim_elementwise_multiply(
        instruction: ElementwiseMultiply,
) -> tuple[str, ...]:
    """Encode one near-bank elementwise multiplication for AiM.

    Args:
        instruction: Elementwise operation to encode.

    Returns:
        One AiM trace record.

    Raises:
        AimSimulatorCompatibilityError: If its column is not encodable by AiM.
    """

    _require_zero_column(instruction, instruction.column)
    return (
        f"AiM EWMUL {instruction.operation_size} "
        f"{render_aim_channel_mask(instruction.channels)} {instruction.row}",
    )


@render_aim_instruction.register(ApplyActivation)
def _render_aim_activation(instruction: ApplyActivation) -> tuple[str, ...]:
    """Encode one activation and its explicit function configuration.

    Args:
        instruction: Activation operation to encode.

    Returns:
        CFR function selection followed by the AiM activation record.

    Raises:
        AimSimulatorCompatibilityError: If its register is not encodable by AiM.
    """

    _require_simulator_register(instruction, instruction.accumulation_register)
    channel_mask = render_aim_channel_mask(instruction.channels)
    return (
        f"W CFR {_ACTIVATION_FUNCTION_CFR} {instruction.activation_function_id}",
        f"AiM AF {channel_mask}",
    )


@render_aim_instruction.register(CopyBankToGlobalBuffer)
@render_aim_instruction.register(CopyGlobalBufferToBank)
def _render_aim_bank_global_buffer_copy(
        instruction: CopyBankToGlobalBuffer | CopyGlobalBufferToBank,
) -> tuple[str, ...]:
    """Encode either direction of a bank and Global Buffer copy.

    Args:
        instruction: Copy operation to encode.

    Returns:
        One AiM trace record whose opcode preserves the transfer direction.

    Raises:
        AimSimulatorCompatibilityError: If its column is not encodable by AiM.
    """

    _require_zero_column(instruction, instruction.column)
    return (
        f"AiM {instruction.opcode.value} {instruction.operation_size} "
        f"{render_aim_channel_mask(instruction.channels)} "
        f"{instruction.bank} {instruction.row}",
    )


@render_aim_instruction.register(WriteBias)
def _render_aim_write_bias(instruction: WriteBias) -> tuple[str, ...]:
    """Encode one MAC-bias initialization for AiM.

    Args:
        instruction: Bias write to encode.

    Returns:
        One AiM trace record.
    """

    return (
        f"AiM WR_BIAS {instruction.source.slot} "
        f"{render_aim_channel_mask(instruction.channels)}",
    )


@render_aim_instruction.register(ReadMac)
@render_aim_instruction.register(ReadActivation)
def _render_aim_register_read(
        instruction: ReadMac | ReadActivation,
) -> tuple[str, ...]:
    """Encode a MAC or activation-result register read for AiM.

    Args:
        instruction: Register read to encode.

    Returns:
        One AiM trace record whose opcode selects the register file.

    Raises:
        AimSimulatorCompatibilityError: If its register is not encodable by AiM.
    """

    _require_simulator_register(instruction, instruction.accumulation_register)
    return (
        f"AiM {instruction.opcode.value} {instruction.destination.slot} "
        f"{render_aim_channel_mask(instruction.channels)}",
    )


@render_aim_instruction.register(WriteGlobalBuffer)
def _render_aim_write_global_buffer(
        instruction: WriteGlobalBuffer,
) -> tuple[str, ...]:
    """Encode one Shared-Buffer-to-Global-Buffer transfer for AiM.

    Args:
        instruction: Global Buffer write to encode.

    Returns:
        One AiM trace record.

    Raises:
        AimSimulatorCompatibilityError: If its column is not encodable by AiM.
    """

    _require_zero_column(instruction, instruction.column)
    return (
        f"AiM WR_GB {instruction.operation_size} {instruction.source.slot} "
        f"{render_aim_channel_mask(instruction.channels)}",
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
