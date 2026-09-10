"""Check typed CENT instructions against one hardware target."""

from ..hardware import CentHardwareSpec
from .address import (
    CentChannelSet,
    CentMemoryAddress,
    CentSharedBufferAddress,
)
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
    ReadMac,
    ReadSingleBank,
    ReceiveCxl,
    SendCxl,
    WriteAllBanks,
    WriteBias,
    WriteGlobalBuffer,
    WriteSingleBank,
)

__all__ = [
    "validate_address",
    "validate_channels",
    "validate_instruction",
    "validate_shared_buffer_address",
]


def validate_address(
    address: CentMemoryAddress, hardware: CentHardwareSpec
) -> None:
    """Check that every part of a DRAM address exists on the target.

    Args:
        address: Channel, bank, row, and column to check.
        hardware: Device that defines the valid ranges.

    Raises:
        ValueError: If an address component is outside the target.
    """

    # Channel, bank, row, and column each have a separate hardware limit.
    if address.channel >= hardware.num_channels:
        raise ValueError("memory channel is outside the target")
    if address.bank >= hardware.num_banks:
        raise ValueError("memory bank is outside the target channel")
    _validate_row_column(address.row, address.column, hardware)


def validate_channels(
    channels: CentChannelSet, hardware: CentHardwareSpec
) -> None:
    """Check that every channel selected by ``CHmask`` exists.

    Args:
        channels: Physical channel numbers selected by the instruction.
        hardware: Device that defines how many channels exist.

    Raises:
        ValueError: If a selected channel is outside the target.
    """

    # CentChannelSet has already checked empty, negative, and duplicate values.
    if any(channel >= hardware.num_channels for channel in channels.channels):
        raise ValueError("selected channel is outside the target")


def validate_shared_buffer_address(
    address: CentSharedBufferAddress, hardware: CentHardwareSpec
) -> None:
    """Check that a Shared Buffer slot exists on the target.

    Args:
        address: Zero-based 256-bit slot, used as ``Rs`` or ``Rd``.
        hardware: Device that defines the number of available slots.

    Raises:
        ValueError: If the slot is outside the target Shared Buffer.
    """

    # slot counts 256-bit entries, not bytes or individual BF16 values.
    if address.slot >= hardware.shared_buffer_slots:
        raise ValueError("Shared Buffer address is outside the target")


def _validate_row_column(
    row: int, column: int, hardware: CentHardwareSpec
) -> None:
    """Check a DRAM row and column against the target.

    Args:
        row: DRAM row number, or ``RO``.
        column: Scalar position in the row, or ``CO``.
        hardware: Device that defines the row and column limits.

    Raises:
        ValueError: If either coordinate is outside the target.
    """

    # Broadcast instructions provide RO/CO without one channel or bank address.
    if row >= hardware.dram_rows:
        raise ValueError("instruction row is outside the target bank")
    if column >= hardware.dram_columns:
        raise ValueError("instruction column is outside the target row")


def _validate_operation_span(
    operation_size: int, column: int, hardware: CentHardwareSpec
) -> None:
    """Check that an ``OPsize`` span fits in one DRAM row.

    Args:
        operation_size: Number of burst-sized operations, or ``OPsize``.
        column: First scalar position in the row, or ``CO``.
        hardware: Device that defines burst and row widths.

    Raises:
        ValueError: If the generated micro-operations cross a row boundary.
    """

    # TODO(paper): We treat CO as a scalar index and stop at row boundaries. We
    # need to confirm its unit, alignment, and whether OPsize may wrap rows.

    # Each operation advances by one burst. Ending exactly at the row width is
    # valid; ending after it crosses the row boundary.
    span_end = column + operation_size * hardware.burst_length
    if span_end > hardware.dram_columns:
        raise ValueError("instruction micro-operations cross a DRAM row")


def _validate_shared_buffer_span(
    address: CentSharedBufferAddress,
    operation_size: int,
    hardware: CentHardwareSpec,
) -> None:
    """Check that consecutive ``Rs`` or ``Rd`` slots fit in the buffer.

    Args:
        address: First 256-bit Shared Buffer slot.
        operation_size: Number of consecutive slots used by the instruction.
        hardware: Device that defines the Shared Buffer capacity.

    Raises:
        ValueError: If the generated slot range exceeds the Shared Buffer.
    """

    # TODO(paper): We assume EXP, RED, ACC, and RISCV use equal-length Rs and Rd
    # ranges. We need each instruction's overlap and in-place rules.

    # Check the first slot, then check the end just after the last used slot.
    validate_shared_buffer_address(address, hardware)
    span_end = address.slot + operation_size
    if span_end > hardware.shared_buffer_slots:
        raise ValueError("instruction exceeds the target Shared Buffer")


def validate_instruction(
    instruction: CentInstruction, hardware: CentHardwareSpec
) -> None:
    """Check one instruction against a CENT target.

    Args:
        instruction: Typed CENT command to check.
        hardware: Device on which the command will run.

    Raises:
        TypeError: If the instruction type is unsupported.
        ValueError: If an operand is incompatible with the target.
    """

    # Exact types force each new instruction to define all of its own checks.
    instruction_type = type(instruction)

    # A single-bank transfer advances through DRAM bursts and Shared Buffer
    # slots. A write reads Rs; a read writes Rd.
    if instruction_type in (WriteSingleBank, ReadSingleBank):
        validate_address(instruction.address, hardware)
        _validate_operation_span(
            instruction.operation_size, instruction.address.column, hardware
        )
        shared_address = (
            instruction.source
            if instruction_type is WriteSingleBank
            else instruction.destination
        )
        _validate_shared_buffer_span(
            shared_address, instruction.operation_size, hardware
        )
        return

    # WR_ABK names one channel and broadcasts within that channel's banks.
    if instruction_type is WriteAllBanks:
        if instruction.channel >= hardware.num_channels:
            raise ValueError("instruction channel is outside the target")
        _validate_row_column(instruction.row, instruction.column, hardware)
        validate_shared_buffer_address(instruction.source, hardware)
        _validate_accumulation_register(
            instruction.accumulation_register, hardware
        )
        return

    # Every instruction in this group contains a CHmask.
    channel_instructions = (
        MacAllBanks,
        ElementwiseMultiply,
        ApplyActivation,
        CopyBankToGlobalBuffer,
        CopyGlobalBufferToBank,
        WriteBias,
        ReadMac,
        WriteGlobalBuffer,
    )
    if instruction_type in channel_instructions:
        validate_channels(instruction.channels, hardware)

    # These instructions start at RO/CO and advance by one burst per operation.
    if instruction_type in (
        MacAllBanks,
        ElementwiseMultiply,
        CopyBankToGlobalBuffer,
        CopyGlobalBufferToBank,
    ):
        _validate_row_column(instruction.row, instruction.column, hardware)
        _validate_operation_span(
            instruction.operation_size, instruction.column, hardware
        )

    # Regid selects a MAC register, not a DRAM or Shared Buffer address.
    if instruction_type in (MacAllBanks, ApplyActivation, ReadMac):
        _validate_accumulation_register(
            instruction.accumulation_register, hardware
        )

    # Check the Shared Buffer fields and any spans implied by OPsize.
    if instruction_type is WriteBias:
        validate_shared_buffer_address(instruction.source, hardware)
    elif instruction_type is ReadMac:
        validate_shared_buffer_address(instruction.destination, hardware)
    elif instruction_type is WriteGlobalBuffer:
        # TODO(architecture): We check this Global Buffer span with the DRAM row
        # width. We need a separate Global Buffer capacity in the hardware spec.

        _validate_shared_buffer_span(
            instruction.source, instruction.operation_size, hardware
        )
        _validate_operation_span(
            instruction.operation_size, instruction.column, hardware
        )
    elif instruction_type in (Exponent, Reduction, Accumulate, RunRiscV):
        # TODO(paper/ABI): We assume equal input and output slot counts. We need
        # separate footprints for EXP, RED, ACC, and each RISC-V routine.

        _validate_shared_buffer_span(
            instruction.source, instruction.operation_size, hardware
        )
        _validate_shared_buffer_span(
            instruction.destination, instruction.operation_size, hardware
        )
    elif instruction_type is SendCxl:
        # TODO(architecture): We incorrectly check remote Rd against the local
        # device. We need the destination device's topology and buffer geometry.

        validate_shared_buffer_address(instruction.source, hardware)
        validate_shared_buffer_address(instruction.destination, hardware)
    elif instruction_type is BroadcastCxl:
        # TODO(architecture): We need a CXL topology to check the 8-bit DVcount
        # and the destination buffers on every receiving device.

        validate_shared_buffer_address(instruction.source, hardware)
        validate_shared_buffer_address(instruction.destination, hardware)

    # Reject a new instruction type until its hardware checks are added above.
    known_types = (
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
        CopyBankToGlobalBuffer,
        CopyGlobalBufferToBank,
        WriteBias,
        ReadMac,
        WriteGlobalBuffer,
    )
    if instruction_type not in known_types:
        raise TypeError(
            f"unsupported CENT instruction type: {type(instruction).__name__}"
        )


def _validate_accumulation_register(
    register: int, hardware: CentHardwareSpec
) -> None:
    """Check a ``Regid`` against the target's MAC register count.

    Args:
        register: Zero-based MAC result register number, or ``Regid``.
        hardware: Device that defines how many registers exist.

    Raises:
        ValueError: If ``register`` is outside the target PU.
    """

    # Regid and Shared Buffer slots are separate address spaces.
    if register >= hardware.accumulator_slots_per_bank:
        raise ValueError("accumulation register is outside the target PU")
