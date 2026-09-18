"""Check typed CENT instructions against one hardware target."""

from functools import singledispatch

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
from ..hardware import CentHardwareSpec

__all__ = [
    "validate_address",
    "validate_channels",
    "validate_instruction",
    "validate_shared_buffer_address",
]


def validate_address(address: CentMemoryAddress, hardware: CentHardwareSpec) -> None:
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


def validate_channels(channels: CentChannelSet, hardware: CentHardwareSpec) -> None:
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


def _validate_row_column(row: int, column: int, hardware: CentHardwareSpec) -> None:
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


def _validate_global_buffer_span(
        operation_size: int,
        column: int,
        hardware: CentHardwareSpec,
) -> None:
    """Check that a burst-sized operation span fits in a Global Buffer.

    Args:
        operation_size: Number of burst-sized operations, or ``OPsize``.
        column: First scalar position in the Global Buffer, or ``CO``.
        hardware: Device that defines burst width and Global Buffer capacity.

    Raises:
        ValueError: If the first column or complete span is outside the Global
            Buffer.
    """

    # Global Buffer and DRAM columns share the ISA's CO spelling, but they are
    # separate address spaces and can have different physical capacities.
    if column >= hardware.global_buffer_columns:
        raise ValueError("instruction column is outside the Global Buffer")
    span_end = column + operation_size * hardware.burst_length
    if span_end > hardware.global_buffer_columns:
        raise ValueError("instruction exceeds the target Global Buffer")


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


@singledispatch
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

    raise TypeError(f"unsupported CENT instruction type: {type(instruction).__name__}")


@validate_instruction.register
def _validate_write_single_bank(
        instruction: WriteSingleBank,
        hardware: CentHardwareSpec,
) -> None:
    """Validate one Shared-Buffer-to-DRAM transfer.

    Args:
        instruction: Single-bank write to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If the DRAM address, row span, or Shared Buffer source span
            is outside the target.
    """

    validate_address(instruction.address, hardware)
    _validate_operation_span(
        instruction.operation_size,
        instruction.address.column,
        hardware,
    )
    _validate_shared_buffer_span(
        instruction.source,
        instruction.operation_size,
        hardware,
    )


@validate_instruction.register
def _validate_read_single_bank(
        instruction: ReadSingleBank,
        hardware: CentHardwareSpec,
) -> None:
    """Validate one DRAM-to-Shared-Buffer transfer.

    Args:
        instruction: Single-bank read to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If the DRAM address, row span, or Shared Buffer destination
            span is outside the target.
    """

    validate_address(instruction.address, hardware)
    _validate_operation_span(
        instruction.operation_size,
        instruction.address.column,
        hardware,
    )
    _validate_shared_buffer_span(
        instruction.destination,
        instruction.operation_size,
        hardware,
    )


@validate_instruction.register
def _validate_write_all_banks(
        instruction: WriteAllBanks,
        hardware: CentHardwareSpec,
) -> None:
    """Validate one all-bank broadcast write.

    Args:
        instruction: All-bank write to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If the channel, DRAM coordinate, Shared Buffer source, or
            accumulation register is outside the target.
    """

    if instruction.channel >= hardware.num_channels:
        raise ValueError("instruction channel is outside the target")
    _validate_row_column(instruction.row, instruction.column, hardware)
    validate_shared_buffer_address(instruction.source, hardware)
    _validate_accumulation_register(instruction.accumulation_register, hardware)


@validate_instruction.register
def _validate_mac_all_banks(
        instruction: MacAllBanks,
        hardware: CentHardwareSpec,
) -> None:
    """Validate one near-bank multiply-accumulate command.

    Args:
        instruction: All-bank MAC to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If a selected channel, DRAM span, or accumulation register
            is outside the target.
    """

    validate_channels(instruction.channels, hardware)
    _validate_row_column(instruction.row, instruction.column, hardware)
    _validate_operation_span(
        instruction.operation_size,
        instruction.column,
        hardware,
    )
    _validate_accumulation_register(instruction.accumulation_register, hardware)


@validate_instruction.register
def _validate_elementwise_multiply(
        instruction: ElementwiseMultiply,
        hardware: CentHardwareSpec,
) -> None:
    """Validate one near-bank elementwise multiplication.

    Args:
        instruction: Elementwise multiplication to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If a selected channel or DRAM span is outside the target.
    """

    validate_channels(instruction.channels, hardware)
    _validate_row_column(instruction.row, instruction.column, hardware)
    _validate_operation_span(
        instruction.operation_size,
        instruction.column,
        hardware,
    )


@validate_instruction.register
def _validate_apply_activation(
        instruction: ApplyActivation,
        hardware: CentHardwareSpec,
) -> None:
    """Validate one near-bank activation command.

    Args:
        instruction: Activation command to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If a selected channel or accumulation register is outside
            the target.
    """

    validate_channels(instruction.channels, hardware)
    _validate_accumulation_register(instruction.accumulation_register, hardware)


def _validate_shared_buffer_input_output(
        *,
        source: CentSharedBufferAddress,
        destination: CentSharedBufferAddress,
        operation_size: int,
        hardware: CentHardwareSpec,
) -> None:
    """Validate equal-size input and output spans for a PNM command.

    Args:
        source: First Shared Buffer input slot.
        destination: First Shared Buffer output slot.
        operation_size: Number of consecutive slots in each span.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If either complete span is outside the Shared Buffer.
    """

    # TODO(paper/ABI): We assume equal input and output slot counts. We need
    # separate footprints for EXP, RED, ACC, and each RISC-V routine.
    _validate_shared_buffer_span(source, operation_size, hardware)
    _validate_shared_buffer_span(destination, operation_size, hardware)


@validate_instruction.register
def _validate_exponent(
        instruction: Exponent,
        hardware: CentHardwareSpec,
) -> None:
    """Validate the Shared Buffer spans used by one exponent command.

    Args:
        instruction: Exponent command to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If its source or destination span is outside the target.
    """

    _validate_shared_buffer_input_output(
        source=instruction.source,
        destination=instruction.destination,
        operation_size=instruction.operation_size,
        hardware=hardware,
    )


@validate_instruction.register
def _validate_reduction(
        instruction: Reduction,
        hardware: CentHardwareSpec,
) -> None:
    """Validate the Shared Buffer spans used by one reduction command.

    Args:
        instruction: Reduction command to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If its source or destination span is outside the target.
    """

    _validate_shared_buffer_input_output(
        source=instruction.source,
        destination=instruction.destination,
        operation_size=instruction.operation_size,
        hardware=hardware,
    )


@validate_instruction.register
def _validate_accumulate(
        instruction: Accumulate,
        hardware: CentHardwareSpec,
) -> None:
    """Validate the Shared Buffer spans used by one accumulation command.

    Args:
        instruction: Accumulation command to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If its source or destination span is outside the target.
    """

    _validate_shared_buffer_input_output(
        source=instruction.source,
        destination=instruction.destination,
        operation_size=instruction.operation_size,
        hardware=hardware,
    )


@validate_instruction.register
def _validate_run_risc_v(
        instruction: RunRiscV,
        hardware: CentHardwareSpec,
) -> None:
    """Validate the Shared Buffer spans used by one RISC-V command.

    Args:
        instruction: RISC-V command to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If its source or destination span is outside the target.
    """

    _validate_shared_buffer_input_output(
        source=instruction.source,
        destination=instruction.destination,
        operation_size=instruction.operation_size,
        hardware=hardware,
    )


@validate_instruction.register
def _validate_send_cxl(
        instruction: SendCxl,
        hardware: CentHardwareSpec,
) -> None:
    """Validate the locally checkable Shared Buffer operands of a CXL send.

    Args:
        instruction: Point-to-point CXL send to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If a checked Shared Buffer address is outside the target.
    """

    # TODO(architecture): We incorrectly check remote Rd against the local
    # device. We need the destination device's topology and buffer geometry.
    validate_shared_buffer_address(instruction.source, hardware)
    validate_shared_buffer_address(instruction.destination, hardware)


@validate_instruction.register
def _validate_receive_cxl(
        instruction: ReceiveCxl,
        hardware: CentHardwareSpec,
) -> None:
    """Accept the operand-free CXL receive command on any CENT target.

    Args:
        instruction: Operand-free CXL receive to validate.
        hardware: Device on which the command will run.
    """

    # Referencing both arguments makes this intentionally operand-free contract
    # visible to static analysis without inventing target checks.
    del instruction, hardware


@validate_instruction.register
def _validate_broadcast_cxl(
        instruction: BroadcastCxl,
        hardware: CentHardwareSpec,
) -> None:
    """Validate locally checkable Shared Buffer operands of a CXL broadcast.

    Args:
        instruction: CXL broadcast to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If a checked Shared Buffer address is outside the target.
    """

    # TODO(architecture): We need a CXL topology to check the 8-bit DVcount and
    # the destination buffers on every receiving device.
    validate_shared_buffer_address(instruction.source, hardware)
    validate_shared_buffer_address(instruction.destination, hardware)


def _validate_bank_global_buffer_copy(
        *,
        channels: CentChannelSet,
        operation_size: int,
        bank: int,
        row: int,
        column: int,
        hardware: CentHardwareSpec,
) -> None:
    """Validate shared geometry for a bank/Global-Buffer copy.

    Args:
        channels: Channels participating in the copy.
        operation_size: Number of burst-sized operations.
        bank: Bank selected in every participating channel.
        row: DRAM row containing the copied span.
        column: First scalar column in DRAM and the Global Buffer.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If a channel, bank, DRAM span, or Global Buffer span is
            outside the target.
    """

    validate_channels(channels, hardware)
    _validate_row_column(row, column, hardware)
    _validate_operation_span(operation_size, column, hardware)
    if bank >= hardware.num_banks:
        raise ValueError("copy bank is outside the target channel")
    # CO selects both address spaces, whose capacities are independent.
    _validate_global_buffer_span(operation_size, column, hardware)


@validate_instruction.register
def _validate_copy_bank_to_global_buffer(
        instruction: CopyBankToGlobalBuffer,
        hardware: CentHardwareSpec,
) -> None:
    """Validate one bank-to-Global-Buffer copy.

    Args:
        instruction: Bank-to-Global-Buffer copy to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If a channel, bank, DRAM span, or Global Buffer span is
            outside the target.
    """

    _validate_bank_global_buffer_copy(
        channels=instruction.channels,
        operation_size=instruction.operation_size,
        bank=instruction.bank,
        row=instruction.row,
        column=instruction.column,
        hardware=hardware,
    )


@validate_instruction.register
def _validate_copy_global_buffer_to_bank(
        instruction: CopyGlobalBufferToBank,
        hardware: CentHardwareSpec,
) -> None:
    """Validate one Global-Buffer-to-bank copy.

    Args:
        instruction: Global-Buffer-to-bank copy to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If a channel, bank, DRAM span, or Global Buffer span is
            outside the target.
    """

    _validate_bank_global_buffer_copy(
        channels=instruction.channels,
        operation_size=instruction.operation_size,
        bank=instruction.bank,
        row=instruction.row,
        column=instruction.column,
        hardware=hardware,
    )


@validate_instruction.register
def _validate_write_bias(
        instruction: WriteBias,
        hardware: CentHardwareSpec,
) -> None:
    """Validate one Shared-Buffer-to-bias-register command.

    Args:
        instruction: Bias write to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If a selected channel or Shared Buffer source is outside
            the target.
    """

    validate_channels(instruction.channels, hardware)
    validate_shared_buffer_address(instruction.source, hardware)


@validate_instruction.register
def _validate_read_activation(
        instruction: ReadActivation,
        hardware: CentHardwareSpec,
) -> None:
    """Validate one activation-register read.

    Args:
        instruction: Activation-register read to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If a channel, accumulation register, or Shared Buffer
            destination is outside the target.
    """

    validate_channels(instruction.channels, hardware)
    _validate_accumulation_register(instruction.accumulation_register, hardware)
    validate_shared_buffer_address(instruction.destination, hardware)


@validate_instruction.register
def _validate_read_mac(
        instruction: ReadMac,
        hardware: CentHardwareSpec,
) -> None:
    """Validate one MAC-register read.

    Args:
        instruction: MAC-register read to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If a channel, accumulation register, or Shared Buffer
            destination is outside the target.
    """

    validate_channels(instruction.channels, hardware)
    _validate_accumulation_register(instruction.accumulation_register, hardware)
    validate_shared_buffer_address(instruction.destination, hardware)


@validate_instruction.register
def _validate_write_global_buffer(
        instruction: WriteGlobalBuffer,
        hardware: CentHardwareSpec,
) -> None:
    """Validate one Shared-Buffer-to-Global-Buffer write.

    Args:
        instruction: Global Buffer write to validate.
        hardware: Device on which the command will run.

    Raises:
        ValueError: If a channel, Shared Buffer source span, or Global Buffer
            destination span is outside the target.
    """

    validate_channels(instruction.channels, hardware)
    _validate_shared_buffer_span(
        instruction.source,
        instruction.operation_size,
        hardware,
    )
    _validate_global_buffer_span(
        instruction.operation_size,
        instruction.column,
        hardware,
    )


def _validate_accumulation_register(register: int, hardware: CentHardwareSpec) -> None:
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
