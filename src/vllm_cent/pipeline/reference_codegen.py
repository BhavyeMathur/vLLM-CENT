"""Expand explicit reference matrix and normalization plans into visible primitives.

Weights remain caller-provided Shared Buffer tensors. Matrix tiles temporarily
occupy DRAM row zero only while their dot products execute; every selected bank
is written before use. No persistent DRAM weight allocation overlaps the scratch
used by copies or elementwise multiplication. The generated reference packing,
accumulator, and scalar instructions cannot be exported as physical paper ISA.
"""

from dataclasses import dataclass

from ..cent import (
    BANKS_PER_GROUP,
    CentChannelSet,
    CentHardwareSpec,
    CentMemoryAddress,
    CentSharedBufferAddress,
    ElementwiseMultiply,
    MacAllBanks,
    MacOperandSource,
    ReadSingleBank,
    WriteGlobalBuffer,
    WriteSingleBank,
    ceil_div,
)
from ..cent.instructions import CentInstruction
from ..cent.reference import (
    InitializeAccumulator,
    ReadAccumulator,
    ReferenceBroadcast,
    ReferenceCopy,
    ReferenceFill,
    ReferenceReduceSum,
    ReferenceRmsScale,
    ReferenceStoreSlice,
)

__all__ = ["LinearCommand", "ReferenceWorkspace", "RmsNormCommand"]


@dataclass(frozen=True, slots=True, kw_only=True)
class LinearCommand:
    """Plan batched row dot products before choosing buffer addresses.

    Attributes:
        source: Storage holding the flattened input rows.
        weights: Storage holding output-major weight rows.
        destination: Storage receiving flattened output rows.
        input_width: Values in each input and weight row.
        output_width: Weight rows, hence output values per input row.
        batch_count: Number of input rows after flattening leading dimensions.
        accumulation_register: Explicit mapping-selected register for running sums.
        tile_width: Scalar values staged per input tile, selected to fit both
            a DRAM row and the Global Buffer.
        channel: Physical channel used for the reference matrix computation.
        bank_count: Banks initialized and used together for one output group.
        operation: Source operation name for diagnostics.
    """

    source: str
    weights: str
    destination: str
    input_width: int
    output_width: int
    batch_count: int
    accumulation_register: int
    tile_width: int
    channel: int
    bank_count: int
    operation: str


@dataclass(frozen=True, slots=True, kw_only=True)
class RmsNormCommand:
    """Plan square, sum, reciprocal-root, broadcast, and learned-scale stages.

    Attributes:
        source: Storage containing flattened input rows.
        scale: Storage containing the learned vector shared by all rows.
        destination: Storage receiving normalized rows in the original order.
        width: Logical values per row; padding is excluded from the mean.
        batch_count: Number of independent rows.
        epsilon: Constant added to each mean square before the reciprocal root.
        tile_slots: Complete slots processed by each elementwise scratch pass.
        operation: Source operation name for diagnostics.
    """

    source: str
    scale: str
    destination: str
    width: int
    batch_count: int
    epsilon: float
    tile_slots: int
    operation: str


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceWorkspace:
    """Reserve reusable Shared Buffer working space for reference computations.

    Attributes:
        name: Unique storage name, separate from every mapped graph tensor.
        slot_count: Maximum temporary slots required by any sequential command.
    """

    name: str
    slot_count: int


def _workspace_slots(
    command: LinearCommand | RmsNormCommand, hardware: CentHardwareSpec
) -> int:
    """Calculate the working slots without reserving addresses.

    Args:
        command: Matrix or normalization command needing temporary packed vectors.
        hardware: Row width, bank count, and scalar lanes per slot.

    Returns:
        Slots for two input staging regions and a result or scalar region.
    """
    if isinstance(command, LinearCommand):
        return 2 * (command.tile_width // hardware.burst_length) + ceil_div(
            command.bank_count, hardware.burst_length
        )
    return 2 * ceil_div(command.width, hardware.burst_length) + 1


def _emit_multiply_chunk(
    hardware: CentHardwareSpec,
    source: CentSharedBufferAddress,
    destination: CentSharedBufferAddress,
    slot_count: int,
) -> tuple[CentInstruction, ...]:
    """Stage both operands in all bank groups and collect one complete product.

    Args:
        hardware: Geometry defining complete four-bank groups and row capacity.
        source: Right operand's first Shared Buffer slot.
        destination: Left operand's first slot, overwritten after both are staged.
        slot_count: Number of complete slots, fitting inside one DRAM row.

    Returns:
        Writes to every consumed bank, one multiply, and a read from bank two.

    Raises:
        ValueError: If the chunk is empty or larger than a scratch row.
    """
    if not 1 <= slot_count <= hardware.dram_columns // hardware.burst_length:
        raise ValueError("multiply chunk must fit one DRAM scratch row")
    instructions: list[CentInstruction] = []
    for group in range(0, hardware.num_banks, BANKS_PER_GROUP):
        for bank, operand in (
            (ElementwiseMultiply.FIRST_OPERAND_BANK, destination),
            (ElementwiseMultiply.SECOND_OPERAND_BANK, source),
        ):
            instructions.append(
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=group + bank, row=0, column=0
                    ),
                    operation_size=slot_count,
                    source=operand,
                )
            )
    instructions.append(
        ElementwiseMultiply(
            channels=CentChannelSet(channels=(0,)),
            operation_size=slot_count,
            row=0,
            column=0,
        )
    )
    instructions.append(
        ReadSingleBank(
            address=CentMemoryAddress(
                channel=0, bank=ElementwiseMultiply.RESULT_BANK, row=0, column=0
            ),
            operation_size=slot_count,
            destination=destination,
        )
    )
    return tuple(instructions)


@dataclass(frozen=True, slots=True, kw_only=True)
class _LinearBuffers:
    """Resolve the four buffer starts used by one matrix-product invocation.

    The graph emitter and nested attention products use the same typed layout,
    without inventing temporary tensor names to look addresses up again.

    Attributes:
        source: First slot of flattened input rows.
        weights: First slot of output-major weight rows.
        destination: First slot receiving flattened output rows.
        workspace: First slot of the two tile buffers and accumulator readback.
    """

    source: CentSharedBufferAddress
    weights: CentSharedBufferAddress
    destination: CentSharedBufferAddress
    workspace: CentSharedBufferAddress


def _emit_linear(
    command: LinearCommand,
    hardware: CentHardwareSpec,
    buffers: _LinearBuffers,
) -> tuple[CentInstruction, ...]:
    """Emit packed row tiles and persistent per-bank dot-product accumulations.

    Args:
        command: Matrix dimensions, symbolic buffers, and selected accumulator.
        hardware: Device geometry determining tile widths and output bank groups.
        buffers: Resolved input, weight, output, and temporary buffer starts.

    Returns:
        Explicit initialization, packing, bank writes, MACs, readback, and gathering.
    """
    width = hardware.burst_length
    row_values = command.tile_width
    row_slots = row_values // width
    workspace = buffers.workspace
    input_stage = workspace
    weight_stage = CentSharedBufferAddress(slot=workspace.slot + row_slots)
    readback = CentSharedBufferAddress(slot=workspace.slot + 2 * row_slots)
    channels = CentChannelSet(channels=(command.channel,))
    output_count = command.batch_count * command.output_width
    instructions: list[CentInstruction] = [
        ReferenceFill(destination=buffers.destination, value_count=output_count)
    ]
    for batch in range(command.batch_count):
        for first_output in range(0, command.output_width, command.bank_count):
            live_banks = min(command.bank_count, command.output_width - first_output)
            instructions.append(
                InitializeAccumulator(
                    channels=channels,
                    accumulation_register=command.accumulation_register,
                )
            )
            for first_input in range(0, command.input_width, row_values):
                count = min(row_values, command.input_width - first_input)
                tile_slots = ceil_div(count, width)
                padded_count = tile_slots * width
                # Scalar offsets allow odd-width rows to start in the middle
                # of a slot. Each tile is packed afresh and its tail is zero.
                instructions.append(
                    ReferenceCopy(
                        source=buffers.source,
                        destination=input_stage,
                        source_offset=batch * command.input_width + first_input,
                        value_count=count,
                        destination_count=padded_count,
                    )
                )
                instructions.append(
                    WriteGlobalBuffer(
                        source=input_stage,
                        channels=channels,
                        column=0,
                        operation_size=tile_slots,
                    )
                )
                for bank in range(command.bank_count):
                    if bank < live_banks:
                        instructions.append(
                            ReferenceCopy(
                                source=buffers.weights,
                                destination=weight_stage,
                                source_offset=(first_output + bank)
                                * command.input_width
                                + first_input,
                                value_count=count,
                                destination_count=padded_count,
                            )
                        )
                    else:
                        # Channel-wide MAC still reads unused output banks.
                        instructions.append(
                            ReferenceFill(
                                destination=weight_stage, value_count=padded_count
                            )
                        )
                    instructions.append(
                        WriteSingleBank(
                            address=CentMemoryAddress(
                                channel=command.channel, bank=bank, row=0, column=0
                            ),
                            source=weight_stage,
                            operation_size=tile_slots,
                        )
                    )
                # The selected register stays live across input tiles. A read
                # does not reset it; only initialization starts a new output group.
                instructions.append(
                    MacAllBanks(
                        channels=channels,
                        operation_size=tile_slots,
                        row=0,
                        column=0,
                        accumulation_register=command.accumulation_register,
                        operand_source=MacOperandSource.GLOBAL_BUFFER,
                    )
                )
            instructions.append(
                ReadAccumulator(
                    channels=channels,
                    accumulation_register=command.accumulation_register,
                    destination=readback,
                )
            )
            instructions.append(
                ReferenceStoreSlice(
                    source=readback,
                    destination=buffers.destination,
                    source_offset=0,
                    destination_offset=batch * command.output_width + first_output,
                    value_count=live_banks,
                    destination_count=output_count,
                )
            )
    return tuple(instructions)


def _emit_rms_norm(
    command: RmsNormCommand,
    hardware: CentHardwareSpec,
    addresses: dict[str, CentSharedBufferAddress],
    workspace: CentSharedBufferAddress,
) -> tuple[CentInstruction, ...]:
    """Normalize each row using visible arithmetic and packing stages.

    Args:
        command: Input, learned scale, row widths, and epsilon.
        hardware: Slot and row widths used to tile elementwise products.
        addresses: Numeric starts of graph buffers.
        workspace: Temporary slots for a packed input row, work vector, and scalar.

    Returns:
        Square, reduction, scalar-scale, broadcast, two products, and output stores.
    """
    slots = ceil_div(command.width, hardware.burst_length)
    original = workspace
    work = CentSharedBufferAddress(slot=workspace.slot + slots)
    scalar = CentSharedBufferAddress(slot=workspace.slot + 2 * slots)
    row_slots = command.tile_slots
    output_count = command.width * command.batch_count
    instructions: list[CentInstruction] = [
        ReferenceFill(
            destination=addresses[command.destination], value_count=output_count
        )
    ]
    for batch in range(command.batch_count):
        for destination in (original, work):
            instructions.append(
                ReferenceCopy(
                    source=addresses[command.source],
                    destination=destination,
                    source_offset=batch * command.width,
                    value_count=command.width,
                    destination_count=command.width,
                )
            )
        for offset in range(0, slots, row_slots):
            count = min(row_slots, slots - offset)
            squared = CentSharedBufferAddress(slot=work.slot + offset)
            instructions.extend(_emit_multiply_chunk(hardware, squared, squared, count))
        # The reduction and denominator use the logical width, never padding.
        # These are declared reference scalar primitives, not invented RED/RISCV operands.
        instructions.append(
            ReferenceReduceSum(
                source=work, destination=scalar, value_count=command.width
            )
        )
        instructions.append(
            ReferenceRmsScale(
                source=scalar,
                destination=scalar,
                value_count=command.width,
                epsilon=command.epsilon,
            )
        )
        instructions.append(
            ReferenceBroadcast(
                source=scalar, destination=work, value_count=command.width
            )
        )
        for source in (original, addresses[command.scale]):
            for offset in range(0, slots, row_slots):
                instructions.extend(
                    _emit_multiply_chunk(
                        hardware,
                        CentSharedBufferAddress(slot=source.slot + offset),
                        CentSharedBufferAddress(slot=work.slot + offset),
                        min(row_slots, slots - offset),
                    )
                )
        instructions.append(
            ReferenceStoreSlice(
                source=work,
                destination=addresses[command.destination],
                source_offset=0,
                destination_offset=batch * command.width,
                value_count=command.width,
                destination_count=output_count,
            )
        )
    return tuple(instructions)
