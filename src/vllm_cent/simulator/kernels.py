"""Read instruction operands and compute proposed writes without applying them.

A kernel is the function implementing one instruction's calculation or copy.
It reads every source before returning the writes to commit. Instruction
``operation_size`` counts transfer bursts, each containing
``hardware.burst_length`` scalars. DRAM means dynamic random-access memory.
"""

from vllm_cent.cent import (
    BANKS_PER_GROUP,
    Accumulate,
    CentGlobalBufferAddress,
    CentMemoryAddress,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    ReadSingleBank,
    WriteGlobalBuffer,
    WriteSingleBank,
)
from vllm_cent.runtime import (
    CentDramRegion,
    CentGlobalBufferRegion,
    CentSharedBufferRegion,
)

from .effects import (
    CentWriteEffect,
    DramWriteEffect,
    GlobalBufferWriteEffect,
    SharedBufferWriteEffect,
    _InstructionEffects,
)
from .errors import CentSimulationLocation
from .state import CentDeviceState

__all__: list[str] = []


class _KernelArithmeticError(Exception):
    """Preserve the destination associated with rejected arithmetic.

    Attributes:
        reason: Numeric-policy failure in plain language.
        location: First destination scalar of the result being computed.
    """

    reason: str
    location: CentSimulationLocation

    def __init__(self, reason: str, *, location: CentSimulationLocation) -> None:
        """Create one internal arithmetic failure with physical context.

        Args:
            reason: Numeric-policy failure in plain language.
            location: Destination associated with the rejected result.
        """

        self.reason = reason
        self.location = location
        super().__init__(reason)


def _prepare_write_single_bank(
    instruction: WriteSingleBank,
    state: CentDeviceState,
) -> _InstructionEffects:
    """Copy Shared Buffer values to one memory bank (``WR_SBK``).

    Args:
        instruction: Source Shared Buffer slot, destination bank address, and
            number of complete slots to copy.
        state: Device state containing the Shared Buffer source.

    Returns:
        Shared Buffer read and one unchanged stored-value DRAM write.

    Raises:
        CentUninitializedReadError: If any source lane is uninitialized.
    """

    # Reading the full source first prevents a missing final lane from leaving
    # a partially copied destination. Transfers preserve stored numbers as-is.
    values = state.read_shared_buffer(
        instruction.source,
        slot_count=instruction.operation_size,
    )
    return _InstructionEffects(
        reads=(
            CentSharedBufferRegion(
                address=instruction.source,
                scalar_count=len(values),
            ),
        ),
        writes=(DramWriteEffect(address=instruction.address, values=values),),
    )


def _prepare_read_single_bank(
    instruction: ReadSingleBank,
    state: CentDeviceState,
) -> _InstructionEffects:
    """Copy consecutive values from one memory bank to Shared Buffer (``RD_SBK``).

    Args:
        instruction: Source bank address, destination Shared Buffer slot, and
            number of bursts to copy.
        state: Device state containing the DRAM source.

    Returns:
        DRAM read and one unchanged stored-value Shared Buffer write.

    Raises:
        CentUninitializedReadError: If any source scalar is uninitialized.
    """

    scalar_count = instruction.operation_size * state.hardware.burst_length
    values = state.read_dram(instruction.address, value_count=scalar_count)
    return _InstructionEffects(
        reads=(
            CentDramRegion(
                address=instruction.address,
                scalar_count=scalar_count,
            ),
        ),
        writes=(
            SharedBufferWriteEffect(
                address=instruction.destination,
                values=values,
            ),
        ),
    )


def _prepare_write_global_buffer(
    instruction: WriteGlobalBuffer,
    state: CentDeviceState,
) -> _InstructionEffects:
    """Copy the same Shared Buffer values to each selected Global Buffer (``WR_GB``).

    Args:
        instruction: Source Shared Buffer slot, selected channels, starting
            Global Buffer column, and number of slots to copy.
        state: Device state containing the Shared Buffer source.

    Returns:
        Shared Buffer read and one stored-value Global Buffer write per channel.

    Raises:
        CentUninitializedReadError: If any source lane is uninitialized.
    """

    # Reading the full source first prevents a missing final lane from leaving
    # a partially copied destination. Transfers preserve stored numbers as-is.
    values = state.read_shared_buffer(
        instruction.source,
        slot_count=instruction.operation_size,
    )
    return _InstructionEffects(
        reads=(
            CentSharedBufferRegion(
                address=instruction.source,
                scalar_count=len(values),
            ),
        ),
        writes=tuple(
            GlobalBufferWriteEffect(
                address=CentGlobalBufferAddress(
                    channel=channel,
                    column=instruction.column,
                ),
                values=values,
            )
            for channel in instruction.channels.channels
        ),
    )


def _prepare_copy_bank_to_global_buffer(
    instruction: CopyBankToGlobalBuffer,
    state: CentDeviceState,
) -> _InstructionEffects:
    """Copy each channel's selected bank into its Global Buffer (``COPY_BKGB``).

    Args:
        instruction: Channels, source bank/row/column, and number of bursts.
            Each channel writes its own Global Buffer at the same column.
        state: Device state containing each selected channel's DRAM source.

    Returns:
        Per-channel DRAM reads and Global Buffer writes in channel order.

    Raises:
        CentUninitializedReadError: If any selected DRAM source is incomplete.
    """

    scalar_count = instruction.operation_size * state.hardware.burst_length
    reads: list[CentDramRegion] = []
    writes: list[CentWriteEffect] = []
    for channel in instruction.channels.channels:
        source = CentMemoryAddress(
            channel=channel,
            bank=instruction.bank,
            row=instruction.row,
            column=instruction.column,
        )
        reads.append(CentDramRegion(address=source, scalar_count=scalar_count))
        writes.append(
            GlobalBufferWriteEffect(
                address=CentGlobalBufferAddress(
                    channel=channel,
                    column=instruction.column,
                ),
                values=state.read_dram(source, value_count=scalar_count),
            )
        )
    return _InstructionEffects(reads=tuple(reads), writes=tuple(writes))


def _prepare_copy_global_buffer_to_bank(
    instruction: CopyGlobalBufferToBank,
    state: CentDeviceState,
) -> _InstructionEffects:
    """Copy each channel's Global Buffer into its selected bank (``COPY_GBBK``).

    Args:
        instruction: Channels, destination bank/row/column, and burst count.
            Each channel reads its own Global Buffer at the same column.
        state: Device state containing each selected channel's Global Buffer.

    Returns:
        Per-channel Global Buffer reads and DRAM writes in channel order.

    Raises:
        CentUninitializedReadError: If any selected Global Buffer is incomplete.
    """

    scalar_count = instruction.operation_size * state.hardware.burst_length
    reads: list[CentGlobalBufferRegion] = []
    writes: list[CentWriteEffect] = []
    for channel in instruction.channels.channels:
        source = CentGlobalBufferAddress(
            channel=channel,
            column=instruction.column,
        )
        reads.append(CentGlobalBufferRegion(address=source, scalar_count=scalar_count))
        writes.append(
            DramWriteEffect(
                address=CentMemoryAddress(
                    channel=channel,
                    bank=instruction.bank,
                    row=instruction.row,
                    column=instruction.column,
                ),
                values=state.read_global_buffer(
                    source,
                    value_count=scalar_count,
                ),
            )
        )
    return _InstructionEffects(reads=tuple(reads), writes=tuple(writes))


def _prepare_elementwise_multiply(
    instruction: ElementwiseMultiply,
    state: CentDeviceState,
) -> _InstructionEffects:
    """Multiply matching scalar positions within each four-bank group (``EW_MUL``).

    Each processing unit (PU) serves four banks. Fixed bank offsets select its
    two operands and result within that group.

    Args:
        instruction: Channels, row, starting column, and burst count shared
            by all four-bank groups participating in the multiplication.
        state: Device state containing all selected operand banks.

    Returns:
        Two operand ranges and one proposed result write per four-bank group.

    Raises:
        CentUninitializedReadError: If any operand scalar is uninitialized.
        _KernelArithmeticError: If the numeric policy rejects multiplication.

    Examples:
        Groups beginning at banks 0 and 4 use the same operand and result
        offsets within their own group; operands from the two groups never mix.
    """

    scalar_count = instruction.operation_size * state.hardware.burst_length
    reads: list[CentDramRegion] = []
    writes: list[CentWriteEffect] = []
    # Banks are grouped by processing unit. Adding the fixed operand offsets
    # to each group start keeps both operands within that same group.
    for channel in instruction.channels.channels:
        for group_start in range(0, state.hardware.num_banks, BANKS_PER_GROUP):
            first_address = CentMemoryAddress(
                channel=channel,
                bank=group_start + ElementwiseMultiply.FIRST_OPERAND_BANK,
                row=instruction.row,
                column=instruction.column,
            )
            second_address = CentMemoryAddress(
                channel=channel,
                bank=group_start + ElementwiseMultiply.SECOND_OPERAND_BANK,
                row=instruction.row,
                column=instruction.column,
            )
            reads.extend(
                (
                    CentDramRegion(
                        address=first_address,
                        scalar_count=scalar_count,
                    ),
                    CentDramRegion(
                        address=second_address,
                        scalar_count=scalar_count,
                    ),
                )
            )
            first_values = state.read_dram(first_address, value_count=scalar_count)
            second_values = state.read_dram(second_address, value_count=scalar_count)
            result_address = CentMemoryAddress(
                channel=channel,
                bank=group_start + ElementwiseMultiply.RESULT_BANK,
                row=instruction.row,
                column=instruction.column,
            )
            try:
                products = tuple(
                    state.numeric.multiply(first, second)
                    for first, second in zip(
                        first_values,
                        second_values,
                        strict=True,
                    )
                )
            except ValueError as error:
                raise _KernelArithmeticError(
                    str(error),
                    location=result_address,
                ) from error
            writes.append(
                DramWriteEffect(
                    address=result_address,
                    values=products,
                )
            )
    return _InstructionEffects(reads=tuple(reads), writes=tuple(writes))


def _prepare_accumulate(
    instruction: Accumulate,
    state: CentDeviceState,
) -> _InstructionEffects:
    """Add matching Shared Buffer positions into the destination (``ACC``).

    Both source ranges are read before any write is applied. If source and
    destination are the same range, each old value is therefore doubled.
    Partly overlapping ranges must already have been rejected by preflight.

    Args:
        instruction: Source slot, destination slot, and number of slots.
            Each destination value is added to the corresponding source value.
        state: Device state containing both input spans.

    Returns:
        The two read ranges and a proposed write replacing destination values.

    Raises:
        CentUninitializedReadError: If either input span is incomplete.
        _KernelArithmeticError: If the numeric policy rejects addition.
    """

    # Snapshot both operands before calculating writes. Exact source/destination
    # overlap must use the old values for every lane, including later lanes.
    destination_values = state.read_shared_buffer(
        instruction.destination,
        slot_count=instruction.operation_size,
    )
    source_values = state.read_shared_buffer(
        instruction.source,
        slot_count=instruction.operation_size,
    )
    try:
        sums = tuple(
            state.numeric.add(destination, source)
            for destination, source in zip(
                destination_values,
                source_values,
                strict=True,
            )
        )
    except ValueError as error:
        raise _KernelArithmeticError(
            str(error),
            location=instruction.destination,
        ) from error
    scalar_count = len(sums)
    return _InstructionEffects(
        reads=(
            CentSharedBufferRegion(
                address=instruction.destination,
                scalar_count=scalar_count,
            ),
            CentSharedBufferRegion(
                address=instruction.source,
                scalar_count=scalar_count,
            ),
        ),
        writes=(
            SharedBufferWriteEffect(
                address=instruction.destination,
                values=sums,
            ),
        ),
    )
