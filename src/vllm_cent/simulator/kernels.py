"""Prepare immutable physical effects for supported CENT instructions."""

from vllm_cent.cent import (
    BANKS_PER_PU,
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
    """Prepare ``WR_SBK`` by snapshotting complete source slots.

    Args:
        instruction: Single-bank write to prepare.
        state: Device state containing the Shared Buffer source.

    Returns:
        Shared Buffer read and one unchanged stored-value DRAM write.

    Raises:
        CentUninitializedReadError: If any source lane is uninitialized.
    """

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
    """Prepare ``RD_SBK`` by snapshotting its complete DRAM source span.

    Args:
        instruction: Single-bank read to prepare.
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
    """Prepare ``WR_GB`` writes for every selected channel.

    Args:
        instruction: Broadcast Global Buffer write to prepare.
        state: Device state containing the Shared Buffer source.

    Returns:
        Shared Buffer read and one stored-value Global Buffer write per channel.

    Raises:
        CentUninitializedReadError: If any source lane is uninitialized.
    """

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
    """Prepare each selected channel's ``COPY_BKGB`` write.

    Args:
        instruction: Bank-to-Global-Buffer copy to prepare.
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
    """Prepare each selected channel's ``COPY_GBBK`` write.

    Args:
        instruction: Global-Buffer-to-bank copy to prepare.
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
    """Prepare every four-bank group's lane-wise ``EW_MUL`` result.

    Args:
        instruction: Elementwise multiplication to prepare.
        state: Device state containing all selected operand banks.

    Returns:
        Two operand reads and one result write per selected PU group.

    Raises:
        CentUninitializedReadError: If any operand scalar is uninitialized.
        _KernelArithmeticError: If the numeric policy rejects multiplication.
    """

    scalar_count = instruction.operation_size * state.hardware.burst_length
    reads: list[CentDramRegion] = []
    writes: list[CentWriteEffect] = []
    for channel in instruction.channels.channels:
        for group_start in range(0, state.hardware.num_banks, BANKS_PER_PU):
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
    """Prepare lane-wise ``ACC`` sums from complete source snapshots.

    Args:
        instruction: Shared Buffer accumulation to prepare.
        state: Device state containing both input spans.

    Returns:
        Both Shared Buffer reads and one in-place destination write.

    Raises:
        CentUninitializedReadError: If either input span is incomplete.
        _KernelArithmeticError: If the numeric policy rejects addition.
    """

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
