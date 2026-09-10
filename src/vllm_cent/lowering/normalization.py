"""Lower reusable normalization operations to CENT instructions."""

from ..cent import (
    BANKS_PER_PU,
    CentProgramBuilder,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    MacAllBanks,
    ReadMac,
    ReadSingleBank,
    WriteBias,
    WriteSingleBank,
    ceil_div,
)
from ..cent.utils import require_positive
from .bindings import CentDramRowRange, CentSharedBufferSpan

__all__ = ["lower_rms_norm"]


def _require_buffer_slots(
    name: str,
    span: CentSharedBufferSpan,
    slot_count: int,
) -> None:
    """Check that a normalization operand has enough staging slots.

    Args:
        name: Operand name used in an error message.
        span: Shared Buffer region assigned to the operand.
        slot_count: Slots accessed by the lowering.

    Raises:
        ValueError: If the assigned span is too small.
    """

    if span.slot_count < slot_count:
        raise ValueError(
            f"{name} needs {slot_count} Shared Buffer slots, "
            f"but its span contains {span.slot_count}"
        )


def lower_rms_norm(
    builder: CentProgramBuilder,
    *,
    input_rows: CentDramRowRange,
    work_rows: CentDramRowRange,
    weight_rows: CentDramRowRange,
    input_buffer: CentSharedBufferSpan,
    scale_buffer: CentSharedBufferSpan,
    partial_sum_buffer: CentSharedBufferSpan,
    output_buffer: CentSharedBufferSpan,
    value_count: int,
) -> None:
    """Emit the currently supported parts of RMS normalization.

    Args:
        builder: Program builder that receives the instructions.
        input_rows: DRAM workspace receiving copies of the input vector.
        work_rows: DRAM workspace receiving the input and RMS scale.
        weight_rows: DRAM rows containing learned weights and the final product.
        input_buffer: Shared Buffer slots containing the input vector.
        scale_buffer: Shared Buffer slots reserved for the repeated RMS scale.
        partial_sum_buffer: Slot carrying the sum-of-squares partial result.
        output_buffer: Shared Buffer slots receiving the normalized vector.
        value_count: Number of values in the vector.

    Raises:
        ValueError: If a size is invalid or a memory region is too small.
    """

    # TODO(RMSNorm): Add the missing RMSNorm steps.
    #
    # The code below produces partial sums of squares. It does not yet combine
    # all banks and channels, divide by vector length, add epsilon, calculate
    # the reciprocal square root, or write that scale into ``scale_buffer``.

    require_positive("value_count", value_count)
    hardware = builder.hardware
    channels_per_block = builder.placement.channels_per_block

    channels = builder.all_channels()

    # Put two copies of the input in neighboring banks. MAC_ABK multiplies the
    # pairs and leaves a partial sum of squares in accumulator register zero.
    neighbor_length = ceil_div(value_count, builder.total_banks // 2)
    neighbor_rows = ceil_div(neighbor_length, hardware.dram_columns)
    if input_rows.row_count < neighbor_rows:
        raise ValueError(
            f"input_rows needs {neighbor_rows} DRAM rows, "
            f"but its range contains {input_rows.row_count}"
        )
    for bank_group in (0, 1):
        builder.emit_neighbor_bank_transfer(
            WriteSingleBank,
            value_count,
            bank_group,
            input_rows.start_row,
            neighbor_length,
            shared_buffer=input_buffer.start,
        )
    builder.append(
        WriteBias(source=partial_sum_buffer.start, channels=channels)
    )
    builder.append(
        MacAllBanks(
            operation_size=ceil_div(
                neighbor_length, hardware.burst_length
            ),
            channels=channels,
            row=input_rows.start_row,
            column=0,
            accumulation_register=0,
        )
    )
    builder.append(
        ReadMac(
            destination=partial_sum_buffer.start,
            accumulation_register=0,
            channels=channels,
        )
    )

    # Place the input and repeated scale in two banks of each four-bank PU.
    # EW_MUL writes the scaled vector into the third bank.
    group_count = builder.total_banks // BANKS_PER_PU
    group_length = ceil_div(value_count, group_count)
    utilized_groups = ceil_div(value_count, group_length)
    # Every PU partition begins at a Shared Buffer slot boundary. Round each
    # partition separately so its padding is included in the capacity check.
    partition_slots = ceil_div(group_length, hardware.burst_length)
    grouped_vector_slots = utilized_groups * partition_slots
    _require_buffer_slots("input_buffer", input_buffer, grouped_vector_slots)
    _require_buffer_slots("scale_buffer", scale_buffer, grouped_vector_slots)
    _require_buffer_slots("partial_sum_buffer", partial_sum_buffer, 1)
    _require_buffer_slots("output_buffer", output_buffer, grouped_vector_slots)
    grouped_rows = ceil_div(group_length, hardware.dram_columns)
    for name, rows in (
        ("work_rows", work_rows),
        ("weight_rows", weight_rows),
    ):
        if rows.row_count < grouped_rows:
            raise ValueError(
                f"{name} needs {grouped_rows} DRAM rows, "
                f"but its range contains {rows.row_count}"
            )
    builder.emit_bank_group_transfer(
        WriteSingleBank,
        channels_per_block,
        utilized_groups,
        0,
        work_rows.start_row,
        group_length,
        shared_buffer=input_buffer.start,
    )
    builder.emit_bank_group_transfer(
        WriteSingleBank,
        channels_per_block,
        utilized_groups,
        1,
        work_rows.start_row,
        group_length,
        shared_buffer=scale_buffer.start,
    )
    operation_size = ceil_div(group_length, hardware.burst_length)
    builder.append(
        ElementwiseMultiply(
            operation_size=operation_size,
            channels=channels,
            row=work_rows.start_row,
            column=0,
        )
    )

    # The copy pair moves the scaled vector beside the learned weights. The
    # second multiplication applies those weights, completing the known path.
    builder.append(
        CopyBankToGlobalBuffer(
            operation_size=operation_size,
            channels=channels,
            row=work_rows.start_row,
            column=0,
        )
    )
    builder.append(
        CopyGlobalBufferToBank(
            operation_size=operation_size,
            channels=channels,
            row=weight_rows.start_row,
            column=0,
        )
    )
    builder.append(
        ElementwiseMultiply(
            operation_size=operation_size,
            channels=channels,
            row=weight_rows.start_row,
            column=0,
        )
    )
    builder.emit_bank_group_transfer(
        ReadSingleBank,
        channels_per_block,
        utilized_groups,
        2,
        weight_rows.start_row,
        group_length,
        shared_buffer=output_buffer.start,
    )
