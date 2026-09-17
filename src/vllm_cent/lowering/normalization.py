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
from .utils import (
    _plan_partitioned_vector,
    _require_dram_row_capacity,
    _require_shared_buffer_capacity,
)

__all__ = ["lower_l2_norm", "lower_rms_norm"]


_INPUT_BANK_GROUP = 0
_SCALE_BANK_GROUP = 1
_OUTPUT_BANK_GROUP = 2
_SUM_OF_SQUARES_REGISTER = 0


def _lower_sum_of_squares(
    builder: CentProgramBuilder,
    *,
    input_rows: CentDramRowRange,
    input_buffer: CentSharedBufferSpan,
    partial_sum_buffer: CentSharedBufferSpan,
    value_count: int,
) -> None:
    """Square vector values and collect the hardware's partial sums.

    Args:
        builder: Program builder that receives the instructions.
        input_rows: DRAM workspace used for two copies of the input vector.
        input_buffer: Shared Buffer slots containing the input vector.
        partial_sum_buffer: Slot receiving the accumulator results.
        value_count: Number of values in the input vector.

    Raises:
        ValueError: If a size is invalid or a memory region is too small.
    """

    require_positive("value_count", value_count)

    hardware = builder.hardware
    pair_count = builder.total_banks // 2
    vector_layout = _plan_partitioned_vector(
        value_count,
        pair_count,
        hardware.burst_length,
    )
    required_rows = ceil_div(
        vector_layout.values_per_partition,
        hardware.dram_columns,
    )
    _require_dram_row_capacity("input_rows", input_rows, required_rows)
    _require_shared_buffer_capacity(
        "input_buffer", input_buffer, vector_layout.slot_count
    )
    _require_shared_buffer_capacity(
        "partial_sum_buffer", partial_sum_buffer, 1
    )

    # Every value is written to both banks in its neighboring pair. MAC_ABK
    # multiplies those matching copies, which squares the original value.
    for bank_group in (_INPUT_BANK_GROUP, _SCALE_BANK_GROUP):
        builder.emit_neighbor_bank_transfer(
            WriteSingleBank,
            value_count,
            bank_group,
            input_rows.start_row,
            vector_layout.values_per_partition,
            shared_buffer=input_buffer.start,
        )

    channels = builder.all_channels()
    builder.append(
        WriteBias(source=partial_sum_buffer.start, channels=channels)
    )
    builder.append(
        MacAllBanks(
            operation_size=ceil_div(
                vector_layout.values_per_partition,
                hardware.burst_length,
            ),
            channels=channels,
            row=input_rows.start_row,
            column=0,
            accumulation_register=_SUM_OF_SQUARES_REGISTER,
        )
    )
    builder.append(
        ReadMac(
            destination=partial_sum_buffer.start,
            accumulation_register=_SUM_OF_SQUARES_REGISTER,
            channels=channels,
        )
    )


def lower_l2_norm(
    builder: CentProgramBuilder,
    *,
    input_rows: CentDramRowRange,
    work_rows: CentDramRowRange,
    input_buffer: CentSharedBufferSpan,
    scale_buffer: CentSharedBufferSpan,
    partial_sum_buffer: CentSharedBufferSpan,
    value_count: int,
) -> None:
    """Emit the supported path that scales a vector by its norm.

    The normalized vector remains in bank group two of ``work_rows``. This
    location lets a following operation consume the result without returning
    it to the Shared Buffer first.

    Args:
        builder: Program builder that receives the instructions.
        input_rows: DRAM workspace used to calculate squared partial sums.
        work_rows: DRAM workspace receiving the input and normalization scale.
        input_buffer: Shared Buffer slots containing the input vector.
        scale_buffer: Shared Buffer slots containing the repeated scale.
        partial_sum_buffer: Slot receiving the sum-of-squares partial results.
        value_count: Number of values in the input vector.

    Raises:
        ValueError: If a size is invalid or a memory region is too small.
    """

    # TODO(L2 norm): Produce the normalization scale from the partial sums.
    #
    # The instructions below calculate squared partial sums, but they do not
    # combine results from every bank and channel or calculate a square root.
    # The caller must currently provide the repeated scale in ``scale_buffer``.

    require_positive("value_count", value_count)

    hardware = builder.hardware
    group_count = builder.total_banks // BANKS_PER_PU
    vector_layout = _plan_partitioned_vector(
        value_count,
        group_count,
        hardware.burst_length,
    )
    _require_shared_buffer_capacity(
        "scale_buffer", scale_buffer, vector_layout.slot_count
    )
    # Calculate the partial sums before consuming the caller-provided scale.
    _lower_sum_of_squares(
        builder,
        input_rows=input_rows,
        input_buffer=input_buffer,
        partial_sum_buffer=partial_sum_buffer,
        value_count=value_count,
    )

    required_rows = ceil_div(
        vector_layout.values_per_partition,
        hardware.dram_columns,
    )
    _require_dram_row_capacity("work_rows", work_rows, required_rows)

    channels_per_block = builder.placement.channels_per_block
    builder.emit_bank_group_transfer(
        WriteSingleBank,
        channels_per_block,
        vector_layout.partition_count,
        _INPUT_BANK_GROUP,
        work_rows.start_row,
        vector_layout.values_per_partition,
        shared_buffer=input_buffer.start,
    )
    builder.emit_bank_group_transfer(
        WriteSingleBank,
        channels_per_block,
        vector_layout.partition_count,
        _SCALE_BANK_GROUP,
        work_rows.start_row,
        vector_layout.values_per_partition,
        shared_buffer=scale_buffer.start,
    )

    # EW_MUL uses the first two banks in each PU as operands and leaves the
    # scaled vector in the third bank. These roles follow the reference code.
    builder.append(
        ElementwiseMultiply(
            operation_size=ceil_div(
                vector_layout.values_per_partition,
                hardware.burst_length,
            ),
            channels=builder.all_channels(),
            row=work_rows.start_row,
            column=0,
        )
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

    # TODO(RMSNorm): Add the missing RMSNorm scale calculation.
    #
    # The code below produces partial sums of squares. It does not yet combine
    # all banks and channels, divide by vector length, add epsilon, calculate
    # the reciprocal square root, or write that scale into ``scale_buffer``.

    require_positive("value_count", value_count)
    hardware = builder.hardware
    channels_per_block = builder.placement.channels_per_block
    channels = builder.all_channels()

    # Plan the learned-weight pass, then let L2 normalization validate and emit
    # the shared prefix. Keeping that prefix in one public lowerer prevents the
    # two normalization paths from drifting apart.
    group_count = builder.total_banks // BANKS_PER_PU
    vector_layout = _plan_partitioned_vector(
        value_count,
        group_count,
        hardware.burst_length,
    )
    lower_l2_norm(
        builder,
        input_rows=input_rows,
        work_rows=work_rows,
        input_buffer=input_buffer,
        scale_buffer=scale_buffer,
        partial_sum_buffer=partial_sum_buffer,
        value_count=value_count,
    )

    _require_shared_buffer_capacity(
        "output_buffer", output_buffer, vector_layout.slot_count
    )
    grouped_rows = ceil_div(
        vector_layout.values_per_partition,
        hardware.dram_columns,
    )
    _require_dram_row_capacity("weight_rows", weight_rows, grouped_rows)

    operation_size = ceil_div(
        vector_layout.values_per_partition,
        hardware.burst_length,
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
        vector_layout.partition_count,
        _OUTPUT_BANK_GROUP,
        weight_rows.start_row,
        vector_layout.values_per_partition,
        shared_buffer=output_buffer.start,
    )
