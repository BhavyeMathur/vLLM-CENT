"""Lower reusable normalization operations to CENT instructions."""

from dataclasses import dataclass

from ..cent import (
    BANKS_PER_PU,
    CentProgramBuilder,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    MacAllBanks,
    MacOperandSource,
    ReadMac,
    ReadSingleBank,
    WriteBias,
    WriteSingleBank,
)
from ..cent.utils import require_nonnegative
from .bindings import (
    CentDramRowRange,
    CentSharedBufferSpan,
    CentSharedBufferVector,
)
from .planning import CentPartitionedVectorLayout
from .utils import (
    _row_operation_sizes,
    _require_dram_row_capacity,
    _require_shared_buffer_capacity,
)

__all__ = [
    "CentL2NormPlan",
    "CentRmsNormPlan",
    "CentSumOfSquaresPlan",
    "lower_l2_norm",
    "lower_rms_norm",
    "lower_sum_of_squares",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class CentSumOfSquaresPlan:
    """Bind a sum-of-squares layout to its working memory.

    Attributes:
        input_rows: DRAM workspace used for two copies of the input vector.
        input_buffer: Zero-padded input vector in the Shared Buffer.
        partial_sum_buffer: Slot receiving the accumulator results.
        layout: Vector split across neighboring bank pairs.
        accumulation_register: Register used to collect squared partial sums.
    """

    input_rows: CentDramRowRange
    input_buffer: CentSharedBufferVector
    partial_sum_buffer: CentSharedBufferSpan
    layout: CentPartitionedVectorLayout
    accumulation_register: int = 0

    def __post_init__(self) -> None:
        """Validate the register and input-vector layout.

        Raises:
            ValueError: If ``accumulation_register`` is negative or the input
                binding uses a different layout.
        """

        require_nonnegative("accumulation_register", self.accumulation_register)
        if self.input_buffer.layout != self.layout:
            raise ValueError("input_buffer layout must match sum-of-squares layout")


@dataclass(frozen=True, slots=True, kw_only=True)
class CentL2NormPlan:
    """Describe the supported L2-normalization dataflow.

    Attributes:
        sum_of_squares: Plan that produces the squared partial sums.
        work_rows: DRAM workspace receiving the input and normalization scale.
        scale_buffer: Zero-padded repeated scale in the Shared Buffer.
        layout: Vector split across four-bank PU groups.
    """

    sum_of_squares: CentSumOfSquaresPlan
    work_rows: CentDramRowRange
    scale_buffer: CentSharedBufferVector
    layout: CentPartitionedVectorLayout

    def __post_init__(self) -> None:
        """Check that both normalization stages describe the same vector.

        Raises:
            ValueError: If the stage layouts describe different vectors or
                incompatible Shared Buffer packing.
        """

        if self.layout.value_count != self.sum_of_squares.layout.value_count:
            raise ValueError("L2 layouts must describe the same value_count")
        if self.layout.burst_length != self.sum_of_squares.layout.burst_length:
            raise ValueError("L2 layouts must use the same burst_length")
        if self.scale_buffer.layout != self.layout:
            raise ValueError("scale_buffer layout must match the L2 layout")
        if self.layout != self.sum_of_squares.layout and not (
            self.layout.is_contiguously_packed
            and self.sum_of_squares.layout.is_contiguously_packed
        ):
            raise ValueError(
                "L2 layouts need repacking when partition padding separates values"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class CentRmsNormPlan:
    """Describe the supported RMS-normalization dataflow.

    Attributes:
        l2_norm: Shared sum-of-squares and scaling prefix.
        weight_rows: DRAM rows containing learned weights and the final product.
        output_buffer: Zero-padded Shared Buffer destination for the result.
    """

    l2_norm: CentL2NormPlan
    weight_rows: CentDramRowRange
    output_buffer: CentSharedBufferVector

    def __post_init__(self) -> None:
        """Require the output to use the final normalization layout.

        Raises:
            ValueError: If the output vector layout differs from the L2 layout.
        """

        if self.output_buffer.layout != self.l2_norm.layout:
            raise ValueError("output_buffer layout must match the L2 layout")


def lower_sum_of_squares(
    builder: CentProgramBuilder,
    plan: CentSumOfSquaresPlan,
) -> None:
    """Emit the currently supported partial-sum stage for squared values.

    This function does not yet combine the bank-pair and channel partials into
    the single scalar implied by "sum of squares."

    Args:
        builder: Program builder that receives the instructions.
        plan: Neighbor-bank vector layout and its memory bindings.

    Raises:
        ValueError: If a size is invalid or a memory region is too small.
    """

    # TODO(sum of squares): Reduce every partial into one scalar result.
    #
    # MAC_ABK produces one partial sum for each active neighboring bank pair.
    # RD_MAC exposes those values, but this function does not yet combine the
    # pairs within one channel or combine the channels holding one vector. We
    # first need the RD_MAC destination layout and the exact RED semantics.

    # TODO(runtime): Initialize the first MAC values to zero.
    #
    # WR_BIAS reads ``partial_sum_buffer`` before RD_MAC writes any result there.
    # The program therefore depends on that slot already containing zeros. The
    # runtime must explicitly provide a zero-filled source slot.

    # TODO(placement): Separate vector partitions from replicated block copies.
    #
    # The transfer helper repeats a block layout in every equal channel region.
    # A final reduction must combine channels that partition one vector without
    # also adding identical results from another block copy.

    # TODO(dataflow): Keep unused bank pairs out of the final sum.
    #
    # The input binding guarantees zero padding inside every occupied partition.
    # MAC_ABK still runs bank pairs that the layout does not use. Those pairs
    # need zero initialization, masking, or exclusion from the later reduction.

    hardware = builder.hardware
    vector_layout = plan.layout
    if vector_layout.burst_length != hardware.burst_length:
        raise ValueError("layout burst_length does not match the target")
    pair_count = builder.total_banks // 2
    if vector_layout.partition_count > pair_count:
        raise ValueError("layout uses more bank pairs than the block owns")

    row_operation_sizes = _row_operation_sizes(
        vector_layout.physical_values_per_partition,
        hardware.dram_columns,
        hardware.burst_length,
    )
    _require_dram_row_capacity("input_rows", plan.input_rows, len(row_operation_sizes))
    # TODO(ISA): Size this span from the confirmed multi-channel RD_MAC layout.
    #
    # One slot is enough only if RD_MAC returns one channel or if every channel
    # has a private Shared Buffer namespace using the same slot number.
    _require_shared_buffer_capacity("partial_sum_buffer", plan.partial_sum_buffer, 1)

    # Every value is written to both banks in its neighboring pair. MAC_ABK
    # multiplies those matching copies, which squares the original value. The
    # input binding also supplies explicit zeros for every padding lane, so the
    # complete final bursts add no padding terms to an active pair's partial.
    for bank_group in (
        MacAllBanks.NEXT_BANK_FIRST_OPERAND_BANK,
        MacAllBanks.NEXT_BANK_SECOND_OPERAND_BANK,
    ):
        builder.emit_neighbor_bank_transfer(
            WriteSingleBank,
            vector_layout.physical_value_count,
            bank_group,
            plan.input_rows.start_row,
            vector_layout.physical_values_per_partition,
            shared_buffer=plan.input_buffer.start,
        )

    channels = builder.all_channels()
    builder.append(WriteBias(source=plan.partial_sum_buffer.start, channels=channels))
    for row_offset, operation_size in enumerate(row_operation_sizes):
        builder.append(
            MacAllBanks(
                operation_size=operation_size,
                channels=channels,
                row=plan.input_rows.row(row_offset),
                column=0,
                accumulation_register=plan.accumulation_register,
                operand_source=MacOperandSource.NEXT_BANK,
            )
        )
    builder.append(
        ReadMac(
            destination=plan.partial_sum_buffer.start,
            accumulation_register=plan.accumulation_register,
            channels=channels,
        )
    )


def lower_l2_norm(
    builder: CentProgramBuilder,
    plan: CentL2NormPlan,
) -> None:
    """Emit the supported path that scales a vector by its norm.

    The normalized vector remains in bank group two of ``work_rows``. This
    location lets a following operation consume the result without returning
    it to the Shared Buffer first.

    Args:
        builder: Program builder that receives the instructions.
        plan: Normalization layouts and their memory bindings.

    Raises:
        ValueError: If a size is invalid or a memory region is too small.
    """

    # TODO(L2 norm): Produce the normalization scale from the partial sums.
    #
    # The instructions below calculate squared partial sums, but they do not
    # combine results from every bank and channel or calculate a square root.
    # The caller must currently provide the repeated scale in ``scale_buffer``.

    # TODO(ISA): Define how unused PU groups are disabled.
    #
    # A layout can transfer data into fewer PU groups, but EW_MUL selects whole
    # channels. The remaining groups still execute unless their input banks are
    # initialized to harmless values or the hardware provides another mask.

    hardware = builder.hardware
    vector_layout = plan.layout
    if vector_layout.burst_length != hardware.burst_length:
        raise ValueError("layout burst_length does not match the target")
    group_count = builder.total_banks // BANKS_PER_PU
    if vector_layout.partition_count > group_count:
        raise ValueError("layout uses more PU groups than the block owns")

    row_operation_sizes = _row_operation_sizes(
        vector_layout.physical_values_per_partition,
        hardware.dram_columns,
        hardware.burst_length,
    )
    _require_dram_row_capacity("work_rows", plan.work_rows, len(row_operation_sizes))

    # Validate every L2 binding before the first instruction is emitted. A bad
    # scale workspace must not leave a completed sum-of-squares prefix behind.
    lower_sum_of_squares(
        builder,
        plan.sum_of_squares,
    )

    channels_per_block = builder.placement.channels_per_block
    # The input and scale bindings contain zero in the same padding lanes.
    # Multiplication therefore leaves zero in the normalized DRAM result.
    builder.emit_bank_group_transfer(
        WriteSingleBank,
        channels_per_block,
        vector_layout.partition_count,
        ElementwiseMultiply.FIRST_OPERAND_BANK,
        plan.work_rows.start_row,
        vector_layout.physical_values_per_partition,
        shared_buffer=plan.sum_of_squares.input_buffer.start,
    )
    builder.emit_bank_group_transfer(
        WriteSingleBank,
        channels_per_block,
        vector_layout.partition_count,
        ElementwiseMultiply.SECOND_OPERAND_BANK,
        plan.work_rows.start_row,
        vector_layout.physical_values_per_partition,
        shared_buffer=plan.scale_buffer.start,
    )

    # EW_MUL uses the first two banks in each PU as operands and leaves the
    # scaled vector in the third bank. These roles follow the reference code.
    for row_offset, operation_size in enumerate(row_operation_sizes):
        builder.append(
            ElementwiseMultiply(
                operation_size=operation_size,
                channels=builder.all_channels(),
                row=plan.work_rows.row(row_offset),
                column=0,
            )
        )


def lower_rms_norm(
    builder: CentProgramBuilder,
    plan: CentRmsNormPlan,
) -> None:
    """Emit the currently supported parts of RMS normalization.

    Args:
        builder: Program builder that receives the instructions.
        plan: RMS-normalization layouts and their memory bindings.

    Raises:
        ValueError: If a size is invalid or a memory region is too small.
    """

    # TODO(RMSNorm): Add the missing RMSNorm scale calculation.
    #
    # The code below produces partial sums of squares. It does not yet combine
    # all banks and channels, divide by vector length, add epsilon, calculate
    # the reciprocal square root, or write that scale into ``scale_buffer``.

    hardware = builder.hardware
    channels_per_block = builder.placement.channels_per_block
    channels = builder.all_channels()

    # Validate the learned-weight pass before emitting the L2 prefix. This keeps
    # a rejected RMS plan from leaving partial instructions in the builder.
    vector_layout = plan.l2_norm.layout
    row_operation_sizes = _row_operation_sizes(
        vector_layout.physical_values_per_partition,
        hardware.dram_columns,
        hardware.burst_length,
    )
    _require_dram_row_capacity(
        "weight_rows", plan.weight_rows, len(row_operation_sizes)
    )
    lower_l2_norm(builder, plan.l2_norm)

    # EW_MUL leaves one result in bank two of each four-bank PU group. AiM's
    # copy instructions name one physical bank, so move every result bank
    # through the Global Buffer into bank one beside the learned weights in
    # bank zero.
    for row_offset, operation_size in enumerate(row_operation_sizes):
        for result_bank in range(
            ElementwiseMultiply.RESULT_BANK,
            hardware.num_banks,
            BANKS_PER_PU,
        ):
            group_start_bank = result_bank - ElementwiseMultiply.RESULT_BANK
            second_operand_bank = (
                group_start_bank + ElementwiseMultiply.SECOND_OPERAND_BANK
            )
            builder.append(
                CopyBankToGlobalBuffer(
                    operation_size=operation_size,
                    channels=channels,
                    bank=result_bank,
                    row=plan.l2_norm.work_rows.row(row_offset),
                    column=0,
                )
            )
            builder.append(
                CopyGlobalBufferToBank(
                    operation_size=operation_size,
                    channels=channels,
                    bank=second_operand_bank,
                    row=plan.weight_rows.row(row_offset),
                    column=0,
                )
            )
        builder.append(
            ElementwiseMultiply(
                operation_size=operation_size,
                channels=channels,
                row=plan.weight_rows.row(row_offset),
                column=0,
            )
        )
    # The normalized operand is still zero in its padding lanes. Multiplying it
    # by the learned weights preserves those zeros. The read then overwrites
    # every lane in each occupied Shared Buffer slot.
    builder.emit_bank_group_transfer(
        ReadSingleBank,
        channels_per_block,
        vector_layout.partition_count,
        ElementwiseMultiply.RESULT_BANK,
        plan.weight_rows.start_row,
        vector_layout.physical_values_per_partition,
        shared_buffer=plan.output_buffer.start,
    )
