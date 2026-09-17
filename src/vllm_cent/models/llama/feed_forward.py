"""Build CENT instructions for the elementwise part of Llama's FFN."""

from dataclasses import dataclass

from ...cent import (
    BANKS_PER_PU,
    CentChannelSet,
    CentProgramBuilder,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    ReadSingleBank,
    WriteSingleBank,
    ceil_div,
)
from ...cent.utils import require_nonnegative, require_positive
from ...lowering import CentSharedBufferSpan
from ...lowering.utils import _require_shared_buffer_capacity

__all__: list[str] = []


@dataclass(frozen=True, slots=True, kw_only=True)
class _SiluProductChunkPlan:
    """Describe one section of the expanded feed-forward vector.

    Attributes:
        row: First DRAM row used by every selected bank.
        value_count: Feed-forward values represented by this section.
        partition_count: Four-bank PU groups used by the section.
        values_per_partition: Padded values stored in each selected PU group.
    """

    row: int
    value_count: int
    partition_count: int
    values_per_partition: int

    def __post_init__(self) -> None:
        """Validate the dimensions of one planned chunk.

        Raises:
            ValueError: If an address is negative or a size is invalid.
        """

        require_nonnegative("row", self.row)
        for name, value in (
            ("value_count", self.value_count),
            ("partition_count", self.partition_count),
            ("values_per_partition", self.values_per_partition),
        ):
            require_positive(name, value)
        if self.partition_count * self.values_per_partition < self.value_count:
            raise ValueError("chunk partitions do not cover every value")


@dataclass(frozen=True, slots=True, kw_only=True)
class _SiluProductPlan:
    """Bind the layout and storage used by the SiLU product lowerer.

    Attributes:
        chunks: Planned sections of the expanded feed-forward vector.
        workspace_buffer: Slots containing the padded vector partitions.
        channels: Channels that execute elementwise multiplication and copies.
        channels_per_copy: Channels occupied by one copy of each chunk layout.
        result_banks: Physical bank numbers holding elementwise results.
    """

    chunks: tuple[_SiluProductChunkPlan, ...]
    workspace_buffer: CentSharedBufferSpan
    channels: CentChannelSet
    channels_per_copy: int
    result_banks: tuple[int, ...]

    def __post_init__(self) -> None:
        """Validate target-independent parts of the SiLU product plan.

        Raises:
            ValueError: If the plan omits work or contains an invalid index.
        """

        if not self.chunks:
            raise ValueError("chunks cannot be empty")
        require_positive("channels_per_copy", self.channels_per_copy)
        if not self.result_banks:
            raise ValueError("result_banks cannot be empty")
        if len(set(self.result_banks)) != len(self.result_banks):
            raise ValueError("result_banks cannot contain duplicates")
        for bank in self.result_banks:
            require_nonnegative("result_bank", bank)
            if bank % BANKS_PER_PU != ElementwiseMultiply.RESULT_BANK:
                raise ValueError("each result bank must hold an EW_MUL result")


def _lower_silu_product(
    builder: CentProgramBuilder,
    plan: _SiluProductPlan,
) -> None:
    """Emit a planned SiLU gate and feed-forward projection product.

    Args:
        builder: Program builder that receives the FFN instructions.
        plan: Checked chunk layout, addresses, and channel selection.

    Raises:
        ValueError: If the plan is incompatible with the target hardware.
    """

    # TODO(dataflow): Connect W1, sigmoid(W1), and W3 to this workspace.
    #
    # The three projections now have separate compiler bindings. This function
    # still needs instructions that repack each RD_MAC result into the common
    # bank-group layout used below. The paper does not define that conversion.
    # Each repacker must write all occupied slots and zero every padding lane.

    # TODO(ISA): Define how unused PU groups are disabled.
    #
    # A chunk can occupy fewer groups than the selected channels contain, but
    # EW_MUL still runs every group. The final copy must not treat an inactive
    # group's stale bank-two value as part of the chunk.

    if builder.hardware.num_channels % plan.channels_per_copy:
        raise ValueError("channels_per_copy must divide num_channels")
    maximum_partitions = builder.total_banks // 4
    if any(chunk.partition_count > maximum_partitions for chunk in plan.chunks):
        raise ValueError("chunk partition_count exceeds available PU groups")
    if any(bank >= builder.hardware.num_banks for bank in plan.result_banks):
        raise ValueError("result bank is outside the target hardware")
    if any(
        ceil_div(chunk.values_per_partition, builder.hardware.burst_length)
        * builder.hardware.burst_length
        > builder.hardware.dram_columns
        for chunk in plan.chunks
    ):
        raise ValueError("a SiLU partition cannot cross a DRAM row")
    builder.channel_set(plan.channels.channels)

    # The same workspace is reused for each chunk. Size it for the largest
    # padded partition layout rather than for the total feed-forward vector.
    required_slots = max(
        chunk.partition_count
        * ceil_div(
            chunk.values_per_partition,
            builder.hardware.burst_length,
        )
        for chunk in plan.chunks
    )
    _require_shared_buffer_capacity(
        "workspace_buffer",
        plan.workspace_buffer,
        required_slots,
    )

    for chunk in plan.chunks:
        # Bank positions 0 and 1 hold W1 output and sigmoid(W1 output). The
        # missing repacking step must stage each value here before its write.
        for bank_group in (
            ElementwiseMultiply.FIRST_OPERAND_BANK,
            ElementwiseMultiply.SECOND_OPERAND_BANK,
        ):
            builder.emit_bank_group_transfer(
                WriteSingleBank,
                plan.channels_per_copy,
                chunk.partition_count,
                bank_group,
                chunk.row,
                chunk.values_per_partition,
                shared_buffer=plan.workspace_buffer.start,
            )
        op_size = ceil_div(
            chunk.values_per_partition,
            builder.hardware.burst_length,
        )
        builder.append(
            ElementwiseMultiply(
                operation_size=op_size,
                channels=plan.channels,
                row=chunk.row,
                column=0,
            )
        )
        # Move bank two's SiLU result to bank one in every four-bank group.
        # AiM COPY instructions operate on one physical bank number at a time.
        for result_bank in plan.result_banks:
            group_start_bank = result_bank - ElementwiseMultiply.RESULT_BANK
            second_operand_bank = (
                group_start_bank + ElementwiseMultiply.SECOND_OPERAND_BANK
            )
            builder.append(
                CopyBankToGlobalBuffer(
                    operation_size=op_size,
                    channels=plan.channels,
                    bank=result_bank,
                    row=chunk.row,
                    column=0,
                )
            )
            builder.append(
                CopyGlobalBufferToBank(
                    operation_size=op_size,
                    channels=plan.channels,
                    bank=second_operand_bank,
                    row=chunk.row,
                    column=0,
                )
            )

    # Put W3 in bank position 0 beside the SiLU result in position 1. The TO-DO
    # above tracks the missing step that places W3 in this workspace first.
    for chunk in plan.chunks:
        builder.emit_bank_group_transfer(
            WriteSingleBank,
            plan.channels_per_copy,
            chunk.partition_count,
            ElementwiseMultiply.FIRST_OPERAND_BANK,
            chunk.row,
            chunk.values_per_partition,
            shared_buffer=plan.workspace_buffer.start,
        )
    for chunk in plan.chunks:
        builder.append(
            ElementwiseMultiply(
                operation_size=ceil_div(
                    chunk.values_per_partition,
                    builder.hardware.burst_length,
                ),
                channels=plan.channels,
                row=chunk.row,
                column=0,
            )
        )
    for chunk in plan.chunks:
        builder.emit_bank_group_transfer(
            ReadSingleBank,
            plan.channels_per_copy,
            chunk.partition_count,
            ElementwiseMultiply.RESULT_BANK,
            chunk.row,
            chunk.values_per_partition,
            shared_buffer=plan.workspace_buffer.start,
        )
