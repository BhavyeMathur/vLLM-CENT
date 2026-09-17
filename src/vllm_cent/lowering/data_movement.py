"""Lower reusable vector transfers between the Shared Buffer and DRAM."""

from dataclasses import dataclass

from ..cent import (
    BANKS_PER_PU,
    CentProgramBuilder,
    ReadSingleBank,
    WriteSingleBank,
    ceil_div,
)
from .bindings import CentDramVector, CentSharedBufferVector
from .utils import _require_dram_row_capacity

__all__ = [
    "CentBankGroupVectorTransferPlan",
    "lower_load_bank_group_vector",
    "lower_store_bank_group_vector",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class CentBankGroupVectorTransferPlan:
    """Bind one zero-padded vector in DRAM and the Shared Buffer.

    Attributes:
        dram: Partitioned DRAM representation of the vector.
        buffer: Shared Buffer representation of the same vector.
    """

    dram: CentDramVector
    buffer: CentSharedBufferVector

    def __post_init__(self) -> None:
        """Require both bindings to describe the same physical vector.

        Raises:
            ValueError: If the DRAM and Shared Buffer layouts differ.
        """

        if self.dram.layout != self.buffer.layout:
            raise ValueError("DRAM and Shared Buffer layouts must match")


def _lower_bank_group_vector_transfer(
    builder: CentProgramBuilder,
    plan: CentBankGroupVectorTransferPlan,
    instruction_type: type[WriteSingleBank] | type[ReadSingleBank],
) -> None:
    """Lower one direction of a partitioned vector transfer.

    Args:
        builder: Program builder that receives the transfer instructions.
        plan: Checked vector layout and its memory bindings.
        instruction_type: Direction of the transfer.

    Raises:
        ValueError: If the plan is incompatible with the target hardware or a
            memory region is too small.
    """

    layout = plan.buffer.layout
    if layout.burst_length != builder.hardware.burst_length:
        raise ValueError("layout burst_length does not match the target")

    available_groups = builder.total_banks // BANKS_PER_PU
    if layout.partition_count > available_groups:
        raise ValueError("layout uses more bank groups than the block owns")

    required_rows = ceil_div(
        layout.physical_values_per_partition,
        builder.hardware.dram_columns,
    )
    _require_dram_row_capacity("rows", plan.dram.rows, required_rows)

    # Each used PU group receives one consecutive partition. The builder maps
    # those partitions to channels and banks and advances the buffer slot.
    builder.emit_bank_group_transfer(
        instruction_type,
        builder.placement.channels_per_block,
        layout.partition_count,
        plan.dram.bank_group,
        plan.dram.rows.start_row,
        layout.physical_values_per_partition,
        shared_buffer=plan.buffer.start,
    )


def lower_store_bank_group_vector(
    builder: CentProgramBuilder,
    plan: CentBankGroupVectorTransferPlan,
) -> None:
    """Store a zero-padded Shared Buffer vector across DRAM bank groups.

    Every occupied slot is copied, so zeros in the source binding become zeros
    in the matching DRAM padding lanes.

    Args:
        builder: Program builder that receives the write instructions.
        plan: Vector layout and the source and destination memory regions.

    Raises:
        ValueError: If the plan is incompatible with the target hardware or a
            memory region is too small.
    """

    _lower_bank_group_vector_transfer(
        builder,
        plan,
        instruction_type=WriteSingleBank,
    )


def lower_load_bank_group_vector(
    builder: CentProgramBuilder,
    plan: CentBankGroupVectorTransferPlan,
) -> None:
    """Load a zero-padded DRAM vector into complete Shared Buffer slots.

    The load writes every occupied lane and preserves the DRAM binding's zero
    padding. It does not rely on previous Shared Buffer contents.

    Args:
        builder: Program builder that receives the read instructions.
        plan: Vector layout and the source and destination memory regions.

    Raises:
        ValueError: If the plan is incompatible with the target hardware or a
            memory region is too small.
    """

    _lower_bank_group_vector_transfer(
        builder,
        plan,
        instruction_type=ReadSingleBank,
    )
