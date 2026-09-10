"""Lower reusable vector transfers between the Shared Buffer and DRAM."""

from ..cent import (
    BANKS_PER_PU,
    CentProgramBuilder,
    ReadSingleBank,
    WriteSingleBank,
    ceil_div,
)
from ..cent.utils import require_positive
from .bindings import CentDramRowRange, CentSharedBufferSpan
from .utils import _plan_partitioned_vector

__all__ = ["lower_load_bank_group_vector", "lower_store_bank_group_vector"]


def _lower_bank_group_vector_transfer(
    builder: CentProgramBuilder,
    *,
    rows: CentDramRowRange,
    buffer: CentSharedBufferSpan,
    value_count: int,
    bank_group: int,
    instruction_type: type[WriteSingleBank] | type[ReadSingleBank],
) -> None:
    """Lower one direction of a partitioned vector transfer.

    Args:
        builder: Program builder that receives the transfer instructions.
        rows: DRAM rows holding the distributed vector.
        buffer: Shared Buffer slots holding the packed vector.
        value_count: Values moved by the transfer.
        bank_group: Bank position within each four-bank PU group.
        instruction_type: Direction of the transfer.

    Raises:
        ValueError: If the vector is empty or the buffer is too small.
    """

    require_positive("value_count", value_count)
    group_count = builder.total_banks // BANKS_PER_PU
    layout = _plan_partitioned_vector(
        value_count,
        group_count,
        builder.hardware.burst_length,
    )
    if buffer.slot_count < layout.slot_count:
        raise ValueError(
            f"buffer needs {layout.slot_count} Shared Buffer slots, "
            f"but its span contains {buffer.slot_count}"
        )
    required_rows = ceil_div(
        layout.values_per_partition,
        builder.hardware.dram_columns,
    )
    if rows.row_count < required_rows:
        raise ValueError(
            f"rows needs {required_rows} DRAM rows, "
            f"but its range contains {rows.row_count}"
        )

    # Each used PU group receives one consecutive partition. The builder maps
    # those partitions to channels and banks and advances the buffer slot.
    builder.emit_bank_group_transfer(
        instruction_type,
        builder.placement.channels_per_block,
        layout.partition_count,
        bank_group,
        rows.start_row,
        layout.values_per_partition,
        shared_buffer=buffer.start,
    )


def lower_store_bank_group_vector(
    builder: CentProgramBuilder,
    *,
    rows: CentDramRowRange,
    buffer: CentSharedBufferSpan,
    value_count: int,
    bank_group: int = 0,
) -> None:
    """Store a packed Shared Buffer vector across DRAM bank groups.

    Args:
        builder: Program builder that receives the write instructions.
        rows: DRAM rows receiving the distributed vector.
        buffer: Shared Buffer slots containing the packed vector.
        value_count: Values stored by the transfer.
        bank_group: Bank position within each four-bank PU group.

    Raises:
        ValueError: If the vector is empty or the buffer is too small.
    """

    _lower_bank_group_vector_transfer(
        builder,
        rows=rows,
        buffer=buffer,
        value_count=value_count,
        bank_group=bank_group,
        instruction_type=WriteSingleBank,
    )


def lower_load_bank_group_vector(
    builder: CentProgramBuilder,
    *,
    rows: CentDramRowRange,
    buffer: CentSharedBufferSpan,
    value_count: int,
    bank_group: int = 0,
) -> None:
    """Load a distributed DRAM vector into packed Shared Buffer slots.

    Args:
        builder: Program builder that receives the read instructions.
        rows: DRAM rows containing the distributed vector.
        buffer: Shared Buffer slots receiving the packed vector.
        value_count: Values loaded by the transfer.
        bank_group: Bank position within each four-bank PU group.

    Raises:
        ValueError: If the vector is empty or the buffer is too small.
    """

    _lower_bank_group_vector_transfer(
        builder,
        rows=rows,
        buffer=buffer,
        value_count=value_count,
        bank_group=bank_group,
        instruction_type=ReadSingleBank,
    )
