"""Internal calculations shared by operation lowerers and planners."""

from dataclasses import dataclass

from ..cent import ceil_div
from ..cent.utils import require_positive
from .bindings import CentDramRowRange, CentSharedBufferSpan

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _PartitionedVectorLayout:
    """Describe an even vector split with per-partition burst padding.

    Attributes:
        values_per_partition: Values assigned to each used partition.
        partition_count: Partitions that contain at least one value.
        slot_count: Shared Buffer slots used by all padded partitions.
    """

    values_per_partition: int
    partition_count: int
    slot_count: int


def _require_shared_buffer_capacity(
    name: str,
    span: CentSharedBufferSpan,
    required_slots: int,
) -> None:
    """Check that a Shared Buffer span covers every accessed slot.

    Args:
        name: Buffer name used in an error message.
        span: Shared Buffer region assigned to an operation.
        required_slots: Slots the operation will access.

    Raises:
        ValueError: If ``required_slots`` is invalid or the span is too small.
    """

    require_positive("required_slots", required_slots)
    if span.slot_count < required_slots:
        raise ValueError(
            f"{name} needs {required_slots} Shared Buffer slots, "
            f"but its span contains {span.slot_count}"
        )


def _require_dram_row_capacity(
    name: str,
    rows: CentDramRowRange,
    required_rows: int,
) -> None:
    """Check that a DRAM row range covers every accessed row.

    Args:
        name: Row-range name used in an error message.
        rows: DRAM row range assigned to an operation.
        required_rows: Rows the operation will access.

    Raises:
        ValueError: If ``required_rows`` is invalid or the range is too small.
    """

    require_positive("required_rows", required_rows)
    if rows.row_count < required_rows:
        raise ValueError(
            f"{name} needs {required_rows} DRAM rows, "
            f"but its range contains {rows.row_count}"
        )


def _plan_partitioned_vector(
    value_count: int,
    available_partitions: int,
    burst_length: int,
) -> _PartitionedVectorLayout:
    """Split a vector evenly and include each partition's burst padding.

    Args:
        value_count: Values divided among the partitions.
        available_partitions: Hardware partitions available to the operation.
        burst_length: Values carried by one Shared Buffer slot.

    Returns:
        Partition width, used partition count, and total padded slot count.

    Raises:
        ValueError: If any argument is less than one.
    """

    values_per_partition = ceil_div(value_count, available_partitions)
    partition_count = ceil_div(value_count, values_per_partition)
    slots_per_partition = ceil_div(values_per_partition, burst_length)
    return _PartitionedVectorLayout(
        values_per_partition=values_per_partition,
        partition_count=partition_count,
        slot_count=partition_count * slots_per_partition,
    )
