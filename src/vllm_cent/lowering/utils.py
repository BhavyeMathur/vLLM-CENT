"""Internal calculations shared by operation lowerers and planners."""

from dataclasses import dataclass

from ..cent import ceil_div

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
