"""Describe vector layouts before lowering them to CENT instructions."""

from collections.abc import Sequence
from dataclasses import dataclass

from ..cent import ceil_div
from ..cent.utils import require_nonnegative, require_positive

__all__ = [
    "CentPartitionedVectorLayout",
    "pack_zero_padded_vector",
    "plan_partitioned_vector",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class CentPartitionedVectorLayout:
    """Describe a vector divided into equal-width physical partitions.

    A partition is the piece assigned to one selected bank or bank group. Every
    partition occupies complete Shared Buffer slots. Logical values come first
    in each partition, and every remaining lane in those slots contains zero.

    Attributes:
        value_count: Number of logical values in the vector.
        partition_count: Number of physical partitions selected by the plan.
        burst_length: Number of values carried by one Shared Buffer slot.
    """

    value_count: int
    partition_count: int
    burst_length: int

    def __post_init__(self) -> None:
        """Validate the dimensions needed to derive the padded layout.

        Raises:
            ValueError: If a dimension is invalid or a partition would contain
                no logical value.
        """

        require_positive("value_count", self.value_count)
        require_positive("partition_count", self.partition_count)
        require_positive("burst_length", self.burst_length)
        if self.partition_count > self.value_count:
            raise ValueError("partition_count cannot exceed value_count")

    @property
    def values_per_partition(self) -> int:
        """Return the logical capacity reserved for each partition.

        Returns:
            Largest logical value count assigned to any partition, before the
            partition is rounded up to complete Shared Buffer slots.
        """

        return ceil_div(self.value_count, self.partition_count)

    def logical_values_in_partition(self, partition: int) -> int:
        """Return the logical values assigned to one partition.

        Values are balanced across the selected partitions. Earlier partitions
        receive one extra value when the vector does not divide evenly.

        Args:
            partition: Zero-based partition index.

        Returns:
            Logical values stored before padding in that partition.

        Raises:
            ValueError: If ``partition`` is outside the layout.
        """

        require_nonnegative("partition", partition)
        if partition >= self.partition_count:
            raise ValueError("partition is outside the vector layout")

        base_count, larger_partition_count = divmod(
            self.value_count, self.partition_count
        )
        return base_count + (partition < larger_partition_count)

    @property
    def slots_per_partition(self) -> int:
        """Return the Shared Buffer slots occupied by one partition.

        Returns:
            Burst slots needed for one padded partition.
        """

        return ceil_div(self.values_per_partition, self.burst_length)

    @property
    def slot_count(self) -> int:
        """Return the Shared Buffer slots occupied by the whole layout.

        Returns:
            Burst slots needed for every padded partition.
        """

        return self.partition_count * self.slots_per_partition

    @property
    def physical_values_per_partition(self) -> int:
        """Return all lanes that one partition owns.

        Returns:
            Scalar lanes in the partition's complete Shared Buffer slots.
            Lanes after the partition's logical values contain zero.
        """

        return self.slots_per_partition * self.burst_length

    @property
    def physical_value_count(self) -> int:
        """Return all lanes occupied by the vector representation.

        Returns:
            Scalar lanes in every complete Shared Buffer slot.
        """

        return self.partition_count * self.physical_values_per_partition

    @property
    def padding_value_count(self) -> int:
        """Return the lanes that must be explicitly set to zero.

        Returns:
            Physical lanes that do not hold logical vector values.
        """

        return self.physical_value_count - self.value_count

    @property
    def is_contiguously_packed(self) -> bool:
        """Return whether partition padding occurs only after the vector.

        Returns:
            ``True`` when joining the partition slots produces the logical
            vector followed only by trailing padding. ``False`` means at least
            one partition inserts padding between logical values.
        """

        # Padding in the final partition is harmless because no later logical
        # values follow it. Every earlier partition must end on a burst boundary
        # for another layout to read the same slots as one continuous vector.
        return all(
            self.logical_values_in_partition(partition) % self.burst_length == 0
            for partition in range(self.partition_count - 1)
        )


def pack_zero_padded_vector[ValueT](
        values: Sequence[ValueT],
        layout: CentPartitionedVectorLayout,
        *,
        zero: ValueT,
) -> tuple[ValueT, ...]:
    """Pack logical values and overwrite every padding lane with zero.

    The runtime and model loader can use this function before writing a logical
    vector to the Shared Buffer or to its partitioned DRAM representation. Each
    partition is padded separately because unused lanes can appear between two
    groups of logical values.

    Args:
        values: Logical vector values in their original order.
        layout: Physical partition and slot layout for the vector.
        zero: Zero value represented in the same host type as ``values``.

    Returns:
        Values for every physical lane, including explicit zero padding.

    Raises:
        ValueError: If ``values`` does not match the layout's logical size.
    """

    if len(values) != layout.value_count:
        raise ValueError(
            f"layout contains {layout.value_count} values, but received "
            f"{len(values)}"
        )

    packed: list[ValueT] = []
    first_value = 0
    for partition in range(layout.partition_count):
        logical_count = layout.logical_values_in_partition(partition)
        next_value = first_value + logical_count
        packed.extend(values[first_value:next_value])

        # A producer owns the complete physical partition. Fill both the
        # equal-partition gap and the final slot's unused lanes with zeros.
        padding_count = layout.physical_values_per_partition - logical_count
        packed.extend(zero for _ in range(padding_count))
        first_value = next_value

    return tuple(packed)


def plan_partitioned_vector(
        value_count: int,
        maximum_partition_count: int,
        burst_length: int,
) -> CentPartitionedVectorLayout:
    """Choose the baseline even layout for a vector.

    This simple policy spreads values across as many useful partitions as the
    supplied limit permits. More advanced planners can construct
    :class:`CentPartitionedVectorLayout` directly and choose a different count.

    Args:
        value_count: Number of logical values in the vector.
        maximum_partition_count: Most partitions this operation may use.
        burst_length: Number of values carried by one Shared Buffer slot.

    Returns:
        Even vector layout selected by the baseline policy.

    Raises:
        ValueError: If any argument is less than one.
    """

    require_positive("value_count", value_count)
    require_positive("maximum_partition_count", maximum_partition_count)
    require_positive("burst_length", burst_length)

    # First choose the smallest equal width that fits the vector within the
    # available partitions. Then omit any trailing partitions that would hold
    # no logical values. This preserves the original lowering policy.
    values_per_partition = ceil_div(value_count, maximum_partition_count)
    partition_count = ceil_div(value_count, values_per_partition)
    return CentPartitionedVectorLayout(
        value_count=value_count,
        partition_count=partition_count,
        burst_length=burst_length,
    )
