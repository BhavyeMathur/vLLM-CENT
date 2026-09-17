"""Physical memory regions shared by operation lowerers."""

from dataclasses import dataclass

from ..cent import BANKS_PER_PU, CentSharedBufferAddress
from ..cent.utils import require_nonnegative, require_positive
from .planning import CentPartitionedVectorLayout

__all__ = [
    "CentDramRowRange",
    "CentDramVector",
    "CentSharedBufferSpan",
    "CentSharedBufferVector",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class CentSharedBufferSpan:
    """Identify consecutive slots in CENT's Shared Buffer.

    A span describes capacity only. It does not promise that any lane has been
    initialized. Use :class:`CentSharedBufferVector` when the slots contain a
    complete zero-padded logical vector.

    Attributes:
        start: Address of the first 256-bit Shared Buffer slot.
        slot_count: Number of consecutive slots in the span.
    """

    start: CentSharedBufferAddress
    slot_count: int

    def __post_init__(self) -> None:
        """Validate the part of the span that does not need a target device.

        Raises:
            ValueError: If the span contains no slots.
        """

        require_positive("slot_count", self.slot_count)

    def address(self, slot_offset: int) -> CentSharedBufferAddress:
        """Return the address of one slot inside this span.

        Args:
            slot_offset: Number of slots after :attr:`start`.

        Returns:
            Shared Buffer address at the requested offset.

        Raises:
            ValueError: If the offset is outside the span.
        """

        require_nonnegative("slot_offset", slot_offset)
        if slot_offset >= self.slot_count:
            raise ValueError("slot_offset is outside the Shared Buffer span")
        return CentSharedBufferAddress(slot=self.start.slot + slot_offset)


@dataclass(frozen=True, slots=True, kw_only=True)
class CentDramRowRange:
    """Identify consecutive row numbers used in every selected DRAM bank.

    An operation decides how values are distributed among channels and banks.
    This type records only the row interval so that its bounds are explicit. It
    makes no promise about stored values or padding.

    Attributes:
        start_row: First row in the range.
        row_count: Number of consecutive rows in the range.
    """

    start_row: int
    row_count: int

    def __post_init__(self) -> None:
        """Validate the row interval.

        Raises:
            ValueError: If the first row is negative or the range is empty.
        """

        require_nonnegative("start_row", self.start_row)
        require_positive("row_count", self.row_count)

    def row(self, row_offset: int) -> int:
        """Return one row number inside this range.

        Args:
            row_offset: Number of rows after :attr:`start_row`.

        Returns:
            DRAM row number at the requested offset.

        Raises:
            ValueError: If the offset is outside the range.
        """

        require_nonnegative("row_offset", row_offset)
        if row_offset >= self.row_count:
            raise ValueError("row_offset is outside the DRAM row range")
        return self.start_row + row_offset


@dataclass(frozen=True, slots=True, kw_only=True)
class CentSharedBufferVector:
    """Bind a zero-padded logical vector to exact Shared Buffer storage.

    This binding carries a data contract, not only a capacity. A valid value in
    ``span`` has every lane written. Within each partition, logical values come
    first and every remaining lane is zero. A producer establishes that
    contract before a consumer uses the binding.

    Attributes:
        span: Exact Shared Buffer slots occupied by the physical vector.
        layout: Logical values, partitions, slots, and zero-padding positions.
    """

    span: CentSharedBufferSpan
    layout: CentPartitionedVectorLayout

    def __post_init__(self) -> None:
        """Require the binding to identify exactly the occupied slots.

        Raises:
            ValueError: If the span includes fewer or more slots than the
                vector layout occupies.
        """

        if self.span.slot_count != self.layout.slot_count:
            raise ValueError(
                "vector span must contain exactly "
                f"{self.layout.slot_count} slots, but contains "
                f"{self.span.slot_count}"
            )

    @property
    def start(self) -> CentSharedBufferAddress:
        """Return the first physical slot occupied by the vector.

        Returns:
            Address of the vector's first Shared Buffer slot.
        """

        return self.span.start

    def address(self, slot_offset: int) -> CentSharedBufferAddress:
        """Return one occupied slot in the vector.

        Args:
            slot_offset: Number of slots after the vector's first slot.

        Returns:
            Shared Buffer address at the requested offset.

        Raises:
            ValueError: If the offset is outside the vector storage.
        """

        return self.span.address(slot_offset)


@dataclass(frozen=True, slots=True, kw_only=True)
class CentDramVector:
    """Bind a zero-padded logical vector to partitioned DRAM rows.

    As with :class:`CentSharedBufferVector`, the binding promises that every
    physical lane is initialized and every non-logical lane is zero.

    Attributes:
        rows: DRAM rows used by every selected bank partition.
        layout: Logical values, partitions, slots, and zero-padding positions.
        bank_group: Bank position selected in each four-bank PU group.
    """

    rows: CentDramRowRange
    layout: CentPartitionedVectorLayout
    bank_group: int = 0

    def __post_init__(self) -> None:
        """Validate the selected position in each four-bank PU group.

        Raises:
            ValueError: If ``bank_group`` is outside a PU's four banks.
        """

        if self.bank_group not in range(BANKS_PER_PU):
            raise ValueError("bank_group must be between 0 and 3")
