"""Physical memory regions shared by operation lowerers."""

from dataclasses import dataclass

from ..cent import CentSharedBufferAddress
from ..cent.utils import require_nonnegative, require_positive

__all__ = ["CentDramRowRange", "CentSharedBufferSpan"]


@dataclass(frozen=True, slots=True, kw_only=True)
class CentSharedBufferSpan:
    """Identify consecutive slots in CENT's Shared Buffer.

    A span gives an operation an explicit input, output, or workspace. This
    prevents unrelated values from silently sharing slot zero.

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
    This type records only the row interval so that its bounds are explicit.

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
