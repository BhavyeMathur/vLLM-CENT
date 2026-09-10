"""Typed addresses for CENT's DRAM and Shared Buffer."""

from dataclasses import dataclass

from ..utils import require_nonnegative

__all__ = [
    "CentChannelSet",
    "CentMemoryAddress",
    "CentSharedBufferAddress",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class CentMemoryAddress:
    """Identify the first DRAM value used by an instruction.

    The paper writes these four fields as ``CHid BK RO CO``. When ``OPsize`` is
    greater than one, later micro-operations move forward from ``column`` by one
    burst at a time.

    Attributes:
        channel: Zero-based physical channel number, or ``CHid``.
        bank: Zero-based bank number within ``channel``, or ``BK``.
        row: Zero-based row number within ``bank``, or ``RO``.
        column: Zero-based scalar position within ``row``, or ``CO``.
    """

    channel: int
    bank: int
    row: int
    column: int

    def __post_init__(self) -> None:
        """Reject address values that are invalid on every target.

        Raises:
            ValueError: If an address component is negative.
        """

        # Upper bounds depend on the target and are checked in validation.py.
        for name in ("channel", "bank", "row", "column"):
            require_nonnegative(name, getattr(self, name))


@dataclass(frozen=True, slots=True, kw_only=True)
class CentChannelSet:
    """List the channels selected by a ``CHmask`` operand.

    Compiler code stores readable channel numbers. The renderer later packs
    those numbers into the paper's bit mask.

    Attributes:
        channels: Unique, zero-based channel numbers. At least one channel is
            required. Order is preserved for display but does not change the
            meaning of the mask.
    """

    channels: tuple[int, ...]

    def __post_init__(self) -> None:
        """Check channel-selection rules that do not need a target.

        Raises:
            ValueError: If the tuple is empty, contains a negative index, or
                contains the same channel more than once.
        """

        # An empty mask would send the instruction to no hardware.
        if not self.channels:
            raise ValueError("channels cannot be empty")

        # The target-specific upper bound is checked later.
        for channel in self.channels:
            require_nonnegative("channel", channel)

        # Each channel contributes one bit, so listing it twice has no meaning.
        if len(set(self.channels)) != len(self.channels):
            raise ValueError("channels cannot contain duplicates")


@dataclass(frozen=True, slots=True, kw_only=True)
class CentSharedBufferAddress:
    """Identify one 256-bit slot in CENT's Shared Buffer.

    The same slot number is called ``Rs`` when an instruction reads it and
    ``Rd`` when an instruction writes it. It is not a byte address or a DRAM
    address.

    Attributes:
        slot: Zero-based index of a 256-bit Shared Buffer entry.
    """

    slot: int

    def __post_init__(self) -> None:
        """Reject a slot number that is invalid on every target.

        Raises:
            ValueError: If ``slot`` is negative.
        """

        # The target-specific upper bound is checked when building a program.
        require_nonnegative("slot", self.slot)
