"""Typed addresses for CENT's independently addressed state regions."""

from dataclasses import dataclass

from ..utils import require_nonnegative

__all__ = [
    "CentBankRegisterAddress",
    "CentChannelSet",
    "CentGlobalBufferAddress",
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
class CentGlobalBufferAddress:
    """Identify one scalar position in a channel's Global Buffer.

    A Global Buffer is channel-local but is not part of a DRAM bank or row.
    Its column is therefore governed by ``global_buffer_columns`` rather than
    the width of a DRAM row.

    Attributes:
        channel: Zero-based physical channel number.
        column: Zero-based scalar position in the channel's Global Buffer.
    """

    channel: int
    column: int

    def __post_init__(self) -> None:
        """Reject coordinates that are invalid on every hardware target.

        Raises:
            ValueError: If either coordinate is negative.
        """

        # Upper bounds belong to the selected hardware target.
        require_nonnegative("channel", self.channel)
        require_nonnegative("column", self.column)


@dataclass(frozen=True, slots=True, kw_only=True)
class CentBankRegisterAddress:
    """Identify one accumulator or activation-result register.

    Accumulator and activation-result files use the same coordinates but are
    distinct state regions. The owning state object decides which file an
    address selects.

    Attributes:
        channel: Zero-based physical channel number.
        bank: Zero-based bank number within ``channel``.
        register: Zero-based result-register number within ``bank``.
    """

    channel: int
    bank: int
    register: int

    def __post_init__(self) -> None:
        """Reject coordinates that are invalid on every hardware target.

        Raises:
            ValueError: If any coordinate is negative.
        """

        # Upper bounds belong to the selected hardware target.
        require_nonnegative("channel", self.channel)
        require_nonnegative("bank", self.bank)
        require_nonnegative("register", self.register)


@dataclass(frozen=True, slots=True, kw_only=True)
class CentChannelSet:
    """List the channels selected by a ``CHmask`` operand.

    Compiler code stores readable channel numbers. The renderer later packs
    those numbers into the paper's bit mask.

    Attributes:
        channels: Unique, zero-based channel numbers. At least one channel is
            required. The tuple is stored in ascending order because order does
            not change the meaning of a channel mask.
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

        # A channel mask is a set. Canonical storage makes equality and hashing
        # reflect that meaning instead of the caller's tuple order.
        # A frozen dataclass rejects normal assignment, even in __post_init__.
        # object.__setattr__ allows this one-time normalization during creation;
        # the completed object remains immutable.
        object.__setattr__(self, "channels", tuple(sorted(self.channels)))


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
