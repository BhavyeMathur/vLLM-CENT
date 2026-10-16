"""Name locations in DRAM, device buffers, and bank result registers."""

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

    The fields correspond to the paper's ``CHid BK RO CO`` labels. This Python
    API currently measures ``column`` in scalar values. A bank and row select
    the storage containing those values.

    This scalar convention is not a verified encoding of hardware ``CO``. AiM
    sources describe burst-column addresses; a unit migration must update
    callers and renderers together.

    Attributes:
        channel: Zero-based physical channel number, or ``CHid``.
        bank: Zero-based bank number within ``channel``, or ``BK``.
        row: Zero-based row number within ``bank``, or ``RO``.
        column: Zero-based scalar position within ``row`` in this Python API.

    Examples:
        With ``burst_length=4``, a transfer at ``column=8`` reads scalar values
        ``[8:12]``. With ``operation_size=2``, it reads ``[8:16]``. Both transfers
        use the same bank and row and must fit within that row.
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


@dataclass(frozen=True, slots=True, kw_only=True)
class CentGlobalBufferAddress:
    """Identify a scalar column in one channel's Global Buffer.

    These runtime coordinates describe storage, not an additional ISA operand.
    Target-specific bounds are checked when binding or accessing storage.

    Attributes:
        channel: Channel whose Global Buffer contains the value.
        column: Scalar offset from the start of that buffer, not a byte offset.
    """

    channel: int
    column: int

    def __post_init__(self) -> None:
        """Reject negative storage coordinates.

        Raises:
            ValueError: If the channel or column is negative.
        """
        require_nonnegative("channel", self.channel)
        require_nonnegative("column", self.column)


@dataclass(frozen=True, slots=True, kw_only=True)
class CentBankRegisterAddress:
    """Identify one bank's accumulator or activation-result register.

    The register file selected by the caller determines which value is accessed.
    These coordinates alone do not specify instruction packing or precision.

    Attributes:
        channel: Channel containing the bank.
        bank: Bank within that channel.
        register: Index within the accumulator or activation-result register set.

    Examples:
        ``CentBankRegisterAddress(channel=1, bank=2, register=0)`` selects register
        zero in bank two of channel one. It does not select Shared Buffer slot zero.
    """

    channel: int
    bank: int
    register: int

    def __post_init__(self) -> None:
        """Reject negative register coordinates.

        Raises:
            ValueError: If the channel, bank, or register index is negative.
        """
        for name in ("channel", "bank", "register"):
            require_nonnegative(name, getattr(self, name))
