"""Describe the parts of CENT hardware that affect compilation."""

from dataclasses import dataclass

# Four neighboring banks form one group for elementwise multiplication. Each
# bank has its own processing unit; four banks do not share a single unit.
# See docs/research/sk-hynix-aim.md for the distinction in the source diagrams.
BANKS_PER_GROUP = 4

# TODO(architecture): We describe DRAM, Global Buffer, and Shared Buffer
# capacities. We need to decide whether compilation also needs CXL topology,
# PNM-unit count, and instruction-buffer capacity.

# TODO(architecture): The paper uses BF16 and moves 16 values in one 256-bit
# transfer. We need to decide whether custom targets may use other formats. If
# they may, scalar width and slot width must become explicit hardware fields.

# TODO(target ABI): We only store the sigmoid AFid today. We need a typed list of
# supported activation functions and IDs so ApplyActivation can be validated.

__all__ = ["BANKS_PER_GROUP", "CentBlockPlacementSpec", "CentHardwareSpec"]


@dataclass(frozen=True, slots=True, kw_only=True)
class CentHardwareSpec:
    """Describe the storage sizes and instruction limits of one CENT device.

    A channel contains banks; each bank contains rows of scalar values. The
    Global Buffer belongs to a channel, while the Shared Buffer belongs to the
    whole device. A burst is the group of values moved by one transfer.
    These are the compiler's current units, not a hardware address encoding.

    Attributes:
        num_channels: Number of independently addressable DRAM channels. The
            paper's ``CHmask`` supports 1 through 32 channels.
        num_banks: Number of DRAM banks in each channel. Four neighboring banks
            form one bank group in the current layout rules.
        dram_rows: Number of rows in each DRAM bank.
        dram_columns: Number of scalar values in each DRAM row. A row must hold
            a whole number of bursts.
        burst_length: Number of scalar values moved by one micro-operation. It
            cannot be wider than a DRAM row.
        accumulator_slots_per_bank: Number of multiply-accumulate (MAC) result
            registers available to each bank. The paper calls an index into these
            registers ``Regid``.
        sigmoid_activation_function_id: Numeric ``AFid`` that this target uses
            for sigmoid. The paper does not publish the numeric mapping.
        global_buffer_columns: Scalar capacity of each channel's Global Buffer.
            The default is a local target configuration, not an ISA guarantee.
        shared_buffer_slots: Number of 256-bit staging slots shared by the
            device. The paper's 64 KiB buffer contains 2,048 slots.

    Examples:
        With ``dram_columns=8`` and ``burst_length=2``, each row holds four
        transfers. Both fields count scalar values, not bytes.
    """

    num_channels: int
    num_banks: int
    dram_rows: int
    dram_columns: int
    burst_length: int
    accumulator_slots_per_bank: int
    sigmoid_activation_function_id: int
    shared_buffer_slots: int = 2_048
    global_buffer_columns: int = 1_024

    def __post_init__(self) -> None:
        """Validate the hardware geometry.

        Raises:
            ValueError: If a dimension is outside its documented range or the
                DRAM row or Global Buffer cannot hold a whole number of bursts.
        """

        # CHmask has one bit for each of the paper's 32 possible channels.
        if not 1 <= self.num_channels <= 32:
            raise ValueError("num_channels must be between 1 and 32")

        # Current placement rules divide each channel into complete four-bank groups.
        if self.num_banks < BANKS_PER_GROUP or self.num_banks % BANKS_PER_GROUP != 0:
            raise ValueError("num_banks must be at least 4 and divisible by 4")
        # One burst cannot cross a row boundary, so a row must contain a whole
        # number of bursts.
        if self.dram_rows < 1:
            raise ValueError("dram_rows must be at least 1")
        if self.dram_columns < 1:
            raise ValueError("dram_columns must be at least 1")
        if self.burst_length < 1 or self.burst_length > self.dram_columns:
            raise ValueError("burst_length must be between 1 and dram_columns")
        if self.dram_columns % self.burst_length != 0:
            raise ValueError("dram_columns must be divisible by burst_length")
        # Regid and Shared Buffer addresses need at least one valid destination.
        if self.accumulator_slots_per_bank < 1:
            raise ValueError("accumulator_slots_per_bank must be at least 1")
        if self.sigmoid_activation_function_id < 0:
            raise ValueError("sigmoid_activation_function_id cannot be negative")
        if self.global_buffer_columns < 1:
            raise ValueError("global_buffer_columns must be at least 1")
        if self.global_buffer_columns % self.burst_length != 0:
            raise ValueError("global_buffer_columns must be divisible by burst_length")
        if self.shared_buffer_slots < 1:
            raise ValueError("shared_buffer_slots must be at least 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class CentBlockPlacementSpec:
    """Describe how many channels are assigned to one model block.

    Attributes:
        channels_per_block: Number of physical channels used by one block. It
            must fit in the target and divide the target's channels evenly.

    Examples:
        With ``channels_per_block=2`` on a four-channel device, the builder can
        place copies in channel groups ``(0, 1)`` and ``(2, 3)``.
    """

    channels_per_block: int

    def __post_init__(self) -> None:
        """Check the placement rule that does not require a hardware target.

        Raises:
            ValueError: If ``channels_per_block`` is less than 1.
        """

        # The builder later checks this count against a specific device.
        if self.channels_per_block < 1:
            raise ValueError("channels_per_block must be at least 1")
