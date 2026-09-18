"""Describe the parts of CENT hardware that affect compilation."""

from dataclasses import dataclass

# One CENT processing unit works with a group of four neighboring DRAM banks.
BANKS_PER_PU = 4

# TODO(architecture): We currently describe DRAM, the Global Buffer, and the
# Shared Buffer. We need to decide whether compilation also needs CXL topology,
# PNM-unit count, and instruction-buffer capacity.

# TODO(architecture): The paper uses BF16 and moves 16 values in one 256-bit
# transfer. We need to decide whether custom targets may use other formats. If
# they may, scalar width and slot width must become explicit hardware fields.

# TODO(target ABI): We only store the sigmoid AFid today. We need a typed list of
# supported activation functions and IDs so ApplyActivation can be validated.

__all__ = ["BANKS_PER_PU", "CentBlockPlacementSpec", "CentHardwareSpec"]


@dataclass(frozen=True, slots=True, kw_only=True)
class CentHardwareSpec:
    """Describe the hardware limits of one CENT device.

    Attributes:
        num_channels: Number of independently addressable DRAM channels. The
            paper's ``CHmask`` supports 1 through 32 channels.
        num_banks: Number of DRAM banks in each channel. Four neighboring banks
            form one processing-unit group, so this must be a multiple of four.
        dram_rows: Number of rows in each DRAM bank.
        dram_columns: Number of scalar values in each DRAM row. A row must hold
            a whole number of bursts.
        global_buffer_columns: Number of scalar positions in each channel's
            Global Buffer. This is an explicit target property because Global
            Buffer capacity is independent of DRAM row width.
        burst_length: Number of scalar values moved by one micro-operation. It
            cannot be wider than a DRAM row or Global Buffer.
        accumulator_slots_per_bank: Number of MAC result registers available to
            each bank. The paper calls an index into these registers ``Regid``.
        sigmoid_activation_function_id: Numeric ``AFid`` that this target uses
            for sigmoid. The paper does not publish the numeric mapping.
        shared_buffer_slots: Number of 256-bit staging slots shared by the
            device. The paper's 64 KiB buffer contains 2,048 slots.
    """

    num_channels: int
    num_banks: int
    dram_rows: int
    dram_columns: int
    global_buffer_columns: int
    burst_length: int
    accumulator_slots_per_bank: int
    sigmoid_activation_function_id: int
    shared_buffer_slots: int = 2_048

    def __post_init__(self) -> None:
        """Validate the hardware geometry.

        Raises:
            ValueError: If a dimension is outside its documented range or the
                row width is not an integer number of bursts, or a burst does
                not fit in the Global Buffer.
        """

        # CHmask has one bit for each of the paper's 32 possible channels.
        if not 1 <= self.num_channels <= 32:
            raise ValueError("num_channels must be between 1 and 32")

        # A partial four-bank group cannot be assigned to a processing unit.
        if self.num_banks < BANKS_PER_PU or self.num_banks % BANKS_PER_PU != 0:
            raise ValueError("num_banks must be at least 4 and divisible by 4")
        # One burst cannot cross a row boundary, so a row must contain a whole
        # number of bursts.
        if self.dram_rows < 1:
            raise ValueError("dram_rows must be at least 1")
        if self.dram_columns < 1:
            raise ValueError("dram_columns must be at least 1")
        if self.global_buffer_columns < 1:
            raise ValueError("global_buffer_columns must be at least 1")
        if self.burst_length < 1 or self.burst_length > self.dram_columns:
            raise ValueError("burst_length must be between 1 and dram_columns")
        if self.dram_columns % self.burst_length != 0:
            raise ValueError("dram_columns must be divisible by burst_length")
        # Global Buffer operations move at least one complete burst. No current
        # instruction requires the total buffer capacity to be burst-aligned.
        if self.burst_length > self.global_buffer_columns:
            raise ValueError("burst_length cannot exceed global_buffer_columns")
        # Regid and Shared Buffer addresses need at least one valid destination.
        if self.accumulator_slots_per_bank < 1:
            raise ValueError("accumulator_slots_per_bank must be at least 1")
        if self.sigmoid_activation_function_id < 0:
            raise ValueError("sigmoid_activation_function_id cannot be negative")
        if self.shared_buffer_slots < 1:
            raise ValueError("shared_buffer_slots must be at least 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class CentBlockPlacementSpec:
    """Describe how many channels are assigned to one model block.

    Attributes:
        channels_per_block: Number of physical channels used by one block. It
            must fit in the target and divide the target's channels evenly.
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
