"""Map logical data transfers to valid CENT instructions."""

from collections.abc import Iterable, Iterator
from typing import TypeAlias

from .hardware import BANKS_PER_PU, CentBlockPlacementSpec, CentHardwareSpec
from .instructions import (
    CentChannelSet,
    CentInstruction,
    CentMemoryAddress,
    CentSharedBufferAddress,
    ReadSingleBank,
    WriteSingleBank,
    validate_instruction,
)
from .program import CentProgram
from .utils import ceil_div, require_nonnegative, require_positive

__all__ = ["CentProgramBuilder"]

SingleBankInstructionType: TypeAlias = type[WriteSingleBank] | type[ReadSingleBank]


def _single_bank_transfers(
    instruction_type: SingleBankInstructionType,
    *,
    channel: int,
    bank: int,
    row: int,
    column: int,
    value_count: int,
    shared_buffer: CentSharedBufferAddress,
    burst_length: int,
    row_width: int,
) -> Iterator[WriteSingleBank | ReadSingleBank]:
    """Split one transfer into instructions that do not cross DRAM rows.

    ``OPsize`` lets one instruction move several bursts from the same row. A
    transfer that reaches another row needs another instruction.

    Args:
        instruction_type: Direction of the transfer: write to or read from DRAM.
        channel: DRAM channel number, called ``CHid`` in the paper.
        bank: Bank number within the channel, called ``BK``.
        row: First DRAM row, called ``RO``.
        column: First scalar position in the row, called ``CO``.
        value_count: Number of scalar values to move.
        shared_buffer: First staging slot. It is ``Rs`` for a DRAM write and
            ``Rd`` for a DRAM read.
        burst_length: Number of scalar values moved by one micro-operation.
        row_width: Number of scalar positions in one DRAM row.

    Yields:
        One instruction for each touched row. Its ``OPsize`` is the number of
        bursts moved in that row.

    Raises:
        ValueError: If a size is invalid or ``column`` is not burst-aligned.
    """

    for name, value in (
        ("value_count", value_count),
        ("burst_length", burst_length),
        ("row_width", row_width),
    ):
        require_positive(name, value)
    require_nonnegative("column", column)
    if column >= row_width:
        raise ValueError("column is outside the starting DRAM row")
    if column % burst_length:
        raise ValueError("column must be aligned to one burst")
    if row_width % burst_length:
        raise ValueError("row_width must be divisible by burst_length")

    # TODO(runtime): We round a partial final burst up to a full slot. We need to
    # define the values placed in unused lanes or add a valid-lane mask.

    # OPsize counts bursts, not scalar values. Each burst uses one Shared Buffer
    # slot, including a final burst that is only partly full.
    remaining_operations = ceil_div(value_count, burst_length)
    completed_operations = 0
    current_row = row
    current_column = column
    while remaining_operations:
        # One instruction stops at the row boundary. With 16 columns and
        # four-value bursts, column 12 has room for one burst; column 0 has four.
        available_operations = (row_width - current_column) // burst_length
        operation_size = min(remaining_operations, available_operations)

        # CentMemoryAddress names the first burst. OPsize tells CENT to continue
        # through later columns, so those bursts need no separate addresses.
        address = CentMemoryAddress(
            channel=channel,
            bank=bank,
            row=current_row,
            column=current_column,
        )

        # Each earlier burst used one 256-bit Shared Buffer slot. Add that count
        # so the next DRAM row continues at the next input or output slot.
        buffer_address = CentSharedBufferAddress(
            slot=shared_buffer.slot + completed_operations
        )

        # A write reads ``Rs``. A read writes ``Rd``. Separate classes make that
        # direction clear even though both walk through DRAM in the same way.
        if instruction_type is WriteSingleBank:
            yield WriteSingleBank(
                address=address,
                operation_size=operation_size,
                source=buffer_address,
            )
        else:
            yield ReadSingleBank(
                address=address,
                operation_size=operation_size,
                destination=buffer_address,
            )
        # The next instruction starts at column zero of the following row.
        remaining_operations -= operation_size
        completed_operations += operation_size
        current_row += 1
        current_column = 0


class CentProgramBuilder:
    """Collect validated instructions for one device and block placement.

    Args:
        hardware: Limits of the CENT device that will run the program.
        placement: Number of channels assigned to one model block.

    Attributes:
        hardware: Device that will run the program.
        placement: Channel allocation for one model block.
        instructions: Commands collected so far, in execution order.
        _finished: Whether :meth:`finish` has frozen the command list.
    """

    # TODO(placement): We currently copy work to channel groups starting at
    # channel zero. We need to decide whether a program represents one block or
    # every replica, and then represent each physical group explicitly.

    def __init__(
        self, hardware: CentHardwareSpec, placement: CentBlockPlacementSpec
    ) -> None:
        """Create an empty builder and check its channel allocation.

        Args:
            hardware: Limits of the target CENT device.
            placement: Number of channels assigned to one model block.

        Raises:
            ValueError: If the placement cannot tile the target channels.
        """

        if placement.channels_per_block > hardware.num_channels:
            raise ValueError("channels_per_block cannot exceed num_channels")
        if hardware.num_channels % placement.channels_per_block:
            raise ValueError("channels_per_block must divide num_channels")

        # Lowering appends to this list. finish() later copies it into a tuple.
        self.hardware = hardware
        self.placement = placement
        self.instructions: list[CentInstruction] = []
        self._finished = False

    @property
    def total_banks(self) -> int:
        """Return the number of banks available to one block.

        Returns:
            ``channels_per_block * num_banks``.
        """

        # The compiler gives these banks one flat index across all block channels.
        return self.placement.channels_per_block * self.hardware.num_banks

    def all_channels(self) -> CentChannelSet:
        """Create a ``CHmask`` selection containing every device channel.

        Returns:
            Channel numbers from zero through ``num_channels - 1``.
        """

        return CentChannelSet(channels=tuple(range(self.hardware.num_channels)))

    def channels_for_matrix(self, utilized_banks: int) -> CentChannelSet:
        """Select all complete channel groups that can hold a matrix copy.

        Args:
            utilized_banks: Number of block-local banks that contain matrix rows.

        Returns:
            Channels belonging to every complete copy of the matrix layout.

        Raises:
            ValueError: If it is outside its valid range.
        """

        # TODO(placement): Keep matrix copies inside their block channel regions.
        #
        # This method currently packs copies every ``channels_per_matrix``. Other
        # transfer helpers space copies every ``channels_per_block``. When those
        # sizes differ, the two methods select different physical layouts.

        if not 1 <= utilized_banks <= self.total_banks:
            raise ValueError("utilized_banks must be between 1 and total_banks")
        # Banks are filled one channel at a time. Round up to find the whole
        # channels needed by one copy, then keep every complete copy on the device.
        channels_per_matrix = ceil_div(utilized_banks, self.hardware.num_banks)
        count = (
            self.hardware.num_channels // channels_per_matrix
        ) * channels_per_matrix
        return CentChannelSet(channels=tuple(range(count)))

    def channel_set(self, channels: range | tuple[int, ...]) -> CentChannelSet:
        """Create a channel selection and check it against this device.

        Args:
            channels: Physical channel numbers to select.

        Returns:
            Validated channel selection used as ``CHmask``.

        Raises:
            ValueError: If a selected channel does not exist.
        """

        selection = CentChannelSet(channels=tuple(channels))
        if any(c >= self.hardware.num_channels for c in selection.channels):
            raise ValueError("channel index is outside the device")
        return selection

    def bank_index(self, logical_bank: int) -> tuple[int, int]:
        """Convert one flat bank number into a channel and bank pair.

        Args:
            logical_bank: Bank number in the block's flattened bank list.

        Returns:
            Local channel number and bank number within that channel.

        Raises:
            ValueError: If it is outside the block allocation.
        """

        if not 0 <= logical_bank < self.total_banks:
            raise ValueError("logical_bank is outside the block allocation")
        # Whole groups of num_banks select the channel; the remainder selects
        # the bank inside that channel.
        return divmod(logical_bank, self.hardware.num_banks)

    def append(self, instruction: CentInstruction) -> None:
        """Check one instruction against the device and append it.

        Args:
            instruction: CENT command to add after the existing commands.

        Raises:
            TypeError: If ``instruction`` is unsupported.
            ValueError: If it is incompatible with the target or the builder
                was already finished.
        """

        self._append_all((instruction,))

    def _append_all(self, instructions: Iterable[CentInstruction]) -> None:
        """Validate and append one instruction batch atomically.

        Args:
            instructions: Commands to append in their execution order.

        Raises:
            TypeError: If any command is unsupported.
            ValueError: If any command is incompatible with the target or the
                builder was already finished.
        """

        if self._finished:
            raise ValueError("cannot append after the builder is finished")

        # Materialize generators before validation. If instruction generation
        # or validation fails, the builder remains exactly as it was before the
        # batch began.
        pending = tuple(instructions)
        for instruction in pending:
            validate_instruction(instruction, self.hardware)
        self.instructions.extend(pending)

    def emit_single_bank_transfer(
        self,
        instruction_type: SingleBankInstructionType,
        channel: int,
        bank: int,
        row: int,
        value_count: int,
        *,
        column: int = 0,
        shared_buffer: CentSharedBufferAddress | None = None,
    ) -> None:
        """Move values between the Shared Buffer and one DRAM bank.

        Args:
            instruction_type: Transfer direction: write to or read from DRAM.
            channel: DRAM channel number, called ``CHid`` in the paper.
            bank: Bank number within the channel, called ``BK``.
            row: First DRAM row, called ``RO``.
            value_count: Number of scalar values to move.
            column: First scalar position in the DRAM row, called ``CO``.
            shared_buffer: First staging slot, called ``Rs`` for a write and
                ``Rd`` for a read. The default is slot zero.

        Raises:
            ValueError: If the transfer is incompatible with the target.
        """

        # Model lowering uses slot zero when it does not request another slot.
        buffer_address = shared_buffer or CentSharedBufferAddress(slot=0)

        # The helper decides where row splits occur. append() then checks each
        # generated instruction against the hardware.
        self._append_all(
            _single_bank_transfers(
                instruction_type,
                channel=channel,
                bank=bank,
                row=row,
                column=column,
                value_count=value_count,
                shared_buffer=buffer_address,
                burst_length=self.hardware.burst_length,
                row_width=self.hardware.dram_columns,
            )
        )

    def emit_neighbor_bank_transfer(
        self,
        instruction_type: SingleBankInstructionType,
        value_count: int,
        bank_group: int,
        row: int,
        size_per_bank: int,
        *,
        shared_buffer: CentSharedBufferAddress | None = None,
    ) -> None:
        """Transfer vector pieces through one bank in each neighboring pair.

        Args:
            instruction_type: Transfer direction: write to or read from DRAM.
            value_count: Number of vector values to divide among banks.
            bank_group: Position within each bank pair, either zero or one.
            row: First DRAM row used by every selected bank.
            size_per_bank: Number of vector values assigned to each bank.
            shared_buffer: First staging slot used by the first vector piece.
                The default is slot zero.

        Raises:
            ValueError: If the group or size is invalid.
        """

        require_positive("value_count", value_count)
        require_positive("size_per_bank", size_per_bank)
        if bank_group not in (0, 1):
            raise ValueError("neighbor bank_group must be 0 or 1")
        # TODO(ISA): Define the roles of the two banks in each pair.
        #
        # Neighboring banks are paired like this:
        #
        # - pair 0: bank 0 and bank 1
        # - pair 1: bank 2 and bank 3
        #
        # The paper does not answer these questions:
        #
        # - Which bank contains each input?
        # - Which bank reads from its neighbor?
        # - Which bank's processing unit stores the result?
        #
        # The current mapping uses the order found in the reference code.

        # Each vector piece uses the same position in the next bank pair.
        partitions = ceil_div(value_count, size_per_bank)

        # Repeat the transfer for each equal-sized copy of the block.
        copies = self.hardware.num_channels // self.placement.channels_per_block

        # A new vector piece starts after the slots used by earlier pieces. An
        # explicit base keeps unrelated tensors from silently sharing slot zero.
        first_buffer = shared_buffer or CentSharedBufferAddress(slot=0)
        slots_per_partition = ceil_div(size_per_bank, self.hardware.burst_length)
        instructions: list[WriteSingleBank | ReadSingleBank] = []
        for partition in range(partitions):
            channel, bank = self.bank_index(partition * 2 + bank_group)
            buffer = CentSharedBufferAddress(
                slot=(first_buffer.slot + partition * slots_per_partition)
            )
            for copy in range(copies):
                # Replicas use the same bank and Shared Buffer slots in another
                # physical channel group.
                instructions.extend(
                    _single_bank_transfers(
                        instruction_type,
                        channel=(channel + self.placement.channels_per_block * copy),
                        bank=bank,
                        row=row,
                        column=0,
                        value_count=size_per_bank,
                        shared_buffer=buffer,
                        burst_length=self.hardware.burst_length,
                        row_width=self.hardware.dram_columns,
                    )
                )
        self._append_all(instructions)

    def emit_bank_group_transfer(
        self,
        instruction_type: SingleBankInstructionType,
        channels_required: int,
        utilized_banks: int,
        bank_group: int,
        row: int,
        size_per_bank: int,
        *,
        shared_buffer: CentSharedBufferAddress | None = None,
    ) -> None:
        """Transfer vector pieces through one bank in each four-bank group.

        Args:
            instruction_type: Transfer direction: write to or read from DRAM.
            channels_required: Number of channels occupied by one copied layout.
            utilized_banks: Number of four-bank groups that contain data.
            bank_group: Position selected within every group, from zero to three.
            row: First DRAM row used by every selected bank.
            size_per_bank: Number of vector values assigned to each selected bank.
            shared_buffer: First staging slot used by the first vector piece.
                The default is slot zero.

        Raises:
            ValueError: If a channel, group, or size value is invalid.
        """

        require_positive("channels_required", channels_required)
        require_positive("utilized_banks", utilized_banks)
        require_positive("size_per_bank", size_per_bank)
        if self.hardware.num_channels % channels_required:
            raise ValueError("channels_required must divide num_channels")
        if channels_required != self.placement.channels_per_block:
            raise ValueError(
                "channels_required must equal placement.channels_per_block"
            )
        if bank_group not in range(BANKS_PER_PU):
            raise ValueError("bank_group must be between 0 and 3")

        # bank_group selects the same position from each four-bank group.
        copies = self.hardware.num_channels // channels_required
        first_buffer = shared_buffer or CentSharedBufferAddress(slot=0)
        slots_per_partition = ceil_div(size_per_bank, self.hardware.burst_length)
        instructions: list[WriteSingleBank | ReadSingleBank] = []
        for partition in range(utilized_banks):
            channel, bank = self.bank_index(partition * BANKS_PER_PU + bank_group)
            buffer = CentSharedBufferAddress(
                slot=(first_buffer.slot + partition * slots_per_partition)
            )
            for copy in range(copies):
                # Each copied layout uses the same Shared Buffer piece.
                instructions.extend(
                    _single_bank_transfers(
                        instruction_type,
                        channel=channel + channels_required * copy,
                        bank=bank,
                        row=row,
                        column=0,
                        value_count=size_per_bank,
                        shared_buffer=buffer,
                        burst_length=self.hardware.burst_length,
                        row_width=self.hardware.dram_columns,
                    )
                )
        self._append_all(instructions)

    def finish(self) -> CentProgram:
        """Finish building and return an immutable program.

        Returns:
            Program containing the collected instructions and target hardware.

        Raises:
            ValueError: If the builder has already been finished.
        """

        if self._finished:
            raise ValueError("builder has already been finished")
        # Copy the mutable list so later code cannot change the returned program.
        program = CentProgram(
            hardware=self.hardware, instructions=tuple(self.instructions)
        )
        self._finished = True
        return program
