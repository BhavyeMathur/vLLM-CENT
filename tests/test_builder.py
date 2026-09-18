"""Tests for CENT address mapping and program construction."""

import unittest

from vllm_cent.cent import (
    CentBlockPlacementSpec,
    CentChannelSet,
    CentHardwareSpec,
    CentMemoryAddress,
    CentProgramBuilder,
    CentSharedBufferAddress,
    ReadSingleBank,
    WriteSingleBank,
    ceil_div,
)
from vllm_cent.cent.builder import _single_bank_transfers


def make_builder(*, num_channels: int = 1) -> CentProgramBuilder:
    """Create the small CENT builder used by these tests.

    Args:
        num_channels: Number of physical channels.

    Returns:
        Builder with four banks, sixteen columns, and four-value bursts.
    """

    return CentProgramBuilder(
        CentHardwareSpec(
            num_channels=num_channels,
            num_banks=4,
            dram_rows=64,
            dram_columns=16,
            global_buffer_columns=16,
            burst_length=4,
            accumulator_slots_per_bank=2,
            sigmoid_activation_function_id=0,
            shared_buffer_slots=32,
        ),
        CentBlockPlacementSpec(channels_per_block=1),
    )


class IntegerMathTests(unittest.TestCase):
    """Test ceiling division."""

    def test_ceil_div_handles_exact_and_partial_groups(self) -> None:
        """Test exact groups, partial groups, and invalid inputs."""

        # Eight values fill exactly two four-value groups. A ninth value needs
        # a third group even though that final group is only partially full.
        self.assertEqual(ceil_div(8, 4), 2)
        self.assertEqual(ceil_div(9, 4), 3)
        for dividend, divisor in ((0, 1), (1, 0), (-1, 1)):
            with self.assertRaises(ValueError):
                ceil_div(dividend, divisor)


class SingleBankExpansionTests(unittest.TestCase):
    """Test how logical transfers become single-bank instructions."""

    def test_combines_bursts_and_splits_only_at_row_boundaries(self) -> None:
        """Combine bursts until the transfer reaches the next row."""

        transfers = tuple(
            _single_bank_transfers(
                WriteSingleBank,
                channel=1,
                bank=2,
                row=3,
                column=12,
                value_count=20,
                shared_buffer=CentSharedBufferAddress(slot=7),
                burst_length=4,
                row_width=16,
            )
        )

        # Five 4-value micro-operations start in the final burst of row 3.
        # OPsize=1 consumes that burst; OPsize=4 covers all of row 4. Rs moves
        # from slot 7 to slot 8 because the first instruction consumed one slot.
        self.assertEqual(
            transfers,
            (
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=1,
                        bank=2,
                        row=3,
                        column=12,
                    ),
                    operation_size=1,
                    source=CentSharedBufferAddress(slot=7),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=1,
                        bank=2,
                        row=4,
                        column=0,
                    ),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=8),
                ),
            ),
        )

    def test_builds_read_destinations_and_rejects_invalid_arguments(self) -> None:
        """Build reads and reject invalid transfer shapes."""

        reads = tuple(
            _single_bank_transfers(
                ReadSingleBank,
                channel=0,
                bank=1,
                row=2,
                column=0,
                value_count=5,
                shared_buffer=CentSharedBufferAddress(slot=3),
                burst_length=4,
                row_width=16,
            )
        )

        # Five values require two four-value burst micro-operations. Both fit
        # in the 16-column row, so they are encoded by one RD_SBK with OPsize 2
        # and Rd remains the caller-supplied Shared Buffer slot 3.
        self.assertEqual(
            reads,
            (
                ReadSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=1,
                        row=2,
                        column=0,
                    ),
                    operation_size=2,
                    destination=CentSharedBufferAddress(slot=3),
                ),
            ),
        )

        # These cases respectively use an unaligned column, a column beyond
        # the row, an empty transfer, and a row that cannot hold whole bursts.
        invalid_calls = (
            lambda: tuple(
                _single_bank_transfers(
                    WriteSingleBank,
                    channel=0,
                    bank=0,
                    row=0,
                    column=2,
                    value_count=1,
                    shared_buffer=CentSharedBufferAddress(slot=0),
                    burst_length=4,
                    row_width=16,
                )
            ),
            lambda: tuple(
                _single_bank_transfers(
                    WriteSingleBank,
                    channel=0,
                    bank=0,
                    row=0,
                    column=16,
                    value_count=1,
                    shared_buffer=CentSharedBufferAddress(slot=0),
                    burst_length=4,
                    row_width=16,
                )
            ),
            lambda: tuple(
                _single_bank_transfers(
                    WriteSingleBank,
                    channel=0,
                    bank=0,
                    row=0,
                    column=0,
                    value_count=0,
                    shared_buffer=CentSharedBufferAddress(slot=0),
                    burst_length=4,
                    row_width=16,
                )
            ),
            lambda: tuple(
                _single_bank_transfers(
                    WriteSingleBank,
                    channel=0,
                    bank=0,
                    row=0,
                    column=4,
                    value_count=1,
                    shared_buffer=CentSharedBufferAddress(slot=0),
                    burst_length=4,
                    row_width=6,
                )
            ),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()


class CentProgramBuilderTests(unittest.TestCase):
    """Test the public program builder."""

    def test_exposes_capacity_and_channel_mapping(self) -> None:
        """Map logical banks and matrices to physical channels."""

        builder = make_builder(num_channels=2)

        # One block owns one channel, and each channel has four banks. Matrix
        # data using all four block-local banks must be replicated across both
        # physical channels, which makes CHmask select channels 0 and 1.
        self.assertEqual(builder.total_banks, 4)
        expected_channels = CentChannelSet(channels=(0, 1))
        self.assertEqual(builder.all_channels(), expected_channels)
        self.assertEqual(builder.channels_for_matrix(4), expected_channels)
        self.assertEqual(builder.channel_set(range(2)), expected_channels)
        self.assertEqual(builder.bank_index(3), (0, 3))
        for call in (
            lambda: builder.channels_for_matrix(0),
            lambda: builder.channels_for_matrix(5),
            lambda: builder.channel_set((2,)),
            lambda: builder.bank_index(-1),
            lambda: builder.bank_index(4),
        ):
            with self.assertRaises(ValueError):
                call()

    def test_emit_single_bank_transfer_uses_one_instruction_per_row(self) -> None:
        """Keep the requested Shared Buffer and DRAM addresses."""

        builder = make_builder()
        builder.emit_single_bank_transfer(
            WriteSingleBank,
            0,
            2,
            7,
            5,
            shared_buffer=CentSharedBufferAddress(slot=4),
        )

        # Five values round up to two four-value bursts. Because both bursts
        # fit in row 7, one WR_SBK carries OPsize 2. Its Rs operand points at
        # Shared Buffer slot 4, the first 256-bit source burst supplied above.
        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=2,
                        row=7,
                        column=0,
                    ),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=4),
                )
            ],
        )
        with self.assertRaises(ValueError):
            builder.emit_single_bank_transfer(WriteSingleBank, 0, 0, 64, 1)

    def test_neighbor_transfer_replicates_banks_and_buffer_partitions(self) -> None:
        """Repeat alternating-bank partitions across block copies."""

        builder = make_builder(num_channels=2)
        builder.emit_neighbor_bank_transfer(WriteSingleBank, 8, 1, 6, 4)

        # Eight values split into two four-value partitions. Bank group 1 uses
        # the odd bank in each neighbor pair, so partitions select banks 1 and
        # 3. Each partition is replicated on both physical channels. The two
        # partitions read distinct Shared Buffer burst slots 0 and 1.
        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=1,
                        row=6,
                        column=0,
                    ),
                    operation_size=1,
                    source=CentSharedBufferAddress(slot=0),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=1,
                        bank=1,
                        row=6,
                        column=0,
                    ),
                    operation_size=1,
                    source=CentSharedBufferAddress(slot=0),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=3,
                        row=6,
                        column=0,
                    ),
                    operation_size=1,
                    source=CentSharedBufferAddress(slot=1),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=1,
                        bank=3,
                        row=6,
                        column=0,
                    ),
                    operation_size=1,
                    source=CentSharedBufferAddress(slot=1),
                ),
            ],
        )

    def test_bank_group_transfer_starts_at_requested_buffer_slot(self) -> None:
        """Offset every partition from an explicit Shared Buffer address."""

        builder = make_builder()
        builder.emit_bank_group_transfer(
            WriteSingleBank,
            channels_required=1,
            utilized_banks=1,
            bank_group=0,
            row=6,
            size_per_bank=8,
            shared_buffer=CentSharedBufferAddress(slot=5),
        )

        # Eight values occupy two four-value slots. The single selected bank
        # therefore reads slots 5 and 6 in one two-burst instruction.
        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=0,
                        row=6,
                        column=0,
                    ),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=5),
                )
            ],
        )
        for arguments in (
            (WriteSingleBank, 8, 2, 6, 4),
            (WriteSingleBank, 0, 0, 6, 4),
        ):
            with self.assertRaises(ValueError):
                builder.emit_neighbor_bank_transfer(
                    *arguments  # type: ignore[arg-type]
                )

    def test_four_bank_group_transfer_maps_reads_and_shared_slots(self) -> None:
        """Send each partition to the selected bank in its group."""

        builder = make_builder(num_channels=2)
        builder.emit_bank_group_transfer(ReadSingleBank, 1, 1, 2, 9, 5)

        # Bank group 2 selects bank 2 out of each four-bank PU. The one-channel
        # block is copied onto both physical channels. Five requested values
        # require OPsize 2, and both copies write the same logical Rd slot 0.
        self.assertEqual(
            builder.instructions,
            [
                ReadSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=2,
                        row=9,
                        column=0,
                    ),
                    operation_size=2,
                    destination=CentSharedBufferAddress(slot=0),
                ),
                ReadSingleBank(
                    address=CentMemoryAddress(
                        channel=1,
                        bank=2,
                        row=9,
                        column=0,
                    ),
                    operation_size=2,
                    destination=CentSharedBufferAddress(slot=0),
                ),
            ],
        )
        # These calls use an invalid channel count, bank count, group, or size.
        invalid_calls = (
            lambda: builder.emit_bank_group_transfer(ReadSingleBank, 0, 1, 0, 0, 4),
            lambda: builder.emit_bank_group_transfer(ReadSingleBank, 3, 1, 0, 0, 4),
            lambda: builder.emit_bank_group_transfer(ReadSingleBank, 1, 0, 0, 0, 4),
            lambda: builder.emit_bank_group_transfer(ReadSingleBank, 1, 1, 4, 0, 4),
            lambda: builder.emit_bank_group_transfer(ReadSingleBank, 1, 1, 0, 0, 0),
        )
        for call in invalid_calls:
            with self.assertRaises(ValueError):
                call()

        two_channel_block = CentProgramBuilder(
            CentHardwareSpec(
                num_channels=4,
                num_banks=4,
                dram_rows=64,
                dram_columns=16,
                global_buffer_columns=16,
                burst_length=4,
                accumulator_slots_per_bank=2,
                sigmoid_activation_function_id=0,
                shared_buffer_slots=32,
            ),
            CentBlockPlacementSpec(channels_per_block=2),
        )
        with self.assertRaisesRegex(ValueError, "placement.channels_per_block"):
            two_channel_block.emit_bank_group_transfer(
                ReadSingleBank,
                channels_required=1,
                utilized_banks=1,
                bank_group=0,
                row=0,
                size_per_bank=4,
            )

    def test_append_validates_and_finish_freezes_the_sequence(self) -> None:
        """Freeze the instruction sequence when the builder finishes."""

        builder = make_builder()
        builder.emit_single_bank_transfer(WriteSingleBank, 0, 0, 0, 4)
        program = builder.finish()

        expected = WriteSingleBank(
            address=CentMemoryAddress(
                channel=0,
                bank=0,
                row=0,
                column=0,
            ),
            operation_size=1,
            source=CentSharedBufferAddress(slot=0),
        )
        self.assertEqual(builder.instructions, [expected])
        self.assertEqual(program.instructions, (expected,))
        with self.assertRaises(ValueError):
            builder.finish()
        with self.assertRaises(ValueError):
            builder.append(program.instructions[0])

    def test_batch_append_rejects_every_instruction_atomically(self) -> None:
        """Leave the builder unchanged when a later command is invalid."""

        builder = make_builder()
        valid = WriteSingleBank(
            address=CentMemoryAddress(
                channel=0,
                bank=0,
                row=0,
                column=0,
            ),
            operation_size=1,
            source=CentSharedBufferAddress(slot=0),
        )
        invalid = WriteSingleBank(
            # The test target provides rows 0 through 63. Row 64 therefore
            # fails target validation after the first command has been checked.
            address=CentMemoryAddress(
                channel=0,
                bank=0,
                row=64,
                column=0,
            ),
            operation_size=1,
            source=CentSharedBufferAddress(slot=1),
        )

        with self.assertRaisesRegex(ValueError, "row"):
            builder._append_all((valid, invalid))
        self.assertEqual(builder.instructions, [])

        builder._append_all((valid,))
        self.assertEqual(builder.instructions, [valid])

    def test_multi_instruction_emitters_do_not_leave_partial_work(self) -> None:
        """Reject invalid later rows or slots without keeping a valid prefix."""

        row_builder = make_builder()
        with self.assertRaisesRegex(ValueError, "row"):
            # Row 63 has room for the first burst at column 12. The second
            # burst would continue into nonexistent row 64.
            row_builder.emit_single_bank_transfer(
                WriteSingleBank,
                channel=0,
                bank=0,
                row=63,
                column=12,
                value_count=8,
            )
        self.assertEqual(row_builder.instructions, [])

        buffer_builder = make_builder()
        with self.assertRaisesRegex(ValueError, "Shared Buffer"):
            # Partition zero uses the final valid slot 31. Partition one would
            # begin at slot 32, outside this target's 32-slot buffer.
            buffer_builder.emit_neighbor_bank_transfer(
                WriteSingleBank,
                value_count=8,
                bank_group=0,
                row=0,
                size_per_bank=4,
                shared_buffer=CentSharedBufferAddress(slot=31),
            )
        self.assertEqual(buffer_builder.instructions, [])


if __name__ == "__main__":
    unittest.main()
