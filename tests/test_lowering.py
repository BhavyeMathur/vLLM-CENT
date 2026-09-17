"""Tests for model-independent CENT lowering components."""

import unittest
from dataclasses import replace

from vllm_cent import CentBlockPlacementSpec, CentHardwareSpec
from vllm_cent.cent import (
    Accumulate,
    ApplyActivation,
    CentChannelSet,
    CentMemoryAddress,
    CentProgramBuilder,
    CentSharedBufferAddress,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    MacAllBanks,
    MacOperandSource,
    ReadActivation,
    ReadMac,
    ReadSingleBank,
    WriteBias,
    WriteGlobalBuffer,
    WriteSingleBank,
)
from vllm_cent.lowering import (
    CentDramRowRange,
    CentSharedBufferSpan,
    lower_accumulate,
    lower_load_bank_group_vector,
    lower_l2_norm,
    lower_rms_norm,
    lower_store_bank_group_vector,
    lower_weight_gemv,
)
from vllm_cent.lowering.data_movement import (
    _lower_bank_group_vector_transfer,
)
from vllm_cent.lowering.normalization import _lower_sum_of_squares
from vllm_cent.lowering.transformer import TransformerAttentionSpec
from vllm_cent.lowering.utils import (
    _require_dram_row_capacity,
    _require_shared_buffer_capacity,
)


def make_builder(
    *,
    dram_columns: int = 16,
    burst_length: int = 4,
    accumulator_slots: int = 4,
) -> CentProgramBuilder:
    """Create the small CENT target used by lowering tests.

    Args:
        dram_columns: Scalar values stored in one DRAM row.
        burst_length: Scalar values moved by one Shared Buffer slot.
        accumulator_slots: MAC result registers available to each bank.

    Returns:
        Empty builder for one channel containing four banks.
    """

    hardware = CentHardwareSpec(
        num_channels=1,
        num_banks=4,
        dram_rows=128,
        dram_columns=dram_columns,
        burst_length=burst_length,
        accumulator_slots_per_bank=accumulator_slots,
        sigmoid_activation_function_id=3,
        shared_buffer_slots=64,
    )
    return CentProgramBuilder(
        hardware,
        CentBlockPlacementSpec(channels_per_block=1),
    )


class LoweringUtilityTests(unittest.TestCase):
    """Test validation shared by reusable lowering operations."""

    def test_shared_buffer_capacity_reports_the_named_span(self) -> None:
        """Accept an exact fit and identify an undersized buffer."""

        span = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=5),
            slot_count=2,
        )
        _require_shared_buffer_capacity("input", span, 2)

        with self.assertRaisesRegex(
            ValueError,
            "input needs 3 Shared Buffer slots, but its span contains 2",
        ):
            _require_shared_buffer_capacity("input", span, 3)
        with self.assertRaisesRegex(ValueError, "required_slots"):
            _require_shared_buffer_capacity("input", span, 0)

    def test_dram_capacity_reports_the_named_range(self) -> None:
        """Accept an exact fit and identify an undersized row range."""

        rows = CentDramRowRange(start_row=7, row_count=2)
        _require_dram_row_capacity("weights", rows, 2)

        with self.assertRaisesRegex(
            ValueError,
            "weights needs 3 DRAM rows, but its range contains 2",
        ):
            _require_dram_row_capacity("weights", rows, 3)
        with self.assertRaisesRegex(ValueError, "required_rows"):
            _require_dram_row_capacity("weights", rows, 0)


class LoweringBindingTests(unittest.TestCase):
    """Test immutable regions passed between operation lowerers."""

    def test_shared_buffer_span_addresses_each_slot(self) -> None:
        """Offset addresses from the span's first Shared Buffer slot."""

        span = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=7),
            slot_count=3,
        )

        self.assertEqual(
            span.address(2),
            CentSharedBufferAddress(slot=9),
        )
        with self.assertRaisesRegex(ValueError, "outside"):
            span.address(3)
        with self.assertRaisesRegex(ValueError, "negative"):
            span.address(-1)

    def test_dram_range_addresses_each_row(self) -> None:
        """Offset row numbers from the range's first DRAM row."""

        rows = CentDramRowRange(start_row=11, row_count=2)

        self.assertEqual(rows.row(1), 12)
        with self.assertRaisesRegex(ValueError, "outside"):
            rows.row(2)
        with self.assertRaisesRegex(ValueError, "negative"):
            rows.row(-1)

    def test_bindings_reject_empty_regions(self) -> None:
        """Reject spans that cannot contain any data."""

        with self.assertRaises(ValueError):
            CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=0),
                slot_count=0,
            )
        with self.assertRaises(ValueError):
            CentDramRowRange(start_row=0, row_count=0)


class LinearLoweringTests(unittest.TestCase):
    """Test model-independent matrix-vector lowering."""

    def test_gemv_uses_explicit_input_and_output_spans(self) -> None:
        """Read the requested input and write the requested result slots."""

        builder = make_builder()
        input_buffer = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=5),
            # Sixteen input values occupy four four-value slots.
            slot_count=4,
        )
        output_buffer = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=20),
            # Eight outputs over four banks produce two accumulator results.
            slot_count=2,
        )

        lower_weight_gemv(
            builder,
            weights=CentDramRowRange(start_row=7, row_count=2),
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            vector_size=16,
            output_size=8,
        )

        channels = CentChannelSet(channels=(0,))
        self.assertEqual(
            builder.instructions,
            [
                WriteGlobalBuffer(
                    operation_size=4,
                    column=0,
                    source=CentSharedBufferAddress(slot=5),
                    channels=channels,
                ),
                WriteBias(
                    source=CentSharedBufferAddress(slot=20),
                    channels=channels,
                ),
                WriteBias(
                    source=CentSharedBufferAddress(slot=21),
                    channels=channels,
                ),
                MacAllBanks(
                    operation_size=4,
                    channels=channels,
                    row=7,
                    column=0,
                    accumulation_register=0,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                ),
                MacAllBanks(
                    operation_size=4,
                    channels=channels,
                    row=8,
                    column=0,
                    accumulation_register=1,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                ),
                ReadMac(
                    destination=CentSharedBufferAddress(slot=20),
                    accumulation_register=0,
                    channels=channels,
                ),
                ReadMac(
                    destination=CentSharedBufferAddress(slot=21),
                    accumulation_register=1,
                    channels=channels,
                ),
            ],
        )

    def test_gemv_keeps_accumulator_groups_in_distinct_slots(self) -> None:
        """Do not overwrite results when outputs exceed register capacity."""

        builder = make_builder(accumulator_slots=2)

        lower_weight_gemv(
            builder,
            weights=CentDramRowRange(start_row=30, row_count=4),
            input_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=1),
                slot_count=4,
            ),
            output_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=10),
                slot_count=4,
            ),
            vector_size=16,
            # Sixteen outputs over four banks require four result slots. Two
            # registers process those results as groups [10, 11] and [12, 13].
            output_size=16,
        )

        reads = [
            instruction
            for instruction in builder.instructions
            if isinstance(instruction, ReadMac)
        ]
        self.assertEqual(
            reads,
            [
                ReadMac(
                    destination=CentSharedBufferAddress(slot=10),
                    accumulation_register=0,
                    channels=CentChannelSet(channels=(0,)),
                ),
                ReadMac(
                    destination=CentSharedBufferAddress(slot=11),
                    accumulation_register=1,
                    channels=CentChannelSet(channels=(0,)),
                ),
                ReadMac(
                    destination=CentSharedBufferAddress(slot=12),
                    accumulation_register=0,
                    channels=CentChannelSet(channels=(0,)),
                ),
                ReadMac(
                    destination=CentSharedBufferAddress(slot=13),
                    accumulation_register=1,
                    channels=CentChannelSet(channels=(0,)),
                ),
            ],
        )

    def test_gemv_reads_each_input_row_from_the_next_slots(self) -> None:
        """Advance the input span when a vector crosses a DRAM row."""

        builder = make_builder()

        lower_weight_gemv(
            builder,
            weights=CentDramRowRange(start_row=40, row_count=2),
            input_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=6),
                # Twenty values occupy five four-value slots.
                slot_count=5,
            ),
            output_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=20),
                slot_count=1,
            ),
            vector_size=20,
            output_size=4,
        )

        writes = [
            instruction
            for instruction in builder.instructions
            if isinstance(instruction, WriteGlobalBuffer)
        ]
        self.assertEqual(
            writes,
            [
                WriteGlobalBuffer(
                    operation_size=4,
                    column=0,
                    source=CentSharedBufferAddress(slot=6),
                    channels=CentChannelSet(channels=(0,)),
                ),
                WriteGlobalBuffer(
                    operation_size=1,
                    column=0,
                    source=CentSharedBufferAddress(slot=10),
                    channels=CentChannelSet(channels=(0,)),
                ),
            ],
        )

    def test_gemv_returns_raw_and_activated_results_separately(self) -> None:
        """Preserve a projection before writing its activated form."""

        builder = make_builder()

        lower_weight_gemv(
            builder,
            weights=CentDramRowRange(start_row=7, row_count=1),
            input_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=2),
                slot_count=4,
            ),
            output_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=12),
                slot_count=1,
            ),
            activated_output_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=18),
                slot_count=1,
            ),
            vector_size=16,
            output_size=4,
        )

        result_instructions = [
            instruction
            for instruction in builder.instructions
            if isinstance(
                instruction, (ReadMac, ReadActivation, ApplyActivation)
            )
        ]
        self.assertEqual(
            result_instructions,
            [
                ReadMac(
                    destination=CentSharedBufferAddress(slot=12),
                    accumulation_register=0,
                    channels=CentChannelSet(channels=(0,)),
                ),
                ApplyActivation(
                    channels=CentChannelSet(channels=(0,)),
                    activation_function_id=3,
                    accumulation_register=0,
                ),
                ReadActivation(
                    destination=CentSharedBufferAddress(slot=18),
                    accumulation_register=0,
                    channels=CentChannelSet(channels=(0,)),
                ),
            ],
        )

    def test_gemv_rejects_regions_that_are_too_small(self) -> None:
        """Reject bindings that cannot contain an operand or result."""

        builder = make_builder()
        weights = CentDramRowRange(start_row=0, row_count=2)
        input_buffer = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=0),
            slot_count=3,
        )
        output_buffer = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=10),
            slot_count=2,
        )

        with self.assertRaisesRegex(ValueError, "input_buffer"):
            lower_weight_gemv(
                builder,
                weights=weights,
                input_buffer=input_buffer,
                output_buffer=output_buffer,
                vector_size=16,
                output_size=8,
            )
        with self.assertRaisesRegex(ValueError, "weights"):
            lower_weight_gemv(
                builder,
                weights=CentDramRowRange(start_row=0, row_count=1),
                input_buffer=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=0),
                    slot_count=4,
                ),
                output_buffer=output_buffer,
                vector_size=16,
                output_size=8,
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            lower_weight_gemv(
                builder,
                weights=weights,
                input_buffer=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=0),
                    slot_count=4,
                ),
                output_buffer=output_buffer,
                activated_output_buffer=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=11),
                    slot_count=2,
                ),
                vector_size=16,
                output_size=8,
            )


class NormalizationLoweringTests(unittest.TestCase):
    """Test reusable vector-normalization lowering."""

    def test_sum_of_squares_emits_neighbor_bank_mac(self) -> None:
        """Square vector partitions and read their partial sums."""

        builder = make_builder()
        _lower_sum_of_squares(
            builder,
            input_rows=CentDramRowRange(start_row=3, row_count=1),
            input_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=4),
                slot_count=4,
            ),
            partial_sum_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=20),
                slot_count=1,
            ),
            value_count=16,
        )

        # Four banks split the vector into two eight-value partitions. Each
        # partition is copied into a neighboring bank pair so MAC_ABK squares
        # matching values and accumulates one partial sum per active PU.
        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=0, row=3, column=0
                    ),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=2, row=3, column=0
                    ),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=6),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=1, row=3, column=0
                    ),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=3, row=3, column=0
                    ),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=6),
                ),
                WriteBias(
                    source=CentSharedBufferAddress(slot=20),
                    channels=CentChannelSet(channels=(0,)),
                ),
                MacAllBanks(
                    operation_size=2,
                    channels=CentChannelSet(channels=(0,)),
                    row=3,
                    column=0,
                    accumulation_register=0,
                    operand_source=MacOperandSource.NEXT_BANK,
                ),
                ReadMac(
                    destination=CentSharedBufferAddress(slot=20),
                    accumulation_register=0,
                    channels=CentChannelSet(channels=(0,)),
                ),
            ],
        )

    def test_l2_norm_scales_the_input_after_sum_of_squares(self) -> None:
        """Place the vector and its scale in neighboring input banks."""

        builder = make_builder()
        lower_l2_norm(
            builder,
            input_rows=CentDramRowRange(start_row=3, row_count=1),
            work_rows=CentDramRowRange(start_row=4, row_count=1),
            input_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=4),
                slot_count=4,
            ),
            scale_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=12),
                slot_count=4,
            ),
            partial_sum_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=20),
                slot_count=1,
            ),
            value_count=16,
        )

        # _lower_sum_of_squares emits the first seven instructions tested
        # above. L2 normalization then writes all 16 input values to bank zero,
        # writes the four-slot repeated scale to bank one, and multiplies both
        # banks into bank two.
        self.assertEqual(
            builder.instructions[7:],
            [
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=0, row=4, column=0
                    ),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=1, row=4, column=0
                    ),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=12),
                ),
                ElementwiseMultiply(
                    operation_size=4,
                    channels=CentChannelSet(channels=(0,)),
                    row=4,
                    column=0,
                ),
            ],
        )

    def test_l2_norm_rejects_undersized_scale_or_work_rows(self) -> None:
        """Reject storage that cannot hold every normalized value."""

        full = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=0),
            slot_count=4,
        )
        partial = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=8),
            slot_count=1,
        )

        with self.assertRaisesRegex(ValueError, "scale_buffer"):
            lower_l2_norm(
                make_builder(),
                input_rows=CentDramRowRange(start_row=3, row_count=1),
                work_rows=CentDramRowRange(start_row=4, row_count=1),
                input_buffer=full,
                # Sixteen values occupy four Shared Buffer slots.
                scale_buffer=replace(full, slot_count=3),
                partial_sum_buffer=partial,
                value_count=16,
            )

        with self.assertRaisesRegex(ValueError, "work_rows needs 2"):
            lower_l2_norm(
                make_builder(dram_columns=8),
                input_rows=CentDramRowRange(start_row=3, row_count=1),
                work_rows=CentDramRowRange(start_row=4, row_count=1),
                input_buffer=full,
                scale_buffer=full,
                partial_sum_buffer=partial,
                # This target has one four-bank PU, so all 16 values occupy its
                # selected bank. Eight columns per row therefore require two
                # work rows.
                value_count=16,
            )

    def test_rms_norm_uses_each_explicit_buffer(self) -> None:
        """Connect input, scale, partial-sum, and output spans."""

        builder = make_builder()
        lower_rms_norm(
            builder,
            input_rows=CentDramRowRange(start_row=3, row_count=1),
            work_rows=CentDramRowRange(start_row=4, row_count=1),
            weight_rows=CentDramRowRange(start_row=5, row_count=1),
            input_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=4),
                slot_count=4,
            ),
            scale_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=12),
                slot_count=4,
            ),
            partial_sum_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=20),
                slot_count=1,
            ),
            output_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=24),
                slot_count=4,
            ),
            value_count=16,
        )

        transfers = [
            instruction
            for instruction in builder.instructions
            if isinstance(instruction, (WriteSingleBank, ReadSingleBank))
        ]
        # The input is split over banks 0 and 2, then copied to bank 1 for the
        # elementwise scale pass. The final bank-2 result is returned to the
        # output span beginning at slot 24.
        self.assertEqual(
            transfers,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=0, row=3, column=0
                    ),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=2, row=3, column=0
                    ),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=6),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=1, row=3, column=0
                    ),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=3, row=3, column=0
                    ),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=6),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=0, row=4, column=0
                    ),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=1, row=4, column=0
                    ),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=12),
                ),
                ReadSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=2, row=5, column=0
                    ),
                    operation_size=4,
                    destination=CentSharedBufferAddress(slot=24),
                ),
            ],
        )
        partial_sum_instructions = [
            instruction
            for instruction in builder.instructions
            if isinstance(instruction, (WriteBias, ReadMac))
        ]
        self.assertEqual(
            partial_sum_instructions,
            [
                WriteBias(
                    source=CentSharedBufferAddress(slot=20),
                    channels=CentChannelSet(channels=(0,)),
                ),
                ReadMac(
                    destination=CentSharedBufferAddress(slot=20),
                    accumulation_register=0,
                    channels=CentChannelSet(channels=(0,)),
                ),
            ],
        )
        copies = [
            instruction
            for instruction in builder.instructions
            if isinstance(
                instruction,
                (CopyBankToGlobalBuffer, CopyGlobalBufferToBank),
            )
        ]
        # AiM identifies one bank per copy. Bank two contains the first EW_MUL
        # result, and bank one receives it beside weights stored in bank zero.
        self.assertEqual(
            copies,
            [
                CopyBankToGlobalBuffer(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=4,
                    bank=2,
                    row=4,
                    column=0,
                ),
                CopyGlobalBufferToBank(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=4,
                    bank=1,
                    row=5,
                    column=0,
                ),
            ],
        )

    def test_rms_norm_rejects_empty_or_undersized_vectors(self) -> None:
        """Reject an empty vector and spans missing partition padding."""

        builder = make_builder()
        rows = CentDramRowRange(start_row=3, row_count=1)
        full = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=0),
            slot_count=4,
        )
        partial = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=8),
            slot_count=1,
        )

        with self.assertRaisesRegex(ValueError, "value_count"):
            lower_rms_norm(
                builder,
                input_rows=rows,
                work_rows=rows,
                weight_rows=rows,
                input_buffer=full,
                scale_buffer=full,
                partial_sum_buffer=partial,
                output_buffer=full,
                value_count=0,
            )
        with self.assertRaisesRegex(ValueError, "input_buffer"):
            lower_rms_norm(
                builder,
                input_rows=rows,
                work_rows=rows,
                weight_rows=rows,
                # Sixteen values need four four-value slots.
                input_buffer=replace(full, slot_count=3),
                scale_buffer=full,
                partial_sum_buffer=partial,
                output_buffer=full,
                value_count=16,
            )
        with self.assertRaisesRegex(ValueError, "input_rows needs 3"):
            lower_rms_norm(
                builder,
                input_rows=rows,
                work_rows=rows,
                weight_rows=rows,
                input_buffer=replace(full, slot_count=20),
                scale_buffer=replace(full, slot_count=20),
                partial_sum_buffer=partial,
                output_buffer=replace(full, slot_count=20),
                # Neighbor-bank normalization splits 80 values over two banks.
                # Each bank receives 40 values, which need three 16-value rows.
                value_count=80,
            )


class DataMovementLoweringTests(unittest.TestCase):
    """Test reusable packed-vector transfers."""

    def test_store_and_load_use_the_requested_bank_position(self) -> None:
        """Move the whole vector in each direction with explicit bindings."""

        builder = make_builder()
        rows = CentDramRowRange(start_row=9, row_count=1)
        buffer = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=6),
            slot_count=4,
        )

        lower_store_bank_group_vector(
            builder,
            rows=rows,
            buffer=buffer,
            value_count=16,
            bank_group=1,
        )
        lower_load_bank_group_vector(
            builder,
            rows=rows,
            buffer=buffer,
            value_count=16,
            bank_group=2,
        )

        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=1, row=9, column=0
                    ),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=6),
                ),
                ReadSingleBank(
                    address=CentMemoryAddress(
                        channel=0, bank=2, row=9, column=0
                    ),
                    operation_size=4,
                    destination=CentSharedBufferAddress(slot=6),
                ),
            ],
        )

    def test_private_transfer_rejects_empty_and_undersized_buffers(self) -> None:
        """Validate the common transfer path used by both directions."""

        builder = make_builder()
        rows = CentDramRowRange(start_row=9, row_count=1)
        buffer = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=0),
            slot_count=1,
        )

        with self.assertRaisesRegex(ValueError, "value_count"):
            _lower_bank_group_vector_transfer(
                builder,
                rows=rows,
                buffer=buffer,
                value_count=0,
                bank_group=0,
                instruction_type=WriteSingleBank,
            )
        with self.assertRaisesRegex(ValueError, "needs 4"):
            _lower_bank_group_vector_transfer(
                builder,
                rows=rows,
                buffer=buffer,
                value_count=16,
                bank_group=0,
                instruction_type=WriteSingleBank,
            )
        with self.assertRaisesRegex(ValueError, "rows needs 2"):
            _lower_bank_group_vector_transfer(
                builder,
                rows=rows,
                buffer=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=0),
                    slot_count=8,
                ),
                # One PU group receives all 32 values. Sixteen values fit in a
                # DRAM row, so the declared one-row range is insufficient.
                value_count=32,
                bank_group=0,
                instruction_type=WriteSingleBank,
            )


class ElementwiseLoweringTests(unittest.TestCase):
    """Test reusable elementwise lowering."""

    def test_accumulate_uses_explicit_vector_spans(self) -> None:
        """Add two vectors from the requested Shared Buffer addresses."""

        builder = make_builder()
        lower_accumulate(
            builder,
            destination=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=7),
                slot_count=4,
            ),
            source=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=20),
                slot_count=4,
            ),
            value_count=16,
        )

        self.assertEqual(
            builder.instructions,
            [
                Accumulate(
                    operation_size=4,
                    destination=CentSharedBufferAddress(slot=7),
                    source=CentSharedBufferAddress(slot=20),
                )
            ],
        )

    def test_accumulate_rejects_empty_and_undersized_vectors(self) -> None:
        """Reject vectors that cannot be represented by their spans."""

        builder = make_builder()
        span = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=0),
            slot_count=1,
        )
        with self.assertRaisesRegex(ValueError, "value_count"):
            lower_accumulate(
                builder,
                destination=span,
                source=span,
                value_count=0,
            )
        with self.assertRaisesRegex(ValueError, "destination"):
            lower_accumulate(
                builder,
                destination=span,
                source=span,
                value_count=8,
            )


class TransformerAttentionSpecTests(unittest.TestCase):
    """Test model-independent attention relationships."""

    def test_accepts_grouped_query_attention_dimensions(self) -> None:
        """Represent four query heads sharing two key/value heads."""

        self.assertEqual(
            TransformerAttentionSpec(
                hidden_size=64,
                num_attention_heads=4,
                num_kv_heads=2,
                head_size=16,
                kv_width=32,
                repeat_count=2,
                sequence_length=5,
                max_sequence_length=16,
            ).repeat_count,
            2,
        )

    def test_rejects_inconsistent_derived_dimensions(self) -> None:
        """Reject head relationships that cannot describe one attention op."""

        valid = TransformerAttentionSpec(
            hidden_size=64,
            num_attention_heads=4,
            num_kv_heads=2,
            head_size=16,
            kv_width=32,
            repeat_count=2,
            sequence_length=5,
            max_sequence_length=16,
        )
        with self.assertRaisesRegex(ValueError, "hidden_size"):
            replace(valid, hidden_size=63)
        with self.assertRaisesRegex(ValueError, "kv_width"):
            replace(valid, kv_width=31)
        with self.assertRaisesRegex(ValueError, "repeat_count"):
            replace(valid, repeat_count=3)
        with self.assertRaisesRegex(ValueError, "sequence_length"):
            replace(valid, sequence_length=17)


if __name__ == "__main__":
    unittest.main()
