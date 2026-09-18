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
    CentAccumulatePlan,
    CentBankGroupVectorTransferPlan,
    CentDramRowRange,
    CentL2NormPlan,
    CentPartitionedVectorLayout,
    CentRmsNormPlan,
    CentDramVector,
    CentSharedBufferVector,
    CentSharedBufferSpan,
    CentSumOfSquaresPlan,
    CentWeightGemvPlan,
    lower_accumulate,
    lower_load_bank_group_vector,
    lower_l2_norm,
    lower_rms_norm,
    lower_sum_of_squares,
    lower_store_bank_group_vector,
    lower_weight_gemv,
    plan_partitioned_vector,
    plan_weight_gemv,
    pack_zero_padded_vector,
)
from vllm_cent.lowering.data_movement import (
    _lower_bank_group_vector_transfer,
)
from vllm_cent.lowering.transformer import TransformerAttentionSpec
from vllm_cent.lowering.utils import (
    _row_operation_sizes,
    _require_dram_row_capacity,
    _require_shared_buffer_capacity,
)


def make_builder(
    *,
    dram_columns: int = 16,
    burst_length: int = 4,
    accumulator_slots: int = 4,
    num_banks: int = 4,
) -> CentProgramBuilder:
    """Create the small CENT target used by lowering tests.

    Args:
        dram_columns: Scalar values stored in one DRAM row.
        burst_length: Scalar values moved by one Shared Buffer slot.
        accumulator_slots: MAC result registers available to each bank.
        num_banks: DRAM banks available in the test channel.

    Returns:
        Empty builder for one channel containing four banks.
    """

    hardware = CentHardwareSpec(
        num_channels=1,
        num_banks=num_banks,
        dram_rows=128,
        dram_columns=dram_columns,
        global_buffer_columns=dram_columns,
        burst_length=burst_length,
        accumulator_slots_per_bank=accumulator_slots,
        sigmoid_activation_function_id=3,
        shared_buffer_slots=64,
    )
    return CentProgramBuilder(
        hardware,
        CentBlockPlacementSpec(channels_per_block=1),
    )


def make_buffer_vector(
    *,
    start_slot: int,
    layout: CentPartitionedVectorLayout,
) -> CentSharedBufferVector:
    """Bind a test vector to the exact slots required by its layout.

    Args:
        start_slot: First Shared Buffer slot occupied by the vector.
        layout: Logical partitioning and zero-padding positions.

    Returns:
        Zero-padded vector binding used by a lowering test.
    """

    return CentSharedBufferVector(
        span=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=start_slot),
            slot_count=layout.slot_count,
        ),
        layout=layout,
    )


def make_gemv_plan(
    builder: CentProgramBuilder,
    *,
    weights: CentDramRowRange,
    input_buffer: CentSharedBufferSpan,
    output_buffer: CentSharedBufferSpan,
    vector_size: int,
    output_size: int,
    activated_output_buffer: CentSharedBufferSpan | None = None,
) -> CentWeightGemvPlan:
    """Create the baseline GEMV plan used by lowering tests.

    Args:
        builder: Builder whose hardware and placement constrain the plan.
        weights: DRAM rows containing the distributed weight matrix.
        input_buffer: Shared Buffer slots containing the input vector.
        output_buffer: Slots receiving raw GEMV results.
        vector_size: Values in the input vector.
        output_size: Values in the output vector.
        activated_output_buffer: Separate slots receiving activated results.

    Returns:
        Explicit GEMV plan matching the repository's baseline policy.
    """

    return plan_weight_gemv(
        builder.hardware,
        builder.placement,
        weights=weights,
        input_buffer=CentSharedBufferVector(
            span=input_buffer,
            layout=plan_partitioned_vector(
                vector_size,
                1,
                builder.hardware.burst_length,
            ),
        ),
        output_buffer=output_buffer,
        vector_size=vector_size,
        output_size=output_size,
        activated_output_buffer=activated_output_buffer,
    )


def make_sum_of_squares_plan(
    builder: CentProgramBuilder,
    *,
    input_rows: CentDramRowRange,
    input_buffer: CentSharedBufferSpan,
    partial_sum_buffer: CentSharedBufferSpan,
    value_count: int,
) -> CentSumOfSquaresPlan:
    """Create the baseline neighboring-bank sum-of-squares plan.

    Args:
        builder: Builder whose target determines the available bank pairs.
        input_rows: DRAM workspace used for both input copies.
        input_buffer: Slots containing the vector partitions.
        partial_sum_buffer: Slot receiving partial sums.
        value_count: Values in the vector.

    Returns:
        Sum-of-squares plan using every useful neighboring bank pair.
    """

    layout = plan_partitioned_vector(
        value_count,
        builder.total_banks // 2,
        builder.hardware.burst_length,
    )
    return CentSumOfSquaresPlan(
        input_rows=input_rows,
        input_buffer=CentSharedBufferVector(
            span=input_buffer,
            layout=layout,
        ),
        partial_sum_buffer=partial_sum_buffer,
        layout=layout,
    )


def make_l2_norm_plan(
    builder: CentProgramBuilder,
    *,
    input_rows: CentDramRowRange,
    work_rows: CentDramRowRange,
    input_buffer: CentSharedBufferSpan,
    scale_buffer: CentSharedBufferSpan,
    partial_sum_buffer: CentSharedBufferSpan,
    value_count: int,
) -> CentL2NormPlan:
    """Create the baseline L2-normalization plan used by tests.

    Args:
        builder: Builder whose target determines available PU groups.
        input_rows: DRAM workspace used by sum of squares.
        work_rows: DRAM workspace used by scale multiplication.
        input_buffer: Slots containing the input vector.
        scale_buffer: Slots containing the repeated scale.
        partial_sum_buffer: Slot receiving partial sums.
        value_count: Values in the vector.

    Returns:
        L2-normalization plan with separate pair and PU-group layouts.
    """

    layout = plan_partitioned_vector(
        value_count,
        builder.total_banks // 4,
        builder.hardware.burst_length,
    )
    return CentL2NormPlan(
        sum_of_squares=make_sum_of_squares_plan(
            builder,
            input_rows=input_rows,
            input_buffer=input_buffer,
            partial_sum_buffer=partial_sum_buffer,
            value_count=value_count,
        ),
        work_rows=work_rows,
        scale_buffer=CentSharedBufferVector(
            span=scale_buffer,
            layout=layout,
        ),
        layout=layout,
    )


def make_rms_norm_plan(
    builder: CentProgramBuilder,
    *,
    input_rows: CentDramRowRange,
    work_rows: CentDramRowRange,
    weight_rows: CentDramRowRange,
    input_buffer: CentSharedBufferSpan,
    scale_buffer: CentSharedBufferSpan,
    partial_sum_buffer: CentSharedBufferSpan,
    output_buffer: CentSharedBufferSpan,
    value_count: int,
) -> CentRmsNormPlan:
    """Create the baseline RMS-normalization plan used by tests.

    Args:
        builder: Builder whose target determines available bank groups.
        input_rows: DRAM workspace used by sum of squares.
        work_rows: DRAM workspace used by scale multiplication.
        weight_rows: DRAM rows containing learned RMS weights.
        input_buffer: Slots containing the input vector.
        scale_buffer: Slots containing the repeated RMS scale.
        partial_sum_buffer: Slot receiving partial sums.
        output_buffer: Slots receiving the normalized vector.
        value_count: Values in the vector.

    Returns:
        Complete RMS-normalization operation plan.
    """

    l2_norm = make_l2_norm_plan(
        builder,
        input_rows=input_rows,
        work_rows=work_rows,
        input_buffer=input_buffer,
        scale_buffer=scale_buffer,
        partial_sum_buffer=partial_sum_buffer,
        value_count=value_count,
    )
    return CentRmsNormPlan(
        l2_norm=l2_norm,
        weight_rows=weight_rows,
        output_buffer=CentSharedBufferVector(
            span=output_buffer,
            layout=l2_norm.layout,
        ),
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

    def test_row_operation_sizes_split_only_at_row_boundaries(self) -> None:
        """Represent exact, partial, and multirow operation counts."""

        self.assertEqual(_row_operation_sizes(16, 16, 4), (4,))
        self.assertEqual(_row_operation_sizes(20, 16, 4), (4, 1))
        for values, width, burst in ((0, 16, 4), (4, 0, 4), (4, 10, 4)):
            with self.subTest(values=values, width=width, burst=burst):
                with self.assertRaises(ValueError):
                    _row_operation_sizes(values, width, burst)


class PartitionedVectorLayoutTests(unittest.TestCase):
    """Test layouts selected before instruction lowering begins."""

    def test_baseline_planner_preserves_the_maximum_partition_policy(
        self,
    ) -> None:
        """Use the old even split while making its choice explicit."""

        # Ten values over at most four partitions reserve three values in each
        # partition. Four-value bursts make every partition occupy one slot.
        self.assertEqual(
            plan_partitioned_vector(10, 4, 4),
            CentPartitionedVectorLayout(
                value_count=10,
                partition_count=4,
                burst_length=4,
            ),
        )

    def test_explicit_layout_can_select_a_different_partition_count(
        self,
    ) -> None:
        """Represent a non-baseline split without asking a lowerer to choose."""

        layout = CentPartitionedVectorLayout(
            value_count=10,
            partition_count=6,
            burst_length=4,
        )

        # Six selected partitions balance ten logical values as 2,2,2,2,1,1.
        # Every partition still occupies one four-value slot. The ten values
        # therefore use 24 physical lanes, and the other 14 lanes must be zero.
        self.assertEqual(
            [
                layout.logical_values_in_partition(partition)
                for partition in range(layout.partition_count)
            ],
            [2, 2, 2, 2, 1, 1],
        )
        self.assertEqual(layout.values_per_partition, 2)
        self.assertEqual(layout.slots_per_partition, 1)
        self.assertEqual(layout.slot_count, 6)
        self.assertEqual(layout.physical_values_per_partition, 4)
        self.assertEqual(layout.physical_value_count, 24)
        self.assertEqual(layout.padding_value_count, 14)

    def test_zero_padded_packer_overwrites_every_physical_lane(self) -> None:
        """Place values in partitions and explicitly zero every unused lane."""

        layout = CentPartitionedVectorLayout(
            value_count=10,
            partition_count=4,
            burst_length=4,
        )

        # Logical partition lengths are 3, 3, 2, and 2. Each partition owns a
        # complete four-lane slot, so zeros appear inside the physical layout,
        # not only once at the end of the complete vector.
        self.assertEqual(
            pack_zero_padded_vector(tuple(range(10)), layout, zero=0),
            (
                0,
                1,
                2,
                0,
                3,
                4,
                5,
                0,
                6,
                7,
                0,
                0,
                8,
                9,
                0,
                0,
            ),
        )

    def test_zero_padded_packer_requires_the_logical_value_count(self) -> None:
        """Reject input that does not match the layout's logical vector size."""

        layout = plan_partitioned_vector(5, 1, 4)

        with self.assertRaisesRegex(ValueError, "received 4"):
            pack_zero_padded_vector((1, 2, 3, 4), layout, zero=0)

    def test_layout_identifies_internal_partition_padding(self) -> None:
        """Distinguish trailing padding from gaps inside a packed vector."""

        exact = CentPartitionedVectorLayout(
            value_count=8,
            partition_count=2,
            burst_length=4,
        )
        self.assertTrue(exact.is_contiguously_packed)
        self.assertEqual(exact.physical_value_count, 8)
        self.assertEqual(exact.padding_value_count, 0)

        self.assertFalse(
            CentPartitionedVectorLayout(
                value_count=6,
                partition_count=2,
                burst_length=4,
            ).is_contiguously_packed
        )

        trailing_padding = CentPartitionedVectorLayout(
            value_count=5,
            partition_count=1,
            burst_length=4,
        )
        self.assertTrue(trailing_padding.is_contiguously_packed)
        self.assertEqual(trailing_padding.physical_value_count, 8)
        self.assertEqual(trailing_padding.padding_value_count, 3)

    def test_layout_rejects_invalid_sizes_and_partition_indices(self) -> None:
        """Reject empty layouts, empty partitions, and invalid lookups."""

        for arguments in (
            {"value_count": 0, "partition_count": 1, "burst_length": 4},
            {"value_count": 4, "partition_count": 0, "burst_length": 4},
            {"value_count": 4, "partition_count": 5, "burst_length": 4},
            {"value_count": 4, "partition_count": 1, "burst_length": 0},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    CentPartitionedVectorLayout(**arguments)

        layout = CentPartitionedVectorLayout(
            value_count=4,
            partition_count=2,
            burst_length=4,
        )
        with self.assertRaises(ValueError):
            layout.logical_values_in_partition(-1)
        with self.assertRaises(ValueError):
            layout.logical_values_in_partition(2)


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

    def test_vector_bindings_require_exact_physical_storage(self) -> None:
        """Bind one zero-padded layout to exact Shared Buffer and DRAM regions."""

        layout = CentPartitionedVectorLayout(
            value_count=10,
            partition_count=4,
            burst_length=4,
        )
        span = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=7),
            # Four partitions each occupy one complete slot.
            slot_count=4,
        )

        self.assertEqual(
            CentSharedBufferVector(span=span, layout=layout).address(3),
            CentSharedBufferAddress(slot=10),
        )
        self.assertEqual(
            CentDramVector(
                rows=CentDramRowRange(start_row=11, row_count=1),
                layout=layout,
                bank_group=2,
            ).layout,
            layout,
        )

        for slot_count in (3, 5):
            with self.subTest(slot_count=slot_count):
                with self.assertRaisesRegex(ValueError, "exactly 4"):
                    CentSharedBufferVector(
                        span=CentSharedBufferSpan(
                            start=CentSharedBufferAddress(slot=0),
                            slot_count=slot_count,
                        ),
                        layout=layout,
                    )

    def test_dram_vector_rejects_an_invalid_bank_group(self) -> None:
        """Keep logical vector bindings inside one four-bank PU group."""

        with self.assertRaisesRegex(ValueError, "bank_group"):
            CentDramVector(
                rows=CentDramRowRange(start_row=0, row_count=1),
                layout=plan_partitioned_vector(4, 1, 4),
                bank_group=4,
            )

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
            make_gemv_plan(
                builder,
                weights=CentDramRowRange(start_row=7, row_count=2),
                input_buffer=input_buffer,
                output_buffer=output_buffer,
                vector_size=16,
                output_size=8,
            ),
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
            make_gemv_plan(
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
                # Sixteen outputs over four banks require four result slots.
                # Two registers process groups [10, 11] and [12, 13].
                output_size=16,
            ),
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

    def test_gemv_obeys_a_plan_that_uses_fewer_banks(self) -> None:
        """Assign more outputs per bank without changing GEMV lowering."""

        builder = make_builder(accumulator_slots=2)
        channels = CentChannelSet(channels=(0,))
        input_layout = plan_partitioned_vector(16, 1, 4)
        plan = CentWeightGemvPlan(
            weights=CentDramRowRange(start_row=10, row_count=4),
            input_buffer=make_buffer_vector(
                start_slot=1,
                layout=input_layout,
            ),
            output_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=20),
                slot_count=4,
            ),
            vector_size=16,
            output_size=8,
            # The baseline uses all four banks and two outputs per bank. This
            # plan uses two banks and assigns four outputs to each one.
            outputs_per_bank=4,
            accumulator_group_size=2,
            channels=channels,
        )

        lower_weight_gemv(builder, plan)

        self.assertEqual(
            builder.instructions,
            [
                WriteGlobalBuffer(
                    operation_size=4,
                    column=0,
                    source=CentSharedBufferAddress(slot=1),
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
                    row=10,
                    column=0,
                    accumulation_register=0,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                ),
                MacAllBanks(
                    operation_size=4,
                    channels=channels,
                    row=11,
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
                WriteBias(
                    source=CentSharedBufferAddress(slot=22),
                    channels=channels,
                ),
                WriteBias(
                    source=CentSharedBufferAddress(slot=23),
                    channels=channels,
                ),
                MacAllBanks(
                    operation_size=4,
                    channels=channels,
                    row=12,
                    column=0,
                    accumulation_register=0,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                ),
                MacAllBanks(
                    operation_size=4,
                    channels=channels,
                    row=13,
                    column=0,
                    accumulation_register=1,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                ),
                ReadMac(
                    destination=CentSharedBufferAddress(slot=22),
                    accumulation_register=0,
                    channels=channels,
                ),
                ReadMac(
                    destination=CentSharedBufferAddress(slot=23),
                    accumulation_register=1,
                    channels=channels,
                ),
            ],
        )

    def test_gemv_reads_each_input_row_from_the_next_slots(self) -> None:
        """Advance the input span when a vector crosses a DRAM row."""

        builder = make_builder()

        lower_weight_gemv(
            builder,
            make_gemv_plan(
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
            ),
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
            make_gemv_plan(
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
            ),
        )

        result_instructions = [
            instruction
            for instruction in builder.instructions
            if isinstance(instruction, (ReadMac, ReadActivation, ApplyActivation))
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

        with self.assertRaisesRegex(ValueError, "exactly 4"):
            lower_weight_gemv(
                builder,
                make_gemv_plan(
                    builder,
                    weights=weights,
                    input_buffer=input_buffer,
                    output_buffer=output_buffer,
                    vector_size=16,
                    output_size=8,
                ),
            )
        with self.assertRaisesRegex(ValueError, "weights"):
            lower_weight_gemv(
                builder,
                make_gemv_plan(
                    builder,
                    weights=CentDramRowRange(start_row=0, row_count=1),
                    input_buffer=CentSharedBufferSpan(
                        start=CentSharedBufferAddress(slot=0),
                        slot_count=4,
                    ),
                    output_buffer=output_buffer,
                    vector_size=16,
                    output_size=8,
                ),
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            make_gemv_plan(
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

    def test_gemv_rejects_invalid_explicit_plan_geometry(self) -> None:
        """Reject bank, channel, register, and placement incompatibilities."""

        builder = make_builder(accumulator_slots=2)
        input_buffer = make_buffer_vector(
            start_slot=0,
            layout=plan_partitioned_vector(16, 1, 4),
        )
        output_buffer = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=16),
            slot_count=8,
        )
        base = CentWeightGemvPlan(
            weights=CentDramRowRange(start_row=0, row_count=8),
            input_buffer=input_buffer,
            output_buffer=output_buffer,
            vector_size=16,
            output_size=8,
            outputs_per_bank=2,
            accumulator_group_size=2,
            channels=CentChannelSet(channels=(0,)),
        )

        with self.assertRaisesRegex(ValueError, "match vector_size"):
            replace(
                base,
                input_buffer=make_buffer_vector(
                    start_slot=0,
                    layout=plan_partitioned_vector(8, 1, 4),
                ),
            )
        with self.assertRaisesRegex(ValueError, "contiguously packed"):
            noncontiguous_layout = CentPartitionedVectorLayout(
                value_count=6,
                partition_count=2,
                burst_length=4,
            )
            replace(
                base,
                vector_size=6,
                input_buffer=make_buffer_vector(
                    start_slot=0,
                    layout=noncontiguous_layout,
                ),
            )
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            replace(base, accumulator_group_size=3)
        with self.assertRaisesRegex(ValueError, "more banks"):
            lower_weight_gemv(
                builder,
                replace(
                    base,
                    output_size=20,
                    outputs_per_bank=1,
                    accumulator_group_size=1,
                ),
            )
        with self.assertRaisesRegex(ValueError, "outside the device"):
            lower_weight_gemv(
                builder,
                replace(base, channels=CentChannelSet(channels=(1,))),
            )
        with self.assertRaisesRegex(ValueError, "target capacity"):
            lower_weight_gemv(
                builder,
                replace(
                    base,
                    output_size=16,
                    outputs_per_bank=4,
                    accumulator_group_size=3,
                ),
            )

        two_channel_builder = CentProgramBuilder(
            replace(builder.hardware, num_channels=2),
            CentBlockPlacementSpec(channels_per_block=2),
        )
        with self.assertRaisesRegex(ValueError, "selected channels"):
            lower_weight_gemv(
                two_channel_builder,
                replace(
                    base,
                    output_size=8,
                    outputs_per_bank=1,
                    accumulator_group_size=1,
                ),
            )

        activated = replace(
            base,
            activated_output_buffer=CentSharedBufferSpan(
                start=CentSharedBufferAddress(slot=32),
                slot_count=8,
            ),
        )
        one_register_builder = make_builder(accumulator_slots=1)
        with self.assertRaisesRegex(ValueError, "at least two"):
            lower_weight_gemv(one_register_builder, activated)
        with self.assertRaisesRegex(ValueError, "at least two"):
            plan_weight_gemv(
                one_register_builder.hardware,
                one_register_builder.placement,
                weights=base.weights,
                input_buffer=input_buffer,
                output_buffer=output_buffer,
                activated_output_buffer=activated.activated_output_buffer,
                vector_size=16,
                output_size=8,
            )

        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            plan_weight_gemv(
                builder.hardware,
                CentBlockPlacementSpec(channels_per_block=2),
                weights=base.weights,
                input_buffer=input_buffer,
                output_buffer=output_buffer,
                vector_size=16,
                output_size=8,
            )
        three_channel_hardware = replace(builder.hardware, num_channels=3)
        with self.assertRaisesRegex(ValueError, "must divide"):
            plan_weight_gemv(
                three_channel_hardware,
                CentBlockPlacementSpec(channels_per_block=2),
                weights=base.weights,
                input_buffer=input_buffer,
                output_buffer=output_buffer,
                vector_size=16,
                output_size=8,
            )


class NormalizationLoweringTests(unittest.TestCase):
    """Test reusable vector-normalization lowering."""

    def test_sum_of_squares_emits_neighbor_bank_mac(self) -> None:
        """Square vector partitions and read their partial sums."""

        builder = make_builder()
        plan = make_sum_of_squares_plan(
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
        lower_sum_of_squares(
            builder,
            # Register one proves this is a plan choice rather than a hidden
            # normalization or instruction-class constant.
            replace(plan, accumulation_register=1),
        )

        # Four banks split the vector into two eight-value partitions. Each
        # partition is copied into a neighboring bank pair so MAC_ABK squares
        # matching values and accumulates one partial sum per active PU.
        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=0, row=3, column=0),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=2, row=3, column=0),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=6),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=1, row=3, column=0),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=3, row=3, column=0),
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
                    accumulation_register=1,
                    operand_source=MacOperandSource.NEXT_BANK,
                ),
                ReadMac(
                    destination=CentSharedBufferAddress(slot=20),
                    accumulation_register=1,
                    channels=CentChannelSet(channels=(0,)),
                ),
            ],
        )
        with self.assertRaisesRegex(ValueError, "accumulation_register"):
            replace(plan, accumulation_register=-1)

    def test_sum_of_squares_splits_a_partition_across_dram_rows(self) -> None:
        """Keep transfers and MAC operations inside individual DRAM rows."""

        builder = make_builder()
        layout = CentPartitionedVectorLayout(
            value_count=20,
            partition_count=1,
            burst_length=4,
        )
        lower_sum_of_squares(
            builder,
            CentSumOfSquaresPlan(
                input_rows=CentDramRowRange(start_row=3, row_count=2),
                input_buffer=make_buffer_vector(
                    start_slot=4,
                    layout=layout,
                ),
                partial_sum_buffer=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=20),
                    slot_count=1,
                ),
                layout=layout,
            ),
        )

        # Twenty values require four bursts in the first row and one in the
        # second. Both banks receive that same five-slot vector before the two
        # row-local MAC instructions square it.
        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=0, row=3, column=0),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=0, row=4, column=0),
                    operation_size=1,
                    source=CentSharedBufferAddress(slot=8),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=1, row=3, column=0),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=1, row=4, column=0),
                    operation_size=1,
                    source=CentSharedBufferAddress(slot=8),
                ),
                WriteBias(
                    source=CentSharedBufferAddress(slot=20),
                    channels=CentChannelSet(channels=(0,)),
                ),
                MacAllBanks(
                    operation_size=4,
                    channels=CentChannelSet(channels=(0,)),
                    row=3,
                    column=0,
                    accumulation_register=0,
                    operand_source=MacOperandSource.NEXT_BANK,
                ),
                MacAllBanks(
                    operation_size=1,
                    channels=CentChannelSet(channels=(0,)),
                    row=4,
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
            make_l2_norm_plan(
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
            ),
        )

        # lower_sum_of_squares emits the first seven instructions tested
        # above. L2 normalization then writes all 16 input values to bank zero,
        # writes the four-slot repeated scale to bank one, and multiplies both
        # banks into bank two.
        self.assertEqual(
            builder.instructions[7:],
            [
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=0, row=4, column=0),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=1, row=4, column=0),
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

    def test_l2_norm_obeys_fewer_planned_bank_partitions(self) -> None:
        """Use one bank pair and one PU group on a larger target."""

        builder = make_builder(num_banks=8)
        vector = CentPartitionedVectorLayout(
            value_count=16,
            partition_count=1,
            burst_length=4,
        )
        plan = CentL2NormPlan(
            sum_of_squares=CentSumOfSquaresPlan(
                input_rows=CentDramRowRange(start_row=3, row_count=1),
                input_buffer=make_buffer_vector(
                    start_slot=4,
                    layout=vector,
                ),
                partial_sum_buffer=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=20),
                    slot_count=1,
                ),
                # Eight banks provide four neighboring pairs, but this plan
                # deliberately assigns the complete vector to only one pair.
                layout=vector,
            ),
            work_rows=CentDramRowRange(start_row=4, row_count=1),
            scale_buffer=make_buffer_vector(
                start_slot=12,
                layout=vector,
            ),
            # Eight banks also provide two PU groups. Selecting one proves the
            # lowerer does not silently restore the baseline two-way split.
            layout=vector,
        )

        lower_l2_norm(builder, plan)

        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=0, row=3, column=0),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=1, row=3, column=0),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteBias(
                    source=CentSharedBufferAddress(slot=20),
                    channels=CentChannelSet(channels=(0,)),
                ),
                MacAllBanks(
                    operation_size=4,
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
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=0, row=4, column=0),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=1, row=4, column=0),
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

        with self.assertRaisesRegex(ValueError, "exactly 4"):
            builder = make_builder()
            lower_l2_norm(
                builder,
                make_l2_norm_plan(
                    builder,
                    input_rows=CentDramRowRange(start_row=3, row_count=1),
                    work_rows=CentDramRowRange(start_row=4, row_count=1),
                    input_buffer=full,
                    # Sixteen values occupy four Shared Buffer slots.
                    scale_buffer=replace(full, slot_count=3),
                    partial_sum_buffer=partial,
                    value_count=16,
                ),
            )

        with self.assertRaisesRegex(ValueError, "work_rows needs 2"):
            builder = make_builder(dram_columns=8)
            lower_l2_norm(
                builder,
                make_l2_norm_plan(
                    builder,
                    input_rows=CentDramRowRange(start_row=3, row_count=1),
                    work_rows=CentDramRowRange(start_row=4, row_count=1),
                    input_buffer=full,
                    scale_buffer=full,
                    partial_sum_buffer=partial,
                    # This target has one four-bank PU, so all 16 values occupy
                    # its selected bank. Two DRAM rows are required.
                    value_count=16,
                ),
            )

    def test_rms_norm_uses_each_explicit_buffer(self) -> None:
        """Connect input, scale, partial-sum, and output spans."""

        builder = make_builder()
        lower_rms_norm(
            builder,
            make_rms_norm_plan(
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
            ),
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
                    address=CentMemoryAddress(channel=0, bank=0, row=3, column=0),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=2, row=3, column=0),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=6),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=1, row=3, column=0),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=3, row=3, column=0),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=6),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=0, row=4, column=0),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=4),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=1, row=4, column=0),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=12),
                ),
                ReadSingleBank(
                    address=CentMemoryAddress(channel=0, bank=2, row=5, column=0),
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
            make_rms_norm_plan(
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
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            lower_rms_norm(
                builder,
                make_rms_norm_plan(
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
                ),
            )
        with self.assertRaisesRegex(ValueError, "input_rows needs 3"):
            lower_rms_norm(
                builder,
                make_rms_norm_plan(
                    builder,
                    input_rows=rows,
                    work_rows=CentDramRowRange(start_row=8, row_count=5),
                    weight_rows=CentDramRowRange(start_row=16, row_count=5),
                    input_buffer=replace(full, slot_count=20),
                    scale_buffer=replace(full, slot_count=20),
                    partial_sum_buffer=partial,
                    output_buffer=replace(full, slot_count=20),
                    # Two bank pairs receive 40 values each. Each partition
                    # therefore needs three sixteen-value DRAM rows.
                    value_count=80,
                ),
            )

    def test_normalization_rejects_incompatible_explicit_layouts(self) -> None:
        """Reject mismatched vectors, bursts, pairs, and PU groups."""

        builder = make_builder()
        rows = CentDramRowRange(start_row=0, row_count=8)
        buffer = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=0),
            slot_count=4,
        )
        partial = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=40),
            slot_count=1,
        )
        sum_plan = make_sum_of_squares_plan(
            builder,
            input_rows=rows,
            input_buffer=buffer,
            partial_sum_buffer=partial,
            value_count=16,
        )

        with self.assertRaisesRegex(ValueError, "input_buffer layout"):
            replace(
                sum_plan,
                input_buffer=make_buffer_vector(
                    start_slot=0,
                    layout=CentPartitionedVectorLayout(
                        value_count=16,
                        partition_count=1,
                        burst_length=4,
                    ),
                ),
            )

        with self.assertRaisesRegex(ValueError, "same value_count"):
            smaller_layout = CentPartitionedVectorLayout(
                value_count=8,
                partition_count=1,
                burst_length=4,
            )
            CentL2NormPlan(
                sum_of_squares=sum_plan,
                work_rows=rows,
                scale_buffer=make_buffer_vector(
                    start_slot=0,
                    layout=smaller_layout,
                ),
                layout=smaller_layout,
            )
        with self.assertRaisesRegex(ValueError, "repacking"):
            partitioned_layout = CentPartitionedVectorLayout(
                value_count=6,
                partition_count=2,
                burst_length=4,
            )
            contiguous_layout = CentPartitionedVectorLayout(
                value_count=6,
                partition_count=1,
                burst_length=4,
            )
            CentL2NormPlan(
                sum_of_squares=replace(
                    sum_plan,
                    input_buffer=make_buffer_vector(
                        start_slot=0,
                        layout=partitioned_layout,
                    ),
                    layout=partitioned_layout,
                ),
                work_rows=rows,
                scale_buffer=make_buffer_vector(
                    start_slot=2,
                    layout=contiguous_layout,
                ),
                layout=contiguous_layout,
            )
        with self.assertRaisesRegex(ValueError, "burst_length"):
            wrong_sum_burst_layout = CentPartitionedVectorLayout(
                value_count=16,
                partition_count=2,
                burst_length=8,
            )
            lower_sum_of_squares(
                builder,
                replace(
                    sum_plan,
                    input_buffer=make_buffer_vector(
                        start_slot=0,
                        layout=wrong_sum_burst_layout,
                    ),
                    layout=wrong_sum_burst_layout,
                ),
            )
        with self.assertRaisesRegex(ValueError, "more bank pairs"):
            too_many_pairs_layout = CentPartitionedVectorLayout(
                value_count=16,
                partition_count=3,
                burst_length=4,
            )
            lower_sum_of_squares(
                builder,
                replace(
                    sum_plan,
                    input_buffer=make_buffer_vector(
                        start_slot=0,
                        layout=too_many_pairs_layout,
                    ),
                    layout=too_many_pairs_layout,
                ),
            )

        l2_plan = make_l2_norm_plan(
            builder,
            input_rows=rows,
            work_rows=rows,
            input_buffer=buffer,
            scale_buffer=buffer,
            partial_sum_buffer=partial,
            value_count=16,
        )
        with self.assertRaisesRegex(ValueError, "scale_buffer layout"):
            replace(
                l2_plan,
                scale_buffer=make_buffer_vector(
                    start_slot=4,
                    layout=CentPartitionedVectorLayout(
                        value_count=16,
                        partition_count=2,
                        burst_length=4,
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "output_buffer layout"):
            CentRmsNormPlan(
                l2_norm=l2_plan,
                weight_rows=rows,
                output_buffer=make_buffer_vector(
                    start_slot=8,
                    layout=CentPartitionedVectorLayout(
                        value_count=16,
                        partition_count=2,
                        burst_length=4,
                    ),
                ),
            )
        wrong_burst_layout = CentPartitionedVectorLayout(
            value_count=16,
            partition_count=1,
            burst_length=8,
        )
        with self.assertRaisesRegex(ValueError, "same burst_length"):
            replace(l2_plan, layout=wrong_burst_layout)
        with self.assertRaisesRegex(ValueError, "burst_length"):
            lower_l2_norm(
                builder,
                CentL2NormPlan(
                    sum_of_squares=replace(
                        l2_plan.sum_of_squares,
                        input_buffer=make_buffer_vector(
                            start_slot=0,
                            layout=wrong_burst_layout,
                        ),
                        layout=wrong_burst_layout,
                    ),
                    work_rows=l2_plan.work_rows,
                    scale_buffer=make_buffer_vector(
                        start_slot=4,
                        layout=wrong_burst_layout,
                    ),
                    layout=wrong_burst_layout,
                ),
            )
        with self.assertRaisesRegex(ValueError, "more PU groups"):
            lower_l2_norm(
                builder,
                replace(
                    l2_plan,
                    scale_buffer=make_buffer_vector(
                        start_slot=4,
                        layout=CentPartitionedVectorLayout(
                            value_count=16,
                            partition_count=2,
                            burst_length=4,
                        ),
                    ),
                    layout=CentPartitionedVectorLayout(
                        value_count=16,
                        partition_count=2,
                        burst_length=4,
                    ),
                ),
            )


class DataMovementLoweringTests(unittest.TestCase):
    """Test reusable packed-vector transfers."""

    def test_store_and_load_use_the_requested_bank_position(self) -> None:
        """Move the whole vector in each direction with explicit bindings."""

        builder = make_builder()
        rows = CentDramRowRange(start_row=9, row_count=1)
        layout = plan_partitioned_vector(16, 1, 4)
        buffer = make_buffer_vector(start_slot=6, layout=layout)

        lower_store_bank_group_vector(
            builder,
            CentBankGroupVectorTransferPlan(
                dram=CentDramVector(rows=rows, layout=layout, bank_group=1),
                buffer=buffer,
            ),
        )
        lower_load_bank_group_vector(
            builder,
            CentBankGroupVectorTransferPlan(
                dram=CentDramVector(rows=rows, layout=layout, bank_group=2),
                buffer=buffer,
            ),
        )

        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=1, row=9, column=0),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=6),
                ),
                ReadSingleBank(
                    address=CentMemoryAddress(channel=0, bank=2, row=9, column=0),
                    operation_size=4,
                    destination=CentSharedBufferAddress(slot=6),
                ),
            ],
        )

    def test_transfer_copies_every_slot_of_each_padded_partition(self) -> None:
        """Write all physical lanes even when partitions end in padding."""

        builder = make_builder(num_banks=16)
        layout = CentPartitionedVectorLayout(
            value_count=10,
            partition_count=4,
            burst_length=4,
        )
        vector = make_buffer_vector(start_slot=5, layout=layout)

        lower_store_bank_group_vector(
            builder,
            CentBankGroupVectorTransferPlan(
                dram=CentDramVector(
                    rows=CentDramRowRange(start_row=9, row_count=1),
                    layout=layout,
                ),
                buffer=vector,
            ),
        )

        # The logical partition lengths are 3, 3, 2, and 2, but each physical
        # partition owns one complete slot. The four writes therefore consume
        # slots 5 through 8 and overwrite all four lanes in every destination.
        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=0, row=9, column=0),
                    operation_size=1,
                    source=CentSharedBufferAddress(slot=5),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=4, row=9, column=0),
                    operation_size=1,
                    source=CentSharedBufferAddress(slot=6),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=8, row=9, column=0),
                    operation_size=1,
                    source=CentSharedBufferAddress(slot=7),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=12, row=9, column=0),
                    operation_size=1,
                    source=CentSharedBufferAddress(slot=8),
                ),
            ],
        )

    def test_vector_transfer_rejects_empty_and_undersized_vectors(self) -> None:
        """Reject an empty layout or a span missing occupied vector slots."""

        with self.assertRaisesRegex(ValueError, "value_count"):
            plan_partitioned_vector(0, 1, 4)

        # A vector binding rejects an undersized span before a transfer can use
        # it, so stale or missing slots cannot be hidden behind raw capacity.
        with self.assertRaisesRegex(ValueError, "exactly 4"):
            CentSharedBufferVector(
                span=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=0),
                    slot_count=1,
                ),
                layout=plan_partitioned_vector(16, 1, 4),
            )

    def test_transfer_obeys_a_supplied_lower_partition_layout(self) -> None:
        """Use one PU group even when the target provides two groups."""

        builder = make_builder(num_banks=8)
        # The baseline planner would use both PU groups. Selecting one here
        # proves that lowering consumes the plan instead of recomputing it.
        layout = CentPartitionedVectorLayout(
            value_count=16,
            partition_count=1,
            burst_length=4,
        )
        plan = CentBankGroupVectorTransferPlan(
            dram=CentDramVector(
                rows=CentDramRowRange(start_row=9, row_count=1),
                layout=layout,
                bank_group=3,
            ),
            buffer=make_buffer_vector(start_slot=2, layout=layout),
        )

        lower_store_bank_group_vector(builder, plan)

        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=3,
                        row=9,
                        column=0,
                    ),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=2),
                )
            ],
        )

    def test_transfer_rejects_incompatible_explicit_layouts(self) -> None:
        """Reject a wrong burst width, unavailable groups, or invalid bank."""

        builder = make_builder()
        rows = CentDramRowRange(start_row=0, row_count=1)
        wrong_burst_layout = CentPartitionedVectorLayout(
            value_count=8,
            partition_count=1,
            burst_length=8,
        )
        with self.assertRaisesRegex(ValueError, "burst_length"):
            lower_store_bank_group_vector(
                builder,
                CentBankGroupVectorTransferPlan(
                    dram=CentDramVector(
                        rows=rows,
                        layout=wrong_burst_layout,
                    ),
                    buffer=make_buffer_vector(
                        start_slot=0,
                        layout=wrong_burst_layout,
                    ),
                ),
            )
        too_many_groups_layout = CentPartitionedVectorLayout(
            value_count=8,
            partition_count=2,
            burst_length=4,
        )
        with self.assertRaisesRegex(ValueError, "more bank groups"):
            lower_store_bank_group_vector(
                builder,
                CentBankGroupVectorTransferPlan(
                    dram=CentDramVector(
                        rows=rows,
                        layout=too_many_groups_layout,
                    ),
                    buffer=make_buffer_vector(
                        start_slot=0,
                        layout=too_many_groups_layout,
                    ),
                ),
            )
        with self.assertRaisesRegex(ValueError, "bank_group"):
            CentDramVector(
                rows=rows,
                layout=plan_partitioned_vector(8, 1, 4),
                bank_group=4,
            )
        multirow_layout = plan_partitioned_vector(32, 1, 4)
        with self.assertRaisesRegex(ValueError, "rows needs 2"):
            _lower_bank_group_vector_transfer(
                builder,
                CentBankGroupVectorTransferPlan(
                    dram=CentDramVector(
                        rows=rows,
                        layout=multirow_layout,
                    ),
                    # One PU group receives all 32 values. Sixteen values fit in
                    # one row, so the declared row range is insufficient.
                    buffer=make_buffer_vector(
                        start_slot=0,
                        layout=multirow_layout,
                    ),
                ),
                instruction_type=WriteSingleBank,
            )

    def test_transfer_requires_matching_zero_padded_layouts(self) -> None:
        """Reject DRAM and Shared Buffer bindings with different padding lanes."""

        with self.assertRaisesRegex(ValueError, "layouts must match"):
            CentBankGroupVectorTransferPlan(
                dram=CentDramVector(
                    rows=CentDramRowRange(start_row=0, row_count=1),
                    layout=CentPartitionedVectorLayout(
                        value_count=8,
                        partition_count=1,
                        burst_length=4,
                    ),
                ),
                buffer=make_buffer_vector(
                    start_slot=0,
                    layout=CentPartitionedVectorLayout(
                        value_count=8,
                        partition_count=2,
                        burst_length=4,
                    ),
                ),
            )


class ElementwiseLoweringTests(unittest.TestCase):
    """Test reusable elementwise lowering."""

    def test_accumulate_uses_explicit_vector_spans(self) -> None:
        """Add two vectors from the requested Shared Buffer addresses."""

        builder = make_builder()
        layout = plan_partitioned_vector(16, 1, 4)
        lower_accumulate(
            builder,
            CentAccumulatePlan(
                destination=make_buffer_vector(start_slot=7, layout=layout),
                source=make_buffer_vector(start_slot=20, layout=layout),
            ),
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

    def test_accumulate_writes_the_complete_zero_padded_final_slot(self) -> None:
        """Process padding lanes so the destination never retains stale data."""

        builder = make_builder()
        layout = CentPartitionedVectorLayout(
            value_count=5,
            partition_count=1,
            burst_length=4,
        )

        lower_accumulate(
            builder,
            CentAccumulatePlan(
                destination=make_buffer_vector(start_slot=3, layout=layout),
                source=make_buffer_vector(start_slot=8, layout=layout),
            ),
        )

        # Five logical values occupy two four-lane slots. ACC handles both
        # complete slots. The two input bindings promise zeros in lanes 5-7, so
        # zero plus zero leaves those three result lanes zero.
        self.assertEqual(
            builder.instructions,
            [
                Accumulate(
                    operation_size=2,
                    destination=CentSharedBufferAddress(slot=3),
                    source=CentSharedBufferAddress(slot=8),
                )
            ],
        )

    def test_accumulate_rejects_different_vector_layouts(self) -> None:
        """Reject operands whose zero-padding lanes do not line up."""

        builder = make_builder()
        contiguous = CentPartitionedVectorLayout(
            value_count=8,
            partition_count=1,
            burst_length=4,
        )
        partitioned = CentPartitionedVectorLayout(
            value_count=8,
            partition_count=2,
            burst_length=4,
        )
        with self.assertRaisesRegex(ValueError, "layouts must match"):
            CentAccumulatePlan(
                destination=make_buffer_vector(
                    start_slot=0,
                    layout=contiguous,
                ),
                source=make_buffer_vector(
                    start_slot=2,
                    layout=partitioned,
                ),
            )

        wrong_burst = CentPartitionedVectorLayout(
            value_count=8,
            partition_count=1,
            burst_length=8,
        )
        with self.assertRaisesRegex(ValueError, "burst_length"):
            lower_accumulate(
                builder,
                CentAccumulatePlan(
                    destination=make_buffer_vector(
                        start_slot=0,
                        layout=wrong_burst,
                    ),
                    source=make_buffer_vector(
                        start_slot=1,
                        layout=wrong_burst,
                    ),
                ),
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
