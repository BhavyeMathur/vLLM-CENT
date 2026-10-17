import math
import unittest
from dataclasses import replace

from vllm_cent.cent import (
    CentBankRegisterAddress,
    CentChannelSet,
    CentGlobalBufferAddress,
    CentHardwareSpec,
    CentInstruction,
    CentMemoryAddress,
    CentProgram,
    CentSharedBufferAddress,
    MacAllBanks,
    MacOperandSource,
    render_instruction,
)
from vllm_cent.cent.aim import render_aim_instruction
from vllm_cent.cent.reference import (
    InitializeAccumulator,
    ReadAccumulator,
    ReferenceBroadcast,
    ReferenceCopy,
    ReferenceFill,
    ReferenceReduceSum,
    ReferenceRmsScale,
    ReferenceStoreSlice,
)
from vllm_cent.runtime import CentAccumulatorRegion
from vllm_cent.simulator import (
    CentExecutionFault,
    CentUninitializedReadError,
    CentUnsupportedSemanticsError,
)
from vllm_cent.simulator.effects import AccumulatorWriteEffect
from vllm_cent.simulator.execution import execute_program
from vllm_cent.simulator.state import CentDeviceState


def hardware() -> CentHardwareSpec:
    """Use two channels, four banks, two lanes, and two independent latches."""
    return CentHardwareSpec(
        num_channels=2,
        num_banks=4,
        dram_rows=2,
        dram_columns=8,
        global_buffer_columns=8,
        burst_length=2,
        accumulator_slots_per_bank=2,
        sigmoid_activation_function_id=0,
        shared_buffer_slots=32,
    )


class ReferenceComputeTests(unittest.TestCase):
    """Check explicit storage effects rather than an opaque host GEMV result."""

    def setUp(self) -> None:
        """Create fresh device memory for each instruction sequence."""
        self.state = CentDeviceState(hardware=hardware())
        self.channels = CentChannelSet(channels=(1,))

    def run_instructions(self, *instructions: CentInstruction) -> None:
        """Execute the provided sequence against this test's current state."""
        execute_program(
            CentProgram(hardware=self.state.hardware, instructions=instructions),
            self.state,
        )

    def test_mac_selected_latch_continuation_read_order_and_isolation(self) -> None:
        """Two dot chunks add to latch one; reads preserve it and latch zero."""
        for channel in range(2):
            for bank in range(4):
                self.state.write_accumulator(
                    CentBankRegisterAddress(channel=channel, bank=bank, register=0),
                    99.0,
                )
        self.state.write_global_buffer(
            CentGlobalBufferAddress(channel=1, column=0), (2.0, -1.0, 3.0, 4.0)
        )
        for bank in range(4):
            self.state.write_dram(
                CentMemoryAddress(channel=1, bank=bank, row=0, column=0),
                (float(bank + 1), 2.0, 1.0, -1.0),
            )
        mac = MacAllBanks(
            channels=self.channels,
            operation_size=1,
            row=0,
            column=0,
            accumulation_register=1,
            operand_source=MacOperandSource.GLOBAL_BUFFER,
        )
        read = ReadAccumulator(
            channels=self.channels,
            accumulation_register=1,
            destination=CentSharedBufferAddress(slot=0),
        )
        self.run_instructions(
            InitializeAccumulator(
                channels=self.channels, accumulation_register=1, value=5.0
            ),
            mac,
            replace(mac, column=2),
            read,
            replace(read, destination=CentSharedBufferAddress(slot=2)),
        )
        # 5 + (bank+1)*2 -2 +3 -4 = 2*bank+4.
        self.assertEqual(
            self.state.read_shared_buffer(
                CentSharedBufferAddress(slot=0), slot_count=4
            ),
            (4.0, 6.0, 8.0, 10.0) * 2,
        )
        for channel in range(2):
            for bank in range(4):
                self.assertEqual(
                    self.state.read_accumulator(
                        CentBankRegisterAddress(channel=channel, bank=bank, register=0)
                    ),
                    99.0,
                )
        with self.assertRaises(CentUninitializedReadError):
            self.state.read_accumulator(
                CentBankRegisterAddress(channel=0, bank=0, register=1)
            )

    def test_mac_global_buffer_capacity_is_checked(self) -> None:
        """A valid DRAM span still needs a matching Global Buffer span."""
        command = MacAllBanks(
            channels=self.channels,
            operation_size=2,
            row=0,
            column=0,
            accumulation_register=0,
            operand_source=MacOperandSource.GLOBAL_BUFFER,
        )
        with self.assertRaises(ValueError):
            CentProgram(
                hardware=replace(hardware(), global_buffer_columns=2),
                instructions=(command,),
            )
        CentProgram(
            hardware=replace(hardware(), global_buffer_columns=2),
            instructions=(replace(command, operand_source=MacOperandSource.NEXT_BANK),),
        )

    def test_neighbor_mac_only_updates_even_banks(self) -> None:
        """Neighbor mode accumulates pair products into the even bank only."""
        for bank in range(4):
            self.state.write_dram(
                CentMemoryAddress(channel=1, bank=bank, row=0, column=0),
                (float(bank + 1), 2.0),
            )
        self.run_instructions(
            InitializeAccumulator(channels=self.channels, accumulation_register=1),
            MacAllBanks(
                channels=self.channels,
                operation_size=1,
                row=0,
                column=0,
                accumulation_register=1,
                operand_source=MacOperandSource.NEXT_BANK,
            ),
            ReadAccumulator(
                channels=self.channels,
                accumulation_register=1,
                destination=CentSharedBufferAddress(slot=0),
            ),
        )
        self.assertEqual(
            self.state.read_shared_buffer(
                CentSharedBufferAddress(slot=0), slot_count=2
            ),
            (6.0, 0.0, 16.0, 0.0),
        )

    def test_failed_mac_and_commit_are_atomic(self) -> None:
        """A late missing operand or bad destination cannot partially update latches."""
        self.run_instructions(
            InitializeAccumulator(
                channels=self.channels, accumulation_register=1, value=7.0
            )
        )
        self.state.write_global_buffer(
            CentGlobalBufferAddress(channel=1, column=0), (1.0, 1.0)
        )
        for bank in range(3):
            self.state.write_dram(
                CentMemoryAddress(channel=1, bank=bank, row=0, column=0), (1.0, 2.0)
            )
        with self.assertRaises(CentUninitializedReadError):
            self.run_instructions(
                MacAllBanks(
                    channels=self.channels,
                    operation_size=1,
                    row=0,
                    column=0,
                    accumulation_register=1,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                )
            )
        register = CentBankRegisterAddress(channel=1, bank=0, register=1)
        self.assertEqual(self.state.read_accumulator(register), 7.0)
        with self.assertRaises(ValueError):
            self.state.commit_effects(
                (
                    AccumulatorWriteEffect(address=register, value=2.0),
                    AccumulatorWriteEffect(
                        address=replace(register, bank=4), value=3.0
                    ),
                )
            )
        self.assertEqual(self.state.read_accumulator(register), 7.0)
        self.assertEqual(
            AccumulatorWriteEffect(address=register, value=2.0).region,
            CentAccumulatorRegion(address=register),
        )

    def test_packing_gather_alias_and_padding(self) -> None:
        """Gathering an odd offset preserves adjacent values without full-output rewrites."""
        source = CentSharedBufferAddress(slot=0)
        destination = CentSharedBufferAddress(slot=10)
        self.state.write_shared_buffer(source, (1.0, 2.0, 3.0, 4.0, 5.0, 6.0))
        self.run_instructions(
            ReferenceCopy(
                source=source,
                destination=destination,
                source_offset=3,
                value_count=2,
                destination_count=5,
            )
        )
        self.assertEqual(
            self.state.read_shared_buffer(destination, slot_count=3),
            (4.0, 5.0, 0.0, 0.0, 0.0, 0.0),
        )
        self.run_instructions(
            ReferenceStoreSlice(
                source=destination,
                destination=destination,
                source_offset=0,
                destination_offset=1,
                value_count=2,
                destination_count=5,
            )
        )
        self.assertEqual(
            self.state.read_shared_buffer(destination, slot_count=3),
            (4.0, 4.0, 5.0, 0.0, 0.0, 0.0),
        )
        self.run_instructions(
            ReferenceFill(destination=destination, value_count=5, value=3.0)
        )
        self.assertEqual(
            self.state.read_shared_buffer(destination, slot_count=3),
            (3.0, 3.0, 3.0, 3.0, 3.0, 0.0),
        )

    def test_reduction_scale_broadcast_excludes_padding(self) -> None:
        """Logical count excludes a deliberately nonzero tail from the RMS mean."""
        source = CentSharedBufferAddress(slot=0)
        scalar = CentSharedBufferAddress(slot=4)
        output = CentSharedBufferAddress(slot=5)
        self.state.write_shared_buffer(source, (1.0, 4.0, 9.0, 1000.0))
        self.run_instructions(
            ReferenceReduceSum(source=source, destination=scalar, value_count=3),
            ReferenceRmsScale(
                source=scalar, destination=scalar, value_count=3, epsilon=1e-5
            ),
            ReferenceBroadcast(source=scalar, destination=output, value_count=3),
        )
        scale = 1 / math.sqrt(14 / 3 + 1e-5)
        self.assertEqual(
            self.state.read_shared_buffer(output, slot_count=2),
            (scale, scale, scale, 0.0),
        )

    def test_negative_root_and_uninitialized_reads_preserve_destination(self) -> None:
        """Rejected scalar math does not overwrite a previously stored result."""
        source = CentSharedBufferAddress(slot=0)
        output = CentSharedBufferAddress(slot=2)
        self.state.write_shared_buffer(source, (-3.0, 0.0))
        self.state.write_shared_buffer(output, (77.0, 88.0))
        with self.assertRaises(CentExecutionFault):
            self.run_instructions(
                ReferenceRmsScale(
                    source=source, destination=output, value_count=1, epsilon=1e-5
                )
            )
        with self.assertRaises(CentUninitializedReadError):
            self.run_instructions(
                ReferenceBroadcast(
                    source=CentSharedBufferAddress(slot=3),
                    destination=output,
                    value_count=2,
                )
            )
        self.assertEqual(
            self.state.read_shared_buffer(output, slot_count=1), (77.0, 88.0)
        )

    def test_reference_validation_and_physical_export_rejection(self) -> None:
        """Reference services never silently acquire paper or AiM encodings."""
        source = CentSharedBufferAddress(slot=0)
        good = ReferenceCopy(
            source=source,
            destination=source,
            source_offset=0,
            value_count=1,
            destination_count=1,
        )
        invalid = (
            replace(good, value_count=0),
            replace(good, source_offset=-1),
            replace(good, destination_count=0),
            replace(good, source_offset=64),
            InitializeAccumulator(channels=self.channels, accumulation_register=2),
            InitializeAccumulator(
                channels=CentChannelSet(channels=(2,)), accumulation_register=0
            ),
            ReferenceRmsScale(
                source=source, destination=source, value_count=1, epsilon=float("nan")
            ),
        )
        for instruction in invalid:
            with self.subTest(instruction=instruction), self.assertRaises(ValueError):
                CentProgram(hardware=hardware(), instructions=(instruction,))
        for renderer in (render_instruction, render_aim_instruction):
            with self.assertRaisesRegex(ValueError, "no physical"):
                renderer(good)

    def test_custom_numeric_policy_and_preflight(self) -> None:
        """MAC honors custom arithmetic; RMS rejects undeclared root semantics."""

        class RejectMultiply:
            """Keep transfers ordinary but reject multiplication deliberately."""

            def store(self, value: float) -> float:
                """Preserve host values."""
                return float(value)

            def add(self, left: float, right: float) -> float:
                """Add stored operands."""
                return left + right

            def multiply(self, left: float, right: float) -> float:
                """Reject an operation to check instruction rollback."""
                raise ValueError("multiplication rejected")

        self.state = CentDeviceState(hardware=hardware(), numeric=RejectMultiply())
        source = CentSharedBufferAddress(slot=0)
        with self.assertRaises(CentUnsupportedSemanticsError):
            self.run_instructions(
                ReferenceFill(destination=source, value_count=2),
                ReferenceRmsScale(
                    source=source, destination=source, value_count=2, epsilon=1e-5
                ),
            )
        with self.assertRaises(CentUninitializedReadError):
            self.state.read_shared_buffer(source, slot_count=1)
        self.run_instructions(
            InitializeAccumulator(
                channels=self.channels, accumulation_register=1, value=3.0
            )
        )
        self.state.write_global_buffer(
            CentGlobalBufferAddress(channel=1, column=0), (1.0, 2.0)
        )
        self.state.write_dram(
            CentMemoryAddress(channel=1, bank=0, row=0, column=0), (1.0, 2.0)
        )
        with self.assertRaises(CentExecutionFault):
            self.run_instructions(
                MacAllBanks(
                    channels=self.channels,
                    operation_size=1,
                    row=0,
                    column=0,
                    accumulation_register=1,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                )
            )
        self.assertEqual(
            self.state.read_accumulator(
                CentBankRegisterAddress(channel=1, bank=0, register=1)
            ),
            3.0,
        )

    def test_channel_major_read_and_tail_padding(self) -> None:
        """Sixteen lanes can hold eight register results and eight zero lanes."""
        self.state = CentDeviceState(
            hardware=replace(
                hardware(), burst_length=16, dram_columns=16, global_buffer_columns=16
            )
        )
        self.run_instructions(
            InitializeAccumulator(
                channels=CentChannelSet(channels=(0,)),
                accumulation_register=1,
                value=2.0,
            ),
            InitializeAccumulator(
                channels=CentChannelSet(channels=(1,)),
                accumulation_register=1,
                value=3.0,
            ),
            ReadAccumulator(
                channels=CentChannelSet(channels=(0, 1)),
                accumulation_register=1,
                destination=CentSharedBufferAddress(slot=0),
            ),
        )
        self.assertEqual(
            self.state.read_shared_buffer(
                CentSharedBufferAddress(slot=0), slot_count=1
            ),
            (2.0,) * 4 + (3.0,) * 4 + (0.0,) * 8,
        )

    def test_aligned_reference_movement_does_not_convert_unused_zero(self) -> None:
        """A complete output slot requires no fabricated padding conversion."""
        from vllm_cent.cent.reference import ReferenceTranspose
        from vllm_cent.simulator.numeric import ReferenceMathSemantics

        class RejectZero(ReferenceMathSemantics):
            """Permit nonzero stored values and reject actual zero initialization."""

            def store(self, value: float) -> float:
                """Make unwanted zero conversion observable during pure movement.

                Args:
                    value: Host value entering simulated storage.

                Returns:
                    The unchanged nonzero value.

                Raises:
                    ValueError: If the caller tries to store zero.
                """
                if value == 0.0:
                    raise ValueError("zero conversion rejected")
                return value

        self.state = CentDeviceState(hardware=hardware(), numeric=RejectZero())
        source = CentSharedBufferAddress(slot=0)
        destination = CentSharedBufferAddress(slot=5)
        self.state.write_shared_buffer(source, (2.0, 3.0))
        commands = (
            ReferenceCopy(
                source=source,
                destination=destination,
                source_offset=0,
                value_count=2,
                destination_count=2,
            ),
            ReferenceBroadcast(source=source, destination=destination, value_count=2),
            ReferenceFill(destination=destination, value_count=2, value=4.0),
            ReferenceTranspose(
                source=source,
                destination=destination,
                source_offset=0,
                row_count=1,
                column_count=2,
            ),
        )
        for command, expected in zip(
            commands, ((2.0, 3.0), (2.0, 2.0), (4.0, 4.0), (2.0, 3.0)), strict=True
        ):
            with self.subTest(command=command):
                self.run_instructions(command)
                self.assertEqual(
                    self.state.read_shared_buffer(destination, slot_count=1), expected
                )
        self.run_instructions(
            InitializeAccumulator(
                channels=self.channels, accumulation_register=1, value=5.0
            ),
            ReadAccumulator(
                channels=self.channels, accumulation_register=1, destination=destination
            ),
        )
        self.assertEqual(
            self.state.read_shared_buffer(destination, slot_count=2), (5.0,) * 4
        )

    def test_rejected_output_conversion_keeps_values_and_reports_destination(
        self,
    ) -> None:
        """Actual fill/padding failures retain structured locations and commit nothing."""
        from vllm_cent.simulator.numeric import ReferenceMathSemantics

        class RejectZero(ReferenceMathSemantics):
            """Reject new zero values while allowing initialized nonzero buffers."""

            def store(self, value: float) -> float:
                """Reject zero and retain every nonzero host value.

                Args:
                    value: Scalar proposed for storage conversion.

                Returns:
                    The unchanged nonzero scalar.

                Raises:
                    ValueError: If zero is requested.
                """
                if value == 0.0:
                    raise ValueError("zero conversion rejected")
                return value

        self.state = CentDeviceState(hardware=hardware(), numeric=RejectZero())
        source = CentSharedBufferAddress(slot=0)
        destination = CentSharedBufferAddress(slot=5)
        self.state.write_shared_buffer(source, (2.0, 3.0))
        self.state.write_shared_buffer(destination, (8.0, 9.0))
        for instruction in (
            ReferenceCopy(
                source=source,
                destination=destination,
                source_offset=0,
                value_count=1,
                destination_count=1,
            ),
            ReferenceFill(destination=destination, value_count=2),
        ):
            with (
                self.subTest(instruction=instruction),
                self.assertRaises(CentExecutionFault) as caught,
            ):
                self.run_instructions(instruction)
            self.assertEqual(caught.exception.location, destination)
            self.assertEqual(caught.exception.instruction_index, 0)
            self.assertEqual(
                self.state.read_shared_buffer(destination, slot_count=1), (8.0, 9.0)
            )

    def test_gather_reads_only_touched_destination_slots(self) -> None:
        """Replacing one initialized slot does not require unrelated slots to exist."""
        source = CentSharedBufferAddress(slot=0)
        destination = CentSharedBufferAddress(slot=5)
        touched = CentSharedBufferAddress(slot=6)
        self.state.write_shared_buffer(source, (2.0, 3.0))
        self.state.write_shared_buffer(touched, (8.0, 9.0))
        self.run_instructions(
            ReferenceStoreSlice(
                source=source,
                destination=destination,
                source_offset=0,
                destination_offset=2,
                value_count=2,
                destination_count=6,
            )
        )
        self.assertEqual(
            self.state.read_shared_buffer(touched, slot_count=1), (2.0, 3.0)
        )
        with self.assertRaises(CentUninitializedReadError):
            self.state.read_shared_buffer(destination, slot_count=1)


if __name__ == "__main__":
    unittest.main()
