"""Check complete runs from named host inputs to outputs and instruction traces."""

import unittest
from unittest.mock import patch

import vllm_cent.simulator as simulator
from vllm_cent import (
    CentDramRegion,
    CentExecutable,
    CentExecutionEvent,
    CentExecutionManifest,
    CentExecutionRequest,
    CentExecutionResult,
    CentGlobalBufferRegion,
    CentHardwareSpec,
    CentInputBinding,
    CentManifestError,
    CentNamedScalars,
    CentNumericProfile,
    CentOutputBinding,
    CentSharedBufferRegion,
    CentSimulatorConfiguration,
    CentTraceLevel,
    CentUnsupportedSemanticsError,
    execute_functionally,
)
from vllm_cent.cent import (
    Accumulate,
    CentChannelSet,
    CentGlobalBufferAddress,
    CentMemoryAddress,
    CentProgram,
    CentSharedBufferAddress,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    Exponent,
    ReadSingleBank,
    WriteGlobalBuffer,
    WriteSingleBank,
)
from vllm_cent.simulator.api import (
    _execution_events,
    _extract_output,
    _materialize_input,
    _numeric_semantics,
)
from vllm_cent.simulator.execution import _execute_preflighted_program, execute_program
from vllm_cent.simulator.numeric import ReferenceMathSemantics
from vllm_cent.simulator.state import CentDeviceState


def make_hardware() -> CentHardwareSpec:
    """Create the tiny target used by public execution tests.

    Returns:
        Two channels with four banks each. Every burst and Shared Buffer
        slot holds two values, so operation_size=2 moves four values. Eight
        Shared Buffer slots provide 16 scalar positions.
    """

    return CentHardwareSpec(
        num_channels=2,
        num_banks=4,
        dram_rows=2,
        dram_columns=8,
        global_buffer_columns=8,
        burst_length=2,
        accumulator_slots_per_bank=2,
        sigmoid_activation_function_id=0,
        shared_buffer_slots=8,
    )


class FunctionalExecutionTests(unittest.TestCase):
    """Check input loading, program execution, output reads, and traces together."""

    def test_trace_selection_controls_retained_values(self) -> None:
        """Discard overwritten values without changing computation or trace order."""

        address = CentSharedBufferAddress(slot=0)
        program = CentProgram(
            hardware=make_hardware(),
            instructions=(
                Accumulate(source=address, destination=address, operation_size=1),
            )
            * 3,
        )
        region = CentSharedBufferRegion(address=address, scalar_count=2)
        executable = CentExecutable(
            program=program,
            manifest=CentExecutionManifest(
                inputs=(CentInputBinding(name="x", region=region),),
                outputs=(CentOutputBinding(name="out", region=region),),
            ),
        )
        for trace_level in CentTraceLevel:
            keep_history = trace_level is CentTraceLevel.SUMMARY
            with self.subTest(trace_level=trace_level):
                # Each instruction doubles the same two lanes. Only the final
                # [8, 16] needs to survive when the caller disables tracing.
                state = CentDeviceState(hardware=program.hardware)
                state.write_shared_buffer(address, (1.0, 2.0))
                summary = _execute_preflighted_program(
                    program, state, retain_effects=keep_history
                )
                self.assertEqual(summary.executed_instruction_count, 3)
                self.assertEqual(
                    len(summary.committed_effects), 3 if keep_history else 0
                )
                self.assertEqual(
                    state.read_shared_buffer_values(address, value_count=2), (8.0, 16.0)
                )
                with patch(
                    "vllm_cent.simulator.api._execute_preflighted_program",
                    wraps=_execute_preflighted_program,
                ) as execute:
                    result = execute_functionally(
                        CentExecutionRequest(
                            executable=executable,
                            inputs=(CentNamedScalars(name="x", values=(1.0, 2.0)),),
                            configuration=CentSimulatorConfiguration(
                                trace_level=trace_level
                            ),
                        )
                    )
                self.assertEqual(
                    execute.call_args.kwargs, {"retain_effects": keep_history}
                )
                self.assertEqual(
                    result.outputs, (CentNamedScalars(name="out", values=(8.0, 16.0)),)
                )
                self.assertEqual(len(result.events), 3 if keep_history else 0)

    def test_executes_the_complete_supported_dataflow_slice(self) -> None:
        """Run copies, multiplication, and addition using named input bindings."""

        hardware = make_hardware()
        channel_zero = CentChannelSet(channels=(0,))
        channel_one = CentChannelSet(channels=(1,))
        program = CentProgram(
            hardware=hardware,
            instructions=(
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=0, row=0, column=0),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=0),
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(channel=0, bank=1, row=0, column=0),
                    operation_size=2,
                    source=CentSharedBufferAddress(slot=2),
                ),
                ElementwiseMultiply(
                    channels=channel_zero,
                    operation_size=2,
                    row=0,
                    column=0,
                ),
                CopyBankToGlobalBuffer(
                    channels=channel_zero,
                    operation_size=2,
                    bank=2,
                    row=0,
                    column=0,
                ),
                CopyGlobalBufferToBank(
                    channels=channel_zero,
                    operation_size=2,
                    bank=3,
                    row=0,
                    column=0,
                ),
                WriteGlobalBuffer(
                    channels=channel_one,
                    operation_size=2,
                    column=0,
                    source=CentSharedBufferAddress(slot=0),
                ),
                CopyGlobalBufferToBank(
                    channels=channel_one,
                    operation_size=2,
                    bank=3,
                    row=0,
                    column=0,
                ),
                ReadSingleBank(
                    address=CentMemoryAddress(channel=0, bank=2, row=0, column=0),
                    operation_size=2,
                    destination=CentSharedBufferAddress(slot=4),
                ),
                Accumulate(
                    operation_size=2,
                    destination=CentSharedBufferAddress(slot=4),
                    source=CentSharedBufferAddress(slot=0),
                ),
            ),
        )
        executable = CentExecutable(
            program=program,
            manifest=CentExecutionManifest(
                inputs=(
                    CentInputBinding(
                        name="left",
                        region=CentSharedBufferRegion(
                            address=CentSharedBufferAddress(slot=0),
                            scalar_count=4,
                        ),
                    ),
                    CentInputBinding(
                        name="right",
                        region=CentSharedBufferRegion(
                            address=CentSharedBufferAddress(slot=2),
                            scalar_count=4,
                        ),
                    ),
                ),
                outputs=(
                    CentOutputBinding(
                        name="sum",
                        region=CentSharedBufferRegion(
                            address=CentSharedBufferAddress(slot=4),
                            scalar_count=4,
                        ),
                    ),
                    CentOutputBinding(
                        name="product_copy",
                        region=CentDramRegion(
                            address=CentMemoryAddress(
                                channel=0,
                                bank=3,
                                row=0,
                                column=0,
                            ),
                            scalar_count=4,
                        ),
                    ),
                    CentOutputBinding(
                        name="global_copy",
                        region=CentDramRegion(
                            address=CentMemoryAddress(
                                channel=1,
                                bank=3,
                                row=0,
                                column=0,
                            ),
                            scalar_count=4,
                        ),
                    ),
                ),
            ),
        )

        result = execute_functionally(
            CentExecutionRequest(
                executable=executable,
                inputs=(
                    CentNamedScalars(
                        name="right",
                        values=(10.0, 20.0, 30.0, 40.0),
                    ),
                    CentNamedScalars(
                        name="left",
                        values=(2.0, 3.0, 4.0, 5.0),
                    ),
                ),
                configuration=CentSimulatorConfiguration(
                    trace_level=CentTraceLevel.SUMMARY
                ),
            )
        )

        # Each operation covers two bursts of two values, hence four scalars
        # in each trace range. Banks 0 and 1 multiply into bank 2; bank 3 gets
        # a copy before the Shared Buffer addition changes the returned sum.
        # Products are (2*10, 3*20, 4*30, 5*40); adding left gives
        # (22, 63, 124, 205). Channel 1 receives an unchanged copy of left.
        scalar_count = 4
        self.assertEqual(
            result,
            CentExecutionResult(
                outputs=(
                    CentNamedScalars(name="sum", values=(22.0, 63.0, 124.0, 205.0)),
                    CentNamedScalars(
                        name="product_copy",
                        values=(20.0, 60.0, 120.0, 200.0),
                    ),
                    CentNamedScalars(
                        name="global_copy",
                        values=(2.0, 3.0, 4.0, 5.0),
                    ),
                ),
                events=(
                    CentExecutionEvent(
                        device_id=0,
                        instruction_index=0,
                        instruction=program.instructions[0],
                        reads=(
                            CentSharedBufferRegion(
                                address=CentSharedBufferAddress(slot=0),
                                scalar_count=scalar_count,
                            ),
                        ),
                        writes=(
                            CentDramRegion(
                                address=CentMemoryAddress(
                                    channel=0,
                                    bank=0,
                                    row=0,
                                    column=0,
                                ),
                                scalar_count=scalar_count,
                            ),
                        ),
                    ),
                    CentExecutionEvent(
                        device_id=0,
                        instruction_index=1,
                        instruction=program.instructions[1],
                        reads=(
                            CentSharedBufferRegion(
                                address=CentSharedBufferAddress(slot=2),
                                scalar_count=scalar_count,
                            ),
                        ),
                        writes=(
                            CentDramRegion(
                                address=CentMemoryAddress(
                                    channel=0,
                                    bank=1,
                                    row=0,
                                    column=0,
                                ),
                                scalar_count=scalar_count,
                            ),
                        ),
                    ),
                    CentExecutionEvent(
                        device_id=0,
                        instruction_index=2,
                        instruction=program.instructions[2],
                        reads=(
                            CentDramRegion(
                                address=CentMemoryAddress(
                                    channel=0,
                                    bank=0,
                                    row=0,
                                    column=0,
                                ),
                                scalar_count=scalar_count,
                            ),
                            CentDramRegion(
                                address=CentMemoryAddress(
                                    channel=0,
                                    bank=1,
                                    row=0,
                                    column=0,
                                ),
                                scalar_count=scalar_count,
                            ),
                        ),
                        writes=(
                            CentDramRegion(
                                address=CentMemoryAddress(
                                    channel=0,
                                    bank=2,
                                    row=0,
                                    column=0,
                                ),
                                scalar_count=scalar_count,
                            ),
                        ),
                    ),
                    CentExecutionEvent(
                        device_id=0,
                        instruction_index=3,
                        instruction=program.instructions[3],
                        reads=(
                            CentDramRegion(
                                address=CentMemoryAddress(
                                    channel=0,
                                    bank=2,
                                    row=0,
                                    column=0,
                                ),
                                scalar_count=scalar_count,
                            ),
                        ),
                        writes=(
                            CentGlobalBufferRegion(
                                address=CentGlobalBufferAddress(
                                    channel=0,
                                    column=0,
                                ),
                                scalar_count=scalar_count,
                            ),
                        ),
                    ),
                    CentExecutionEvent(
                        device_id=0,
                        instruction_index=4,
                        instruction=program.instructions[4],
                        reads=(
                            CentGlobalBufferRegion(
                                address=CentGlobalBufferAddress(
                                    channel=0,
                                    column=0,
                                ),
                                scalar_count=scalar_count,
                            ),
                        ),
                        writes=(
                            CentDramRegion(
                                address=CentMemoryAddress(
                                    channel=0,
                                    bank=3,
                                    row=0,
                                    column=0,
                                ),
                                scalar_count=scalar_count,
                            ),
                        ),
                    ),
                    CentExecutionEvent(
                        device_id=0,
                        instruction_index=5,
                        instruction=program.instructions[5],
                        reads=(
                            CentSharedBufferRegion(
                                address=CentSharedBufferAddress(slot=0),
                                scalar_count=scalar_count,
                            ),
                        ),
                        writes=(
                            CentGlobalBufferRegion(
                                address=CentGlobalBufferAddress(
                                    channel=1,
                                    column=0,
                                ),
                                scalar_count=scalar_count,
                            ),
                        ),
                    ),
                    CentExecutionEvent(
                        device_id=0,
                        instruction_index=6,
                        instruction=program.instructions[6],
                        reads=(
                            CentGlobalBufferRegion(
                                address=CentGlobalBufferAddress(
                                    channel=1,
                                    column=0,
                                ),
                                scalar_count=scalar_count,
                            ),
                        ),
                        writes=(
                            CentDramRegion(
                                address=CentMemoryAddress(
                                    channel=1,
                                    bank=3,
                                    row=0,
                                    column=0,
                                ),
                                scalar_count=scalar_count,
                            ),
                        ),
                    ),
                    CentExecutionEvent(
                        device_id=0,
                        instruction_index=7,
                        instruction=program.instructions[7],
                        reads=(
                            CentDramRegion(
                                address=CentMemoryAddress(
                                    channel=0,
                                    bank=2,
                                    row=0,
                                    column=0,
                                ),
                                scalar_count=scalar_count,
                            ),
                        ),
                        writes=(
                            CentSharedBufferRegion(
                                address=CentSharedBufferAddress(slot=4),
                                scalar_count=scalar_count,
                            ),
                        ),
                    ),
                    CentExecutionEvent(
                        device_id=0,
                        instruction_index=8,
                        instruction=program.instructions[8],
                        reads=(
                            CentSharedBufferRegion(
                                address=CentSharedBufferAddress(slot=4),
                                scalar_count=scalar_count,
                            ),
                            CentSharedBufferRegion(
                                address=CentSharedBufferAddress(slot=0),
                                scalar_count=scalar_count,
                            ),
                        ),
                        writes=(
                            CentSharedBufferRegion(
                                address=CentSharedBufferAddress(slot=4),
                                scalar_count=scalar_count,
                            ),
                        ),
                    ),
                ),
                # Two input copies, multiply, two bank/buffer copies, a
                # channel-1 broadcast and copy, read-back, then addition.
                executed_instruction_count=9,
            ),
        )

    def test_manifest_errors_use_the_simulator_error_hierarchy(self) -> None:
        """Translate request-data mismatches into a stable public error."""

        executable = CentExecutable(
            program=CentProgram(
                hardware=make_hardware(),
                instructions=(
                    Accumulate(
                        operation_size=1,
                        destination=CentSharedBufferAddress(slot=0),
                        source=CentSharedBufferAddress(slot=0),
                    ),
                ),
            ),
            manifest=CentExecutionManifest(
                inputs=(
                    CentInputBinding(
                        name="input",
                        region=CentSharedBufferRegion(
                            address=CentSharedBufferAddress(slot=0),
                            scalar_count=2,
                        ),
                    ),
                )
            ),
        )

        with self.assertRaisesRegex(CentManifestError, "missing.*input"):
            execute_functionally(CentExecutionRequest(executable=executable, inputs=()))

    def test_materializes_and_extracts_each_raw_address_space(self) -> None:
        """Use DRAM and Global Buffer bindings with default trace suppression."""

        hardware = make_hardware()
        channel_zero = CentChannelSet(channels=(0,))
        executable = CentExecutable(
            program=CentProgram(
                hardware=hardware,
                instructions=(
                    CopyGlobalBufferToBank(
                        channels=channel_zero,
                        operation_size=1,
                        bank=1,
                        row=0,
                        column=0,
                    ),
                    CopyBankToGlobalBuffer(
                        channels=channel_zero,
                        operation_size=1,
                        bank=0,
                        row=0,
                        column=0,
                    ),
                ),
            ),
            manifest=CentExecutionManifest(
                inputs=(
                    CentInputBinding(
                        name="dram_input",
                        region=CentDramRegion(
                            address=CentMemoryAddress(
                                channel=0,
                                bank=0,
                                row=0,
                                column=0,
                            ),
                            scalar_count=2,
                        ),
                    ),
                    CentInputBinding(
                        name="global_input",
                        region=CentGlobalBufferRegion(
                            address=CentGlobalBufferAddress(channel=0, column=0),
                            scalar_count=2,
                        ),
                    ),
                ),
                outputs=(
                    CentOutputBinding(
                        name="dram_output",
                        region=CentDramRegion(
                            address=CentMemoryAddress(
                                channel=0,
                                bank=1,
                                row=0,
                                column=0,
                            ),
                            scalar_count=2,
                        ),
                    ),
                    CentOutputBinding(
                        name="global_output",
                        region=CentGlobalBufferRegion(
                            address=CentGlobalBufferAddress(channel=0, column=0),
                            scalar_count=2,
                        ),
                    ),
                ),
            ),
        )

        result = execute_functionally(
            CentExecutionRequest(
                executable=executable,
                inputs=(
                    CentNamedScalars(name="dram_input", values=(1.0, 2.0)),
                    CentNamedScalars(name="global_input", values=(9.0, 10.0)),
                ),
            )
        )

        # Copy the original Global Buffer values to bank 1 before replacing
        # that buffer with bank 0. Reversing the copies would lose (9, 10).
        self.assertEqual(
            result.outputs,
            (
                CentNamedScalars(name="dram_output", values=(9.0, 10.0)),
                CentNamedScalars(name="global_output", values=(1.0, 2.0)),
            ),
        )
        self.assertEqual(result.events, ())

    def test_execution_event_rejects_negative_coordinates(self) -> None:
        """Keep trace event device and program positions well formed."""

        instruction = Accumulate(
            operation_size=1,
            destination=CentSharedBufferAddress(slot=0),
            source=CentSharedBufferAddress(slot=0),
        )
        for device_id, instruction_index in ((-1, 0), (0, -1)):
            with self.subTest(
                device_id=device_id,
                instruction_index=instruction_index,
            ):
                with self.assertRaises(ValueError):
                    CentExecutionEvent(
                        device_id=device_id,
                        instruction_index=instruction_index,
                        instruction=instruction,
                        reads=(),
                        writes=(),
                    )

    def test_unsupported_instruction_fails_during_preflight(self) -> None:
        """Reject exponentiation during support checks, before trying to run it."""

        executable = CentExecutable(
            program=CentProgram(
                hardware=make_hardware(),
                instructions=(
                    Exponent(
                        operation_size=1,
                        destination=CentSharedBufferAddress(slot=1),
                        source=CentSharedBufferAddress(slot=0),
                    ),
                ),
            ),
            manifest=CentExecutionManifest(
                inputs=(
                    CentInputBinding(
                        name="input",
                        region=CentSharedBufferRegion(
                            address=CentSharedBufferAddress(slot=0),
                            scalar_count=2,
                        ),
                    ),
                )
            ),
        )

        with self.assertRaises(CentUnsupportedSemanticsError):
            execute_functionally(
                CentExecutionRequest(
                    executable=executable,
                    inputs=(CentNamedScalars(name="input", values=(1.0, 2.0)),),
                )
            )


class FunctionalApiHelperTests(unittest.TestCase):
    """Check input loading, output reads, numeric selection, and trace helpers."""

    def test_package_keeps_state_level_execution_private(self) -> None:
        """Expose requests with named inputs, without exposing mutable device state."""

        for internal_name in (
            "CentDeviceState",
            "CentExecutionSummary",
            "CentNumericSemantics",
            "ReferenceMathSemantics",
            "execute_program",
            "preflight_program",
        ):
            with self.subTest(internal_name=internal_name):
                self.assertFalse(hasattr(simulator, internal_name))

    def test_materialization_and_extraction_cover_every_region_type(self) -> None:
        """Write and read the same named values in all three supported memory spaces."""

        state = CentDeviceState(hardware=make_hardware())
        regions = (
            CentDramRegion(
                address=CentMemoryAddress(channel=0, bank=0, row=0, column=0),
                scalar_count=2,
            ),
            CentSharedBufferRegion(
                address=CentSharedBufferAddress(slot=1),
                scalar_count=2,
            ),
            CentGlobalBufferRegion(
                address=CentGlobalBufferAddress(channel=1, column=2),
                scalar_count=2,
            ),
        )
        for index, region in enumerate(regions):
            with self.subTest(region=type(region).__name__):
                name = f"value_{index}"
                value = CentNamedScalars(name=name, values=(1.0, 2.0))
                _materialize_input(
                    state,
                    CentInputBinding(name=name, region=region),
                    value,
                )
                self.assertEqual(
                    _extract_output(
                        state,
                        CentOutputBinding(name=name, region=region),
                    ),
                    value,
                )

    def test_numeric_and_event_helpers_return_the_selected_contract(self) -> None:
        """Select Python float arithmetic and trace the range actually written."""

        self.assertIsInstance(
            _numeric_semantics(CentNumericProfile.REFERENCE_MATH),
            ReferenceMathSemantics,
        )
        hardware = make_hardware()
        source = CentSharedBufferAddress(slot=0)
        destination = CentMemoryAddress(channel=0, bank=0, row=0, column=0)
        instruction = WriteSingleBank(
            address=destination,
            operation_size=1,
            source=source,
        )
        executable = CentExecutable(
            program=CentProgram(hardware=hardware, instructions=(instruction,)),
            manifest=CentExecutionManifest(),
        )
        state = CentDeviceState(hardware=hardware)
        state.write_shared_buffer(source, (1.0, 2.0))
        summary = execute_program(executable.program, state)
        no_trace_request = CentExecutionRequest(executable=executable)

        self.assertEqual(_execution_events(no_trace_request, state, summary), ())
        trace_request = CentExecutionRequest(
            executable=executable,
            configuration=CentSimulatorConfiguration(
                trace_level=CentTraceLevel.SUMMARY
            ),
        )
        self.assertEqual(
            _execution_events(trace_request, state, summary),
            (
                CentExecutionEvent(
                    device_id=0,
                    instruction_index=0,
                    instruction=instruction,
                    reads=(
                        CentSharedBufferRegion(
                            address=source,
                            scalar_count=2,
                        ),
                    ),
                    writes=(
                        CentDramRegion(
                            address=destination,
                            scalar_count=2,
                        ),
                    ),
                ),
            ),
        )


if __name__ == "__main__":
    unittest.main()
