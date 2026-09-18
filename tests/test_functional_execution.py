"""End-to-end tests for the public functional-execution boundary."""

import unittest

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
from vllm_cent.simulator.execution import execute_program
from vllm_cent.simulator.numeric import ReferenceMathSemantics
from vllm_cent.simulator.state import CentDeviceState


def make_hardware() -> CentHardwareSpec:
    """Create the tiny target used by public execution tests.

    Returns:
        Two-channel target with two values in each burst.
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
    """Exercise materialization, execution, extraction, and tracing together."""

    def test_executes_the_complete_supported_dataflow_slice(self) -> None:
        """Run every currently supported instruction through the public API."""

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
        """Never guess semantics for an opcode outside the supported slice."""

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
    """Directly exercise nontrivial private API boundary helpers."""

    def test_package_keeps_state_level_execution_private(self) -> None:
        """Expose only the manifest-facing simulator API and stable errors."""

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
        """Move named host scalars through each raw physical region."""

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
        """Construct reference math and trace one exact committed effect."""

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
