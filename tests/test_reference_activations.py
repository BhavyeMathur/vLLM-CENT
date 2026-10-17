"""Check nonlinear reference services, masking, and complete output padding."""

import math
import unittest
from dataclasses import replace

from vllm_cent.activations import ActivationKind
from vllm_cent.cent import CentHardwareSpec, CentProgram, CentSharedBufferAddress
from vllm_cent.cent.aim import render_aim_instruction
from vllm_cent.cent.reference import (
    ReferenceActivation,
    ReferenceFill,
    ReferenceSoftmax,
)
from vllm_cent.cent.render import render_instruction
from vllm_cent.simulator import (
    CentExecutionFault,
    CentUninitializedReadError,
    CentUnsupportedSemanticsError,
)
from vllm_cent.simulator.execution import execute_program
from vllm_cent.simulator.reference import _activate_scalar, _softmax_row
from vllm_cent.simulator.state import CentDeviceState
from vllm_cent.simulator.numeric import ReferenceMathSemantics


def hardware() -> CentHardwareSpec:
    """Use four lanes per slot to expose logical tails and unaligned rows."""
    return CentHardwareSpec(
        num_channels=1,
        num_banks=4,
        dram_rows=1,
        dram_columns=8,
        global_buffer_columns=8,
        burst_length=4,
        accumulator_slots_per_bank=2,
        sigmoid_activation_function_id=0,
        shared_buffer_slots=8,
    )


class ReferenceActivationTests(unittest.TestCase):
    """Verify each explicit formula and the reference execution boundary."""

    def test_scalar_formulas(self) -> None:
        """Moderate inputs agree with the definitions without overflow branches."""
        for x in (-3.0, -0.5, 0.0, 0.5, 3.0):
            expected = {
                ActivationKind.RELU: max(0.0, x),
                ActivationKind.SIGMOID: 1 / (1 + math.exp(-x)),
                ActivationKind.TANH: math.tanh(x),
                ActivationKind.SILU: x / (1 + math.exp(-x)),
                ActivationKind.GELU_ERF: 0.5 * x * (1 + math.erf(x / math.sqrt(2))),
                ActivationKind.GELU_TANH: 0.5
                * x
                * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x * x * x))),
            }
            for function, value in expected.items():
                with self.subTest(x=x, function=function):
                    self.assertAlmostEqual(
                        _activate_scalar(x, function), value, places=14
                    )

    def test_extremes_and_nan(self) -> None:
        """Finite extremes do not overflow; infinities use explicit limiting values."""
        negative = {
            ActivationKind.RELU: 0.0,
            ActivationKind.SIGMOID: 0.0,
            ActivationKind.TANH: -1.0,
            ActivationKind.SILU: 0.0,
            ActivationKind.GELU_ERF: 0.0,
            ActivationKind.GELU_TANH: 0.0,
        }
        for function in ActivationKind:
            for x in (-math.inf, -1e300):
                self.assertEqual(_activate_scalar(x, function), negative[function])
            positive = (
                1.0
                if function in (ActivationKind.SIGMOID, ActivationKind.TANH)
                else math.inf
            )
            self.assertEqual(_activate_scalar(math.inf, function), positive)
            finite = (
                1.0
                if function in (ActivationKind.SIGMOID, ActivationKind.TANH)
                else 1e300
            )
            self.assertEqual(_activate_scalar(1e300, function), finite)
            self.assertTrue(math.isnan(_activate_scalar(math.nan, function)))

    def test_alias_and_padding(self) -> None:
        """Sigmoid zero is half, but tail lanes must remain zero, even in place."""
        for function in ActivationKind:
            state = CentDeviceState(hardware=hardware())
            address = CentSharedBufferAddress(slot=0)
            state.write_shared_buffer(address, (-2.0, 0.0, 3.0, 999.0))
            command = ReferenceActivation(
                source=address, destination=address, value_count=3, function=function
            )
            execute_program(
                CentProgram(hardware=hardware(), instructions=(command,)), state
            )
            expected = (*(_activate_scalar(x, function) for x in (-2.0, 0.0, 3.0)), 0.0)
            self.assertEqual(state.read_shared_buffer(address, slot_count=1), expected)

    def test_softmax_stability_masking_and_special_values(self) -> None:
        """Mask lanes receive zero; rows without a finite normalizer are rejected."""
        expected = (1 / (1 + math.exp(-1)), math.exp(-1) / (1 + math.exp(-1)), 0.0)
        self.assertEqual(_softmax_row((1000.0, 999.0, -math.inf)), expected)
        for row in (
            (-math.inf, -math.inf),
            (math.inf, 1.0, math.inf, -math.inf),
            (0.0, math.nan, -math.inf),
        ):
            with self.assertRaises(ValueError):
                _softmax_row(row)
        self.assertEqual(_softmax_row((1e300, 1e300)), (0.5, 0.5))
        with self.assertRaises(ValueError):
            _softmax_row(())

    def test_softmax_rows_alias_and_final_padding(self) -> None:
        """Three-wide rows are adjacent scalars, not separately padded slots."""
        state = CentDeviceState(hardware=hardware())
        address = CentSharedBufferAddress(slot=0)
        state.write_shared_buffer(
            address, (0.0, 0.0, -math.inf, 1000.0, 1000.0, 1000.0, 99.0, 99.0)
        )
        command = ReferenceSoftmax(
            source=address, destination=address, row_count=2, row_width=3
        )
        execute_program(
            CentProgram(hardware=hardware(), instructions=(command,)), state
        )
        self.assertEqual(
            state.read_shared_buffer(address, slot_count=2),
            (0.5, 0.5, 0.0, 1 / 3, 1 / 3, 1 / 3, 0.0, 0.0),
        )

    def test_validation_export_and_uninitialized_atomicity(self) -> None:
        """Reject invalid spans and physical export; a failed read leaves output intact."""
        source = CentSharedBufferAddress(slot=0)
        destination = CentSharedBufferAddress(slot=2)
        activation = ReferenceActivation(
            source=source,
            destination=destination,
            value_count=3,
            function=ActivationKind.RELU,
        )
        softmax = ReferenceSoftmax(
            source=source, destination=destination, row_count=2, row_width=3
        )
        for command in (
            replace(activation, value_count=0),
            replace(activation, value_count=33),
            replace(softmax, row_count=0),
            replace(softmax, row_width=-1),
            replace(softmax, row_count=11),
        ):
            with self.assertRaises(ValueError):
                CentProgram(hardware=hardware(), instructions=(command,))
        for command in (activation, softmax):
            for renderer in (render_instruction, render_aim_instruction):
                with self.assertRaisesRegex(ValueError, "no physical"):
                    renderer(command)
            state = CentDeviceState(hardware=hardware())
            state.write_shared_buffer(
                destination, (7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0)
            )
            with self.assertRaises(CentUninitializedReadError):
                execute_program(
                    CentProgram(hardware=hardware(), instructions=(command,)), state
                )
            self.assertEqual(
                state.read_shared_buffer(destination, slot_count=2),
                (7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0),
            )

    def test_foreign_numeric_policy_rejected_before_mutation(self) -> None:
        """Nonlinear Python arithmetic cannot silently substitute for another policy."""

        class ForeignPolicy:
            """Supply basic numeric protocol methods without reference root semantics."""

            def store(self, value: float) -> float:
                """Store unchanged."""
                return value

            def add(self, left: float, right: float) -> float:
                """Add unchanged."""
                return left + right

            def multiply(self, left: float, right: float) -> float:
                """Multiply unchanged."""
                return left * right

        address = CentSharedBufferAddress(slot=0)
        for command in (
            ReferenceActivation(
                source=address,
                destination=address,
                value_count=3,
                function=ActivationKind.SILU,
            ),
            ReferenceSoftmax(
                source=address, destination=address, row_count=1, row_width=3
            ),
        ):
            state = CentDeviceState(hardware=hardware(), numeric=ForeignPolicy())
            program = CentProgram(
                hardware=hardware(),
                instructions=(
                    ReferenceFill(destination=address, value_count=3),
                    command,
                ),
            )
            with self.assertRaises(CentUnsupportedSemanticsError):
                execute_program(program, state)
            with self.assertRaises(CentUninitializedReadError):
                state.read_shared_buffer(address, slot_count=1)

    def test_late_softmax_row_failure_is_atomic(self) -> None:
        """An invalid second row cannot overwrite a valid first row or its input."""
        address = CentSharedBufferAddress(slot=0)
        for invalid in ((-math.inf,) * 3, (0.0, math.inf, 0.0), (0.0, math.nan, 0.0)):
            state = CentDeviceState(hardware=hardware())
            original = (1.0, 2.0, 3.0, *invalid, 99.0, 99.0)
            state.write_shared_buffer(address, original)
            command = ReferenceSoftmax(
                source=address, destination=address, row_count=2, row_width=3
            )
            with self.assertRaises(CentExecutionFault):
                execute_program(
                    CentProgram(hardware=hardware(), instructions=(command,)), state
                )
            observed = state.read_shared_buffer(address, slot_count=2)
            for expected, actual in zip(original, observed, strict=True):
                if math.isnan(expected):
                    self.assertTrue(math.isnan(actual))
                else:
                    self.assertEqual(actual, expected)

    def test_exact_gelu_preserves_small_negative_tail(self) -> None:
        """The erfc form keeps a tiny tail that one-plus-erf cancels to zero."""
        expected = -5.0 * math.erfc(10.0 / math.sqrt(2.0))
        self.assertLess(expected, 0.0)
        self.assertEqual(_activate_scalar(-10.0, ActivationKind.GELU_ERF), expected)

    def test_reference_subclass_converts_results_and_preserves_atomicity(self) -> None:
        """Custom result conversion is honored without partially committing a row."""

        class RejectPositive(ReferenceMathSemantics):
            """Reject positive stored results to expose deferred writes."""

            def store(self, value: float) -> float:
                """Reject positive outputs; permit negative input initialization."""
                if value > 0.0:
                    raise ValueError("positive result rejected")
                return value

        state = CentDeviceState(hardware=hardware(), numeric=RejectPositive())
        address = CentSharedBufferAddress(slot=0)
        state.write_shared_buffer(address, (-1.0, -2.0, -3.0, 0.0))
        command = ReferenceActivation(
            source=address,
            destination=address,
            value_count=3,
            function=ActivationKind.SIGMOID,
        )
        with self.assertRaises(CentExecutionFault):
            execute_program(
                CentProgram(hardware=hardware(), instructions=(command,)), state
            )
        self.assertEqual(
            state.read_shared_buffer(address, slot_count=1), (-1.0, -2.0, -3.0, 0.0)
        )
