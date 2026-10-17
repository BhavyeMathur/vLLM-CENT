"""Verify the small reference services used between attention MAC stages."""

import math
import unittest
from dataclasses import replace

from vllm_cent.cent import CentHardwareSpec, CentProgram, CentSharedBufferAddress
from vllm_cent.cent.aim import render_aim_instruction
from vllm_cent.cent.reference import ReferenceAttentionScores, ReferenceTranspose
from vllm_cent.cent.render import render_instruction
from vllm_cent.simulator import CentExecutionFault, CentUninitializedReadError
from vllm_cent.simulator.execution import execute_program
from vllm_cent.simulator.state import CentDeviceState


def hardware() -> CentHardwareSpec:
    """Provide four-lane slots so six-value transposes need a padded tail."""
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


class ReferenceAttentionTests(unittest.TestCase):
    """Check offsets, mask precedence, padding, and failed-instruction rollback."""

    def test_transpose_offset_and_alias_snapshot(self) -> None:
        """Transpose a two-by-three slice from lane one before overwriting it."""
        state = CentDeviceState(hardware=hardware())
        address = CentSharedBufferAddress(slot=0)
        state.write_shared_buffer(address, (99.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 99.0))
        command = ReferenceTranspose(
            source=address,
            destination=address,
            source_offset=1,
            row_count=2,
            column_count=3,
        )
        summary = execute_program(
            CentProgram(hardware=hardware(), instructions=(command,)), state
        )
        self.assertEqual(
            state.read_shared_buffer(address, slot_count=2),
            (1.0, 4.0, 2.0, 5.0, 3.0, 6.0, 0.0, 0.0),
        )
        self.assertEqual(summary.executed_instruction_count, 1)

    def test_scale_add_then_causal_mask_with_mask_offset(self) -> None:
        """Scale first, then add; the inclusive causal limit suppresses later keys."""
        state = CentDeviceState(hardware=hardware())
        address = CentSharedBufferAddress(slot=0)
        mask = CentSharedBufferAddress(slot=3)
        state.write_shared_buffer(address, (2.0, 4.0, math.inf, 123.0))
        state.write_shared_buffer(mask, (99.0, 3.0, -math.inf, 5.0))
        command = ReferenceAttentionScores(
            source=address,
            destination=address,
            value_count=3,
            scale=0.5,
            mask=mask,
            mask_offset=1,
            causal_limit=1,
        )
        execute_program(
            CentProgram(hardware=hardware(), instructions=(command,)), state
        )
        self.assertEqual(
            state.read_shared_buffer(address, slot_count=1),
            (4.0, -math.inf, -math.inf, 0.0),
        )

    def test_additive_mask_suppresses_infinite_score(self) -> None:
        """A masked position becomes negative infinity without evaluating inf+-inf."""
        state = CentDeviceState(hardware=hardware())
        address = CentSharedBufferAddress(slot=0)
        mask = CentSharedBufferAddress(slot=1)
        state.write_shared_buffer(address, (math.inf, 2.0, 3.0, 0.0))
        state.write_shared_buffer(mask, (-math.inf, 1.0, 0.0, 0.0))
        command = ReferenceAttentionScores(
            source=address, destination=address, value_count=3, scale=2.0, mask=mask
        )
        execute_program(
            CentProgram(hardware=hardware(), instructions=(command,)), state
        )
        self.assertEqual(
            state.read_shared_buffer(address, slot_count=1), (-math.inf, 5.0, 6.0, 0.0)
        )

    def test_no_mask_and_no_causal_limit(self) -> None:
        """Ordinary scaling preserves logical order and zeroes padding."""
        state = CentDeviceState(hardware=hardware())
        address = CentSharedBufferAddress(slot=0)
        state.write_shared_buffer(address, (1.0, -2.0, 3.0, 9.0))
        command = ReferenceAttentionScores(
            source=address, destination=address, value_count=3, scale=-2.0
        )
        execute_program(
            CentProgram(hardware=hardware(), instructions=(command,)), state
        )
        self.assertEqual(
            state.read_shared_buffer(address, slot_count=1), (-2.0, 4.0, -6.0, 0.0)
        )

    def test_bad_mask_is_rejected_even_when_causally_hidden(self) -> None:
        """Validate every supplied mask value before committing any score output."""
        for bad in (math.inf, math.nan):
            state = CentDeviceState(hardware=hardware())
            address = CentSharedBufferAddress(slot=0)
            mask = CentSharedBufferAddress(slot=1)
            state.write_shared_buffer(address, (1.0, 2.0, 3.0, 4.0))
            state.write_shared_buffer(mask, (0.0, 0.0, bad, 0.0))
            command = ReferenceAttentionScores(
                source=address,
                destination=address,
                value_count=3,
                scale=1.0,
                mask=mask,
                causal_limit=0,
            )
            with self.assertRaises(CentExecutionFault):
                execute_program(
                    CentProgram(hardware=hardware(), instructions=(command,)), state
                )
            self.assertEqual(
                state.read_shared_buffer(address, slot_count=1), (1.0, 2.0, 3.0, 4.0)
            )

    def test_bounds_export_and_uninitialized_mask(self) -> None:
        """Reject malformed spans before execution and missing mask values on read."""
        address = CentSharedBufferAddress(slot=0)
        transpose = ReferenceTranspose(
            source=address,
            destination=address,
            source_offset=0,
            row_count=2,
            column_count=3,
        )
        scores = ReferenceAttentionScores(
            source=address, destination=address, value_count=3, scale=1.0
        )
        invalid = (
            replace(transpose, row_count=0),
            replace(transpose, column_count=-1),
            replace(transpose, source_offset=-1),
            replace(transpose, source_offset=30),
            replace(scores, scale=math.inf),
            replace(scores, value_count=0),
            replace(scores, mask_offset=-1),
            replace(scores, mask_offset=1),
            replace(scores, causal_limit=-1),
            replace(scores, mask=address, mask_offset=31),
        )
        for command in invalid:
            with self.subTest(command=command), self.assertRaises(ValueError):
                CentProgram(hardware=hardware(), instructions=(command,))
        for command in (transpose, scores):
            for renderer in (render_instruction, render_aim_instruction):
                with self.assertRaisesRegex(ValueError, "no physical"):
                    renderer(command)
        state = CentDeviceState(hardware=hardware())
        state.write_shared_buffer(address, (1.0, 2.0, 3.0, 4.0))
        with self.assertRaises(CentUninitializedReadError):
            execute_program(
                CentProgram(
                    hardware=hardware(),
                    instructions=(
                        replace(scores, mask=CentSharedBufferAddress(slot=3)),
                    ),
                ),
                state,
            )
        self.assertEqual(
            state.read_shared_buffer(address, slot_count=1), (1.0, 2.0, 3.0, 4.0)
        )

    def test_zero_scale_and_unbounded_causal_prefix(self) -> None:
        """A limit beyond the keys permits all positions; zero scale is explicit."""
        state = CentDeviceState(hardware=hardware())
        address = CentSharedBufferAddress(slot=0)
        state.write_shared_buffer(address, (1e300, -1e300, 3.0, 99.0))
        command = ReferenceAttentionScores(
            source=address,
            destination=address,
            value_count=3,
            scale=0.0,
            causal_limit=100,
        )
        execute_program(
            CentProgram(hardware=hardware(), instructions=(command,)), state
        )
        self.assertEqual(
            state.read_shared_buffer(address, slot_count=1), (0.0, 0.0, 0.0, 0.0)
        )

    def test_offsets_skip_uninitialized_prior_slots(self) -> None:
        """Selecting a later matrix or mask does not read earlier tensor slots."""
        state = CentDeviceState(hardware=hardware())
        source = CentSharedBufferAddress(slot=0)
        destination = CentSharedBufferAddress(slot=4)
        state.write_shared_buffer(CentSharedBufferAddress(slot=2), (1.0, 2.0, 3.0, 4.0))
        transpose = ReferenceTranspose(
            source=source,
            destination=destination,
            source_offset=8,
            row_count=2,
            column_count=2,
        )
        scores = ReferenceAttentionScores(
            source=destination,
            destination=destination,
            value_count=4,
            scale=1.0,
            mask=source,
            mask_offset=8,
        )
        execute_program(
            CentProgram(hardware=hardware(), instructions=(transpose, scores)), state
        )
        self.assertEqual(
            state.read_shared_buffer(destination, slot_count=1), (2.0, 5.0, 5.0, 8.0)
        )

    def test_numeric_policy_is_used_and_rejection_is_atomic(self) -> None:
        """Scaling uses the supplied multiply/add rules without a nonlinear gate."""

        class CheckedArithmetic:
            """Reject a later product while ordinary transfers remain valid."""

            def store(self, value: float) -> float:
                """Keep host and mask constants unchanged."""
                return value

            def add(self, left: float, right: float) -> float:
                """Add an extra one so using this method is observable."""
                return left + right + 1.0

            def multiply(self, left: float, right: float) -> float:
                """Reject the last test score; scale every other value."""
                if left == 3.0:
                    raise ValueError("third product rejected")
                return left * right

        state = CentDeviceState(hardware=hardware(), numeric=CheckedArithmetic())
        address = CentSharedBufferAddress(slot=0)
        mask = CentSharedBufferAddress(slot=1)
        state.write_shared_buffer(address, (1.0, 2.0, 3.0, 99.0))
        state.write_shared_buffer(mask, (0.0, 0.0, 0.0, 0.0))
        command = ReferenceAttentionScores(
            source=address, destination=address, value_count=3, scale=2.0, mask=mask
        )
        with self.assertRaises(CentExecutionFault):
            execute_program(
                CentProgram(hardware=hardware(), instructions=(command,)), state
            )
        self.assertEqual(
            state.read_shared_buffer(address, slot_count=1), (1.0, 2.0, 3.0, 99.0)
        )
        # Hiding key two bypasses the rejected multiply and commits the prefix.
        execute_program(
            CentProgram(
                hardware=hardware(), instructions=(replace(command, causal_limit=1),)
            ),
            state,
        )
        self.assertEqual(
            state.read_shared_buffer(address, slot_count=1), (3.0, 5.0, -math.inf, 0.0)
        )

    def test_immediate_scale_converted_once_and_rejection_is_atomic(self) -> None:
        """Immediate scale enters stored form once before any score multiplication."""

        class ImmediatePolicy:
            """Quantize one immediate or reject it while recording conversions."""

            def __init__(self, reject: bool) -> None:
                """Select rejection and start an empty conversion history."""
                self.reject = reject
                self.conversions: list[float] = []
                self.multipliers: list[float] = []

            def store(self, value: float) -> float:
                """Map the scale from one quarter to one half, or reject it."""
                self.conversions.append(value)
                if value == 0.25:
                    if self.reject:
                        raise ValueError("immediate conversion rejected")
                    return 0.5
                return value

            def add(self, left: float, right: float) -> float:
                """Add operands already in stored form."""
                return left + right

            def multiply(self, left: float, right: float) -> float:
                """Record the stored multiplier without converting it again."""
                self.multipliers.append(right)
                return left * right

        for reject in (False, True):
            policy = ImmediatePolicy(reject)
            state = CentDeviceState(hardware=hardware(), numeric=policy)
            address = CentSharedBufferAddress(slot=0)
            state.write_shared_buffer(address, (2.0, 4.0, 6.0, 99.0))
            policy.conversions.clear()
            command = ReferenceAttentionScores(
                source=address, destination=address, value_count=3, scale=0.25
            )
            program = CentProgram(hardware=hardware(), instructions=(command,))
            if reject:
                with self.assertRaisesRegex(CentExecutionFault, "immediate conversion"):
                    execute_program(program, state)
                self.assertEqual(
                    state.read_shared_buffer(address, slot_count=1),
                    (2.0, 4.0, 6.0, 99.0),
                )
                self.assertEqual(policy.multipliers, [])
            else:
                execute_program(program, state)
                self.assertEqual(
                    state.read_shared_buffer(address, slot_count=1),
                    (1.0, 2.0, 3.0, 0.0),
                )
                self.assertEqual(policy.multipliers, [0.5, 0.5, 0.5])
            self.assertEqual(policy.conversions.count(0.25), 1)
