"""Check explicit reference GEMV and normalization against ordinary host arithmetic."""

import math
import unittest
from dataclasses import replace

from vllm_cent.cent import CentHardwareSpec
from vllm_cent.ir import DType, Function, LinearOp, RMSNormOp, TensorType, Value
from vllm_cent.pipeline import (
    AccumulatorPlacement,
    CentTargetProfile,
    ExecutionContract,
    compile_graph,
)
from vllm_cent.runtime import CentNamedScalars


def _fp32(name: str, *shape: int) -> Value:
    """Create a named fp32 function argument, the only format the pipeline runs.

    Args:
        name: Argument name; the compiled program's input of the same name.
        shape: Tensor dimensions, outermost first.

    Returns:
        A new graph input value with the requested shape.
    """
    return Value(type=TensorType(shape=shape, dtype=DType.FP32), name=name)


def target() -> CentTargetProfile:
    """Return a tiny explicit-reference target with two accumulator registers.

    Returns:
        Two-lane slots, four-value rows, and enough working buffer for tail tests.
    """
    return CentTargetProfile(
        name="reference",
        execution_contract=ExecutionContract.REFERENCE,
        max_accumulate_slots=2,
        hardware=CentHardwareSpec(
            num_channels=1,
            num_banks=4,
            dram_rows=1,
            dram_columns=4,
            burst_length=2,
            accumulator_slots_per_bank=2,
            sigmoid_activation_function_id=0,
            shared_buffer_slots=128,
        ),
    )


def linear_graph() -> Function:
    """Return a batched matrix-vector graph with odd widths and preserved inputs.

    Returns:
        Two five-value vectors multiplied by a three-row weight matrix. The
        ``(2, 3)`` result is named ``out``; that name also names the linear
        operation in mapping schedules and accumulator placements.
    """
    x, w = _fp32("x", 2, 5), _fp32("w", 3, 5)
    linear = LinearOp(operands=(x, w), name_hint="out")
    return Function(
        name="main",
        operands=(x, w),
        operations=(linear,),
        results=(*linear.results, x, w),
    )


class GemvNormTests(unittest.TestCase):
    """Require complete initialization, shape preservation, and explicit opt-in."""

    def test_gemv_crosses_rows_and_preserves_values(self) -> None:
        """Five input values cross a four-value row; three outputs leave an inactive bank."""
        x = (1.0, -2.0, 3.0, 4.0, -5.0, 2.0, 1.0, 0.0, -3.0, 4.0)
        w = (
            1.0,
            2.0,
            3.0,
            4.0,
            5.0,
            -1.0,
            0.0,
            2.0,
            -2.0,
            3.0,
            2.0,
            -1.0,
            0.0,
            1.0,
            -1.0,
        )
        expected = tuple(
            sum(x[b * 5 + k] * w[o * 5 + k] for k in range(5))
            for b in range(2)
            for o in range(3)
        )
        artifact = compile_graph(linear_graph(), target())
        self.assertEqual(
            artifact.execute(
                (
                    CentNamedScalars(name="x", values=x),
                    CentNamedScalars(name="w", values=w),
                )
            ),
            (
                CentNamedScalars(name="out", values=expected),
                CentNamedScalars(name="x", values=x),
                CentNamedScalars(name="w", values=w),
            ),
        )
        from vllm_cent.cent import WriteGlobalBuffer

        self.assertEqual(
            tuple(
                instruction.operation_size
                for instruction in artifact.executable.program.instructions
                if isinstance(instruction, WriteGlobalBuffer)
            ),
            (2, 1, 2, 1),
        )
        mapping = replace(
            artifact.target_ir.mapping,
            accumulators=(AccumulatorPlacement(operation="out", register=1),),
        )
        selected = compile_graph(linear_graph(), target(), mapping=mapping)
        self.assertEqual(
            selected.execute(
                (
                    CentNamedScalars(name="x", values=x),
                    CentNamedScalars(name="w", values=w),
                )
            ),
            artifact.execute(
                (
                    CentNamedScalars(name="x", values=x),
                    CentNamedScalars(name="w", values=w),
                )
            ),
        )
        self.assertNotEqual(
            selected.executable.program.instructions,
            artifact.executable.program.instructions,
        )

    def test_rms_uses_logical_width_gamma_and_epsilon(self) -> None:
        """Normalize each row independently, excluding padding from the mean."""
        epsilon = 0.25
        x_value, gamma_value = _fp32("x", 2, 5), _fp32("gamma", 5)
        norm = RMSNormOp(operands=(x_value, gamma_value), eps=epsilon, name_hint="out")
        graph = Function(
            name="main",
            operands=(x_value, gamma_value),
            operations=(norm,),
            results=(*norm.results, x_value),
        )
        artifact = compile_graph(graph, target())
        gamma = (1.0, -2.0, 0.5, 3.0, 0.0)
        for x in ((0.0,) * 10, (1.0, -2.0, 3.0, -4.0, 5.0, 2.0, 0.0, -1.0, 3.0, -2.0)):
            expected = []
            for row in (x[:5], x[5:]):
                scale = 1 / math.sqrt(sum(v * v for v in row) / 5 + epsilon)
                expected.extend(
                    value * scale * weight
                    for value, weight in zip(row, gamma, strict=True)
                )
            result = artifact.execute(
                (
                    CentNamedScalars(name="x", values=x),
                    CentNamedScalars(name="gamma", values=gamma),
                )
            )
            for actual, wanted in zip(result[0].values, expected, strict=True):
                self.assertAlmostEqual(actual, wanted)
            self.assertEqual(result[1], CentNamedScalars(name="x", values=x))

    def test_reference_compute_requires_opt_in_and_register_capacity(self) -> None:
        """Do not silently interpret missing physical accumulator/scalar contracts."""
        with self.assertRaisesRegex(ValueError, "reference|REFERENCE"):
            compile_graph(
                linear_graph(),
                replace(target(), execution_contract=ExecutionContract.STANDARD),
            )
        artifact = compile_graph(linear_graph(), target())
        for register in (-1, 2):
            with self.subTest(register=register), self.assertRaises(ValueError):
                compile_graph(
                    linear_graph(),
                    target(),
                    mapping=replace(
                        artifact.target_ir.mapping,
                        accumulators=(
                            AccumulatorPlacement(operation="out", register=register),
                        ),
                    ),
                )
        with self.assertRaises(ValueError):
            compile_graph(
                linear_graph(),
                target(),
                mapping=replace(artifact.target_ir.mapping, accumulators=()),
            )

    def test_global_buffer_limits_tiles_and_output_groups_continue(self) -> None:
        """A small Global Buffer limits input tiles; extra outputs start a new bank group."""
        x_value, w_value = _fp32("x", 1, 2, 5), _fp32("w", 7, 5)
        # The (1, 2, 5) input and (7, 5) weight infer a (1, 2, 7) result.
        linear = LinearOp(operands=(x_value, w_value), name_hint="out")
        graph = Function(
            name="main",
            operands=(x_value, w_value),
            operations=(linear,),
            results=linear.results,
        )
        device = target()
        device = replace(
            device, hardware=replace(device.hardware, global_buffer_columns=2)
        )
        x = tuple(float(i - 4) for i in range(10))
        w = tuple(float((i % 7) - 3) for i in range(35))
        expected = tuple(
            sum(x[batch * 5 + k] * w[row * 5 + k] for k in range(5))
            for batch in range(2)
            for row in range(7)
        )
        artifact = compile_graph(graph, device)
        self.assertEqual(
            artifact.execute(
                (
                    CentNamedScalars(name="x", values=x),
                    CentNamedScalars(name="w", values=w),
                )
            ),
            (CentNamedScalars(name="out", values=expected),),
        )
        from vllm_cent.cent import WriteGlobalBuffer

        # Two-value tiles fit the Global Buffer even though a DRAM row holds four.
        writes = tuple(
            instruction
            for instruction in artifact.executable.program.instructions
            if isinstance(instruction, WriteGlobalBuffer)
        )
        self.assertEqual(
            tuple(instruction.operation_size for instruction in writes), (1,) * 12
        )

    def test_workspace_counts_toward_capacity_and_cannot_be_tampered(self) -> None:
        """Graph tensors alone fit, but the two staging rows and readback need six more slots."""
        from vllm_cent.pipeline import compile_target

        artifact = compile_graph(linear_graph(), target())
        # Inputs occupy 5 + 8 slots, output 3, and working memory 2 + 2 + 2.
        with self.assertRaisesRegex(ValueError, "capacity"):
            compile_graph(
                linear_graph(),
                replace(
                    target(),
                    hardware=replace(target().hardware, shared_buffer_slots=21),
                ),
            )
        boundary = compile_graph(
            linear_graph(),
            replace(
                target(), hardware=replace(target().hardware, shared_buffer_slots=22)
            ),
        )
        self.assertEqual(boundary.target_ir.reference_workspace.slot_count, 6)
        with self.assertRaises(ValueError):
            compile_target(replace(artifact.target_ir, reference_workspace=None))

    def test_linear_never_overwrites_an_operand_mid_dot_product(self) -> None:
        """Equal input/output storage sizes do not make a matrix transform safely in-place."""
        from vllm_cent.pipeline import ValuePlacement

        x_value, w_value = _fp32("x", 5), _fp32("w", 5, 5)
        linear = LinearOp(operands=(x_value, w_value), name_hint="out")
        graph = Function(
            name="main",
            operands=(x_value, w_value),
            operations=(linear,),
            results=linear.results,
        )
        with self.assertRaisesRegex(ValueError, "live"):
            compile_graph(
                graph,
                target(),
                constraints=(
                    ValuePlacement(value="x", storage="input"),
                    ValuePlacement(value="out", storage="input"),
                ),
            )

    def test_short_vector_does_not_reserve_a_full_hardware_row(self) -> None:
        """A one-value dot product needs two staging slots, even on a wide device."""
        from vllm_cent.cent import WriteGlobalBuffer

        x_value, w_value = _fp32("x", 1), _fp32("w", 1, 1)
        linear = LinearOp(operands=(x_value, w_value), name_hint="out")
        graph = Function(
            name="main",
            operands=(x_value, w_value),
            operations=(linear,),
            results=linear.results,
        )
        device = replace(
            target(),
            hardware=replace(
                target().hardware,
                dram_columns=1024,
                global_buffer_columns=1024,
                shared_buffer_slots=7,
            ),
        )
        artifact = compile_graph(graph, device)
        # Three tensor slots plus two staging slots and two bank-readback slots.
        self.assertEqual(artifact.target_ir.reference_workspace.slot_count, 4)
        self.assertEqual(
            artifact.execute(
                (
                    CentNamedScalars(name="x", values=(3.0,)),
                    CentNamedScalars(name="w", values=(-2.0,)),
                )
            ),
            (CentNamedScalars(name="out", values=(-6.0,)),),
        )
        writes = tuple(
            instruction.operation_size
            for instruction in artifact.executable.program.instructions
            if isinstance(instruction, WriteGlobalBuffer)
        )
        self.assertEqual(writes, (1,))
