"""Check nonlinear tensor operations against independent scalar arithmetic."""

import math
import unittest
from dataclasses import replace

from tests.test_pipeline_gemv_norm import target
from vllm_cent.activations import ActivationKind
from vllm_cent.ir import ActivationOp, DType, Function, SoftmaxOp, TensorType, Value
from vllm_cent.pipeline import ExecutionContract, compile_graph
from vllm_cent.runtime import CentNamedScalars


def _fp32(name: str, *shape: int) -> Value:
    """Create a named fp32 function argument, the only format the pipeline runs.

    Args:
        name: Argument name; the compiled program's input of the same name.
        shape: Tensor dimensions, outermost first. No dimensions make a scalar.

    Returns:
        A new graph input value with the requested shape.
    """
    return Value(type=TensorType(shape=shape, dtype=DType.FP32), name=name)


def _softmax_graph(x: Value, *, preserve: bool) -> Function:
    """Apply softmax to one input and name the result ``out``.

    Args:
        x: Graph input normalized along its last dimension.
        preserve: Whether the function also returns ``x`` after ``out``, which
            keeps the input's storage alive and observable.

    Returns:
        Function computing ``out = softmax(x)``.
    """
    softmax = SoftmaxOp(operands=(x,), name_hint="out")
    return Function(
        name="main",
        operands=(x,),
        operations=(softmax,),
        results=(*softmax.results, x) if preserve else softmax.results,
    )


class ActivationTests(unittest.TestCase):
    """Exercise tail lanes, preserved inputs, and last-use storage reuse."""

    def test_activation_values_and_storage(self) -> None:
        """Both independent and reused output buffers implement every formula."""
        values = (-3.0, -1.0, 0.0, 1.0, 3.0)
        formulas = {
            ActivationKind.RELU: lambda x: max(0.0, x),
            ActivationKind.SIGMOID: lambda x: 1 / (1 + math.exp(-x)),
            ActivationKind.TANH: math.tanh,
            ActivationKind.SILU: lambda x: x / (1 + math.exp(-x)),
            ActivationKind.GELU_ERF: lambda x: (
                0.5 * x * (1 + math.erf(x / math.sqrt(2)))
            ),
            ActivationKind.GELU_TANH: lambda x: (
                0.5
                * x
                * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x**3)))
            ),
        }
        for function, formula in formulas.items():
            for preserve in (False, True):
                with self.subTest(function=function, preserve=preserve):
                    x = _fp32("x", 5)
                    activation = ActivationOp(
                        operands=(x,), function=function, name_hint="out"
                    )
                    graph = Function(
                        name="main",
                        operands=(x,),
                        operations=(activation,),
                        results=(
                            (*activation.results, x) if preserve else activation.results
                        ),
                    )
                    artifact = compile_graph(graph, target())
                    actual = artifact.execute(
                        (CentNamedScalars(name="x", values=values),)
                    )
                    for result, value in zip(actual[0].values, values, strict=True):
                        self.assertAlmostEqual(result, formula(value))
                    if preserve:
                        self.assertEqual(actual[1].values, values)
                    placements = {
                        item.value: item.storage
                        for item in artifact.target_ir.mapping.placements
                    }
                    self.assertEqual(placements["x"] == placements["out"], not preserve)

    def test_softmax_rows_are_stable_and_independent(self) -> None:
        """Odd-width rows start inside slots and large logits do not overflow."""
        graph = _softmax_graph(_fp32("x", 2, 3), preserve=True)
        values = (1000.0, 1001.0, 1002.0, -1000.0, -1002.0, -1004.0)
        artifact = compile_graph(graph, target())
        actual = artifact.execute((CentNamedScalars(name="x", values=values),))
        expected = []
        for row in (values[:3], values[3:]):
            exps = [math.exp(value - max(row)) for value in row]
            expected.extend(value / sum(exps) for value in exps)
        for result, wanted in zip(actual[0].values, expected, strict=True):
            self.assertAlmostEqual(result, wanted)
        self.assertEqual(actual[1].values, values)

    def test_reference_gate_and_semantic_shape(self) -> None:
        """Reject missing opt-in and softmax without a last dimension."""
        graph = _softmax_graph(_fp32("x", 3), preserve=False)
        with self.assertRaisesRegex(ValueError, "REFERENCE"):
            compile_graph(
                graph, replace(target(), execution_contract=ExecutionContract.STANDARD)
            )
        # A scalar has no last dimension to normalize over.
        with self.assertRaisesRegex(ValueError, "dimension"):
            SoftmaxOp(operands=(_fp32("x"),))

    def test_masks_singletons_and_in_place_rows(self) -> None:
        """Masks get zero probability and last-use rows can reuse input storage."""
        for shape, values, expected in (
            (
                (2, 3),
                (0.0, -math.inf, 0.0, -math.inf, 7.0, -math.inf),
                (0.5, 0.0, 0.5, 0.0, 1.0, 0.0),
            ),
            ((2, 1), (10000.0, -10000.0), (1.0, 1.0)),
        ):
            graph = _softmax_graph(_fp32("x", *shape), preserve=False)
            artifact = compile_graph(graph, target())
            placements = {
                item.value: item.storage
                for item in artifact.target_ir.mapping.placements
            }
            self.assertEqual(placements["x"], placements["out"])
            self.assertEqual(
                artifact.execute((CentNamedScalars(name="x", values=values),))[
                    0
                ].values,
                expected,
            )

    def test_invalid_activation_shapes_and_device_tampering(self) -> None:
        """Function selection survives legalization and cannot change after verification."""
        from vllm_cent.pipeline import compile_target
        from vllm_cent.pipeline.ir import CapabilityProfile, DeviceOperationKind

        x = _fp32("x", 3)
        operation = ActivationOp(
            operands=(x,), function=ActivationKind.GELU_ERF, name_hint="out"
        )
        graph = Function(
            name="main",
            operands=(x,),
            operations=(operation,),
            results=operation.results,
        )
        # An activation reads exactly one tensor.
        for operands in ((), (x, x)):
            with self.subTest(count=len(operands)), self.assertRaises(ValueError):
                ActivationOp(operands=operands, function=ActivationKind.GELU_ERF)
        with self.assertRaises(NotImplementedError):
            compile_graph(
                graph,
                target(),
                capabilities=CapabilityProfile(
                    frozenset({DeviceOperationKind.PNM_ADD})
                ),
            )
        artifact = compile_graph(graph, target())
        command = artifact.target_ir.commands[0]
        with self.assertRaises(ValueError):
            compile_target(
                replace(
                    artifact.target_ir,
                    commands=(replace(command, function=ActivationKind.GELU_TANH),),
                )
            )
