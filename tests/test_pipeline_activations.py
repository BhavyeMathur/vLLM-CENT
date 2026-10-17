"""Check nonlinear tensor operations against independent scalar arithmetic."""

import math
import unittest
from dataclasses import replace

from vllm_cent.pipeline import ExecutionContract, compile_graph
from vllm_cent.pipeline.ir import SemanticGraph, TensorType, Value
from vllm_cent.pipeline.operations import (
    ActivationKind,
    ActivationOperation,
    SoftmaxOperation,
)
from vllm_cent.runtime import CentNamedScalars
from tests.test_pipeline_gemv_norm import target


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
                    tensor = TensorType((5,))
                    graph = SemanticGraph(
                        inputs=(Value("x", tensor),),
                        operations=(
                            ActivationOperation(
                                name="act",
                                inputs=("x",),
                                output=Value("out", tensor),
                                function=function,
                            ),
                        ),
                        outputs=("out", "x") if preserve else ("out",),
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
        tensor = TensorType((2, 3))
        graph = SemanticGraph(
            inputs=(Value("x", tensor),),
            operations=(
                SoftmaxOperation(
                    name="softmax", inputs=("x",), output=Value("out", tensor)
                ),
            ),
            outputs=("out", "x"),
        )
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
        tensor = TensorType((3,))
        graph = SemanticGraph(
            inputs=(Value("x", tensor),),
            operations=(
                SoftmaxOperation(
                    name="softmax", inputs=("x",), output=Value("out", tensor)
                ),
            ),
            outputs=("out",),
        )
        with self.assertRaisesRegex(ValueError, "REFERENCE"):
            compile_graph(
                graph, replace(target(), execution_contract=ExecutionContract.STANDARD)
            )
        with self.assertRaises(ValueError):
            SoftmaxOperation(
                name="bad", inputs=("x",), output=Value("out", TensorType(()))
            ).verify((TensorType(()),))

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
            tensor = TensorType(shape)
            graph = SemanticGraph(
                inputs=(Value("x", tensor),),
                operations=(
                    SoftmaxOperation(
                        name="softmax", inputs=("x",), output=Value("out", tensor)
                    ),
                ),
                outputs=("out",),
            )
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

        tensor = TensorType((3,))
        operation = ActivationOperation(
            name="act",
            inputs=("x",),
            output=Value("out", tensor),
            function=ActivationKind.GELU_ERF,
        )
        graph = SemanticGraph(
            inputs=(Value("x", tensor),), operations=(operation,), outputs=("out",)
        )
        for types in ((), (TensorType((2,)),)):
            with self.assertRaises(ValueError):
                operation.verify(types)
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
