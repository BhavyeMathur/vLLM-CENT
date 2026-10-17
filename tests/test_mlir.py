"""Check native MLIR import/export, verification, optimization, and execution."""

import importlib.util
import unittest
from dataclasses import replace
from unittest.mock import patch

from vllm_cent.ir_mlir import from_mlir, optimize_mlir, to_mlir
from vllm_cent.pipeline.ir import (
    ScalarType,
    SemanticGraph,
    TensorType,
    Value,
)
from vllm_cent.pipeline.operations import (
    AddOperation,
    MultiplyOperation,
    RmsNormOperation,
)


def _graph() -> SemanticGraph:
    """Build sum = left + right with a model-block label to preserve on export.

    Returns:
        One addition on three-element F32 tensors, labelled decoder.0.
    """
    tensor = TensorType((3,), ScalarType.F32)
    return SemanticGraph(
        inputs=(Value("left", tensor), Value("right", tensor)),
        operations=(
            AddOperation(
                name="residual",
                inputs=("left", "right"),
                output=Value("sum", tensor),
                block="decoder.0",
            ),
        ),
        outputs=("sum",),
    )


class MlirDependencyTests(unittest.TestCase):
    """Keep native MLIR an explicit, optional compiler dependency."""

    def test_missing_dependency_is_actionable(self) -> None:
        """Report the toolchain requirement instead of substituting text parsing."""
        with patch(
            "vllm_cent.ir_mlir.bridge.import_module",
            side_effect=ModuleNotFoundError("iree"),
        ):
            with self.assertRaisesRegex(RuntimeError, "iree-base-compiler"):
                to_mlir(_graph())


@unittest.skipUnless(importlib.util.find_spec("iree"), "optional native MLIR toolchain")
class NativeMlirTests(unittest.TestCase):
    """Exercise the actual native parser, builders, verifier, and pass manager."""

    def test_roundtrip_preserves_typed_graph(self) -> None:
        """Keep shapes, input connections, names, and the model-block label through text export."""
        text = to_mlir(_graph())
        self.assertIn("arith.addf", text)
        self.assertIn("tensor<3xf32>", text)
        self.assertEqual(from_mlir(text), _graph())

    def test_multiplication_roundtrip_and_fastmath_rejection(self) -> None:
        """Preserve multiplication and reject unsupported floating-point promises."""
        graph = _graph()
        graph = replace(
            graph,
            operations=(
                MultiplyOperation(
                    name="product",
                    inputs=("left", "right"),
                    output=graph.operations[0].output,
                    block="decoder.0",
                ),
            ),
        )
        text = to_mlir(graph)
        self.assertIn("arith.mulf", text)
        self.assertEqual(from_mlir(text), graph)
        self.assertEqual(from_mlir(optimize_mlir(text)), graph)
        relaxed = text.replace(
            "arith.mulf %arg0, %arg1", "arith.mulf %arg0, %arg1 fastmath<fast>"
        )
        with self.assertRaisesRegex(ValueError, "fastmath"):
            from_mlir(relaxed)

    def test_native_pass_pipeline_preserves_semantics(self) -> None:
        """Run native simplification passes and recover the same single-addition graph."""
        self.assertEqual(from_mlir(optimize_mlir(to_mlir(_graph()))), _graph())

    def test_import_without_cent_metadata(self) -> None:
        """Give ordinary MLIR inputs and results stable names when CENT attributes are absent."""
        text = """module {
          func.func @main(%a: tensor<3xf32>, %b: tensor<3xf32>) -> tensor<3xf32> {
            %sum = arith.addf %a, %b : tensor<3xf32>
            return %sum : tensor<3xf32>
          }
        }"""
        graph = from_mlir(text)
        self.assertEqual(graph.inputs[0].name, "arg0")
        self.assertEqual(graph.operations[0].inputs, ("arg0", "arg1"))
        self.assertEqual(graph.outputs, ("value0",))

    def test_partial_metadata_keeps_generated_names_unique(self) -> None:
        """Reserve later explicit names before naming unannotated inputs and results."""
        text = """module {
          func.func @main(%a: tensor<3xf32>) -> tensor<3xf32> {
            %x = arith.addf %a, %a : tensor<3xf32>
            %y = arith.addf %x, %a {cent.output = "arg0", cent.name = "add0"}
              : tensor<3xf32>
            return %y : tensor<3xf32>
          }
        }"""
        graph = from_mlir(text)
        self.assertEqual(graph.inputs[0].name, "arg0_0")
        self.assertEqual(graph.operations[0].name, "add0_0")
        self.assertEqual(graph.operations[1].name, "add0")
        self.assertEqual(graph.outputs, ("arg0",))
        annotated = to_mlir(_graph()).replace(
            '["left", "right"]', '["value0", "right"]'
        )
        annotated = annotated.replace(', cent.output = "sum"', "")
        self.assertEqual(from_mlir(annotated).outputs, ("value0_0",))

    def test_rejects_explicit_rounding_metadata(self) -> None:
        """Do not erase rounding choices that the semantic arithmetic cannot retain."""
        # Pinned IREE predates native arith rounding-mode attributes. A plain
        # attribute exercises the import guard without depending on a newer parser.
        text = to_mlir(_graph()).replace(
            'cent.name = "residual"',
            'roundingmode = "toward_zero", cent.name = "residual"',
        )
        with self.assertRaisesRegex(ValueError, "rounding"):
            from_mlir(text)

    def test_rejects_unregistered_operation(self) -> None:
        """Reject an operation family that the native MLIR context does not know."""
        with self.assertRaises(ValueError):
            from_mlir('module { "imaginary.add"() : () -> () }')

    def test_rejects_registered_but_unsupported_operation(self) -> None:
        """Reject subtraction until the graph and backend support it."""
        text = to_mlir(_graph()).replace("arith.addf", "arith.subf")
        with self.assertRaisesRegex(ValueError, "arith.subf"):
            from_mlir(text)

    def test_rejects_invalid_tensor_types(self) -> None:
        """Reject unsupported element types after native parsing succeeds."""
        text = to_mlir(_graph()).replace("f32", "f64")
        with self.assertRaisesRegex(ValueError, "f32"):
            from_mlir(text)

    def test_native_verifier_rejects_mismatched_return(self) -> None:
        """Reject a function returning three elements when its declared result has four."""
        text = to_mlir(_graph()).replace("-> tensor<3xf32>", "-> tensor<4xf32>")
        with self.assertRaises(ValueError):
            from_mlir(text)

    def test_native_canonicalization_removes_dead_addition(self) -> None:
        """Remove the addition when the function returns only an unchanged input."""
        graph = replace(_graph(), outputs=("left",))
        optimized = optimize_mlir(to_mlir(graph))
        self.assertNotIn("arith.addf", optimized)
        self.assertEqual(from_mlir(optimized), replace(graph, operations=()))

    def test_rejects_fastmath_assumptions(self) -> None:
        """Do not silently erase floating-point assumptions during import."""
        text = to_mlir(_graph()).replace(
            "arith.addf %arg0, %arg1", "arith.addf %arg0, %arg1 fastmath<fast>"
        )
        with self.assertRaisesRegex(ValueError, "fastmath"):
            from_mlir(text)

    def test_rejects_unsupported_export(self) -> None:
        """Reject normalization until native export and device lowering support it."""
        graph = _graph()
        operation = RmsNormOperation(
            name="norm",
            inputs=("left", "right"),
            output=graph.operations[0].output,
            epsilon=1e-5,
        )
        with self.assertRaisesRegex(ValueError, "addition and multiplication"):
            to_mlir(replace(graph, operations=(operation,)))

    def test_rejects_dynamic_and_scalar_mlir_types(self) -> None:
        """Reject unknown tensor dimensions and bare scalar types instead of tensor types."""
        for tensor_type in ("tensor<?xf32>", "f32"):
            with self.subTest(tensor_type=tensor_type):
                text = to_mlir(_graph()).replace("tensor<3xf32>", tensor_type)
                with self.assertRaisesRegex(ValueError, "static f32|ranked tensors"):
                    from_mlir(text)

    def test_rejects_missing_function(self) -> None:
        """An empty module has no executable semantic graph."""
        with self.assertRaisesRegex(ValueError, "exactly one"):
            from_mlir("module {}")

    def test_rejects_function_declaration(self) -> None:
        """Reject a function name and signature with no body to compile."""
        with self.assertRaisesRegex(ValueError, "body block"):
            from_mlir("module {func.func private @external(tensor<3xf32>)}")

    def test_rejects_inconsistent_provenance(self) -> None:
        """Require one saved input name for each actual MLIR function argument."""
        text = to_mlir(_graph()).replace('["left", "right"]', '["left"]')
        with self.assertRaisesRegex(ValueError, "match function arguments"):
            from_mlir(text)

    def test_optimization_rejects_invalid_mlir(self) -> None:
        """Native pass execution requires verified input assembly."""
        with self.assertRaisesRegex(ValueError, "Native MLIR pass"):
            optimize_mlir("not a module")


@unittest.skipUnless(importlib.util.find_spec("iree"), "optional native MLIR toolchain")
class MlirExecutionTests(unittest.TestCase):
    """Exercise native MLIR through every CENT stage to numerical execution."""

    def test_mlir_and_python_paths_match(self) -> None:
        """Compile the same graph from MLIR and Python, then verify the resulting sum."""
        from vllm_cent.cent import CentHardwareSpec
        from vllm_cent.pipeline import CentTargetProfile, compile_graph, compile_mlir
        from vllm_cent.runtime import CentNamedScalars

        # Three values occupy two two-lane slots. The one-slot ACC limit makes
        # both frontends exercise the same two-command addition during codegen.
        target = CentTargetProfile(
            name="mlir-proof",
            max_accumulate_slots=1,
            hardware=CentHardwareSpec(
                num_channels=1,
                num_banks=4,
                dram_rows=1,
                dram_columns=4,
                burst_length=2,
                accumulator_slots_per_bank=1,
                sigmoid_activation_function_id=0,
                shared_buffer_slots=8,
            ),
        )
        artifact = compile_mlir(to_mlir(_graph()), target)
        self.assertEqual(artifact, compile_graph(_graph(), target))
        self.assertEqual(
            artifact.execute(
                (
                    CentNamedScalars(name="left", values=(1.0, -2.0, 3.0)),
                    CentNamedScalars(name="right", values=(10.0, 20.0, -30.0)),
                )
            ),
            (CentNamedScalars(name="sum", values=(11.0, 18.0, -27.0)),),
        )

    def test_mlir_multiply_add_chain_executes(self) -> None:
        """Run native multiplication through DRAM staging and then Shared Buffer addition."""
        from vllm_cent.pipeline import compile_graph, compile_mlir
        from vllm_cent.runtime import CentNamedScalars
        from tests.test_pipeline import target

        graph = _graph()
        product = Value("product", graph.inputs[0].type)
        graph = replace(
            graph,
            operations=(
                MultiplyOperation(
                    name="multiply", inputs=("left", "right"), output=product
                ),
                AddOperation(
                    name="add",
                    inputs=("product", "left"),
                    output=graph.operations[0].output,
                ),
            ),
            outputs=("sum", "left"),
        )
        artifact = compile_mlir(to_mlir(graph), target())
        self.assertEqual(artifact, compile_graph(graph, target()))
        self.assertEqual(
            artifact.execute(
                (
                    CentNamedScalars(name="left", values=(1.0, -2.0, 3.0)),
                    CentNamedScalars(name="right", values=(10.0, 20.0, -30.0)),
                )
            ),
            (
                CentNamedScalars(name="sum", values=(11.0, -42.0, -87.0)),
                CentNamedScalars(name="left", values=(1.0, -2.0, 3.0)),
            ),
        )
