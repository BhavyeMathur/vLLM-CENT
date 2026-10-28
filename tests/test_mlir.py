"""Check native MLIR import/export, verification, optimization, and execution."""

import importlib.util
import unittest
from unittest.mock import patch

from vllm_cent.ir import (
    AddOp,
    DType,
    Function,
    MulOp,
    RMSNormOp,
    TensorType,
    Value,
    print_function,
    value_names,
)
from vllm_cent.ir_mlir import from_mlir, optimize_mlir, to_mlir
from vllm_cent.pipeline import CompiledArtifact


def _arguments() -> tuple[Value, Value]:
    """Create the two named three-element fp32 inputs shared by these graphs.

    Returns:
        ``left`` and ``right`` arguments, in that order.
    """
    tensor = TensorType(shape=(3,), dtype=DType.FP32)
    return Value(type=tensor, name="left"), Value(type=tensor, name="right")


def _graph() -> Function:
    """Build sum = left + right with a model-block label to preserve on export.

    Returns:
        Function ``main`` with one addition on three-element fp32 tensors. Its
        result is named ``decoder.0.sum``: the ``decoder.0`` prefix is the
        model-block label, and the whole name travels as ``cent.name``.
    """
    left, right = _arguments()
    residual = AddOp(operands=(left, right), name_hint="decoder.0.sum")
    return Function(
        name="main",
        operands=(left, right),
        operations=(residual,),
        results=residual.results,
    )


def _assert_same_artifact(
    test: unittest.TestCase, actual: CompiledArtifact, expected: CompiledArtifact
) -> None:
    """Check that two compilations produced the same program from equal sources.

    IR values compare by identity, so two separately built functions are never
    equal objects even when they describe the same calculation. Compare their
    printed text instead, then every other compiled field exactly.

    Args:
        test: Test case that reports the failed assertion.
        actual: Artifact compiled through the MLIR frontend.
        expected: Artifact compiled directly from the Python IR function.
    """
    actual_request = actual.target_ir.request
    expected_request = expected.target_ir.request
    test.assertEqual(
        print_function(actual_request.graph.source),
        print_function(expected_request.graph.source),
    )
    test.assertEqual(actual_request.graph.ranks, expected_request.graph.ranks)
    test.assertEqual(
        actual_request.graph.capabilities, expected_request.graph.capabilities
    )
    test.assertEqual(actual_request.target, expected_request.target)
    test.assertEqual(actual_request.constraints, expected_request.constraints)
    test.assertEqual(actual.target_ir.mapping, expected.target_ir.mapping)
    test.assertEqual(actual.target_ir.commands, expected.target_ir.commands)
    test.assertEqual(
        actual.target_ir.reference_workspace, expected.target_ir.reference_workspace
    )
    test.assertEqual(actual.version, expected.version)
    test.assertEqual(actual.inputs, expected.inputs)
    test.assertEqual(actual.outputs, expected.outputs)
    test.assertEqual(actual.executable, expected.executable)


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
        self.assertIn('cent.name = "decoder.0.sum"', text)
        self.assertIn('cent.input_names = ["left", "right"]', text)
        imported = from_mlir(text)
        self.assertEqual(print_function(imported), print_function(_graph()))
        self.assertEqual(imported.operations[0].name_hint, "decoder.0.sum")

    def test_multiplication_roundtrip_and_fastmath_rejection(self) -> None:
        """Preserve multiplication and reject unsupported floating-point promises."""
        left, right = _arguments()
        product = MulOp(operands=(left, right), name_hint="decoder.0.product")
        graph = Function(
            name="main",
            operands=(left, right),
            operations=(product,),
            results=product.results,
        )
        text = to_mlir(graph)
        self.assertIn("arith.mulf", text)
        self.assertEqual(print_function(from_mlir(text)), print_function(graph))
        self.assertEqual(
            print_function(from_mlir(optimize_mlir(text))), print_function(graph)
        )
        relaxed = text.replace(
            "arith.mulf %arg0, %arg1", "arith.mulf %arg0, %arg1 fastmath<fast>"
        )
        with self.assertRaisesRegex(ValueError, "fastmath"):
            from_mlir(relaxed)

    def test_native_pass_pipeline_preserves_semantics(self) -> None:
        """Run native simplification passes and recover the same single-addition graph."""
        self.assertEqual(
            print_function(from_mlir(optimize_mlir(to_mlir(_graph())))),
            print_function(_graph()),
        )

    def test_import_without_cent_metadata(self) -> None:
        """Give ordinary MLIR inputs and results stable names when CENT attributes are absent."""
        text = """module {
          func.func @main(%a: tensor<3xf32>, %b: tensor<3xf32>) -> tensor<3xf32> {
            %sum = arith.addf %a, %b : tensor<3xf32>
            return %sum : tensor<3xf32>
          }
        }"""
        graph = from_mlir(text)
        names = value_names(graph)
        self.assertEqual(
            tuple(names[value] for value in graph.operands), ("arg0", "arg1")
        )
        self.assertEqual(graph.operations[0].operands, graph.operands)
        self.assertEqual(graph.operations[0].name_hint, "value0")
        self.assertEqual(tuple(names[value] for value in graph.results), ("value0",))

    def test_partial_metadata_keeps_generated_names_unique(self) -> None:
        """Reserve later explicit names before naming unannotated inputs and results."""
        # The later results claim "value0" and "arg0", the names the importer
        # would otherwise generate for the first result and the input. Both
        # generated names therefore take the first free suffix, "_0".
        text = """module {
          func.func @main(%a: tensor<3xf32>) -> tensor<3xf32> {
            %x = arith.addf %a, %a : tensor<3xf32>
            %y = arith.addf %x, %a {cent.name = "value0"} : tensor<3xf32>
            %z = arith.addf %y, %a {cent.name = "arg0"} : tensor<3xf32>
            return %z : tensor<3xf32>
          }
        }"""
        graph = from_mlir(text)
        names = value_names(graph)
        self.assertEqual(names[graph.operands[0]], "arg0_0")
        self.assertEqual(
            [names[operation.results[0]] for operation in graph.operations],
            ["value0_0", "value0", "arg0"],
        )
        self.assertEqual(tuple(names[value] for value in graph.results), ("arg0",))
        # An explicit input name also reserves the generated result name.
        annotated = to_mlir(_graph()).replace(
            '["left", "right"]', '["value0", "right"]'
        )
        annotated = annotated.replace(' {cent.name = "decoder.0.sum"}', "")
        imported = from_mlir(annotated)
        self.assertEqual(
            tuple(value_names(imported)[value] for value in imported.results),
            ("value0_0",),
        )

    def test_rejects_explicit_rounding_metadata(self) -> None:
        """Do not erase rounding choices that the semantic arithmetic cannot retain."""
        # Pinned IREE predates native arith rounding-mode attributes. A plain
        # attribute exercises the import guard without depending on a newer parser.
        text = to_mlir(_graph()).replace(
            'cent.name = "decoder.0.sum"',
            'roundingmode = "toward_zero", cent.name = "decoder.0.sum"',
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
        graph = _graph()
        graph = Function(
            name=graph.name,
            operands=graph.operands,
            operations=graph.operations,
            results=graph.operands[:1],
        )
        optimized = optimize_mlir(to_mlir(graph))
        self.assertNotIn("arith.addf", optimized)
        self.assertEqual(
            print_function(from_mlir(optimized)),
            print_function(
                Function(
                    name=graph.name,
                    operands=graph.operands,
                    operations=(),
                    results=graph.operands[:1],
                )
            ),
        )

    def test_rejects_fastmath_assumptions(self) -> None:
        """Do not silently erase floating-point assumptions during import."""
        text = to_mlir(_graph()).replace(
            "arith.addf %arg0, %arg1", "arith.addf %arg0, %arg1 fastmath<fast>"
        )
        with self.assertRaisesRegex(ValueError, "fastmath"):
            from_mlir(text)

    def test_rejects_unsupported_export(self) -> None:
        """Reject normalization until native export and device lowering support it."""
        left, right = _arguments()
        # ``right`` serves as the learned per-element scale of length three.
        norm = RMSNormOp(operands=(left, right), eps=1e-5, name_hint="norm")
        graph = Function(
            name="main",
            operands=(left, right),
            operations=(norm,),
            results=norm.results,
        )
        with self.assertRaisesRegex(ValueError, "addition and multiplication"):
            to_mlir(graph)

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
        _assert_same_artifact(self, artifact, compile_graph(_graph(), target))
        # The result name's parent path survives MLIR as the device block label.
        (operation,) = artifact.target_ir.request.graph.ranks[0].operations
        self.assertEqual(operation.block, "decoder.0")
        self.assertEqual(
            artifact.execute(
                (
                    CentNamedScalars(name="left", values=(1.0, -2.0, 3.0)),
                    CentNamedScalars(name="right", values=(10.0, 20.0, -30.0)),
                )
            ),
            (CentNamedScalars(name="decoder.0.sum", values=(11.0, 18.0, -27.0)),),
        )

    def test_mlir_multiply_add_chain_executes(self) -> None:
        """Run native multiplication through DRAM staging and then Shared Buffer addition."""
        from tests.test_pipeline import target
        from vllm_cent.pipeline import compile_graph, compile_mlir
        from vllm_cent.runtime import CentNamedScalars

        left, right = _arguments()
        product = MulOp(operands=(left, right), name_hint="product")
        total = AddOp(operands=(product.results[0], left), name_hint="sum")
        graph = Function(
            name="main",
            operands=(left, right),
            operations=(product, total),
            results=(total.results[0], left),
        )
        artifact = compile_mlir(to_mlir(graph), target())
        _assert_same_artifact(self, artifact, compile_graph(graph, target()))
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
