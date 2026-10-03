"""Tests for AnnotateRolesAndAxes: model annotations become Semantic IR."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from vllm_cent.ir.annotate import AnnotateRolesAndAxes, TensorAnnotation
from vllm_cent.ir.base import Function, Module, Operation, Value
from vllm_cent.ir.passes import PassManager
from vllm_cent.ir.printer import print_module
from vllm_cent.ir.semantic import LinearOp, RMSNormOp
from vllm_cent.ir.types import DType, TensorRole, TensorType

# A tiny RMSNorm -> linear block: hidden size 4, output size 2.
HIDDEN = 4
OUT = 2


@dataclass(frozen=True, eq=False)
class _DropsAxes(Operation):
    """Keeps the operand's shape but forgets its axes, like an op whose
    infer_result_types was written before axes existed."""

    NAME = "test.drops_axes"
    NUM_OPERANDS = 1

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return the operand's shape and dtype only."""

        operand = self.operands[0].type
        return (TensorType(shape=operand.shape, dtype=operand.dtype),)


def _type(*shape: int) -> TensorType:
    """Create an unannotated fp16 type, as the importer produces.

    Args:
        *shape: Tensor dimensions.

    Returns:
        The tensor type.
    """

    return TensorType(shape=shape, dtype=DType.FP16)


def _block_function(name: str = "main") -> Function:
    """Build ``proj(norm(x, gamma), w)`` without annotations.

    Args:
        name: Function name.

    Returns:
        The function.
    """

    x = Value(type=_type(1, HIDDEN), name="x")
    gamma = Value(type=_type(HIDDEN), name="gamma")
    w = Value(type=_type(OUT, HIDDEN), name="w")
    norm = RMSNormOp(operands=(x, gamma), eps=1e-5, name_hint="norm")
    proj = LinearOp(operands=(norm.results[0], w), name_hint="proj")
    return Function(
        name=name,
        operands=(x, gamma, w),
        operations=(norm, proj),
        results=(proj.results[0],),
    )


def _block_annotations() -> dict[str, TensorAnnotation]:
    """Annotate every argument of ``_block_function``.

    Returns:
        A new annotation mapping.
    """

    return {
        "x": TensorAnnotation(TensorRole.ACTIVATION, ("batch", "hidden")),
        "gamma": TensorAnnotation(TensorRole.WEIGHT, ("hidden",)),
        "w": TensorAnnotation(TensorRole.WEIGHT, ("out", "hidden")),
    }


def _annotate(
    module: Module, annotations: dict[str, TensorAnnotation] | None = None
) -> Module:
    """Run the pass through a verifying PassManager.

    Args:
        module: Module to annotate.
        annotations: Annotations; the block's full set by default.

    Returns:
        The annotated module.
    """

    if annotations is None:
        annotations = _block_annotations()
    return PassManager((AnnotateRolesAndAxes(annotations),)).run(module)


class TensorAnnotationTest(unittest.TestCase):
    """The annotation value type."""

    def test_axes_are_stored_as_tuple(self) -> None:
        """A list is accepted and frozen into a tuple."""

        annotation = TensorAnnotation(TensorRole.WEIGHT, ["out", "hidden"])  # type: ignore[arg-type]
        self.assertEqual(annotation.axes, ("out", "hidden"))


class AnnotateRolesAndAxesTest(unittest.TestCase):
    """Arguments get the model's annotations; ops propagate the axes."""

    def test_block_golden(self) -> None:
        """Weights get role=weight; results get axes derived by each op.

        rms_norm keeps x's axes [batch, hidden]; linear contracts over hidden
        and takes w's output axis, giving [batch, out].
        """

        module = _annotate(Module(name="toy", functions=(_block_function(),)))
        expected = (
            "module @toy\n"
            "\n"
            "func @main(%x: tensor<1x4xfp16, axes=[batch, hidden]>, "
            "%gamma: tensor<4xfp16, role=weight, axes=[hidden]>, "
            "%w: tensor<2x4xfp16, role=weight, axes=[out, hidden]>) "
            "-> (tensor<1x2xfp16, axes=[batch, out]>) {\n"
            "  %norm = aloi.rms_norm(%x, %gamma) {eps = 1e-05} "
            ": tensor<1x4xfp16, axes=[batch, hidden]>\n"
            "  %proj = aloi.linear(%norm, %w) : tensor<1x2xfp16, axes=[batch, out]>\n"
            "  return %proj\n"
            "}\n"
        )
        self.assertEqual(print_module(module), expected)

    def test_input_module_is_unchanged(self) -> None:
        """The pass is pure: the imported module still prints the same."""

        module = Module(name="toy", functions=(_block_function(),))
        before = print_module(module)
        _annotate(module)
        self.assertEqual(print_module(module), before)

    def test_missing_annotation_lists_every_argument(self) -> None:
        """M2 requires every tensor to be annotated."""

        annotations = _block_annotations()
        del annotations["gamma"]
        del annotations["w"]
        module = Module(name="toy", functions=(_block_function(),))
        with self.assertRaisesRegex(ValueError, r"\['gamma', 'w'\]"):
            _annotate(module, annotations)

    def test_unnamed_argument_counts_as_missing(self) -> None:
        """An argument without a name cannot be looked up."""

        x = Value(type=_type(1, HIDDEN))
        function = Function(name="main", operands=(x,), operations=(), results=(x,))
        with self.assertRaisesRegex(ValueError, r"\[None\]"):
            _annotate(Module(name="toy", functions=(function,)), {})

    def test_unknown_annotation_is_rejected(self) -> None:
        """A typo such as 'gama' must not be silently ignored."""

        annotations = _block_annotations()
        annotations["gama"] = annotations["gamma"]
        module = Module(name="toy", functions=(_block_function(),))
        with self.assertRaisesRegex(ValueError, "gama"):
            _annotate(module, annotations)

    def test_annotation_may_belong_to_another_function(self) -> None:
        """Unused keys are checked per module, not per function."""

        x = Value(type=_type(1, HIDDEN), name="other_input")
        helper = Function(name="helper", operands=(x,), operations=(), results=(x,))
        annotations = _block_annotations()
        annotations["other_input"] = TensorAnnotation(
            TensorRole.ACTIVATION, ("batch", "hidden")
        )
        module = Module(name="toy", functions=(_block_function(), helper))
        annotated = _annotate(module, annotations)
        (other_input,) = annotated.get_function("helper").operands
        self.assertEqual(other_input.type.axes, ("batch", "hidden"))

    def test_axes_must_match_argument_rank(self) -> None:
        """Two axis names cannot describe the rank-1 gamma."""

        annotations = _block_annotations()
        annotations["gamma"] = TensorAnnotation(TensorRole.WEIGHT, ("out", "hidden"))
        module = Module(name="toy", functions=(_block_function(),))
        with self.assertRaisesRegex(ValueError, "%gamma"):
            _annotate(module, annotations)

    def test_op_rejects_inconsistent_annotations(self) -> None:
        """w contracting over 'out' cannot read x's 'hidden' axis."""

        annotations = _block_annotations()
        annotations["w"] = TensorAnnotation(TensorRole.WEIGHT, ("hidden", "out"))
        module = Module(name="toy", functions=(_block_function(),))
        with self.assertRaisesRegex(ValueError, "aloi.linear"):
            _annotate(module, annotations)

    def test_op_without_axis_propagation_is_reported(self) -> None:
        """The error names the op that left its result without axes."""

        x = Value(type=_type(1, HIDDEN), name="x")
        op = _DropsAxes(operands=(x,), name_hint="mystery")
        function = Function(
            name="main", operands=(x,), operations=(op,), results=(op.results[0],)
        )
        annotations = {"x": _block_annotations()["x"]}
        with self.assertRaisesRegex(
            ValueError, r"test\.drops_axes \(mystery\) does not propagate axes"
        ):
            _annotate(Module(name="toy", functions=(function,)), annotations)

    def test_annotations_are_copied(self) -> None:
        """Editing the caller's mapping later does not change the pass."""

        annotations = _block_annotations()
        annotate = AnnotateRolesAndAxes(annotations)
        annotations.clear()
        module = annotate.run(Module(name="toy", functions=(_block_function(),)))
        self.assertEqual(
            module.get_function("main").operands[0].type.axes, ("batch", "hidden")
        )


if __name__ == "__main__":
    unittest.main()
