"""Tests for the deterministic IR printer."""

from __future__ import annotations

import unittest
from dataclasses import dataclass
from enum import IntEnum

from vllm_cent.ir.base import Function, Module, Operation, Value, clone_function
from vllm_cent.ir.printer import format_type, print_function, print_module
from vllm_cent.ir.semantic import LinearOp, RMSNormOp
from vllm_cent.ir.types import DType, TensorRole, TensorType


def _type(*shape: int, **annotations: object) -> TensorType:
    """Create an fp16 tensor type.

    Args:
        *shape: Tensor dimensions.
        **annotations: Optional role, axes, or global_shape.

    Returns:
        The tensor type.
    """

    return TensorType(shape=shape, dtype=DType.FP16, **annotations)  # type: ignore[arg-type]


VECTOR = _type(4)
VECTOR_TEXT = "tensor<4xfp16>"


# ---------------------------------------------------------------------------
# Test-only operations covering shapes of op lines that the semantic dialect
# does not produce yet: several results, no result, and many attribute types.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Split(Operation):
    """One operand, two results of the same type."""

    NAME = "test.split"
    NUM_OPERANDS = 1

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return the operand type twice."""

        return (self.operands[0].type, self.operands[0].type)


@dataclass(frozen=True, eq=False)
class _Sink(Operation):
    """One operand and no result."""

    NAME = "test.sink"
    NUM_OPERANDS = 1

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Produce nothing."""

        return ()


class _Mode(IntEnum):
    """An IntEnum attribute; it must print its value, not its repr."""

    FAST = 3


@dataclass(frozen=True, eq=False)
class _Attributes(Operation):
    """One attribute of every supported kind, in declaration order."""

    NAME = "test.attributes"
    NUM_OPERANDS = 1

    flag: bool = True
    count: int = 4
    ratio: float = 0.5
    label: str = 'say "hi"'
    mode: _Mode = _Mode.FAST
    dtype: DType = DType.BF16
    dims: tuple[int, ...] = (1, 2)
    missing: None = None

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Pass the operand type through."""

        return (self.operands[0].type,)


@dataclass(frozen=True, eq=False)
class _Opaque(Operation):
    """An attribute whose type has no deterministic text form."""

    NAME = "test.opaque"
    NUM_OPERANDS = 1

    thing: object = None

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Pass the operand type through."""

        return (self.operands[0].type,)


def _block() -> Function:
    """Build the M0 example: RMSNorm followed by the query projection.

    Returns:
        ``block(x, gamma, wq)`` returning the query.
    """

    x = Value(type=_type(1, 1, 8192), name="x")
    gamma = Value(type=_type(8192), name="gamma")
    wq = Value(type=_type(8192, 8192), name="wq")
    norm = RMSNormOp(operands=(x, gamma), eps=1e-5)
    query = LinearOp(operands=(norm.results[0], wq), name_hint="q")
    return Function(
        name="block",
        operands=(x, gamma, wq),
        operations=(norm, query),
        results=(query.results[0],),
    )


# The normalized value has no name hint, so it is numbered %0; the projection
# has name_hint="q". eps prints with repr(), so 1e-5 appears as 1e-05.
BLOCK_TEXT = """\
func @block(%x: tensor<1x1x8192xfp16>, %gamma: tensor<8192xfp16>, \
%wq: tensor<8192x8192xfp16>) -> (tensor<1x1x8192xfp16>) {
  %0 = aloi.rms_norm(%x, %gamma) {eps = 1e-05} : tensor<1x1x8192xfp16>
  %q = aloi.linear(%0, %wq) : tensor<1x1x8192xfp16>
  return %q
}"""


class FormatTypeTest(unittest.TestCase):
    """Text form of tensor types."""

    def test_shape_and_dtype(self) -> None:
        """Dimensions and the DType value are joined with 'x'."""

        self.assertEqual(format_type(_type(1, 1, 8192)), "tensor<1x1x8192xfp16>")

    def test_dtype_uses_enum_value(self) -> None:
        """The dtype prints exactly as its DType value."""

        bf16 = TensorType(shape=(4,), dtype=DType.BF16)
        self.assertEqual(format_type(bf16), "tensor<4xbf16>")

    def test_rank_zero(self) -> None:
        """A scalar has no dimensions, only a dtype."""

        self.assertEqual(format_type(_type()), "tensor<fp16>")

    def test_default_annotations_are_omitted(self) -> None:
        """An explicit ACTIVATION role is the default and is not printed."""

        self.assertEqual(format_type(_type(4, role=TensorRole.ACTIVATION)), VECTOR_TEXT)

    def test_role_and_axes(self) -> None:
        """Non-default annotations follow the shape: role, then axes."""

        weight = _type(
            1024,
            8192,
            role=TensorRole.WEIGHT,
            axes=("kv_head*head_dim", "hidden"),
        )
        self.assertEqual(
            format_type(weight),
            "tensor<1024x8192xfp16, role=weight, axes=[kv_head*head_dim, hidden]>",
        )

    def test_axes_and_global_shape(self) -> None:
        """A TP shard shows its global shape after the axes.

        With 8-way TP, each rank holds 8192 / 8 = 1024 of the hidden values.
        """

        shard = _type(1, 1024, axes=("batch", "hidden_size"), global_shape=(1, 8192))
        self.assertEqual(
            format_type(shard),
            "tensor<1x1024xfp16, axes=[batch, hidden_size], global_shape=1x8192>",
        )


class PrintFunctionTest(unittest.TestCase):
    """Text form of functions: names, op lines, and errors."""

    def test_m0_golden(self) -> None:
        """The hand-written RMSNorm + linear example prints exactly."""

        self.assertEqual(print_function(_block()), BLOCK_TEXT)

    def test_output_is_deterministic(self) -> None:
        """Printing twice, or printing a clone, gives identical text."""

        function = _block()
        self.assertEqual(print_function(function), print_function(function))
        self.assertEqual(print_function(clone_function(function)), BLOCK_TEXT)

    def test_repeated_hints_get_suffixes(self) -> None:
        """Two values hinted 'x' print as %x and %x_1."""

        first = Value(type=VECTOR, name="x")
        second = Value(type=VECTOR, name="x")
        function = Function(
            name="f", operands=(first, second), operations=(), results=(second,)
        )
        self.assertEqual(
            print_function(function),
            f"func @f(%x: {VECTOR_TEXT}, %x_1: {VECTOR_TEXT}) -> ({VECTOR_TEXT}) {{\n"
            "  return %x_1\n"
            "}",
        )

    def test_numbering_skips_claimed_names(self) -> None:
        """An argument named '0' forces unnamed results to start at %1."""

        zero = Value(type=VECTOR, name="0")
        split = _Split(operands=(zero,))
        function = Function(
            name="f", operands=(zero,), operations=(split,), results=split.results
        )
        self.assertIn("  %1, %2 = test.split(%0)", print_function(function))

    def test_multiple_and_zero_results(self) -> None:
        """Several results share one line; an op without results has no '='."""

        x = Value(type=VECTOR, name="x")
        split = _Split(operands=(x,), name_hint="half")
        sink = _Sink(operands=(split.results[0],))
        function = Function(
            name="f",
            operands=(x,),
            operations=(split, sink),
            results=(split.results[1],),
        )
        self.assertEqual(
            print_function(function),
            f"func @f(%x: {VECTOR_TEXT}) -> ({VECTOR_TEXT}) {{\n"
            f"  %half, %half_1 = test.split(%x) : ({VECTOR_TEXT}, {VECTOR_TEXT})\n"
            "  test.sink(%half)\n"
            "  return %half_1\n"
            "}",
        )

    def test_attribute_formats(self) -> None:
        """Each supported attribute type has one fixed text form."""

        x = Value(type=VECTOR, name="x")
        op = _Attributes(operands=(x,))
        function = Function(
            name="f", operands=(x,), operations=(op,), results=op.results
        )
        expected_attributes = (
            '{flag = true, count = 4, ratio = 0.5, label = "say \\"hi\\"", '
            "mode = 3, dtype = bf16, dims = [1, 2], missing = none}"
        )
        self.assertIn(
            f"  %0 = test.attributes(%x) {expected_attributes} : {VECTOR_TEXT}",
            print_function(function),
        )

    def test_empty_function(self) -> None:
        """No arguments, no ops, and a bare return."""

        function = Function(name="empty", operands=(), operations=(), results=())
        self.assertEqual(print_function(function), "func @empty() -> () {\n  return\n}")

    def test_rejects_undefined_operand(self) -> None:
        """A value used without a definition is an error, not a new name."""

        stranger = Value(type=VECTOR, name="stranger")
        op = _Split(operands=(stranger,))
        function = Function(name="f", operands=(), operations=(op,), results=())
        with self.assertRaisesRegex(ValueError, "stranger"):
            print_function(function)

    def test_rejects_undefined_return(self) -> None:
        """Returning an undefined value is also rejected."""

        function = Function(
            name="f", operands=(), operations=(), results=(Value(type=VECTOR),)
        )
        with self.assertRaises(ValueError):
            print_function(function)

    def test_rejects_value_defined_twice(self) -> None:
        """Listing one op twice would print the same value twice."""

        x = Value(type=VECTOR, name="x")
        op = _Split(operands=(x,))
        function = Function(name="f", operands=(x,), operations=(op, op), results=())
        with self.assertRaisesRegex(ValueError, "defined twice"):
            print_function(function)

    def test_rejects_attribute_without_stable_text(self) -> None:
        """A generic object's repr may contain an address, so it is refused."""

        x = Value(type=VECTOR, name="x")
        op = _Opaque(operands=(x,), thing=object())
        function = Function(name="f", operands=(x,), operations=(op,), results=())
        with self.assertRaises(TypeError):
            print_function(function)


class PrintModuleTest(unittest.TestCase):
    """Text form of modules."""

    def test_layout(self) -> None:
        """Header, a blank line between sections, and a final newline.

        Names restart in every function, so both functions use %x.
        """

        def identity_function(name: str) -> Function:
            """Return ``name(x) -> x``."""

            x = Value(type=VECTOR, name="x")
            return Function(name=name, operands=(x,), operations=(), results=(x,))

        module = Module(
            name="toy",
            functions=(identity_function("first"), identity_function("second")),
        )
        self.assertEqual(
            print_module(module),
            "module @toy\n"
            "\n"
            f"func @first(%x: {VECTOR_TEXT}) -> ({VECTOR_TEXT}) {{\n"
            "  return %x\n"
            "}\n"
            "\n"
            f"func @second(%x: {VECTOR_TEXT}) -> ({VECTOR_TEXT}) {{\n"
            "  return %x\n"
            "}\n",
        )

    def test_empty_module(self) -> None:
        """A module without functions is just its header line."""

        self.assertEqual(
            print_module(Module(name="toy", functions=())), "module @toy\n"
        )


if __name__ == "__main__":
    unittest.main()
