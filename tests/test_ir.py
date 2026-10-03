"""Tests for the ALOI IR core: types, values, operations, functions, modules."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from vllm_cent.ir.base import Function, Module, Operation, Value, clone_function
from vllm_cent.ir.types import DType, TensorRole, TensorType

# A small rank-2 tensor type shared by most tests. The exact shape does not
# matter; it only has to be the same everywhere so test operations can pass
# their operand type through unchanged.
VECTOR = TensorType(shape=(1, 4), dtype=DType.FP16)


# ---------------------------------------------------------------------------
# Test-only operations
#
# These exercise the Operation base class without depending on semantic.py.
# Subclasses only assign NAME and NUM_OPERANDS; the ClassVar declaration in
# the base class keeps dataclass from treating them as fields.
# ---------------------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class _Identity(Operation):
    """One operand; the result has the operand's type."""

    NAME = "test.identity"
    NUM_OPERANDS = 1

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Pass the operand type through unchanged."""

        return (self.operands[0].type,)


@dataclass(frozen=True, eq=False)
class _Scale(Operation):
    """One operand plus one typed attribute that has a default value."""

    NAME = "test.scale"
    NUM_OPERANDS = 1

    factor: float = 2.0

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Pass the operand type through unchanged."""

        return (self.operands[0].type,)


@dataclass(frozen=True, eq=False)
class _Add(Operation):
    """Two operands that must have the same type."""

    NAME = "test.add"
    NUM_OPERANDS = 2

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Require matching operand types and return that type."""

        lhs, rhs = self.operands
        if lhs.type != rhs.type:
            raise ValueError(f"test.add operands differ: {lhs.type} vs {rhs.type}")
        return (lhs.type,)


def _chain() -> tuple[Function, Value, _Identity, _Identity]:
    """Build ``x -> first -> second`` and return ``second``'s result.

    Returns:
        The function, its single argument, and its two operations in order.
    """

    x = Value(type=VECTOR, name="x")
    first = _Identity(operands=(x,))
    second = _Identity(operands=(first.results[0],))
    function = Function(
        name="main",
        operands=(x,),
        operations=(first, second),
        results=(second.results[0],),
    )
    return function, x, first, second


# ---------------------------------------------------------------------------
# types.py
# ---------------------------------------------------------------------------


class DTypeTest(unittest.TestCase):
    """Bit widths and accepted spellings of element types."""

    def test_bits_for_every_member(self) -> None:
        """Every dtype reports its storage width in bits."""

        expected = {
            DType.INT64: 64,
            DType.INT32: 32,
            DType.FP32: 32,
            DType.BF16: 16,
            DType.FP16: 16,
            DType.INT8: 8,
            DType.FP8_E4M3: 8,
            DType.FP8_E5M2: 8,
            DType.FP4: 4,
        }
        # Comparing the key set with the enum makes this test fail when a new
        # dtype is added without updating the expected widths.
        self.assertEqual(set(expected), set(DType))
        for dtype, bits in expected.items():
            with self.subTest(dtype=dtype):
                self.assertEqual(dtype.bits, bits)

    def test_normalize_accepts_aliases_and_canonical_names(self) -> None:
        """PyTorch-style and short spellings map to one canonical member."""

        cases = {
            "torch.float16": DType.FP16,
            "half": DType.FP16,
            "fp16": DType.FP16,
            "torch.bfloat16": DType.BF16,
            "float32": DType.FP32,
            "torch.int64": DType.INT64,
            DType.FP8_E4M3: DType.FP8_E4M3,
        }
        for spelling, dtype in cases.items():
            with self.subTest(spelling=spelling):
                self.assertIs(DType.normalize(spelling), dtype)

    def test_normalize_rejects_unknown_dtype(self) -> None:
        """An unsupported spelling raises instead of guessing."""

        with self.assertRaises(ValueError):
            DType.normalize("complex64")


class TensorTypeTest(unittest.TestCase):
    """Immutable tensor types compare by content."""

    def test_equal_content_means_equal_and_same_hash(self) -> None:
        """Two separately built identical types are interchangeable."""

        first = TensorType(shape=(1, 8192), dtype=DType.FP16)
        second = TensorType(shape=(1, 8192), dtype=DType.FP16)
        self.assertEqual(first, second)
        self.assertEqual(hash(first), hash(second))

    def test_dtype_is_part_of_equality(self) -> None:
        """Same shape with a different dtype is a different type."""

        self.assertNotEqual(
            TensorType(shape=(1, 8192), dtype=DType.FP16),
            TensorType(shape=(1, 8192), dtype=DType.BF16),
        )

    def test_dtype_is_required(self) -> None:
        """Omitting dtype is an error rather than a silent default."""

        with self.assertRaises(TypeError):
            TensorType(shape=(1, 8192))  # type: ignore[call-arg]

    def test_list_shape_is_stored_as_hashable_tuple(self) -> None:
        """A list shape is copied into a tuple so the type stays hashable."""

        tensor_type = TensorType(shape=[1, 4], dtype=DType.FP16)  # type: ignore[arg-type]
        self.assertEqual(tensor_type.shape, (1, 4))
        self.assertIsInstance(tensor_type.shape, tuple)
        self.assertEqual(hash(tensor_type), hash(VECTOR))

    def test_rank_counts_dimensions(self) -> None:
        """Rank is the number of dimensions, including size-1 ones."""

        self.assertEqual(TensorType(shape=(1, 1, 8192), dtype=DType.FP16).rank, 3)

    def test_defaults_describe_unannotated_activation(self) -> None:
        """Without annotations a type is an activation with unknown axes."""

        self.assertIs(VECTOR.role, TensorRole.ACTIVATION)
        self.assertEqual(VECTOR.axes, ())
        self.assertIsNone(VECTOR.global_shape)

    def test_rejects_negative_dimension(self) -> None:
        """A negative extent cannot describe a tensor."""

        with self.assertRaises(ValueError):
            TensorType(shape=(1, -4), dtype=DType.FP16)

    def test_rejects_axes_that_do_not_match_rank(self) -> None:
        """When axes are given there must be one name per dimension."""

        with self.assertRaises(ValueError):
            TensorType(shape=(1, 4), dtype=DType.FP16, axes=("hidden",))

    def test_rejects_global_shape_with_different_rank(self) -> None:
        """Sharding changes extents, never the number of dimensions."""

        with self.assertRaises(ValueError):
            TensorType(shape=(1, 4), dtype=DType.FP16, global_shape=(32,))


# ---------------------------------------------------------------------------
# base.py: Value
# ---------------------------------------------------------------------------


class ValueTest(unittest.TestCase):
    """SSA values compare by identity, not by content."""

    def test_values_with_same_type_are_distinct(self) -> None:
        """Two values of one type are still two different SSA values."""

        first = Value(type=VECTOR)
        second = Value(type=VECTOR)
        self.assertNotEqual(first, second)
        self.assertEqual(first, first)

    def test_values_are_independent_dict_keys(self) -> None:
        """Identity hashing lets a value map hold same-typed values."""

        first = Value(type=VECTOR, name="x")
        second = Value(type=VECTOR, name="x")
        mapping = {first: "first", second: "second"}
        self.assertEqual(len(mapping), 2)
        self.assertEqual(mapping[first], "first")


# ---------------------------------------------------------------------------
# base.py: Operation
# ---------------------------------------------------------------------------


class OperationTest(unittest.TestCase):
    """Behavior shared by every operation through the base class."""

    def test_results_are_created_from_inferred_types(self) -> None:
        """Construction creates one new result value per inferred type."""

        x = Value(type=VECTOR, name="x")
        op = _Identity(operands=(x,))
        self.assertEqual(len(op.results), 1)
        self.assertIsInstance(op.results[0], Value)
        self.assertEqual(op.results[0].type, VECTOR)
        self.assertIsNot(op.results[0], x)

    def test_each_operation_creates_fresh_results(self) -> None:
        """Two operations on the same operand never share a result value."""

        x = Value(type=VECTOR)
        self.assertIsNot(
            _Identity(operands=(x,)).results[0],
            _Identity(operands=(x,)).results[0],
        )

    def test_operands_are_stored_as_tuple(self) -> None:
        """Any iterable of operands is copied into an immutable tuple."""

        x = Value(type=VECTOR)
        op = _Identity(operands=[x])  # type: ignore[arg-type]
        self.assertIsInstance(op.operands, tuple)
        self.assertEqual(op.operands, (x,))

    def test_rejects_wrong_operand_count(self) -> None:
        """NUM_OPERANDS is enforced before result inference runs."""

        with self.assertRaises(ValueError):
            _Identity(operands=(Value(type=VECTOR), Value(type=VECTOR)))

    def test_inference_errors_raise_during_construction(self) -> None:
        """An ill-typed operation cannot be created at all."""

        other = TensorType(shape=(1, 8), dtype=DType.FP16)
        with self.assertRaises(ValueError):
            _Add(operands=(Value(type=VECTOR), Value(type=other)))

    def test_subclass_without_inference_cannot_be_instantiated(self) -> None:
        """Forgetting infer_result_types fails when the op is created."""

        @dataclass(frozen=True, eq=False)
        class Incomplete(Operation):
            NAME = "test.incomplete"
            NUM_OPERANDS = 1

        with self.assertRaises(TypeError):
            Incomplete(operands=(Value(type=VECTOR),))  # type: ignore[abstract]

    def test_with_operands_rebuilds_with_fresh_results(self) -> None:
        """Rebuilding returns a new op of the same class with new results."""

        x = Value(type=VECTOR, name="x")
        y = Value(type=VECTOR, name="y")
        original = _Identity(operands=(x,))
        rebuilt = original.with_operands([y])

        self.assertIsInstance(rebuilt, _Identity)
        self.assertIsNot(rebuilt, original)
        self.assertEqual(rebuilt.operands, (y,))
        self.assertIsNot(rebuilt.results[0], original.results[0])
        # The original operation is untouched.
        self.assertEqual(original.operands, (x,))

    def test_with_operands_preserves_attributes(self) -> None:
        """Typed attributes survive a rebuild unchanged."""

        rebuilt = _Scale(operands=(Value(type=VECTOR),), factor=3.0).with_operands(
            [Value(type=VECTOR)]
        )
        self.assertEqual(rebuilt.factor, 3.0)

    def test_attributes_lists_only_declared_attributes(self) -> None:
        """Operands and results are structure, not printable attributes."""

        x = Value(type=VECTOR)
        self.assertEqual(_Scale(operands=(x,), factor=3.0).attributes(), {"factor": 3.0})
        self.assertEqual(_Identity(operands=(x,)).attributes(), {})

    def test_subclass_may_declare_required_attribute(self) -> None:
        """A subclass attribute without a default is allowed.

        RMSNorm's ``eps`` is a realistic example: it should be required so a
        frontend cannot forget it. The class is defined inside the test so a
        failure here does not prevent the rest of this module from importing.
        """

        @dataclass(frozen=True, eq=False)
        class WithEps(Operation):
            NAME = "test.with_eps"
            NUM_OPERANDS = 1

            eps: float

            def infer_result_types(self) -> tuple[TensorType, ...]:
                return (self.operands[0].type,)

        op = WithEps(operands=(Value(type=VECTOR),), eps=1e-5)
        self.assertEqual(op.eps, 1e-5)
        self.assertEqual(op.attributes(), {"eps": 1e-5})


# ---------------------------------------------------------------------------
# base.py: Function
# ---------------------------------------------------------------------------


class FunctionTest(unittest.TestCase):
    """Function construction, verification, and use-def queries."""

    def test_constructor_accepts_any_iterable(self) -> None:
        """Lists and generators are copied into tuples."""

        _, x, first, second = _chain()
        function = Function(
            name="main",
            operands=[x],  # type: ignore[arg-type]
            # A generator can be consumed only once, so storing it unchanged
            # would make the operation list empty on the second iteration.
            operations=(op for op in (first, second)),  # type: ignore[arg-type]
            results=[second.results[0]],  # type: ignore[arg-type]
        )
        self.assertEqual(function.operands, (x,))
        self.assertEqual(function.operations, (first, second))
        self.assertEqual(function.results, (second.results[0],))
        # Iterating twice proves the generator was materialized.
        self.assertEqual(list(function.operations), list(function.operations))

    def test_verify_accepts_ordered_chain(self) -> None:
        """Every value is defined before use, so verification passes."""

        function, *_ = _chain()
        function.verify()

    def test_verify_rejects_undefined_operand(self) -> None:
        """An op may only use arguments or earlier results."""

        x = Value(type=VECTOR, name="x")
        stranger = Value(type=VECTOR, name="stranger")
        op = _Identity(operands=(stranger,))
        function = Function(
            name="main", operands=(x,), operations=(op,), results=(op.results[0],)
        )
        with self.assertRaises(ValueError):
            function.verify()

    def test_verify_rejects_repeated_argument(self) -> None:
        """Listing one argument twice defines it twice, breaking SSA."""

        x = Value(type=VECTOR, name="x")
        op = _Identity(operands=(x,))
        function = Function(
            name="main", operands=(x, x), operations=(op,), results=(op.results[0],)
        )
        with self.assertRaisesRegex(ValueError, "argument %x twice"):
            function.verify()

    def test_verify_error_names_op_kind_and_hint(self) -> None:
        """The message names the op kind and, when present, its name hint."""

        stranger = Value(type=VECTOR, name="stranger")
        cases = (
            (None, "test.identity uses"),
            ("q_proj", r"test\.identity \(q_proj\) uses"),
        )
        for name_hint, expected in cases:
            with self.subTest(name_hint=name_hint):
                op = _Identity(operands=(stranger,), name_hint=name_hint)
                function = Function(
                    name="main", operands=(), operations=(op,), results=()
                )
                with self.assertRaisesRegex(ValueError, expected):
                    function.verify()

    def test_verify_rejects_use_before_definition(self) -> None:
        """Swapping the chain makes ``second`` read a value not yet defined."""

        _, x, first, second = _chain()
        function = Function(
            name="main",
            operands=(x,),
            operations=(second, first),
            results=(second.results[0],),
        )
        with self.assertRaises(ValueError):
            function.verify()

    def test_verify_rejects_value_defined_twice(self) -> None:
        """Listing one op twice defines its result twice, breaking SSA."""

        _, x, first, _ = _chain()
        function = Function(
            name="main",
            operands=(x,),
            operations=(first, first),
            results=(first.results[0],),
        )
        with self.assertRaises(ValueError):
            function.verify()

    def test_verify_rejects_undefined_return(self) -> None:
        """A function can only return values it defines."""

        _, x, first, _ = _chain()
        function = Function(
            name="main",
            operands=(x,),
            operations=(first,),
            results=(Value(type=VECTOR, name="stranger"),),
        )
        with self.assertRaises(ValueError):
            function.verify()

    def test_defining_op(self) -> None:
        """Arguments have no defining op; results point to their producer."""

        function, x, first, second = _chain()
        self.assertIsNone(function.defining_op(x))
        self.assertIs(function.defining_op(first.results[0]), first)
        self.assertIs(function.defining_op(second.results[0]), second)

    def test_users_in_program_order(self) -> None:
        """Users are the ops that consume a value, in function order."""

        x = Value(type=VECTOR, name="x")
        left = _Identity(operands=(x,))
        right = _Identity(operands=(x,))
        total = _Add(operands=(left.results[0], right.results[0]))
        function = Function(
            name="main",
            operands=(x,),
            operations=(left, right, total),
            results=(total.results[0],),
        )
        self.assertEqual(function.users(x), (left, right))
        self.assertEqual(function.users(left.results[0]), (total,))
        self.assertEqual(function.users(total.results[0]), ())


# ---------------------------------------------------------------------------
# base.py: Module
# ---------------------------------------------------------------------------


class ModuleTest(unittest.TestCase):
    """Top-level lookup of functions by name."""

    def test_get_function_by_name(self) -> None:
        """The function with the requested name is returned."""

        function, *_ = _chain()
        module = Module(name="block", functions=(function,))
        self.assertIs(module.get_function("main"), function)

    def test_missing_function_raises_key_error(self) -> None:
        """Looking up an absent name fails loudly."""

        function, *_ = _chain()
        module = Module(name="block", functions=(function,))
        with self.assertRaises(KeyError):
            module.get_function("missing")

    def test_functions_are_stored_as_tuple(self) -> None:
        """A list of functions is copied into an immutable tuple."""

        function, *_ = _chain()
        module = Module(name="block", functions=[function])  # type: ignore[arg-type]
        self.assertIsInstance(module.functions, tuple)

    def test_verify_accepts_valid_functions(self) -> None:
        """A module of valid functions verifies without error."""

        function, *_ = _chain()
        Module(name="block", functions=(function,)).verify()

    def test_verify_names_the_invalid_function(self) -> None:
        """Function errors gain the function's name as context."""

        _, x, first, _ = _chain()
        broken = Function(
            name="broken",
            operands=(x,),
            operations=(first,),
            results=(Value(type=VECTOR, name="stranger"),),
        )
        valid, *_ = _chain()
        module = Module(name="block", functions=(valid, broken))
        with self.assertRaisesRegex(ValueError, "@broken"):
            module.verify()

    def test_verify_rejects_duplicate_function_names(self) -> None:
        """get_function looks up by name, so names must be unique."""

        first, *_ = _chain()
        second, *_ = _chain()
        module = Module(name="block", functions=(first, second))
        with self.assertRaisesRegex(ValueError, "main"):
            module.verify()


# ---------------------------------------------------------------------------
# base.py: clone_function
# ---------------------------------------------------------------------------


def _all_values(function: Function) -> list[Value]:
    """List arguments and every op result in definition order.

    Args:
        function: Function whose defined values are collected.

    Returns:
        Arguments followed by each operation's results.
    """

    values = list(function.operands)
    for op in function.operations:
        values.extend(op.results)
    return values


class CloneFunctionTest(unittest.TestCase):
    """Copying a function maps every old value to a new one."""

    def test_clone_is_valid_and_structurally_equal(self) -> None:
        """Same op classes, types, and names; still passes verification."""

        original, *_ = _chain()
        clone = clone_function(original)

        clone.verify()
        self.assertEqual(clone.name, original.name)
        self.assertEqual(
            [type(op) for op in clone.operations],
            [type(op) for op in original.operations],
        )
        self.assertEqual(
            [value.type for value in _all_values(clone)],
            [value.type for value in _all_values(original)],
        )
        self.assertEqual(clone.operands[0].name, "x")

    def test_clone_shares_no_values_with_original(self) -> None:
        """Every argument, result, and return is a brand-new object."""

        original, *_ = _chain()
        clone = clone_function(original)
        original_ids = {id(value) for value in _all_values(original)}
        clone_ids = {id(value) for value in _all_values(clone)}
        self.assertEqual(original_ids & clone_ids, set())
        self.assertNotIn(id(clone.results[0]), original_ids)

    def test_clone_rewires_dataflow(self) -> None:
        """Each new op reads the new value that replaced its old operand."""

        original, *_ = _chain()
        clone = clone_function(original)
        first, second = clone.operations
        self.assertIs(first.operands[0], clone.operands[0])
        self.assertIs(second.operands[0], first.results[0])
        self.assertIs(clone.results[0], second.results[0])

    def test_clone_leaves_original_unchanged(self) -> None:
        """The input function keeps its own ops and values."""

        original, x, first, second = _chain()
        clone_function(original)
        self.assertEqual(original.operands, (x,))
        self.assertEqual(original.operations, (first, second))
        self.assertEqual(original.results, (second.results[0],))
        original.verify()

    def test_clone_preserves_attributes(self) -> None:
        """Rebuilt ops keep their typed attributes."""

        x = Value(type=VECTOR, name="x")
        scale = _Scale(operands=(x,), factor=3.0)
        original = Function(
            name="main",
            operands=(x,),
            operations=(scale,),
            results=(scale.results[0],),
        )
        (cloned_scale,) = clone_function(original).operations
        self.assertIsInstance(cloned_scale, _Scale)
        self.assertEqual(cloned_scale.factor, 3.0)


if __name__ == "__main__":
    unittest.main()
