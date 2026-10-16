"""Check that each operation owns its attributes and mathematical shape rules."""

import unittest
from dataclasses import FrozenInstanceError, dataclass, fields

from vllm_cent.pipeline.ir import (
    CapabilityProfile,
    ParallelPlan,
    SemanticGraph,
    legalize,
    shard,
    verify_semantic,
)
from vllm_cent.pipeline.operations import (
    AddOperation,
    LinearOperation,
    Operation,
    RmsNormOperation,
    TensorType,
    Value,
)


@dataclass(frozen=True, slots=True, kw_only=True)
class UnaryIdentity(Operation):
    """Supply an independently defined operation without changing graph verification.

    Inherited fields identify one input and a result of the same tensor type.
    This class exists only to test that graph verification delegates shape rules.
    """

    def verify(self, input_types: tuple[TensorType, ...]) -> None:
        """Require one input whose type is preserved in the output.

        Args:
            input_types: Operand types resolved by graph verification, in order.

        Raises:
            ValueError: If the input count or output tensor type differs.
        """
        if input_types != (self.output.type,):
            raise ValueError("identity requires one input matching its output")


class OperationTests(unittest.TestCase):
    """Keep operation-specific behavior separate from shared graph bookkeeping."""

    def test_base_is_abstract_and_has_only_shared_fields(self) -> None:
        """The common operation interface carries neither kind nor epsilon."""
        self.assertEqual(
            tuple(field.name for field in fields(Operation)),
            ("name", "inputs", "output", "block"),
        )
        with self.assertRaises(TypeError):
            Operation(
                name="abstract", inputs=("x",), output=Value("out", TensorType((3,)))
            )

    def test_add_cannot_accept_normalization_attributes(self) -> None:
        """An irrelevant epsilon fails at construction, before any graph pass."""
        with self.assertRaisesRegex(TypeError, "epsilon"):
            AddOperation(
                name="add",
                inputs=("x", "y"),
                output=Value("out", TensorType((3,))),
                epsilon=1e-5,
            )

    def test_rms_requires_epsilon_and_validates_it_at_construction(self) -> None:
        """RMSNorm owns a required finite positive stabilization constant."""
        output = Value("out", TensorType((2, 3)))
        with self.assertRaisesRegex(TypeError, "epsilon"):
            RmsNormOperation(name="norm", inputs=("x", "scale"), output=output)
        for epsilon in (0.0, -1.0, float("nan"), float("inf"), -float("inf")):
            with (
                self.subTest(epsilon=epsilon),
                self.assertRaisesRegex(ValueError, "epsilon"),
            ):
                RmsNormOperation(
                    name="norm", inputs=("x", "scale"), output=output, epsilon=epsilon
                )
        norm = RmsNormOperation(
            name="norm", inputs=("x", "scale"), output=output, epsilon=1e-5
        )
        self.assertEqual(norm.epsilon, 1e-5)
        with self.assertRaises(FrozenInstanceError):
            norm.epsilon = 1e-4

    def test_add_shape_rules(self) -> None:
        """Addition accepts exact equal types, including scalars, without broadcasting."""
        for shape in ((), (3,), (2, 3)):
            tensor = TensorType(shape)
            operation = AddOperation(
                name="add", inputs=("x", "y"), output=Value("out", tensor)
            )
            operation.verify((tensor, tensor))
            for operands in (
                (),
                (tensor,),
                (tensor, tensor, tensor),
                (tensor, TensorType((7,))),
            ):
                with (
                    self.subTest(shape=shape, operands=operands),
                    self.assertRaises(ValueError),
                ):
                    operation.verify(operands)
        operation = AddOperation(
            name="add", inputs=("x", "y"), output=Value("out", TensorType((2, 3)))
        )
        with self.assertRaises(ValueError):
            operation.verify((TensorType((3,)), TensorType((3,))))

    def test_rms_shape_rules(self) -> None:
        """The scale matches the final input dimension and normalization preserves shape."""
        tensor = TensorType((2, 3))
        operation = RmsNormOperation(
            name="norm",
            inputs=("x", "scale"),
            output=Value("out", tensor),
            epsilon=1e-5,
        )
        operation.verify((tensor, TensorType((3,))))
        for operands in (
            (),
            (tensor,),
            (tensor, tensor),
            (TensorType(()), TensorType(())),
        ):
            with self.subTest(operands=operands), self.assertRaises(ValueError):
                operation.verify(operands)
        wrong_output = RmsNormOperation(
            name="norm",
            inputs=("x", "scale"),
            output=Value("out", TensorType((3,))),
            epsilon=1e-5,
        )
        with self.assertRaises(ValueError):
            wrong_output.verify((tensor, TensorType((3,))))

    def test_linear_shape_rules(self) -> None:
        """Weights replace only the last input dimension and must be a matrix."""
        # Six input features and five output rows preserve both leading dimensions.
        operation = LinearOperation(
            name="linear",
            inputs=("x", "weight"),
            output=Value("out", TensorType((2, 4, 5))),
        )
        operation.verify((TensorType((2, 4, 6)), TensorType((5, 6))))
        for operands in (
            (),
            (TensorType((2, 4, 6)),),
            (TensorType((2, 4, 6)), TensorType((5,))),
            (TensorType((2, 4, 6)), TensorType((5, 7))),
            (TensorType(()), TensorType((5, 6))),
            (TensorType((2, 4, 6)), TensorType((7, 6))),
        ):
            with self.subTest(operands=operands), self.assertRaises(ValueError):
                operation.verify(operands)

    def test_independent_unary_operation_uses_common_graph_verifier(self) -> None:
        """A new operation supplies its own arity and type rule without an enum branch."""
        tensor = TensorType((3,))
        operation = UnaryIdentity(
            name="identity", inputs=("x",), output=Value("out", tensor)
        )
        graph = SemanticGraph(
            inputs=(Value("x", tensor),), operations=(operation,), outputs=("out",)
        )
        verify_semantic(graph)
        self.assertEqual(shard(graph, ParallelPlan()).shards[0].graph, graph)
        # A semantic extension does not automatically acquire device support.
        with self.assertRaises(NotImplementedError):
            legalize(shard(graph, ParallelPlan()), CapabilityProfile())
        invalid = SemanticGraph(
            inputs=(Value("x", TensorType((4,))),),
            operations=(operation,),
            outputs=("out",),
        )
        with self.assertRaisesRegex(ValueError, "identity"):
            verify_semantic(invalid)


class TensorMovementAndMultiplyTests(unittest.TestCase):
    """Check tensor operations without mixing their shape rules into the graph."""

    def test_multiply_requires_equal_shapes_and_two_inputs(self) -> None:
        """Multiplication permits repeated operands for squaring, but no broadcasting."""
        from vllm_cent.pipeline.operations import MultiplyOperation

        tensor = TensorType((2, 3))
        operation = MultiplyOperation(
            name="square", inputs=("x", "x"), output=Value("out", tensor)
        )
        operation.verify((tensor, tensor))
        for types in ((tensor,), (tensor, TensorType((3,))), (tensor, tensor, tensor)):
            with self.subTest(types=types), self.assertRaises(ValueError):
                operation.verify(types)
        scalar = TensorType(())
        MultiplyOperation(
            name="scalar", inputs=("x", "y"), output=Value("out", scalar)
        ).verify((scalar, scalar))

    def test_copy_preserves_type_and_requires_one_operand(self) -> None:
        """Copy creates a named result with the same shape, without changing values."""
        from vllm_cent.pipeline.operations import CopyOperation

        tensor = TensorType((2, 3))
        operation = CopyOperation(
            name="copy", inputs=("x",), output=Value("out", tensor)
        )
        operation.verify((tensor,))
        for types in ((), (tensor, tensor), (TensorType((6,)),)):
            with self.subTest(types=types), self.assertRaises(ValueError):
                operation.verify(types)

    def test_reshape_preserves_element_count(self) -> None:
        """Reshape changes dimensions while keeping the flattened element order."""
        from vllm_cent.pipeline.operations import ReshapeOperation

        operation = ReshapeOperation(
            name="reshape", inputs=("x",), output=Value("out", TensorType((3, 2)))
        )
        operation.verify((TensorType((2, 3)),))
        for types in (
            (),
            (TensorType((2, 3)), TensorType((2, 3))),
            (TensorType((5,)),),
        ):
            with self.subTest(types=types), self.assertRaises(ValueError):
                operation.verify(types)
        ReshapeOperation(
            name="scalar", inputs=("x",), output=Value("out", TensorType(()))
        ).verify((TensorType((1,)),))

    def test_unary_operations_check_operand_names_too(self) -> None:
        """Direct verification rejects a type tuple inconsistent with the declared arity."""
        from vllm_cent.pipeline.operations import CopyOperation, ReshapeOperation

        tensor = TensorType((3,))
        for operation_type in (CopyOperation, ReshapeOperation):
            with (
                self.subTest(operation_type=operation_type),
                self.assertRaises(ValueError),
            ):
                operation_type(
                    name="invalid", inputs=("x", "y"), output=Value("out", tensor)
                ).verify((tensor,))
