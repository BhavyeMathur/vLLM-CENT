"""Tests for decomposing composite semantic ops: SoftmaxOp and DecomposeOps."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from vllm_cent.activations import ActivationKind
from vllm_cent.ir import (
    ActivationOp,
    BroadcastOp,
    CopyOp,
    DivOp,
    DType,
    ExpOp,
    Function,
    LinearOp,
    MaxOp,
    Module,
    MulOp,
    Operation,
    PassManager,
    Shard,
    SoftmaxOp,
    SubOp,
    SumOp,
    TensorType,
    Value,
    print_function,
)
from vllm_cent.ir.base import Decomposable
from vllm_cent.ir.passes import DecomposeOps

# One rank's attention scores under tp=8: 8 of 64 query heads, all 2048
# cached positions. Softmax normalizes over the positions (last dimension).
SCORES = TensorType(
    shape=(1, 1, 8, 2048),
    dtype=DType.FP32,
    axes=("batch", "seq_len", "q_head", "context"),
    global_shape=(1, 1, 64, 2048),
    placement=Shard("q_head"),
)


def _softmax_function(x_type: TensorType) -> Function:
    """Return ``main(x) = softmax(x)``, with the softmax named ``attn.softmax``.

    Args:
        x_type: Type of the argument.

    Returns:
        A one-op function.
    """

    x = Value(type=x_type, name="x")
    softmax = SoftmaxOp(operands=(x,), name_hint="attn.softmax")
    return Function(
        name="main", operands=(x,), operations=(softmax,), results=softmax.results
    )


@dataclass(frozen=True, eq=False, slots=True)
class _Once(CopyOp, Decomposable):
    """A copy that decomposes into a plain copy: one level of nesting."""

    NAME = "test.once"

    def decompose(self) -> tuple[Operation, ...]:
        """Return a plain copy of the operand.

        Returns:
            One CopyOp.
        """
        return (CopyOp(operands=self.operands),)


@dataclass(frozen=True, eq=False, slots=True)
class _Twice(CopyOp, Decomposable):
    """A copy that decomposes into ``_Once``: two levels, so two rounds."""

    NAME = "test.twice"

    def decompose(self) -> tuple[Operation, ...]:
        """Return a ``_Once`` of the operand.

        Returns:
            One _Once.
        """
        return (_Once(operands=self.operands),)


@dataclass(frozen=True, eq=False, slots=True)
class _Forever(CopyOp, Decomposable):
    """A copy that decomposes into itself, so decomposition never ends."""

    NAME = "test.forever"

    def decompose(self) -> tuple[Operation, ...]:
        """Return a new ``_Forever`` of the operand.

        Returns:
            One _Forever.
        """
        return (_Forever(operands=self.operands),)


def _unary_function(op_class: type[Operation]) -> Function:
    """Return ``main(x) = op_class(x)`` over a [4] fp32 vector.

    Args:
        op_class: A one-operand op class without attributes.

    Returns:
        A one-op function.
    """

    x = Value(type=TensorType(shape=(4,), dtype=DType.FP32), name="x")
    op = op_class(operands=(x,))
    return Function(name="main", operands=(x,), operations=(op,), results=op.results)


class SoftmaxDecompositionTest(unittest.TestCase):
    """softmax(x) = exp(x - max(x)) / sum(exp(x - max(x))), row by row."""

    def test_ops_and_names(self) -> None:
        """Seven ops named after the softmax; the last keeps its name."""

        (softmax,) = _softmax_function(SCORES).operations
        assert isinstance(softmax, SoftmaxOp)
        ops = softmax.decompose()
        self.assertEqual(
            [type(op) for op in ops],
            [MaxOp, BroadcastOp, SubOp, ExpOp, SumOp, BroadcastOp, DivOp],
        )
        self.assertEqual(
            [op.name_hint for op in ops],
            [
                "attn.softmax.max",
                "attn.softmax.max_rows",
                "attn.softmax.shifted",
                "attn.softmax.exp",
                "attn.softmax.sum",
                "attn.softmax.sum_rows",
                "attn.softmax",
            ],
        )
        # Both broadcasts restore the reduced dimension: 2048 positions.
        broadcasts = [op for op in ops if isinstance(op, BroadcastOp)]
        self.assertEqual(
            [(b.size, b.axis) for b in broadcasts], [(2048, "context")] * 2
        )

    def test_result_type_equals_softmax_type(self) -> None:
        """Unannotated, annotated, and tensor-parallel inputs keep their type.

        The sharded case also checks every sub-op's placement rule: the
        q_head split must survive max, broadcast, sub, exp, sum and div.
        """

        unannotated = TensorType(shape=(2, 5), dtype=DType.FP32)
        annotated = TensorType(
            shape=(2, 5), dtype=DType.FP32, axes=("seq_len", "context")
        )
        for x_type in (unannotated, annotated, SCORES):
            with self.subTest(x_type=x_type):
                (softmax,) = _softmax_function(x_type).operations
                assert isinstance(softmax, Decomposable)
                last = softmax.decompose()[-1]
                self.assertEqual(last.results[0].type, softmax.results[0].type)


class DecomposeOpsTest(unittest.TestCase):
    """The pass replaces selected decomposable ops, in rounds."""

    def test_decomposes_softmax_through_the_pass_manager(self) -> None:
        """The pass output verifies, and the result keeps the softmax's name."""

        module = Module(name="m", functions=(_softmax_function(SCORES),))
        decomposed = PassManager((DecomposeOps(),)).run(module)
        function = decomposed.get_function("main")
        self.assertEqual(
            [op.NAME for op in function.operations],
            [
                "aloi.reduce_max",
                "aloi.broadcast",
                "aloi.sub",
                "aloi.exp",
                "aloi.reduce_sum",
                "aloi.broadcast",
                "aloi.div",
            ],
        )
        self.assertEqual(function.results[0].type, SCORES)
        self.assertTrue(print_function(function).endswith("  return %attn.softmax\n}"))
        # The input module is untouched.
        self.assertIsInstance(module.get_function("main").operations[0], SoftmaxOp)

    def test_unselected_and_plain_ops_stay(self) -> None:
        """select=False keeps the softmax; a function without one is returned as is."""

        function = _softmax_function(SCORES)
        kept = DecomposeOps(select=lambda op: False).run_on_function(function)
        self.assertIs(kept, function)
        x = Value(type=TensorType(shape=(1, 4), dtype=DType.FP32), name="x")
        w = Value(type=TensorType(shape=(3, 4), dtype=DType.FP32), name="w")
        linear = LinearOp(operands=(x, w))
        plain = Function(
            name="main", operands=(x, w), operations=(linear,), results=linear.results
        )
        self.assertIs(DecomposeOps().run_on_function(plain), plain)

    def test_nested_decomposition_takes_one_round_per_level(self) -> None:
        """_Twice -> _Once -> copy needs exactly two rounds, and two suffice."""

        function = DecomposeOps(max_rounds=2).run_on_function(_unary_function(_Twice))
        self.assertEqual([type(op) for op in function.operations], [CopyOp])
        with self.assertRaisesRegex(ValueError, r"after 1 rounds.*'test\.once'"):
            DecomposeOps(max_rounds=1).run_on_function(_unary_function(_Twice))

    def test_self_decomposition_is_reported(self) -> None:
        """An op that decomposes into itself never finishes."""

        with self.assertRaisesRegex(ValueError, "produce each other"):
            DecomposeOps().run_on_function(_unary_function(_Forever))

    def test_rejects_no_rounds(self) -> None:
        """At least one round is needed to decompose anything."""

        with self.assertRaisesRegex(ValueError, "at least 1"):
            DecomposeOps(max_rounds=0)


class SiluDecompositionTest(unittest.TestCase):
    """SiLU decomposes into x * sigmoid(x); other activations do not."""

    def test_only_silu_can_decompose(self) -> None:
        """can_decompose() is True for SiLU alone."""

        x = Value(type=SCORES, name="x")
        self.assertEqual(
            {
                kind: ActivationOp(operands=(x,), function=kind).can_decompose()
                for kind in ActivationKind
            },
            {kind: kind is ActivationKind.SILU for kind in ActivationKind},
        )

    def test_decomposition(self) -> None:
        """A sigmoid of x, then x times it; the product keeps the SiLU's name."""

        x = Value(type=SCORES, name="x")
        silu = ActivationOp(
            operands=(x,), function=ActivationKind.SILU, name_hint="mlp.silu"
        )
        sigmoid, product = silu.decompose()

        assert isinstance(sigmoid, ActivationOp)
        self.assertEqual(
            (sigmoid.operands, sigmoid.function, sigmoid.name_hint),
            ((x,), ActivationKind.SIGMOID, "mlp.silu.sigmoid"),
        )
        self.assertIsInstance(product, MulOp)
        self.assertEqual(
            (product.operands, product.name_hint), ((x, *sigmoid.results), "mlp.silu")
        )
        # Same type as the SiLU result: shape, axes and the q_head shard.
        self.assertEqual(product.results[0].type, silu.results[0].type)

    def test_other_functions_raise(self) -> None:
        """decompose() on ReLU names the function."""

        relu = ActivationOp(
            operands=(Value(type=SCORES),), function=ActivationKind.RELU
        )
        with self.assertRaisesRegex(ValueError, "relu has no decomposition"):
            relu.decompose()

    def test_pass_decomposes_silu_and_keeps_gelu(self) -> None:
        """DecomposeOps asks can_decompose() before select."""

        x = Value(type=SCORES, name="x")
        silu = ActivationOp(operands=(x,), function=ActivationKind.SILU)
        gelu = ActivationOp(operands=silu.results, function=ActivationKind.GELU_TANH)
        function = Function(
            name="main", operands=(x,), operations=(silu, gelu), results=gelu.results
        )
        decomposed = DecomposeOps().run_on_function(function)
        self.assertEqual(
            [op.NAME for op in decomposed.operations],
            ["aloi.activation", "aloi.mul", "aloi.activation"],
        )
        self.assertEqual(
            [
                op.function
                for op in decomposed.operations
                if isinstance(op, ActivationOp)
            ],
            [ActivationKind.SIGMOID, ActivationKind.GELU_TANH],
        )


if __name__ == "__main__":
    unittest.main()
