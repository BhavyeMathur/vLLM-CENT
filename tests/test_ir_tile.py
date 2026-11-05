"""Tests for structured loops: Block, tile.for, scoping, and printing."""

from __future__ import annotations

import unittest
from dataclasses import dataclass

from vllm_cent.ir import Block, Function, Value, clone_function, print_function
from vllm_cent.ir.base import Operation, rebuild_function
from vllm_cent.ir.printer import value_names
from vllm_cent.ir.tile import INDEX, TileForOp
from vllm_cent.ir.types import DType, TensorType

TILE = TensorType(shape=(4,), dtype=DType.BF16)
OTHER = TensorType(shape=(8,), dtype=DType.BF16)


@dataclass(frozen=True, eq=False, slots=True)
class _StepOp(Operation):
    """``test.step(a, b)``: one loop step; the result has ``a``'s type.

    Stands in for a real tile op, so these tests depend on no target.
    """

    NAME = "test.step"
    NUM_OPERANDS = 2

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return ``a``'s type."""
        return (self.operands[0].type,)


def _loop(
    initial: Value, extent: int = 4, other: Value | None = None, hint: str = "loop"
) -> tuple[TileForOp, Value, Value, _StepOp]:
    """Build ``for index: carried = step(carried, other or index)``.

    Returns:
        ``(loop, index, carried, step)``.
    """

    index = Value(type=INDEX, name="i")
    carried = Value(type=initial.type, name="a")
    step = _StepOp(operands=(carried, other or index), name_hint="step")
    body = Block((index, carried), (step,), step.results)
    loop = TileForOp(operands=(initial,), extent=extent, body=body, name_hint=hint)
    return loop, index, carried, step


class TileForTypeTest(unittest.TestCase):
    """tile.for checks its carried types by induction when it is built."""

    def test_results_have_the_carried_types(self) -> None:
        """The loop returns one value per carried value, of the same type."""

        x = Value(type=TILE, name="x")
        loop, index, carried, _ = _loop(x)
        self.assertEqual([r.type for r in loop.results], [TILE])
        self.assertIs(loop.index, index)
        self.assertEqual(loop.carried, (carried,))
        self.assertEqual(loop.blocks(), (loop.body,))
        self.assertEqual(loop.attributes(), {"extent": 4})

    def test_loop_without_carried_values(self) -> None:
        """A loop may carry nothing; its body takes only the index."""

        loop = TileForOp(
            operands=(), extent=2, body=Block((Value(type=INDEX),), (), ())
        )
        self.assertEqual(loop.results, ())

    def test_rejects_bad_loops(self) -> None:
        """Each broken rule is reported when the op is built."""

        x = Value(type=TILE, name="x")

        def body(arguments: tuple[TensorType, ...], yields: TensorType) -> Block:
            values = tuple(Value(type=t) for t in arguments)
            return Block(values, (), (Value(type=yields),))

        int64_index = TensorType(shape=(), dtype=DType.INT64)
        # (message, extent, body)
        cases = (
            ("extent must be at least 1", 0, body((INDEX, TILE), TILE)),
            ("need a body with 2 arguments", 2, body((INDEX,), TILE)),
            (
                "need a body with 1 results",
                2,
                Block((Value(type=INDEX), Value(type=TILE)), (), ()),
            ),
            ("the index must be", 2, body((int64_index, TILE), TILE)),
            # The first iteration would read x as a value of the wrong type.
            ("starts as tensor<4xbf16>", 2, body((INDEX, OTHER), OTHER)),
            # A later iteration would read the yield as the wrong type.
            ("yielded as tensor<8xbf16>", 2, body((INDEX, TILE), OTHER)),
        )
        for message, extent, loop_body in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    TileForOp(operands=(x,), extent=extent, body=loop_body)


class TileScopeTest(unittest.TestCase):
    """Function.verify checks which values a block may use."""

    def test_nested_loop_sees_outer_values(self) -> None:
        """A body may use the function's arguments and an outer loop's index."""

        x = Value(type=TILE, name="x")
        w = Value(type=TILE, name="w")
        outer_index = Value(type=INDEX, name="i")
        outer_carried = Value(type=TILE, name="a")
        inner, *_ = _loop(outer_carried, other=outer_index, hint="inner")
        # The outer body also reads w, a function argument.
        mix = _StepOp(operands=(inner.results[0], w), name_hint="mix")
        outer = TileForOp(
            operands=(x,),
            extent=2,
            body=Block((outer_index, outer_carried), (inner, mix), mix.results),
            name_hint="outer",
        )
        Function("main", (x, w), (outer,), outer.results).verify()

    def test_body_cannot_use_a_later_value(self) -> None:
        """A body sees the values defined before its loop, not after it."""

        x = Value(type=TILE, name="x")
        later = _StepOp(operands=(x, x), name_hint="later")
        loop, *_ = _loop(x, other=later.results[0])
        function = Function("main", (x,), (loop, later), later.results)
        with self.assertRaisesRegex(ValueError, r"test\.step \(step\) uses"):
            function.verify()

    def test_outer_code_cannot_use_a_body_value(self) -> None:
        """Only the loop's results leave the body."""

        x = Value(type=TILE, name="x")
        loop, _, _, step = _loop(x)
        leak = _StepOp(operands=(step.results[0], x), name_hint="leak")
        function = Function("main", (x,), (loop, leak), leak.results)
        with self.assertRaisesRegex(ValueError, r"test\.step \(leak\) uses"):
            function.verify()

    def test_body_cannot_return_an_outer_value_it_cannot_see(self) -> None:
        """A yield follows the same rule as an operand."""

        x = Value(type=TILE, name="x")
        stranger = Value(type=TILE, name="stranger")
        index = Value(type=INDEX, name="i")
        carried = Value(type=TILE, name="a")
        loop = TileForOp(
            operands=(x,), extent=2, body=Block((index, carried), (), (stranger,))
        )
        function = Function("main", (x,), (loop,), loop.results)
        with self.assertRaisesRegex(ValueError, "body of tile.for returns"):
            function.verify()

    def test_one_block_cannot_be_used_twice(self) -> None:
        """Two loops sharing a body would define its values twice."""

        x = Value(type=TILE, name="x")
        first, *_ = _loop(x)
        second = TileForOp(operands=first.results, extent=2, body=first.body)
        function = Function("main", (x,), (first, second), second.results)
        with self.assertRaisesRegex(ValueError, "argument %i twice"):
            function.verify()

    def test_block_argument_cannot_redefine_an_outer_value(self) -> None:
        """A block argument is a new value, never one of the code around it."""

        x = Value(type=TILE, name="x")
        index = Value(type=INDEX, name="i")
        loop = TileForOp(operands=(x,), extent=2, body=Block((index, x), (), (x,)))
        function = Function("main", (x,), (loop,), loop.results)
        with self.assertRaisesRegex(ValueError, "argument %x twice"):
            function.verify()

    def test_rebuild_rejects_nested_blocks(self) -> None:
        """No pass rewrites loop bodies yet, so rebuilding fails loudly."""

        x = Value(type=TILE, name="x")
        loop, *_ = _loop(x)
        function = Function("main", (x,), (loop,), loop.results)
        with self.assertRaisesRegex(NotImplementedError, "nested blocks"):
            rebuild_function(function, (Value(type=TILE, name="y"),))
        with self.assertRaises(NotImplementedError):
            clone_function(function)


class TilePrinterTest(unittest.TestCase):
    """The printer prints each block under the op that owns it."""

    def _nested(self) -> Function:
        x = Value(type=TILE, name="x")
        outer_index = Value(type=INDEX, name="i")
        outer_carried = Value(type=TILE, name="a")
        inner, *_ = _loop(outer_carried, extent=3, other=outer_index, hint="inner")
        outer = TileForOp(
            operands=(x,),
            extent=2,
            body=Block((outer_index, outer_carried), (inner,), inner.results),
            name_hint="outer",
        )
        return Function("main", (x,), (outer,), outer.results)

    def test_nested_loops(self) -> None:
        """Names are unique in the whole function: the inner %i becomes %i_1."""

        function = self._nested()
        function.verify()
        self.assertEqual(
            print_function(function),
            "\n".join(
                [
                    "func @main(%x: tensor<4xbf16>) -> (tensor<4xbf16>) {",
                    "  %outer = tile.for(%x) {extent = 2} : tensor<4xbf16> {",
                    "  ^(%i: tensor<int32>, %a: tensor<4xbf16>):",
                    "    %inner = tile.for(%a) {extent = 3} : tensor<4xbf16> {",
                    "    ^(%i_1: tensor<int32>, %a_1: tensor<4xbf16>):",
                    "      %step = test.step(%a_1, %i) : tensor<4xbf16>",
                    "      yield %step",
                    "    }",
                    "    yield %inner",
                    "  }",
                    "  return %outer",
                    "}",
                ]
            ),
        )

    def test_loop_without_results(self) -> None:
        """A loop that carries nothing prints no results and a bare yield."""

        loop = TileForOp(
            operands=(), extent=2, body=Block((Value(type=INDEX),), (), ())
        )
        function = Function("main", (), (loop,), ())
        self.assertEqual(
            print_function(function),
            "\n".join(
                [
                    "func @main() -> () {",
                    "  tile.for() {extent = 2} {",
                    "  ^(%0: tensor<int32>):",
                    "    yield",
                    "  }",
                    "  return",
                    "}",
                ]
            ),
        )

    def test_value_names_match_the_dump(self) -> None:
        """value_names gives block values the names they print with."""

        function = self._nested()
        names = value_names(function)
        outer = function.operations[0]
        assert isinstance(outer, TileForOp)
        inner = outer.body.operations[0]
        assert isinstance(inner, TileForOp)
        self.assertEqual(names[outer.index], "i")
        self.assertEqual(names[inner.index], "i_1")
        self.assertEqual(names[inner.carried[0]], "a_1")
        self.assertEqual(names[inner.body.results[0]], "step")
        self.assertEqual(names[outer.results[0]], "outer")


if __name__ == "__main__":
    unittest.main()
