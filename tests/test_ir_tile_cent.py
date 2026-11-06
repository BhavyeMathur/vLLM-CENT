"""Tests for CENT Tile IR: CentTensorType, the printer hook, and CentTileOp."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, replace

from vllm_cent.ir import Block, Value
from vllm_cent.ir.printer import format_type
from vllm_cent.ir.tile import INDEX, CentTensorType, CentTileOp, TileForOp
from vllm_cent.ir.tile.cent import _expect, _expect_indices
from vllm_cent.ir.types import DType, ExecutionDomain, MemorySpace, TensorType


def _cent(space: MemorySpace, shape: tuple[int, ...] = (16,)) -> CentTensorType:
    return CentTensorType(shape=shape, dtype=DType.BF16, memory_space=space)


@dataclass(frozen=True, eq=False, slots=True)
class _CopyOp(CentTileOp):
    """``test.cent_copy(x)``: x's type, unchecked; only the channels matter."""

    NAME = "test.cent_copy"
    NUM_OPERANDS = 1

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return the operand's type."""
        return (self.operands[0].type,)


class CentTensorTypeTest(unittest.TestCase):
    """The memory space refines the domain and takes part in equality."""

    def test_domain_comes_from_the_space(self) -> None:
        """Each space fills in the domain before its dot."""

        self.assertIs(_cent(MemorySpace.PIM_MAC).domain, ExecutionDomain.PIM)
        self.assertIs(_cent(MemorySpace.PNM_SB).domain, ExecutionDomain.PNM)

    def test_matching_domain_is_accepted(self) -> None:
        """A domain that agrees with the space may be given explicitly."""

        tile = CentTensorType(
            shape=(16,),
            dtype=DType.BF16,
            domain=ExecutionDomain.PIM,
            memory_space=MemorySpace.PIM_GB,
        )
        self.assertEqual(tile, _cent(MemorySpace.PIM_GB))

    def test_conflicting_domain_is_rejected(self) -> None:
        """The Shared Buffer is PNM storage, not PIM."""

        with self.assertRaisesRegex(
            ValueError, "pnm.sb belongs to domain pnm, but the type says pim"
        ):
            CentTensorType(
                shape=(16,),
                dtype=DType.BF16,
                domain=ExecutionDomain.PIM,
                memory_space=MemorySpace.PNM_SB,
            )

    def test_equality_compares_the_space(self) -> None:
        """Same shape, other space or no space: different types."""

        sb = _cent(MemorySpace.PNM_SB)
        self.assertEqual(sb, _cent(MemorySpace.PNM_SB))
        self.assertNotEqual(sb, _cent(MemorySpace.PIM_MAC))
        plain = TensorType(shape=(16,), dtype=DType.BF16, domain=ExecutionDomain.PNM)
        self.assertNotEqual(sb, plain)
        self.assertNotEqual(plain, sb)

    def test_replace_keeps_the_space(self) -> None:
        """dataclasses.replace builds the same class, with the same space."""

        moved = replace(_cent(MemorySpace.PIM_GB), shape=(32,))
        self.assertIsInstance(moved, CentTensorType)
        self.assertIs(moved.memory_space, MemorySpace.PIM_GB)

    def test_loop_rejects_a_change_of_space(self) -> None:
        """tile.for's induction check sees a value yielded in another space."""

        initial = Value(type=_cent(MemorySpace.PIM_MAC), name="acc")
        body = Block(
            (Value(type=INDEX), Value(type=initial.type)),
            (),
            (Value(type=_cent(MemorySpace.PNM_SB)),),
        )
        with self.assertRaisesRegex(ValueError, "memory_space=pnm.sb"):
            TileForOp(operands=(initial,), extent=2, body=body)

    def test_printed_after_the_domain(self) -> None:
        """The printer appends the subclass's annotation last."""

        self.assertEqual(
            format_type(_cent(MemorySpace.PIM_MAC, (32, 16))),
            "tensor<32x16xbf16, domain=pim, memory_space=pim.mac>",
        )
        self.assertEqual(
            format_type(TensorType(shape=(4,), dtype=DType.BF16)),
            "tensor<4xbf16>",
        )


class CentTileOpTest(unittest.TestCase):
    """Every CENT op checks its channel list before its operands."""

    def test_channels_are_kept_as_a_tuple(self) -> None:
        """A list is accepted and stored as a tuple, like operands."""

        x = Value(type=_cent(MemorySpace.PNM_SB), name="x")
        op = _CopyOp(operands=(x,), channels=[0, 2, 5])  # type: ignore[arg-type]
        self.assertEqual(op.channels, (0, 2, 5))
        self.assertEqual(op.attributes(), {"channels": (0, 2, 5)})
        self.assertEqual(op.results[0].type, x.type)

    def test_rejects_bad_channels(self) -> None:
        """Empty, negative, unsorted and repeated channel lists are errors."""

        x = Value(type=_cent(MemorySpace.PNM_SB), name="x")
        cases = (
            ((), "at least one"),
            ((-1, 0), "non-negative"),
            ((1, 0), "ascending without repeats"),
            ((0, 0), "ascending without repeats"),
        )
        for channels, message in cases:
            with self.subTest(channels=channels):
                with self.assertRaisesRegex(ValueError, message):
                    _CopyOp(operands=(x,), channels=channels)


class OperandCheckTest(unittest.TestCase):
    """The helpers that the cent.* ops use for their operands."""

    def test_expect_returns_the_narrowed_type(self) -> None:
        """An operand in the right space passes, with its CentTensorType."""

        gb = Value(type=_cent(MemorySpace.PIM_GB))
        self.assertIs(_expect("op", "input", gb, MemorySpace.PIM_GB), gb.type)

    def test_expect_rejects_other_spaces(self) -> None:
        """Another space, or no space at all, is reported with the role."""

        sb = Value(type=_cent(MemorySpace.PNM_SB))
        with self.assertRaisesRegex(ValueError, "op: input must be in pim.gb, got"):
            _expect("op", "input", sb, MemorySpace.PIM_GB)
        plain = Value(type=TensorType(shape=(16,), dtype=DType.BF16))
        with self.assertRaisesRegex(ValueError, "has no memory space"):
            _expect("op", "input", plain, MemorySpace.PIM_GB)

    def test_expect_indices(self) -> None:
        """One distinct name per INDEX operand."""

        i, j = Value(type=INDEX), Value(type=INDEX)
        _expect_indices("op", ("n_wave", "k_row"), (i, j))
        bad = Value(type=TensorType(shape=(), dtype=DType.INT64))
        cases = (
            (("n_wave",), (i, j), "names 1 dimensions"),
            (("n_wave", "n_wave"), (i, j), "distinct"),
            (("",), (i,), "distinct"),
            (("n_wave",), (bad,), "the index for 'n_wave' must be"),
        )
        for at, indices, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _expect_indices("op", at, indices)


if __name__ == "__main__":
    unittest.main()
