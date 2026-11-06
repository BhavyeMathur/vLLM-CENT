"""Tests for CENT Tile IR: CentTensorType, the printer hook, and the cent ops."""

from __future__ import annotations

import unittest
from dataclasses import dataclass, replace

from vllm_cent.ir import Block, Function, Value, print_function
from vllm_cent.ir.printer import format_type
from vllm_cent.ir.tile import (
    INDEX,
    CentMacAbkOp,
    CentRdMacOp,
    CentTensorType,
    CentTileOp,
    CentWrBiasOp,
    CentWrGbOp,
    TileForOp,
)
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


# The small GEMV of the tests: 4 channels of 16 banks, so one wave computes
# 64 outputs; N = 128 outputs in 2 waves, K = 64 in 2 Global Buffer tiles
# of 32 values.
CHANNELS = (0, 1, 2, 3)
MAC = (4, 16)
N, K, GB = 128, 64, 32


def _value(name: str, space: MemorySpace, shape: tuple[int, ...]) -> Value:
    return Value(type=_cent(space, shape), name=name)


def _index(name: str) -> Value:
    return Value(type=INDEX, name=name)


class _Operands:
    """One fresh set of GEMV operands, in their memory spaces."""

    def __init__(self) -> None:
        self.x = _value("x", MemorySpace.PNM_SB, (K,))
        self.w = _value("w", MemorySpace.PIM_DRAM, (N, K))
        self.y = _value("y", MemorySpace.PNM_SB, (N,))
        self.zeros = _value("zeros", MemorySpace.PNM_SB, (16,))
        self.gb = _value("gb", MemorySpace.PIM_GB, (GB,))
        self.acc = _value("acc", MemorySpace.PIM_MAC, MAC)
        self.n = _index("n")
        self.kt = _index("kt")


class CentOpTypeTest(unittest.TestCase):
    """Each cent op checks its operands and gives its result type."""

    def setUp(self) -> None:
        """Give each test fresh operands."""
        self.v = _Operands()

    def test_wr_bias(self) -> None:
        """MAC registers of the given shape, in the source's dtype."""

        v = self.v
        for stride, operands, at in ((0, (v.zeros,), ()), (1, (v.y, v.n), ("n",))):
            with self.subTest(stride=stride):
                op = CentWrBiasOp(
                    operands=operands,
                    channels=CHANNELS,
                    at=at,
                    stride=stride,
                    shape=list(MAC),  # type: ignore[arg-type]
                )
                self.assertEqual(op.shape, MAC)
                self.assertEqual(op.results[0].type, _cent(MemorySpace.PIM_MAC, MAC))
                self.assertIs(op.source, operands[0])
                self.assertEqual(op.indices, operands[1:])

    def test_wr_gb(self) -> None:
        """One channel's Global Buffer tile, of the given shape."""

        v = self.v
        op = CentWrGbOp(
            operands=(v.x, v.kt),
            channels=CHANNELS,
            at=["k_tile"],  # type: ignore[arg-type]
            shape=(GB,),
        )
        self.assertEqual(op.at, ("k_tile",))
        self.assertEqual(op.results[0].type, _cent(MemorySpace.PIM_GB, (GB,)))

    def test_mac_abk_returns_the_accumulator_type(self) -> None:
        """The result has acc's type, so a loop can carry it."""

        v = self.v
        op = CentMacAbkOp(
            operands=(v.w, v.gb, v.acc, v.n, v.kt),
            channels=CHANNELS,
            at=("n_wave", "k_row"),
        )
        self.assertIs(op.results[0].type, v.acc.type)
        self.assertEqual(
            (op.weight, op.global_buffer, op.accumulator), (v.w, v.gb, v.acc)
        )
        self.assertEqual(op.indices, (v.n, v.kt))

    def test_rd_mac_returns_a_new_version_of_y(self) -> None:
        """The result has y's type but is a new value."""

        v = self.v
        op = CentRdMacOp(operands=(v.acc, v.y, v.n), channels=CHANNELS, at=("n_wave",))
        self.assertIs(op.results[0].type, v.y.type)
        self.assertIsNot(op.results[0], v.y)
        self.assertEqual((op.accumulator, op.destination), (v.acc, v.y))

    def test_rejects_bad_operands(self) -> None:
        """Each rule has one counterexample; the message names the rule."""

        v = self.v
        bf16_dram = _value("w16", MemorySpace.PIM_DRAM, (N, K))
        fp16_gb = Value(
            type=CentTensorType(
                shape=(GB,), dtype=DType.FP16, memory_space=MemorySpace.PIM_GB
            )
        )
        three = (0, 1, 2)
        cases = (
            # wr_bias
            (
                "source must be in pnm.sb",
                lambda: CentWrBiasOp(
                    operands=(v.gb,), channels=CHANNELS, at=(), stride=0, shape=MAC
                ),
            ),
            (
                "stride must be 0 or 1",
                lambda: CentWrBiasOp(
                    operands=(v.zeros,), channels=CHANNELS, at=(), stride=2, shape=MAC
                ),
            ),
            (
                r"MAC registers of 3 channels have shape \[3, banks\]",
                lambda: CentWrBiasOp(
                    operands=(v.zeros,), channels=three, at=(), stride=0, shape=MAC
                ),
            ),
            (
                "MAC registers of 4 channels",
                lambda: CentWrBiasOp(
                    operands=(v.zeros,),
                    channels=CHANNELS,
                    at=(),
                    stride=0,
                    shape=(4, 16, 1),
                ),
            ),
            # wr_gb
            (
                "shape must have positive dimensions",
                lambda: CentWrGbOp(
                    operands=(v.x, v.kt), channels=CHANNELS, at=("k",), shape=(0,)
                ),
            ),
            (
                "at names 0 dimensions",
                lambda: CentWrGbOp(
                    operands=(v.x, v.kt), channels=CHANNELS, at=(), shape=(GB,)
                ),
            ),
            # mac_abk
            (
                "expects 3 operands before its indices, got 2",
                lambda: CentMacAbkOp(operands=(v.w, v.gb), channels=CHANNELS, at=()),
            ),
            (
                "weight must be in pim.dram, got pnm.sb",
                lambda: CentMacAbkOp(
                    operands=(v.y, v.gb, v.acc), channels=CHANNELS, at=()
                ),
            ),
            (
                "global buffer must be in pim.gb",
                lambda: CentMacAbkOp(
                    operands=(bf16_dram, v.x, v.acc), channels=CHANNELS, at=()
                ),
            ),
            (
                "accumulator must be in pim.mac",
                lambda: CentMacAbkOp(
                    operands=(v.w, v.gb, v.y), channels=CHANNELS, at=()
                ),
            ),
            (
                "share one dtype, got weight bf16, global_buffer fp16",
                lambda: CentMacAbkOp(
                    operands=(v.w, fp16_gb, v.acc), channels=CHANNELS, at=()
                ),
            ),
            (
                "MAC registers of 3 channels",
                lambda: CentMacAbkOp(
                    operands=(v.w, v.gb, v.acc), channels=three, at=()
                ),
            ),
            # rd_mac
            (
                "destination must be in pnm.sb",
                lambda: CentRdMacOp(operands=(v.acc, v.gb), channels=CHANNELS, at=()),
            ),
            (
                "the index for 'n_wave' must be",
                lambda: CentRdMacOp(
                    operands=(v.acc, v.y, v.x), channels=CHANNELS, at=("n_wave",)
                ),
            ),
        )
        for message, build in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    build()

    def test_channels_are_checked_first(self) -> None:
        """A bad channel list is reported before any operand rule."""

        v = self.v
        with self.assertRaisesRegex(ValueError, "ascending"):
            CentMacAbkOp(operands=(v.y,), channels=(1, 0), at=())


def _output_stationary() -> Function:
    """gemv.py's schedule 1: per wave, the MAC registers reduce all of K."""

    v = _Operands()
    y_now, acc = _value("y_now", MemorySpace.PNM_SB, (N,)), v.acc
    acc0 = CentWrBiasOp(
        operands=(v.zeros,),
        channels=CHANNELS,
        at=(),
        stride=0,
        shape=MAC,
        name_hint="acc0",
    )
    gb = CentWrGbOp(
        operands=(v.x, v.kt),
        channels=CHANNELS,
        at=("k_tile",),
        shape=(GB,),
        name_hint="gb",
    )
    mac = CentMacAbkOp(
        operands=(v.w, gb.results[0], acc, v.n, v.kt),
        channels=CHANNELS,
        at=("n_wave", "k_row"),
        name_hint="mac",
    )
    reduce = TileForOp(
        operands=acc0.results,
        extent=K // GB,
        body=Block((v.kt, acc), (gb, mac), mac.results),
        name_hint="reduce",
    )
    save = CentRdMacOp(
        operands=(reduce.results[0], y_now, v.n),
        channels=CHANNELS,
        at=("n_wave",),
        name_hint="y_next",
    )
    waves = TileForOp(
        operands=(v.y,),
        extent=N // (len(CHANNELS) * MAC[1]),
        body=Block((v.n, y_now), (acc0, reduce, save), save.results),
        name_hint="waves",
    )
    return Function("gemv", (v.x, v.w, v.y, v.zeros), (waves,), waves.results)


class CentKernelTest(unittest.TestCase):
    """The GEMV kernels of examples/Tile_DSL/gemv.py, written in IR."""

    def test_output_stationary_gemv(self) -> None:
        """Schedule 1 verifies and prints as below."""

        function = _output_stationary()
        function.verify()
        self.assertEqual(print_function(function), OUTPUT_STATIONARY)

    def test_gb_stationary_gemv(self) -> None:
        """Schedule 2: partial sums go to y and back between k_tiles."""

        v = _Operands()
        partial = _value("partial", MemorySpace.PNM_SB, (N,))
        partial_now = _value("partial_now", MemorySpace.PNM_SB, (N,))
        gb = CentWrGbOp(
            operands=(v.x, v.kt), channels=CHANNELS, at=("k_tile",), shape=(GB,)
        )
        restore = CentWrBiasOp(
            operands=(partial_now, v.n),
            channels=CHANNELS,
            at=("n_wave",),
            stride=1,
            shape=MAC,
        )
        mac = CentMacAbkOp(
            operands=(v.w, gb.results[0], restore.results[0], v.n, v.kt),
            channels=CHANNELS,
            at=("n_wave", "k_row"),
        )
        save = CentRdMacOp(
            operands=(mac.results[0], partial_now, v.n),
            channels=CHANNELS,
            at=("n_wave",),
        )
        waves = TileForOp(
            operands=(partial,),
            extent=2,
            body=Block((v.n, partial_now), (restore, mac, save), save.results),
        )
        reduce = TileForOp(
            operands=(v.y,),
            extent=2,
            body=Block((v.kt, partial), (gb, waves), waves.results),
        )
        Function("gemv", (v.x, v.w, v.y), (reduce,), reduce.results).verify()

    def test_loop_rejects_registers_of_other_channels(self) -> None:
        """A body that yields MAC registers of 3 channels breaks the carry."""

        v = _Operands()
        three = CentWrBiasOp(
            operands=(v.zeros,), channels=(0, 1, 2), at=(), stride=0, shape=(3, 16)
        )
        acc0 = _value("acc0", MemorySpace.PIM_MAC, MAC)
        with self.assertRaisesRegex(ValueError, "yielded as tensor<3x16xbf16"):
            TileForOp(
                operands=(acc0,),
                extent=2,
                body=Block((v.kt, v.acc), (three,), three.results),
            )


OUTPUT_STATIONARY = "\n".join(
    [
        (
            "func @gemv(%x: tensor<64xbf16, domain=pnm, memory_space=pnm.sb>, "
            "%w: tensor<128x64xbf16, domain=pim, memory_space=pim.dram>, "
            "%y: tensor<128xbf16, domain=pnm, memory_space=pnm.sb>, "
            "%zeros: tensor<16xbf16, domain=pnm, "
            "memory_space=pnm.sb>) -> (tensor<128xbf16, domain=pnm, "
            "memory_space=pnm.sb>) {"
        ),
        (
            "  %waves = tile.for(%y) {extent = 2} : tensor<128xbf16, domain=pnm, "
            "memory_space=pnm.sb> {"
        ),
        (
            "  ^(%n: tensor<int32>, %y_now: tensor<128xbf16, domain=pnm, "
            "memory_space=pnm.sb>):"
        ),
        (
            "    %acc0 = cent.wr_bias(%zeros) {channels = [0, 1, 2, 3], at = [], "
            "stride = 0, shape = [4, 16]} : tensor<4x16xbf16, domain=pim, "
            "memory_space=pim.mac>"
        ),
        (
            "    %reduce = tile.for(%acc0) {extent = 2} : tensor<4x16xbf16, "
            "domain=pim, memory_space=pim.mac> {"
        ),
        (
            "    ^(%kt: tensor<int32>, %acc: tensor<4x16xbf16, domain=pim, "
            "memory_space=pim.mac>):"
        ),
        (
            "      %gb = cent.wr_gb(%x, %kt) {channels = [0, 1, 2, 3], "
            'at = ["k_tile"], shape = [32]} : tensor<32xbf16, domain=pim, '
            "memory_space=pim.gb>"
        ),
        (
            "      %mac = cent.mac_abk(%w, %gb, %acc, %n, %kt) {channels = [0, "
            '1, 2, 3], at = ["n_wave", "k_row"]} : tensor<4x16xbf16, domain=pim, '
            "memory_space=pim.mac>"
        ),
        "      yield %mac",
        "    }",
        (
            "    %y_next = cent.rd_mac(%reduce, %y_now, %n) {channels = [0, 1, "
            '2, 3], at = ["n_wave"]} : tensor<128xbf16, domain=pnm, '
            "memory_space=pnm.sb>"
        ),
        "    yield %y_next",
        "  }",
        "  return %waves",
        "}",
    ]
)


if __name__ == "__main__":
    unittest.main()
