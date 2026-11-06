"""Tests for the Tile DSL: what it records, the shapes it derives, its errors."""

from __future__ import annotations

import contextlib
import io
import runpy
import unittest
from collections.abc import Callable

from vllm_cent.ir import print_function
from vllm_cent.ir.tile import CentWrBiasOp, CentWrGbOp, TileForOp
from vllm_cent.ir.types import DType, MemorySpace
from vllm_cent.tile import Bank, Channel, Column, Lane, Row, Serial, Slot, split
from vllm_cent.tile.dsl import TileKernel

from test_tile_verify import SMALL_ARCH

SB, DRAM = MemorySpace.PNM_SB, MemorySpace.PIM_DRAM


class _Gemv:
    """The small GEMV of the verifier tests, declared with the DSL."""

    def __init__(self, name: str = "gemv") -> None:
        self.k = k = TileKernel(name, SMALL_ARCH)
        self.x = k.tensor("x", (64,), DType.BF16, SB)
        self.w = k.tensor("w", (128, 64), "bf16", DRAM)
        self.y = k.tensor("y", (128,), DType.BF16, SB)
        self.zeros = k.tensor("zeros", (16,), DType.BF16, SB)
        k.bind_layout(self.x, split(k=(Serial("k_tile", 2), Slot(), Lane())))
        k.bind_layout(
            self.w,
            split(
                n=(Serial("n_wave", 2), Channel(), Bank()),
                k=(Row("k_row", 2), Column()),
            ),
        )
        k.bind_layout(
            self.y, split(n=(Serial("n_wave", 2), Slot(of=Channel()), Lane(of=Bank())))
        )
        k.bind_layout(self.zeros, split(b=(Slot(1), Lane(of=Bank()))))
        self.chans = k.all_channels()

    def output_stationary(self) -> None:
        """Record gemv.py's schedule 1 and its results."""

        k, chans = self.k, self.chans
        with k.serial("n_wave", 2, carry=(self.y,)) as waves:
            n, (y_now,) = waves.index, waves.carried
            acc0 = k.wr_bias(self.zeros, stride=0, channels=chans)
            with k.serial("k_tile", 2, carry=(acc0,)) as reduce:
                kt, (acc,) = reduce.index, reduce.carried
                gb = k.wr_gb(self.x.tile(k_tile=kt), channels=chans)
                reduce.yield_(
                    k.mac_abk(self.w.tile(n_wave=n, k_row=kt), gb, acc, channels=chans)
                )
            waves.yield_(
                k.rd_mac(reduce.results[0], into=y_now.tile(n_wave=n), channels=chans)
            )
        k.returns(*waves.results)


class TileDslBuildTest(unittest.TestCase):
    """The DSL records exactly the Tile IR the kernel describes."""

    def test_output_stationary_gemv(self) -> None:
        """Schedule 1 builds, verifies and prints as the hand-written IR does."""

        gemv = _Gemv()
        gemv.output_stationary()
        kernel = gemv.k.build()
        self.assertEqual(print_function(kernel.function), GOLDEN)
        # Bases follow the declarations; y is aligned to the 4 channels.
        bases = {a.name: layout.base for a, layout in kernel.layouts.items()}
        self.assertEqual(bases, {"x": 0, "w": 0, "y": 4, "zeros": 12})
        self.assertIs(gemv.k.build(), kernel)

    def test_shapes_come_from_the_layout_and_device(self) -> None:
        """wr_gb takes its tile's size, wr_bias (channels, banks)."""

        gemv = _Gemv()
        gemv.output_stationary()
        kernel = gemv.k.build()
        waves = kernel.function.operations[0]
        assert isinstance(waves, TileForOp)
        acc0, reduce = waves.body.operations[:2]
        assert isinstance(acc0, CentWrBiasOp) and isinstance(reduce, TileForOp)
        gb = reduce.body.operations[0]
        assert isinstance(gb, CentWrGbOp)
        self.assertEqual(acc0.shape, (4, 16))
        self.assertEqual(gb.shape, (32,))

    def test_results_keep_their_tensor(self) -> None:
        """Loop results and rd_mac results are versions of the tensor they carry."""

        gemv = _Gemv()
        gemv.output_stationary()
        k = gemv.k
        with k.serial("again", 2, carry=(gemv.y,)) as loop:
            loop.yield_(*loop.carried)
        (result,) = loop.results
        self.assertIs(result.root, gemv.y.value)
        self.assertEqual(result.hint, "y")


class TileDslErrorTest(unittest.TestCase):
    """Mistakes are reported where they are made, with the rule broken."""

    def test_rejects_misuse(self) -> None:
        """Each misuse raises at the call that makes it."""

        def other_kernel_value(g: _Gemv) -> None:
            other = TileKernel("other", SMALL_ARCH)
            stranger = other.tensor("s", (16,), DType.BF16, SB)
            g.k.wr_bias(stranger, stride=0, channels=g.chans)

        def tile_without_layout(g: _Gemv) -> None:
            t = g.k.tensor("t", (16,), DType.BF16, SB)
            with g.k.serial("i", 2) as loop:
                t.tile(i=loop.index)

        def wrong_index_names(g: _Gemv) -> None:
            with g.k.serial("i", 2) as loop:
                g.w.tile(n_wave=loop.index)

        def missing_yield(g: _Gemv) -> None:
            with g.k.serial("i", 2, carry=(g.y,)):
                pass

        def wrong_yield_count(g: _Gemv) -> None:
            with g.k.serial("i", 2, carry=(g.y,)) as loop:
                loop.yield_()

        def results_too_early(g: _Gemv) -> None:
            with g.k.serial("i", 2, carry=(g.y,)) as loop:
                loop.yield_(*loop.results)

        def build_inside_a_loop(g: _Gemv) -> None:
            with g.k.serial("i", 2):
                g.k.build()

        def returns_inside_a_loop(g: _Gemv) -> None:
            with g.k.serial("i", 2):
                g.k.returns(g.y)

        def illegal_kernel(g: _Gemv) -> None:
            # Two live MAC values on the same channels: verify_kernel rejects.
            with g.k.serial("n_wave", 2, carry=(g.y,)) as waves:
                g.k.wr_bias(g.zeros, stride=0, channels=g.chans, name="first")
                second = g.k.wr_bias(g.zeros, stride=0, channels=g.chans)
                n, (y_now,) = waves.index, waves.carried
                waves.yield_(
                    g.k.rd_mac(second, into=y_now.tile(n_wave=n), channels=g.chans)
                )
            g.k.returns(*waves.results)
            g.k.build()

        cases: tuple[tuple[str, Callable[[_Gemv], object], type[Exception]], ...] = (
            (
                "must be in pim.dram or pnm.sb",
                lambda g: g.k.tensor("gb", (16,), DType.BF16, MemorySpace.PIM_GB),
                ValueError,
            ),
            (
                "already has a tensor named 'x'",
                lambda g: g.k.tensor("x", (16,), DType.BF16, SB),
                ValueError,
            ),
            (
                "already has a layout",
                lambda g: g.k.bind_layout(g.x, split(k=(Slot(), Lane()))),
                ValueError,
            ),
            ("is not a value of kernel 'gemv'", other_kernel_value, ValueError),
            ("bind a layout to tensor 't'", tile_without_layout, ValueError),
            ("needs one index for each of", wrong_index_names, ValueError),
            ("never called yield_", missing_yield, ValueError),
            ("carries 1 values, but yield_ got 0", wrong_yield_count, ValueError),
            ("exist only after its with block", results_too_early, RuntimeError),
            ("loop 'i' is still open", build_inside_a_loop, ValueError),
            ("outside every loop", returns_inside_a_loop, ValueError),
            ("already holds live MAC value %first", illegal_kernel, ValueError),
        )
        for message, misuse, error in cases:
            with self.subTest(error=message), self.assertRaisesRegex(error, message):
                misuse(_Gemv())

    def test_needs_a_cent_device(self) -> None:
        """A device without a CENT architecture cannot run CENT ops."""

        with self.assertRaisesRegex(TypeError, "need a CENT architecture"):
            TileKernel("k", object())  # type: ignore[arg-type]


class ExamplesTest(unittest.TestCase):
    """The Tile DSL examples build and verify on the paper spec."""

    def test_examples_run(self) -> None:
        """gemv.py and gemm.py build all four kernels without errors."""

        for name in ("gemv", "gemm"):
            with (
                self.subTest(example=name),
                contextlib.redirect_stdout(io.StringIO()) as out,
            ):
                runpy.run_path(f"examples/Tile_DSL/{name}.py", run_name="__main__")
            self.assertEqual(out.getvalue().count("func @"), 2)


GOLDEN = "\n".join(
    [
        (
            "func @gemv(%x: tensor<64xbf16, domain=pnm, "
            "memory_space=pnm.sb>, %w: tensor<128x64xbf16, domain=pim, "
            "memory_space=pim.dram>, %y: tensor<128xbf16, domain=pnm, "
            "memory_space=pnm.sb>, %zeros: tensor<16xbf16, domain=pnm, "
            "memory_space=pnm.sb>) -> (tensor<128xbf16, domain=pnm, "
            "memory_space=pnm.sb>) {"
        ),
        (
            "  %y_1 = tile.for(%y) {extent = 2} : tensor<128xbf16, "
            "domain=pnm, memory_space=pnm.sb> {"
        ),
        (
            "  ^(%n_wave: tensor<int32>, %y_2: tensor<128xbf16, domain=pnm, "
            "memory_space=pnm.sb>):"
        ),
        (
            "    %acc = cent.wr_bias(%zeros) {channels = [0, 1, 2, 3], "
            "at = [], stride = 0, shape = [4, 16]} : tensor<4x16xbf16, "
            "domain=pim, memory_space=pim.mac>"
        ),
        (
            "    %acc_1 = tile.for(%acc) {extent = 2} : tensor<4x16xbf16, "
            "domain=pim, memory_space=pim.mac> {"
        ),
        (
            "    ^(%k_tile: tensor<int32>, %acc_2: tensor<4x16xbf16, "
            "domain=pim, memory_space=pim.mac>):"
        ),
        (
            "      %gb = cent.wr_gb(%x, %k_tile) {channels = [0, 1, 2, 3], "
            'at = ["k_tile"], shape = [32]} : tensor<32xbf16, domain=pim, '
            "memory_space=pim.gb>"
        ),
        (
            "      %acc_3 = cent.mac_abk(%w, %gb, %acc_2, %n_wave, "
            '%k_tile) {channels = [0, 1, 2, 3], at = ["n_wave", '
            '"k_row"]} : tensor<4x16xbf16, domain=pim, '
            "memory_space=pim.mac>"
        ),
        "      yield %acc_3",
        "    }",
        (
            "    %y_3 = cent.rd_mac(%acc_1, %y_2, %n_wave) {channels = [0, "
            '1, 2, 3], at = ["n_wave"]} : tensor<128xbf16, domain=pnm, '
            "memory_space=pnm.sb>"
        ),
        "    yield %y_3",
        "  }",
        "  return %y_1",
        "}",
    ]
)


if __name__ == "__main__":
    unittest.main()
