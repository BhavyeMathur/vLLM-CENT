"""Tests for CENT Tile IR kernel verification: layouts, aliasing, ops, liveness, and SB versions."""

from __future__ import annotations

import unittest
from dataclasses import replace

from vllm_cent.device import CentArchitecture, MemorySpace
from vllm_cent.ir import Block, DType, Function, Value
from vllm_cent.ir.tile import (
    INDEX,
    CentMacAbkOp,
    CentRdMacOp,
    CentTensorType,
    CentWrBiasOp,
    CentWrGbOp,
    TileForOp,
)
from vllm_cent.tile import (
    Bank,
    CentKernel,
    Channel,
    Column,
    Lane,
    Mac,
    PhysicalLayout,
    Row,
    Serial,
    Slot,
    allocate,
    resolve,
    split,
    verify_kernel,
)

# Small architecture matching tests/test_ir_tile_cent.py:
# 4 channels, 16 banks/channel, 64B rows (32 bf16 values), 64B GB (32 bf16 values),
# 32B bursts (16 bf16 values), 1024B SB (32 slots), 1 MAC accumulator per bank.
SMALL_ARCH = CentArchitecture(
    channels=4,
    banks_per_channel=16,
    banks_per_group=4,
    rows_per_bank=64,
    row_size=64,
    burst_size=32,
    global_buffer_size=64,
    shared_buffer_size=1024,
    accumulators_per_bank=1,
)

N, K, GB = 128, 64, 32
CHANNELS = (0, 1, 2, 3)
MAC = (4, 16)


def _val(name: str, space: MemorySpace, shape: tuple[int, ...]) -> Value:
    return Value(
        type=CentTensorType(shape=shape, dtype=DType.BF16, memory_space=space),
        name=name,
    )


def _idx(name: str) -> Value:
    return Value(type=INDEX, name=name)


def _default_layouts(
    x: Value, w: Value, y: Value, zeros: Value | None = None
) -> dict[Value, PhysicalLayout]:
    raw = {
        x: resolve(split(k=(Serial("k_tile", 2), Slot(), Lane())), x.type, SMALL_ARCH),
        w: resolve(
            split(
                n=(Serial("n_wave", 2), Channel(), Bank()),
                k=(Row("k_row", 2), Column()),
            ),
            w.type,
            SMALL_ARCH,
        ),
        y: resolve(
            split(n=(Serial("n_wave", 2), Slot(of=Channel()), Lane(of=Bank()))),
            y.type,
            SMALL_ARCH,
        ),
    }
    if zeros is not None:
        raw[zeros] = resolve(
            split(b=(Slot(1), Lane(of=Bank()))), zeros.type, SMALL_ARCH
        )
    return allocate(raw, SMALL_ARCH)


def _output_stationary_kernel() -> CentKernel:
    x = _val("x", MemorySpace.PNM_SB, (K,))
    w = _val("w", MemorySpace.PIM_DRAM, (N, K))
    y = _val("y", MemorySpace.PNM_SB, (N,))
    zeros = _val("zeros", MemorySpace.PNM_SB, (16,))
    n, kt = _idx("n"), _idx("kt")
    y_now = _val("y_now", MemorySpace.PNM_SB, (N,))
    acc = _val("acc", MemorySpace.PIM_MAC, MAC)

    acc0 = CentWrBiasOp(
        operands=(zeros,),
        channels=CHANNELS,
        at=(),
        stride=0,
        shape=MAC,
        name_hint="acc0",
    )
    gb = CentWrGbOp(
        operands=(x, kt),
        channels=CHANNELS,
        at=("k_tile",),
        shape=(GB,),
        name_hint="gb",
    )
    mac = CentMacAbkOp(
        operands=(gb.results[0], w, acc, n, kt),
        channels=CHANNELS,
        at=("n_wave", "k_row"),
        name_hint="mac",
    )
    reduce_loop = TileForOp(
        operands=acc0.results,
        extent=2,
        body=Block((kt, acc), (gb, mac), mac.results),
        name_hint="reduce",
    )
    save = CentRdMacOp(
        operands=(reduce_loop.results[0], y_now, n),
        channels=CHANNELS,
        at=("n_wave",),
        name_hint="y_next",
    )
    waves = TileForOp(
        operands=(y,),
        extent=2,
        body=Block((n, y_now), (acc0, reduce_loop, save), save.results),
        name_hint="waves",
    )
    func = Function("gemv_os", (x, w, y, zeros), (waves,), waves.results)
    return CentKernel(func, _default_layouts(x, w, y, zeros))


def _gb_stationary_kernel() -> CentKernel:
    x = _val("x", MemorySpace.PNM_SB, (K,))
    w = _val("w", MemorySpace.PIM_DRAM, (N, K))
    y = _val("y", MemorySpace.PNM_SB, (N,))
    n, kt = _idx("n"), _idx("kt")
    partial = _val("partial", MemorySpace.PNM_SB, (N,))
    partial_now = _val("partial_now", MemorySpace.PNM_SB, (N,))

    gb = CentWrGbOp(
        operands=(x, kt),
        channels=CHANNELS,
        at=("k_tile",),
        shape=(GB,),
        name_hint="gb",
    )
    restore = CentWrBiasOp(
        operands=(partial_now, n),
        channels=CHANNELS,
        at=("n_wave",),
        stride=1,
        shape=MAC,
        name_hint="acc0",
    )
    mac = CentMacAbkOp(
        operands=(gb.results[0], w, restore.results[0], n, kt),
        channels=CHANNELS,
        at=("n_wave", "k_row"),
        name_hint="mac",
    )
    save = CentRdMacOp(
        operands=(mac.results[0], partial_now, n),
        channels=CHANNELS,
        at=("n_wave",),
        name_hint="y_next",
    )
    waves = TileForOp(
        operands=(partial,),
        extent=2,
        body=Block((n, partial_now), (restore, mac, save), save.results),
        name_hint="waves",
    )
    reduce_loop = TileForOp(
        operands=(y,),
        extent=2,
        body=Block((kt, partial), (gb, waves), waves.results),
        name_hint="reduce",
    )
    func = Function("gemv_gb", (x, w, y), (reduce_loop,), reduce_loop.results)
    return CentKernel(func, _default_layouts(x, w, y))


class VerifyKernelTest(unittest.TestCase):
    """Both legal schedules pass; each rule catches its counterexample."""

    def test_both_schedules_are_legal(self) -> None:
        """gemv.py's output-stationary and GB-stationary schedules verify."""

        verify_kernel(_output_stationary_kernel(), SMALL_ARCH)
        verify_kernel(_gb_stationary_kernel(), SMALL_ARCH)

    def test_rule_b_layout_checks(self) -> None:
        """Layouts must be complete, fit this device, be aligned and not overlap."""

        kernel = _gb_stationary_kernel()
        x, w, y = kernel.function.arguments

        # Missing layout for y.
        with (
            self.subTest(error="missing a PhysicalLayout"),
            self.assertRaisesRegex(ValueError, "missing a PhysicalLayout"),
        ):
            verify_kernel(
                CentKernel(
                    kernel.function,
                    {x: kernel.layouts[x], w: kernel.layouts[w]},
                ),
                SMALL_ARCH,
            )

        # Unaligned base on y (needs alignment to 4 channels).
        bad_y = replace(kernel.layouts[y], base=5)
        with (
            self.subTest(error="aligned to 4 channels"),
            self.assertRaisesRegex(ValueError, "aligned to 4 channels"),
        ):
            verify_kernel(
                CentKernel(kernel.function, {**kernel.layouts, y: bad_y}),
                SMALL_ARCH,
            )

        # Overlapping SB intervals between x [0, 4) and y [0, 8).
        overlap_y = replace(kernel.layouts[y], base=0)
        with (
            self.subTest(error="overlap"),
            self.assertRaisesRegex(ValueError, "overlap"),
        ):
            verify_kernel(
                CentKernel(kernel.function, {**kernel.layouts, y: overlap_y}),
                SMALL_ARCH,
            )

    def test_rule_b_layouts_must_fit_this_device(self) -> None:
        """Layouts resolved for another device are rejected, even if aligned."""

        kernel = _output_stationary_kernel()
        y = kernel.function.arguments[2]
        # y at slot 0 stays aligned on 8 channels, so only the re-resolve
        # can tell that its 4-channel slots do not match the device.
        layouts = {**kernel.layouts, y: replace(kernel.layouts[y], base=0)}
        x = kernel.function.arguments[0]
        layouts[x] = replace(kernel.layouts[x], base=8)
        zeros = kernel.function.arguments[3]
        layouts[zeros] = replace(kernel.layouts[zeros], base=12)
        cases = (
            (replace(SMALL_ARCH, channels=8), "%w: its layout does not fit"),
            # 64-byte bursts still resolve x, but to 32 lanes instead of 16.
            (
                replace(SMALL_ARCH, burst_size=64),
                "%x: its layout was resolved for another device",
            ),
        )
        for arch, message in cases:
            with (
                self.subTest(error=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                verify_kernel(CentKernel(kernel.function, layouts), arch)

    def test_rule_c_loop_yields_wrong_storage_root(self) -> None:
        """A loop must yield the storage it carries, even if the types agree."""

        x = _val("x", MemorySpace.PNM_SB, (N,))
        w = _val("w", MemorySpace.PIM_DRAM, (N, K))
        y = _val("y", MemorySpace.PNM_SB, (N,))
        n = _idx("n")
        y_now = _val("y_now", MemorySpace.PNM_SB, (N,))

        # Loop carries y, but yields x (which has the same type as y!).
        waves = TileForOp(
            operands=(y,),
            extent=2,
            body=Block((n, y_now), (), (x,)),
            name_hint="waves",
        )
        func = Function("bad_root", (x, w, y), (waves,), waves.results)
        layouts = allocate(
            {
                x: resolve(
                    split(n=(Serial("n_wave", 2), Slot(of=Channel()), Lane(of=Bank()))),
                    x.type,
                    SMALL_ARCH,
                ),
                w: resolve(
                    split(
                        n=(Serial("n_wave", 2), Channel(), Bank()),
                        k=(Row("k_row", 2), Column()),
                    ),
                    w.type,
                    SMALL_ARCH,
                ),
                y: resolve(
                    split(n=(Serial("n_wave", 2), Slot(of=Channel()), Lane(of=Bank()))),
                    y.type,
                    SMALL_ARCH,
                ),
            },
            SMALL_ARCH,
        )
        with (
            self.subTest(error="belongs to storage of %y, but body yields %x"),
            self.assertRaisesRegex(
                ValueError, "belongs to storage of %y, but body yields %x"
            ),
        ):
            verify_kernel(CentKernel(func, layouts), SMALL_ARCH)

    def test_rule_d_op_vs_layout_checks(self) -> None:
        """Each op is checked against the layouts of the tensors it indexes."""

        x = _val("x", MemorySpace.PNM_SB, (K,))
        w = _val("w", MemorySpace.PIM_DRAM, (N, K))
        y = _val("y", MemorySpace.PNM_SB, (N,))
        zeros = _val("zeros", MemorySpace.PNM_SB, (16,))
        layouts = _default_layouts(x, w, y, zeros)

        # 1. Channel index out of range (channel 4 on a 4-channel device).
        n, kt = _idx("n"), _idx("kt")
        bad_ch = CentWrBiasOp(
            operands=(zeros,), channels=(0, 1, 2, 4), at=(), stride=0, shape=MAC
        )
        y_now = _val("y_now", MemorySpace.PNM_SB, (N,))
        save2 = CentRdMacOp(
            operands=(bad_ch.results[0], y_now, n),
            channels=(0, 1, 2, 4),
            at=("n_wave",),
        )
        w2 = TileForOp(
            operands=(y,),
            extent=2,
            body=Block((n, y_now), (bad_ch, save2), save2.results),
        )
        with (
            self.subTest(error="channel 4 out of range"),
            self.assertRaisesRegex(ValueError, "channel 4 out of range"),
        ):
            verify_kernel(
                CentKernel(Function("f", (x, w, y, zeros), (w2,), w2.results), layouts),
                SMALL_ARCH,
            )

        # 2. Missing index factor name in `at` (w needs both n_wave and k_row).
        y_now = _val("y_now", MemorySpace.PNM_SB, (N,))
        acc0 = CentWrBiasOp(
            operands=(zeros,), channels=CHANNELS, at=(), stride=0, shape=MAC
        )
        gb = CentWrGbOp(operands=(x, n), channels=CHANNELS, at=("k_tile",), shape=(GB,))
        bad_mac = CentMacAbkOp(
            operands=(gb.results[0], w, acc0.results[0], n),
            channels=CHANNELS,
            at=("n_wave",),
        )
        save3 = CentRdMacOp(
            operands=(bad_mac.results[0], y_now, n),
            channels=CHANNELS,
            at=("n_wave",),
        )
        w3 = TileForOp(
            operands=(y,),
            extent=2,
            body=Block((n, y_now), (acc0, gb, bad_mac, save3), save3.results),
        )
        with (
            self.subTest(error="does not match layout index factors"),
            self.assertRaisesRegex(ValueError, "does not match layout index factors"),
        ):
            verify_kernel(
                CentKernel(Function("f", (x, w, y, zeros), (w3,), w3.results), layouts),
                SMALL_ARCH,
            )

        # 3. Loop extent exceeds layout factor extent (loop extent 4 > n_wave extent 2).
        y_now2 = _val("y_now", MemorySpace.PNM_SB, (N,))
        acc0_b = CentWrBiasOp(
            operands=(zeros,), channels=CHANNELS, at=(), stride=0, shape=MAC
        )
        save4 = CentRdMacOp(
            operands=(acc0_b.results[0], y_now2, n),
            channels=CHANNELS,
            at=("n_wave",),
        )
        w4 = TileForOp(
            operands=(y,),
            extent=4,
            body=Block((n, y_now2), (acc0_b, save4), save4.results),
        )
        with (
            self.subTest(error="loop extent 4, exceeding factor extent 2"),
            self.assertRaisesRegex(
                ValueError, "loop extent 4, exceeding factor extent 2"
            ),
        ):
            verify_kernel(
                CentKernel(Function("f", (x, w, y, zeros), (w4,), w4.results), layouts),
                SMALL_ARCH,
            )

        # 4. mac_abk channels exceed gb channels.
        y_now3 = _val("y_now", MemorySpace.PNM_SB, (N,))
        acc0_c = CentWrBiasOp(
            operands=(zeros,), channels=CHANNELS, at=(), stride=0, shape=MAC
        )
        gb_partial = CentWrGbOp(
            operands=(x, kt), channels=(0, 1), at=("k_tile",), shape=(GB,)
        )
        mac_full = CentMacAbkOp(
            operands=(gb_partial.results[0], w, acc0_c.results[0], n, kt),
            channels=CHANNELS,
            at=("n_wave", "k_row"),
        )
        save5 = CentRdMacOp(
            operands=(mac_full.results[0], y_now3, n),
            channels=CHANNELS,
            at=("n_wave",),
        )
        # kt must be a loop index in scope, so an outer k loop carries y.
        y_outer = _val("y_outer", MemorySpace.PNM_SB, (N,))
        w5_in = TileForOp(
            operands=(y_outer,),
            extent=2,
            body=Block(
                (n, y_now3), (acc0_c, gb_partial, mac_full, save5), save5.results
            ),
        )
        outer2 = TileForOp(
            operands=(y,),
            extent=2,
            body=Block((kt, y_outer), (w5_in,), w5_in.results),
        )
        with (
            self.subTest(error="not covered by Global Buffer"),
            self.assertRaisesRegex(ValueError, "not covered by Global Buffer"),
        ):
            verify_kernel(
                CentKernel(
                    Function("f", (x, w, y, zeros), (outer2,), outer2.results),
                    layouts,
                ),
                SMALL_ARCH,
            )

        # 5. wr_bias stride=0 with multi-slot source x (4 slots) instead of 1-slot zeros.
        y_now4 = _val("y_now", MemorySpace.PNM_SB, (N,))
        bad_bias = CentWrBiasOp(
            operands=(x, n),
            channels=CHANNELS,
            at=("k_tile",),
            stride=0,
            shape=MAC,
        )
        save6 = CentRdMacOp(
            operands=(bad_bias.results[0], y_now4, n),
            channels=CHANNELS,
            at=("n_wave",),
        )
        w6 = TileForOp(
            operands=(y,),
            extent=2,
            body=Block((n, y_now4), (bad_bias, save6), save6.results),
        )
        with (
            self.subTest(error="stride=0 source %x must be a single-slot"),
            self.assertRaisesRegex(
                ValueError, "stride=0 source %x must be a single-slot"
            ),
        ):
            verify_kernel(
                CentKernel(Function("f", (x, w, y, zeros), (w6,), w6.results), layouts),
                SMALL_ARCH,
            )

    def test_rule_d_rejects_a_mac_array_wider_than_one(self) -> None:
        """A weight laid out for a 2-output MAC array has no op to use it."""

        arch = replace(SMALL_ARCH, mac_array_width=2)
        x = _val("x", MemorySpace.PNM_SB, (K,))
        w = _val("w", MemorySpace.PIM_DRAM, (2 * N, K))
        zeros = _val("zeros", MemorySpace.PNM_SB, (16,))
        layouts = allocate(
            {
                x: resolve(
                    split(k=(Serial("k_tile", 2), Slot(), Lane())), x.type, arch
                ),
                w: resolve(
                    split(
                        n=(Serial("n_wave", 2), Channel(), Bank(), Mac()),
                        k=(Row("k_row", 4), Column(16)),
                    ),
                    w.type,
                    arch,
                ),
                zeros: resolve(split(b=(Slot(1), Lane(of=Bank()))), zeros.type, arch),
            },
            arch,
        )
        kt = _idx("kt")
        gb = CentWrGbOp(
            operands=(x, kt), channels=CHANNELS, at=("k_tile",), shape=(GB,)
        )
        acc0 = CentWrBiasOp(
            operands=(zeros,), channels=CHANNELS, at=(), stride=0, shape=MAC
        )
        mac = CentMacAbkOp(
            operands=(gb.results[0], w, acc0.results[0], kt, kt),
            channels=CHANNELS,
            at=("n_wave", "k_row"),
            name_hint="mac",
        )
        loop = TileForOp(operands=(), extent=2, body=Block((kt,), (gb, acc0, mac), ()))
        function = Function("f", (x, w, zeros), (loop,), ())
        with self.assertRaisesRegex(
            ValueError, "MAC array 2 outputs wide, but cent.mac_abk supports only one"
        ):
            verify_kernel(CentKernel(function, layouts), arch)

    def test_rule_e_liveness_rejects_two_live_macs_or_gbs(self) -> None:
        """A channel holds one live Global Buffer tile and one live MAC value."""

        x = _val("x", MemorySpace.PNM_SB, (K,))
        w = _val("w", MemorySpace.PIM_DRAM, (N, K))
        y = _val("y", MemorySpace.PNM_SB, (N,))
        zeros = _val("zeros", MemorySpace.PNM_SB, (16,))
        layouts = _default_layouts(x, w, y, zeros)
        n, kt = _idx("n"), _idx("kt")

        # Two consecutive wr_bias without consuming the first acc0.
        y_now = _val("y_now", MemorySpace.PNM_SB, (N,))
        acc0 = CentWrBiasOp(
            operands=(zeros,),
            channels=CHANNELS,
            at=(),
            stride=0,
            shape=MAC,
            name_hint="acc0",
        )
        acc1 = CentWrBiasOp(
            operands=(zeros,),
            channels=CHANNELS,
            at=(),
            stride=0,
            shape=MAC,
            name_hint="acc1",
        )
        save = CentRdMacOp(
            operands=(acc1.results[0], y_now, n),
            channels=CHANNELS,
            at=("n_wave",),
        )
        waves = TileForOp(
            operands=(y,),
            extent=2,
            body=Block((n, y_now), (acc0, acc1, save), save.results),
        )
        with (
            self.subTest(error="channel 0 already holds live MAC value %acc0"),
            self.assertRaisesRegex(
                ValueError, "channel 0 already holds live MAC value %acc0"
            ),
        ):
            verify_kernel(
                CentKernel(
                    Function("f", (x, w, y, zeros), (waves,), waves.results),
                    layouts,
                ),
                SMALL_ARCH,
            )

        # Two live GB values at the same time.
        y_now2 = _val("y_now", MemorySpace.PNM_SB, (N,))
        gb0 = CentWrGbOp(
            operands=(x, kt),
            channels=CHANNELS,
            at=("k_tile",),
            shape=(GB,),
            name_hint="gb0",
        )
        gb1 = CentWrGbOp(
            operands=(x, kt),
            channels=CHANNELS,
            at=("k_tile",),
            shape=(GB,),
            name_hint="gb1",
        )
        acc_b = CentWrBiasOp(
            operands=(zeros,), channels=CHANNELS, at=(), stride=0, shape=MAC
        )
        mac_b = CentMacAbkOp(
            operands=(gb1.results[0], w, acc_b.results[0], kt, kt),
            channels=CHANNELS,
            at=("n_wave", "k_row"),
        )
        save_b = CentRdMacOp(
            operands=(mac_b.results[0], y_now2, kt),
            channels=CHANNELS,
            at=("n_wave",),
        )
        waves_b = TileForOp(
            operands=(y,),
            extent=2,
            body=Block((kt, y_now2), (gb0, gb1, acc_b, mac_b, save_b), save_b.results),
        )
        with (
            self.subTest(error="channel 0 already holds live Global Buffer value %gb0"),
            self.assertRaisesRegex(
                ValueError, "channel 0 already holds live Global Buffer value %gb0"
            ),
        ):
            verify_kernel(
                CentKernel(
                    Function("f", (x, w, y, zeros), (waves_b,), waves_b.results),
                    layouts,
                ),
                SMALL_ARCH,
            )

    def test_rule_f_sb_version_checks(self) -> None:
        """An overwritten Shared Buffer version is never read or yielded again."""

        x = _val("x", MemorySpace.PNM_SB, (K,))
        w = _val("w", MemorySpace.PIM_DRAM, (N, K))
        y = _val("y", MemorySpace.PNM_SB, (N,))
        zeros = _val("zeros", MemorySpace.PNM_SB, (16,))
        layouts = _default_layouts(x, w, y, zeros)
        n = _idx("n")

        # 1. Yielding the old y_now instead of the newly written y_next.
        y_now = _val("y_now", MemorySpace.PNM_SB, (N,))
        acc0 = CentWrBiasOp(
            operands=(zeros,), channels=CHANNELS, at=(), stride=0, shape=MAC
        )
        save = CentRdMacOp(
            operands=(acc0.results[0], y_now, n),
            channels=CHANNELS,
            at=("n_wave",),
            name_hint="y_next",
        )
        bad_waves = TileForOp(
            operands=(y,),
            extent=2,
            body=Block((n, y_now), (acc0, save), (y_now,)),
        )
        with (
            self.subTest(
                error="Shared Buffer value %y_now is read or yielded after being overwritten"
            ),
            self.assertRaisesRegex(
                ValueError,
                "Shared Buffer value %y_now is read or yielded after being overwritten",
            ),
        ):
            verify_kernel(
                CentKernel(
                    Function("f", (x, w, y, zeros), (bad_waves,), bad_waves.results),
                    layouts,
                ),
                SMALL_ARCH,
            )

        # 2. Reading y_now with wr_bias AFTER rd_mac already overwrote y_now.
        y_now2 = _val("y_now", MemorySpace.PNM_SB, (N,))
        acc_a = CentWrBiasOp(
            operands=(zeros,), channels=CHANNELS, at=(), stride=0, shape=MAC
        )
        save_a = CentRdMacOp(
            operands=(acc_a.results[0], y_now2, n),
            channels=CHANNELS,
            at=("n_wave",),
        )
        stale_read = CentWrBiasOp(
            operands=(y_now2, n),
            channels=CHANNELS,
            at=("n_wave",),
            stride=1,
            shape=MAC,
        )
        save_a2 = CentRdMacOp(
            operands=(stale_read.results[0], save_a.results[0], n),
            channels=CHANNELS,
            at=("n_wave",),
        )
        bad_waves2 = TileForOp(
            operands=(y,),
            extent=2,
            body=Block(
                (n, y_now2), (acc_a, save_a, stale_read, save_a2), save_a2.results
            ),
        )
        with (
            self.subTest(
                error="Shared Buffer value %y_now is read or yielded after being overwritten"
            ),
            self.assertRaisesRegex(
                ValueError,
                "Shared Buffer value %y_now is read or yielded after being overwritten",
            ),
        ):
            verify_kernel(
                CentKernel(
                    Function("f", (x, w, y, zeros), (bad_waves2,), bad_waves2.results),
                    layouts,
                ),
                SMALL_ARCH,
            )


if __name__ == "__main__":
    unittest.main()
