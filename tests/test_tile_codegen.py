"""Tests for Tile IR expansion and AiM trace generation."""

from __future__ import annotations

import contextlib
import io
import os
import re
import runpy
import subprocess
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from typing import Any

from vllm_cent.cent import AimSimulatorCompatibilityError
from vllm_cent.cent.instructions import ReadMac, WriteBias
from vllm_cent.tile import CentKernel
from vllm_cent.tile.codegen import aim_trace
from vllm_cent.tile.expand import ZERO_GPR, ChannelIssue, iter_instructions

from test_tile_dsl import _Gemv
from test_tile_verify import SMALL_ARCH

MASK = "0xf0000000"  # channels 0-3: channel 0 is the most significant bit

# Small output-stationary GEMV (tests/test_tile_dsl.py): x at slots 0-3 (two
# tiles of two slots), y at 4-11 (two waves of four), zeros at 12; w row
# = n_wave * 2 + k_row.
OUTPUT_STATIONARY = [
    "W CFR 0 1",
    *(
        line
        for n in range(2)
        for line in (
            f"AiM WR_BIAS 12 {MASK}",
            f"AiM WR_GB 2 0 {MASK}",
            f"AiM MAC_ABK 2 {MASK} {n * 2}",
            f"AiM WR_GB 2 2 {MASK}",
            f"AiM MAC_ABK 2 {MASK} {n * 2 + 1}",
            f"AiM RD_MAC {4 + 4 * n} {MASK}",
        )
    ),
    "AiM EOC",
]


def _gb_stationary() -> CentKernel:
    """gemv.py's schedule 2 on the small device."""

    g = _Gemv("gemv_gb")
    k, chans = g.k, g.chans
    with k.serial("k_tile", 2, carry=(g.y,)) as reduce:
        kt, (partial,) = reduce.index, reduce.carried
        gb = k.wr_gb(g.x.tile(k_tile=kt), channels=chans)
        with k.serial("n_wave", 2, carry=(partial,)) as waves:
            n, (now,) = waves.index, waves.carried
            acc = k.wr_bias(now.tile(n_wave=n), stride=1, channels=chans)
            acc = k.mac_abk(gb, g.w.tile(n_wave=n, k_row=kt), acc, channels=chans)
            waves.yield_(k.rd_mac(acc, into=now.tile(n_wave=n), channels=chans))
        reduce.yield_(*waves.results)
    k.returns(*reduce.results)
    return k.build()


def _output_stationary() -> CentKernel:
    g = _Gemv()
    g.output_stationary()
    return g.k.build()


class ExpandTest(unittest.TestCase):
    """Each op becomes the instructions its layouts and loop indices give."""

    def test_output_stationary_trace(self) -> None:
        """One mask per instruction; addresses from the layouts."""

        self.assertEqual(
            aim_trace(_output_stationary(), SMALL_ARCH).splitlines(), OUTPUT_STATIONARY
        )

    def test_serialize(self) -> None:
        """SERIALIZE: one transfer per channel; channel c's slot is base + stride*c."""

        lines = aim_trace(
            _output_stationary(), SMALL_ARCH, issue=ChannelIssue.SERIALIZE
        ).splitlines()
        one_channel = ["0x80000000", "0x40000000", "0x20000000", "0x10000000"]
        # WR_BIAS from zeros (stride 0) and WR_GB: every channel, the same slot.
        self.assertEqual(lines[1:5], [f"AiM WR_BIAS 12 {m}" for m in one_channel])
        self.assertEqual(lines[5:9], [f"AiM WR_GB 2 0 {m}" for m in one_channel])
        # MAC_ABK is always one instruction for all channels.
        self.assertEqual(lines[9], f"AiM MAC_ABK 2 {MASK} 0")
        # RD_MAC: channel c writes slot Rd + c.
        first = lines.index("AiM RD_MAC 4 0x80000000")
        self.assertEqual(
            lines[first : first + 4],
            [f"AiM RD_MAC {4 + c} {m}" for c, m in enumerate(one_channel)],
        )

    def test_wr_bias_zero_reads_no_slot(self) -> None:
        """A sum started by wr_bias_zero needs no tensor of zeros."""

        g = _Gemv("gemv_zero")
        k, chans = g.k, g.chans
        with k.serial("n_wave", 2, carry=(g.y,)) as waves:
            n, (y_now,) = waves.index, waves.carried
            acc = k.wr_bias_zero(channels=chans)
            with k.serial("k_tile", 2, carry=(acc,)) as reduce:
                kt, (a,) = reduce.index, reduce.carried
                gb = k.wr_gb(g.x.tile(k_tile=kt), channels=chans)
                reduce.yield_(
                    k.mac_abk(gb, g.w.tile(n_wave=n, k_row=kt), a, channels=chans)
                )
            waves.yield_(
                k.rd_mac(reduce.results[0], into=y_now.tile(n_wave=n), channels=chans)
            )
        k.returns(*waves.results)
        lines = aim_trace(k.build(), SMALL_ARCH).splitlines()
        # The GPR field must be a non-negative number for the simulator.
        self.assertEqual(lines[1], f"AiM WR_BIAS {ZERO_GPR} {MASK}")
        self.assertEqual([line for line in lines if "WR_BIAS" in line], [lines[1]] * 2)

    def test_stride_one_wr_bias(self) -> None:
        """Restoring partial sums reads slot Rs + c, issued either way."""

        kernel = _gb_stationary()
        one = [
            i for i in iter_instructions(kernel, SMALL_ARCH) if isinstance(i, WriteBias)
        ]
        self.assertEqual([i.source.slot for i in one], [4, 8, 4, 8])
        self.assertEqual([i.channels.channels for i in one], [(0, 1, 2, 3)] * 4)

        per = [
            i
            for i in iter_instructions(kernel, SMALL_ARCH, issue=ChannelIssue.SERIALIZE)
            if isinstance(i, (WriteBias, ReadMac))
        ]
        # Wave 1 of the first k_tile: restore slots 8-11, then save them.
        wave = per[8:16]
        self.assertEqual(
            [(type(i).__name__, i.channels.channels) for i in wave[:4]],
            [("WriteBias", (c,)) for c in range(4)],
        )
        self.assertEqual(
            [i.source.slot for i in wave[:4] if isinstance(i, WriteBias)],
            [8, 9, 10, 11],
        )
        self.assertEqual(
            [i.destination.slot for i in wave[4:] if isinstance(i, ReadMac)],
            [8, 9, 10, 11],
        )

    def test_rejects_what_the_simulator_cannot_model(self) -> None:
        """The simulator models 16 banks per channel; this is checked first."""

        eight_banks = replace(SMALL_ARCH, banks_per_channel=8)
        with self.assertRaisesRegex(AimSimulatorCompatibilityError, "16 banks"):
            aim_trace(_output_stationary(), eight_banks)


def _paper_gemv() -> dict[str, Any]:
    with contextlib.redirect_stdout(io.StringIO()):
        return runpy.run_path("examples/Tile_DSL/gemv.py")


def _cent_gemv_trace(base_row: int) -> list[str]:
    """CENT's no-reuse GEMV order for 4096 x 4096 on 32 channels.

    Written after ``Vector_Matrix_Mul_weight_pim_only_trace`` in
    ``../ALOI/cent_simulation/TransformerBlock.py``: per 1024-value input
    row, one WR_GB, then WR_BIAS / MAC_ABK / RD_MAC per output of each bank.
    CENT writes GPR 0 throughout.
    """

    mask = "0xffffffff"
    lines = []
    for row_index in range(4):
        lines.append(f"AiM WR_GB 64 0 {mask}")
        for output in range(8):
            lines += [
                f"AiM WR_BIAS 0 {mask}",
                f"AiM MAC_ABK 64 {mask} {base_row + output * 4 + row_index}",
                f"AiM RD_MAC 0 {mask}",
            ]
    return [*lines, "AiM EOC"]


def _without_gpr(line: str) -> str:
    """Set the GPR field to 0, as CENT writes it."""

    fields = line.split()
    if fields[1] in ("WR_BIAS", "RD_MAC"):
        fields[2] = "0"
    elif fields[1] == "WR_GB":
        fields[3] = "0"
    return " ".join(fields)


class CentTraceTest(unittest.TestCase):
    """On the paper spec, the GB-stationary GEMV is CENT's own GEMV trace."""

    def test_gb_stationary_matches_cent(self) -> None:
        """Same records in the same order, but for the GPR field and the CFR."""

        ns = _paper_gemv()
        kernel = ns["gb_stationary"]
        base_row = kernel.layouts[kernel.function.arguments[1]].base
        ours = [
            _without_gpr(line)
            for line in aim_trace(kernel, ns["arch"]).splitlines()
            if not line.startswith("W CFR")
        ]
        self.assertEqual(ours, _cent_gemv_trace(base_row))


_SIMULATOR = Path(os.environ.get("AIM_SIMULATOR", "../ALOI/aim_simulator"))


def _simulate(trace: str) -> int:
    """Run the AiM simulator on ``trace``; return its memory system cycles."""

    with tempfile.NamedTemporaryFile("w", suffix=".trace", delete=False) as f:
        f.write(trace)
    try:
        result = subprocess.run(
            [
                str(_SIMULATOR / "build" / "ramulator2"),
                "-f",
                str(_SIMULATOR / "test" / "example_GDDR6.yaml"),
                "-t",
                f.name,
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=600,
        )
    finally:
        os.unlink(f.name)
    match = re.search(r"memory_system_cycles: (\d+)", result.stdout)
    assert match is not None, result.stdout[-2000:]
    return int(match.group(1))


@unittest.skipUnless(
    (_SIMULATOR / "build" / "ramulator2").exists(),
    "AiM simulator not built; set AIM_SIMULATOR to its directory",
)
class SimulatorTest(unittest.TestCase):
    """The traces run on the AiM simulator."""

    def test_gb_stationary_times_like_cent(self) -> None:
        """Our trace takes CENT's cycles, plus at most the one CFR write."""

        ns = _paper_gemv()
        kernel = ns["gb_stationary"]
        base_row = kernel.layouts[kernel.function.arguments[1]].base
        ours = _simulate(aim_trace(kernel, ns["arch"]))
        cent = _simulate("\n".join(_cent_gemv_trace(base_row)) + "\n")
        self.assertLessEqual(abs(ours - cent), 1)

    def test_per_channel_issue_costs_more(self) -> None:
        """One blocking RD_MAC per channel is slower than one full-mask RD_MAC."""

        ns = _paper_gemv()
        kernel = ns["output_stationary"]
        one = _simulate(aim_trace(kernel, ns["arch"]))
        per = _simulate(aim_trace(kernel, ns["arch"], issue=ChannelIssue.SERIALIZE))
        self.assertLess(one, per)


if __name__ == "__main__":
    unittest.main()
