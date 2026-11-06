"""Tests for the Llama 3.1 70B kernel mappings: CENT's records, Device IR shapes."""

from __future__ import annotations

import os
import re
import subprocess
import tempfile
import unittest
from math import ceil
from pathlib import Path
from typing import ClassVar

from examples.llama_70b.kernel_mapping import analytical
from examples.llama_70b.kernel_mapping.gemv import kv_proj, reuse_gb_gemv
from examples.llama_70b.kernel_mapping.llama3_1_70b import (
    DEVICE,
    RUN,
    WEIGHT_ROWS,
    LayerMapping,
    _check_gemv,
    device_ir,
    layer_trace,
    map_layer,
)
from vllm_cent.device import CentArchitecture, load_device_spec
from vllm_cent.driver import load_run_spec
from vllm_cent.ir import Function
from vllm_cent.tile.codegen import iter_aim_records
from vllm_cent.tile.dsl import TileKernel
from vllm_cent.tile.expand import ChannelIssue

ALL = "0xffffffff"
GOLDEN = Path("tests/golden/cent/llama3_1_70b_kvtp8_s2304.trace")


def _arch(path: Path) -> CentArchitecture:
    arch = load_device_spec(path).architecture
    assert isinstance(arch, CentArchitecture)
    return arch


PAPER = _arch(DEVICE)  # 32 MAC registers per bank: CENT's reuse-GB groups
AIM = _arch(Path("examples/cent/cent.yaml"))  # one MAC register per bank


def _cent_gemv(
    n: int, k: int, base: int, reuse: int = 32, af: bool = False
) -> list[str]:
    """CENT's reuse-GB GEMV records for 32 x 16 banks, bf16, GPR 0.

    Written after ``Vector_Matrix_Mul_weight_pim_only_trace`` and its
    ``_af_`` variant (``../ALOI/cent_simulation/TransformerBlock.py:1745-1829``):
    with ``af``, groups are half the size and the last K row's ``AF`` and
    ``RD_AF`` come before its ``RD_MAC``.
    """

    cols, rows = ceil(n / 512), ceil(k / 1024)
    groups = ceil(cols / (reuse // 2 if af else reuse))
    size = ceil(cols / groups)
    lines = []
    for r in range(rows):
        op = 64 if r < rows - 1 else ceil((k - 1024 * r) / 16)
        lines.append(f"AiM WR_GB {op} 0 {ALL}")
        for g in range(groups):
            members = range(g * size, min(cols, (g + 1) * size))
            lines += [f"AiM WR_BIAS 0 {ALL}"] * len(members)
            lines += [f"AiM MAC_ABK {op} {ALL} {base + c * rows + r}" for c in members]
            if af and r == rows - 1:
                lines += [f"AiM AF {ALL}", f"AiM RD_AF 0 {ALL}"] * len(members)
            lines += [f"AiM RD_MAC 0 {ALL}"] * len(members)
    return lines


def _cent_kv_proj(k_row: int, v_row: int) -> list[str]:
    """CENT's K/V projection records at TP = 8 (128 outputs each).

    Written after ``Vector_Matrix_Mul_kv_projection_pim_only_trace``
    (``TransformerBlock.py:1831-1882``): K on channels 0-7, V on 16-23.
    """

    km, vm = "0xff000000", "0xff00"
    lines = []
    for r in range(8):
        lines += [f"AiM WR_GB 64 0 {km}", f"AiM WR_GB 64 0 {vm}"]
        lines += [f"AiM WR_BIAS 0 {km}", f"AiM WR_BIAS 0 {vm}"]
        lines += [
            f"AiM MAC_ABK 64 {km} {k_row + r}",
            f"AiM MAC_ABK 64 {vm} {v_row + r}",
        ]
        lines += [f"AiM RD_MAC 0 {km}", f"AiM RD_MAC 0 {vm}"]
    return lines


def _as_cent(lines: list[str]) -> list[str]:
    """Drop comments and CFR writes; set GPR fields to 0, as CENT writes them."""

    records = []
    for line in lines:
        if line.startswith(("#", "W CFR")):
            continue
        fields = line.split()
        if fields[1] in ("WR_BIAS", "RD_MAC", "RD_AF"):
            fields[2] = "0"
        elif fields[1] == "WR_GB":
            fields[3] = "0"
        elif fields[1] == "EWADD":
            fields[3] = fields[4] = "0"
        records.append(" ".join(fields))
    return records


class CentGemvTest(unittest.TestCase):
    """The GEMV kernels issue CENT's records, in CENT's order."""

    def _records(self, k: TileKernel, arch: CentArchitecture) -> list[str]:
        return _as_cent(list(iter_aim_records(k.build(), arch)))

    def test_layer_gemvs_match_cent(self) -> None:
        """Every TP = 8 GEMV of the layer, with CENT's weight rows."""

        cases = (
            ("q_proj", 1024, 8192, "self_attn.q_proj.weight", False),
            ("o_proj", 8192, 1024, "self_attn.o_proj.weight", False),
            ("gate_proj", 3584, 8192, "mlp.gate_proj.weight", True),
            ("up_proj", 3584, 8192, "mlp.up_proj.weight", False),
            # K = 3584: the last K row is half full (opsize 32).
            ("down_proj", 8192, 3584, "mlp.down_proj.weight", False),
        )
        for name, n, kdim, weight, af in cases:
            with self.subTest(gemv=name):
                with TileKernel(name, PAPER) as k:
                    reuse_gb_gemv(
                        k,
                        n=n,
                        kdim=kdim,
                        w_row=WEIGHT_ROWS[weight],
                        activation="sigmoid" if af else None,
                    )
                self.assertEqual(
                    self._records(k, PAPER),
                    _cent_gemv(n, kdim, WEIGHT_ROWS[weight], af=af),
                )

    def test_kv_proj_matches_cent(self) -> None:
        """K and V on disjoint channel groups, their records interleaved."""

        with TileKernel("kv", PAPER) as k:
            kv_proj(
                k,
                n=128,
                kdim=8192,
                k_names=("x", "wk", "k"),
                v_names=("x", "wv", "v"),
                k_row=19,
                v_row=27,
            )
        self.assertEqual(self._records(k, PAPER), _cent_kv_proj(19, 27))

    def test_aim_issues_one_output_at_a_time(self) -> None:
        """With one MAC register per bank, groups have one output: no-reuse."""

        with TileKernel("q_proj", AIM) as k:
            reuse_gb_gemv(k, n=1024, kdim=8192, w_row=3)
        self.assertEqual(self._records(k, AIM), _cent_gemv(1024, 8192, 3, reuse=1))


class AnalyticalTest(unittest.TestCase):
    """The ported formulas give CENT's numbers for the evaluated cell."""

    def test_cent_values(self) -> None:
        """Values from ../ALOI's own functions (KV-head TP = 8, seqlen 2304)."""

        self.assertEqual(2 * analytical.rms_norm_cycles(8192, 32), 66.0)
        self.assertEqual(analytical.softmax_cycles(2304, 8, 32), 1046.0)
        self.assertEqual(analytical.rope_cycles(1024 + 128), 432.0)
        self.assertEqual(
            analytical.sv_reduction_cycles(2304, 8, 128, 4, 1024, 32), 16.0
        )
        # cxl_latency.kv_head_tp_latency(8192, 4, 8, 32) and vector_latency.
        self.assertAlmostEqual(
            2 * analytical.cxl_all_reduce_ms(8192, 4, 8), 0.021223997802734373
        )
        self.assertAlmostEqual(
            analytical.pipeline_handoff_ms(8192, 4), 0.0014614998626708985
        )


class LayerMappingTest(unittest.TestCase):
    """The mapping reads the real Device IR of the evaluated run."""

    function: ClassVar[Function]
    mapping: ClassVar[LayerMapping]
    context: ClassVar[int]

    @classmethod
    def setUpClass(cls) -> None:
        """Lower the evaluated run once and map its layer."""
        run = load_run_spec(RUN)
        cls.context = run.workload.context_len
        cls.function = device_ir(run, load_device_spec(DEVICE))
        cls.mapping = map_layer(cls.function, PAPER, cls.context)

    def test_every_op_has_a_kernel_or_a_comment(self) -> None:
        """PIM ops get kernels; PNM, CXL and data-moving ops get comments."""

        kernels = sorted(op.name_hint or "" for op in self.mapping.kernels)
        self.assertEqual(
            kernels,
            [
                "add",
                "add_1",
                "input_layernorm",
                "mlp.down_proj",
                "mlp.gate_proj",
                "mlp.silu",
                "mlp.up_proj",
                "post_attention_layernorm",
                "self_attn.gqa_qk",
                "self_attn.gqa_sv",
                "self_attn.k_proj",
                "self_attn.kv_update",
                "self_attn.kv_update_1",
                "self_attn.o_proj",
                "self_attn.q_proj",
            ],
        )
        covered = {
            op.name_hint: by.name_hint for op, by in self.mapping.covered_by.items()
        }
        self.assertEqual(
            covered,
            {
                "self_attn.v_proj": "self_attn.k_proj",
                "mlp.silu.sigmoid": "mlp.gate_proj",
                "mlp.mul": "mlp.silu",
            },
        )
        for op in self.function.operations:
            with self.subTest(op=op.name_hint or op.NAME):
                self.assertTrue(
                    op in self.mapping.kernels
                    or op in self.mapping.covered_by
                    or op in self.mapping.comments
                )

    def test_layer_trace_is_cents(self) -> None:
        """The whole layer gives CENT's 7237 records, but for one known row.

        CENT stores the FFN norm's squared input in row 61, which its EWMUL
        operands later overwrite; this mapping uses the unused row 60.
        """

        ours = _as_cent(list(layer_trace(self.mapping, PAPER)))
        cents = _as_cent(GOLDEN.read_text().splitlines())
        self.assertEqual(len(ours), len(cents))
        differ = [(a, b) for a, b in zip(ours, cents, strict=True) if a != b]
        self.assertEqual(len(differ), 1025)
        for a, b in differ:
            self.assertEqual(a.rsplit(" ", 1), [b.rsplit(" ", 1)[0], "60"])
            self.assertEqual(b.rsplit(" ", 1)[1], "61")

    def test_analytical_comments(self) -> None:
        """PNM and CXL work is named in comments with CENT's latencies."""

        comments = [
            line for line in layer_trace(self.mapping, PAPER) if line.startswith("#")
        ]
        text = "\n".join(comments)
        self.assertEqual(text.count("RMSNorm sum of squares and rsqrt: 33 cycles"), 2)
        self.assertIn("RoPE: 384 cycles", text)
        self.assertIn("RoPE: 48 cycles", text)
        self.assertIn("softmax (max, exp, sum, div): 1046 cycles", text)
        self.assertIn("SV partial-sum reduction: 16 cycles", text)
        self.assertEqual(text.count("CXL all-reduce over 8 devices: 0.0106120 ms"), 2)

    def test_serialize_issues_one_transfer_per_channel(self) -> None:
        """The run spec's other channel_issue also expands the whole layer."""

        broadcast = _as_cent(list(layer_trace(self.mapping, PAPER)))
        serialize = _as_cent(
            list(layer_trace(self.mapping, PAPER, ChannelIssue.SERIALIZE))
        )
        self.assertGreater(len(serialize), len(broadcast))
        masks = {
            line.split()[-1] for line in serialize if line.startswith("AiM RD_MAC")
        }
        self.assertTrue(all(bin(int(m, 16)).count("1") == 1 for m in masks))

    def test_shape_mismatch_is_reported(self) -> None:
        """A kernel built for other sizes does not pass the check."""

        q_op = next(
            op for op in self.function.operations if op.name_hint == "self_attn.q_proj"
        )
        with TileKernel("wrong", PAPER) as k:
            reuse_gb_gemv(
                k, n=2048, kdim=8192, names=("x", "self_attn.q_proj.weight", "y")
            )
        with self.assertRaisesRegex(ValueError, "Device IR value is"):
            _check_gemv(q_op, k.build(), ("x", "self_attn.q_proj.weight", "y"))


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
class LayerSimulationTest(unittest.TestCase):
    """The layer trace runs on the AiM simulator in CENT's time."""

    def test_layer_cycles_match_cent(self) -> None:
        """Within 0.5% of CENT's trace (the FFN norm row, the CFR writes)."""

        run = load_run_spec(RUN)
        mapping = map_layer(
            device_ir(run, load_device_spec(DEVICE)), PAPER, run.workload.context_len
        )
        ours = _simulate("\n".join(layer_trace(mapping, PAPER)) + "\n")
        cents = _simulate(GOLDEN.read_text())
        self.assertLess(abs(ours - cents) / cents, 0.005)


if __name__ == "__main__":
    unittest.main()
