"""Tests for the Llama 3.1 70B kernel mappings: CENT's records, Device IR shapes."""

from __future__ import annotations

import unittest
from math import ceil
from pathlib import Path
from typing import ClassVar

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
from vllm_cent.ir.device import PIMGemvOp
from vllm_cent.tile.codegen import iter_aim_records
from vllm_cent.tile.dsl import TileKernel

ALL = "0xffffffff"


def _arch(path: Path) -> CentArchitecture:
    arch = load_device_spec(path).architecture
    assert isinstance(arch, CentArchitecture)
    return arch


PAPER = _arch(DEVICE)  # 32 MAC registers per bank: CENT's reuse-GB groups
AIM = _arch(Path("examples/cent/cent.yaml"))  # one MAC register per bank


def _cent_gemv(n: int, k: int, base: int, reuse: int = 32) -> list[str]:
    """CENT's reuse-GB GEMV records for 32 x 16 banks, bf16, GPR 0.

    Written after ``Vector_Matrix_Mul_weight_pim_only_trace``
    (``../ALOI/cent_simulation/TransformerBlock.py:1745-1783``).
    """

    cols, rows = ceil(n / 512), ceil(k / 1024)
    groups = ceil(cols / reuse)
    size = ceil(cols / groups)
    lines = []
    for r in range(rows):
        op = 64 if r < rows - 1 else ceil((k - 1024 * r) / 16)
        lines.append(f"AiM WR_GB {op} 0 {ALL}")
        for g in range(groups):
            members = range(g * size, min(cols, (g + 1) * size))
            lines += [f"AiM WR_BIAS 0 {ALL}"] * len(members)
            lines += [f"AiM MAC_ABK {op} {ALL} {base + c * rows + r}" for c in members]
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


def _gpr_zero(line: str) -> str:
    """Set the GPR field to 0, as CENT writes it."""

    fields = line.split()
    if fields[1] in ("WR_BIAS", "RD_MAC"):
        fields[2] = "0"
    elif fields[1] == "WR_GB":
        fields[3] = "0"
    return " ".join(fields)


class CentGemvTest(unittest.TestCase):
    """The GEMV kernels issue CENT's records, in CENT's order."""

    def _records(self, k: TileKernel, arch: CentArchitecture) -> list[str]:
        return [_gpr_zero(line) for line in iter_aim_records(k.build(), arch)]

    def test_layer_gemvs_match_cent(self) -> None:
        """Every TP = 8 GEMV of the layer, with CENT's weight rows."""

        cases = (
            ("q_proj", 1024, 8192, "self_attn.q_proj.weight"),
            ("o_proj", 8192, 1024, "self_attn.o_proj.weight"),
            ("up_proj", 3584, 8192, "mlp.up_proj.weight"),
            # K = 3584: the last K row is half full (opsize 32).
            ("down_proj", 8192, 3584, "mlp.down_proj.weight"),
        )
        for name, n, kdim, weight in cases:
            with self.subTest(gemv=name):
                with TileKernel(name, PAPER) as k:
                    reuse_gb_gemv(k, n=n, kdim=kdim, w_row=WEIGHT_ROWS[weight])
                self.assertEqual(
                    self._records(k, PAPER),
                    _cent_gemv(n, kdim, WEIGHT_ROWS[weight]),
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


class LayerMappingTest(unittest.TestCase):
    """The mapping reads the real Device IR of the evaluated run."""

    function: ClassVar[Function]
    mapping: ClassVar[LayerMapping]

    @classmethod
    def setUpClass(cls) -> None:
        """Lower the evaluated run once and map its layer."""
        run = load_run_spec(RUN)
        cls.function = device_ir(run, load_device_spec(DEVICE))
        cls.mapping = map_layer(cls.function, PAPER)

    def test_gemvs_are_mapped_with_their_shapes(self) -> None:
        """q, k (with v), o, up and down get kernels whose shapes match."""

        mapped = sorted(op.name_hint or "" for op in self.mapping.kernels)
        self.assertEqual(
            mapped,
            [
                "mlp.down_proj",
                "mlp.up_proj",
                "self_attn.k_proj",
                "self_attn.o_proj",
                "self_attn.q_proj",
            ],
        )
        covered = {
            op.name_hint: by.name_hint for op, by in self.mapping.covered_by.items()
        }
        self.assertEqual(covered, {"self_attn.v_proj": "self_attn.k_proj"})

    def test_shape_mismatch_is_reported(self) -> None:
        """A kernel built for other sizes does not pass the check."""

        q_op = next(
            op
            for op in self.function.operations
            if isinstance(op, PIMGemvOp) and op.name_hint == "self_attn.q_proj"
        )
        with TileKernel("wrong", PAPER) as k:
            reuse_gb_gemv(
                k, n=2048, kdim=8192, names=("x", "self_attn.q_proj.weight", "y")
            )
        with self.assertRaisesRegex(ValueError, "Device IR operand is"):
            _check_gemv(q_op, k.build(), ("x", "self_attn.q_proj.weight", "y"))

    def test_layer_trace(self) -> None:
        """Mapped ops give records; the others are named in comments."""

        lines = list(layer_trace(self.mapping, PAPER))
        self.assertEqual(lines[0], "W CFR 0 1")
        self.assertEqual(lines[-1], "AiM EOC")
        self.assertEqual(sum(line.startswith("AiM ") for line in lines[:-1]), 541)
        self.assertIn("# pim.qk (self_attn.gqa_qk): not mapped yet", lines)


if __name__ == "__main__":
    unittest.main()
