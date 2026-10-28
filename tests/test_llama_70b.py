"""Tests for the Llama 2 / 3.1 70B decoder layer example, end to end.

Run from the repository root (``python -m unittest tests.test_llama_70b``) so
that the ``examples`` package is importable.
"""

from __future__ import annotations

import dataclasses
import json
import math
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from examples.llama_70b import (
    LlamaConfig,
    LlamaDecoderLayer,
    axis_sizes,
    build_meta_decoder_layer,
    example_inputs,
    semantic_annotations,
)
from vllm_cent.driver import STAGES, compile_stages, load_model, load_run_spec
from vllm_cent.frontend import export_model, import_exported_program
from vllm_cent.ir import (
    AllReduceOp,
    AnnotateRolesAndAxes,
    Module,
    Partial,
    PassManager,
    Replicate,
    Shard,
)
from vllm_cent.ir.passes import ApplyTensorParallel

# The example's model files (HF config.json) and run specs.
EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "llama_70b"
LLAMA2_70B_JSON = EXAMPLE_DIR / "llama2_70b.json"
LLAMA3_1_70B_JSON = EXAMPLE_DIR / "llama3_1_70b.json"
LLAMA2_70B = LlamaConfig.from_hf(json.loads(LLAMA2_70B_JSON.read_text()))
LLAMA3_1_70B = LlamaConfig.from_hf(json.loads(LLAMA3_1_70B_JSON.read_text()))
LLAMA3_1_70B_RUN = load_run_spec(EXAMPLE_DIR / "runs" / "llama3_1_70b_tp8_decode.yaml")
LLAMA2_70B_RUN = load_run_spec(EXAMPLE_DIR / "runs" / "llama2_70b_tp8_decode.yaml")

# Golden IR of Llama 3.1 70B, tp=8, context 2048, written by
# ``aloi-compile --model examples/llama_70b/llama3_1_70b.json
# --run examples/llama_70b/runs/llama3_1_70b_tp8_decode.yaml
# --out-dir tests/golden/llama3_1_70b_tp8`` and reviewed by hand. Regenerate
# with the same command after an intended IR change.
GOLDEN_DIR = Path(__file__).parent / "golden" / "llama3_1_70b_tp8"

# A layer small enough to run on CPU in fp32, with the same structure as the
# 70B models: GQA with 2 query heads per KV head, head_dim 8.
SMALL_DIMENSIONS = {
    "hidden_size": 32,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 8,
    "intermediate_size": 48,
    "max_position_embeddings": 16,
    "dtype": torch.float32,
}
# Llama 3 scaling with a pretraining context of 64, so head_dim 8 has low,
# medium and high frequencies.
SMALL_LLAMA3_SCALING = {
    "rope_type": "llama3",
    "factor": 8.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 64,
}


def _small_config(base: LlamaConfig) -> LlamaConfig:
    """Shrink a 70B config, keeping its RoPE kind.

    Args:
        base: The Llama 2 or Llama 3.1 70B config.

    Returns:
        The small config.
    """

    scaling = None if base.rope_scaling is None else SMALL_LLAMA3_SCALING
    return dataclasses.replace(base, rope_scaling=scaling, **SMALL_DIMENSIONS)  # type: ignore[arg-type]


def _reference_inverse_frequencies(config: LlamaConfig) -> Tensor:
    """Rotary frequencies, with Meta's loop form of the llama3 scaling.

    Args:
        config: Model config.

    Returns:
        ``[head_dim // 2]`` fp32 frequencies.
    """

    dim = config.head_dim
    frequencies = 1.0 / (config.rope_theta ** (torch.arange(0, dim, 2).float() / dim))
    scaling = config.rope_scaling
    if scaling is None:
        return frequencies
    factor = float(scaling["factor"])
    low_freq_factor = float(scaling["low_freq_factor"])
    high_freq_factor = float(scaling["high_freq_factor"])
    original = float(scaling["original_max_position_embeddings"])
    low_wavelength = original / low_freq_factor
    high_wavelength = original / high_freq_factor
    scaled = []
    for frequency in frequencies.tolist():
        wavelength = 2 * math.pi / frequency
        if wavelength < high_wavelength:
            scaled.append(frequency)
        elif wavelength > low_wavelength:
            scaled.append(frequency / factor)
        else:
            smooth = (original / wavelength - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            scaled.append((1 - smooth) * frequency / factor + smooth * frequency)
    return torch.tensor(scaled)


def _reference_layer(
    layer: LlamaDecoderLayer,
    x: Tensor,
    positions: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Hugging Face style Llama decoder layer in plain torch.

    It shares only the weights with the custom-op layer: heads are moved with
    view/transpose, KV heads repeated with expand (``repeat_kv``), attention
    is a batched matmul, and RoPE uses ``rotate_half`` with Meta's scaling.

    Args:
        layer: Layer whose parameters are used.
        x: ``[batch, 1, hidden]``.
        positions: ``[batch, 1]``.
        k_cache: ``[batch, context, kv_heads, head_dim]``.
        v_cache: ``[batch, context, kv_heads, head_dim]``.

    Returns:
        ``(output, new k_cache, new v_cache)``.
    """

    config = layer.config
    weights = dict(layer.named_parameters())
    heads, kv_heads, dim = (
        config.num_attention_heads,
        config.num_key_value_heads,
        config.head_dim,
    )
    batch, tokens, _ = x.shape

    def rms_norm(hidden: Tensor, weight: Tensor) -> Tensor:
        """Divide by the root mean square, then scale."""

        mean_square = hidden.pow(2).mean(-1, keepdim=True)
        return hidden * torch.rsqrt(mean_square + config.rms_norm_eps) * weight

    def rotate_half(t: Tensor) -> Tensor:
        """Map halves (a, b) to (-b, a)."""

        return torch.cat((-t[..., dim // 2 :], t[..., : dim // 2]), dim=-1)

    def project(hidden: Tensor, name: str, count: int) -> Tensor:
        """Linear projection, then [batch, heads, tokens, head_dim]."""

        out = hidden @ weights[f"self_attn.{name}.weight"].T
        return out.view(batch, tokens, count, dim).transpose(1, 2)

    normalized = rms_norm(x, weights["input_layernorm.weight"])
    q = project(normalized, "q_proj", heads)
    k = project(normalized, "k_proj", kv_heads)
    v = project(normalized, "v_proj", kv_heads)

    angles = positions[..., None].float() * _reference_inverse_frequencies(config)
    angles = torch.cat((angles, angles), dim=-1)[:, None]  # [batch, 1, tokens, dim]
    q = q * angles.cos() + rotate_half(q) * angles.sin()
    k = k * angles.cos() + rotate_half(k) * angles.sin()

    k_cache, v_cache = k_cache.clone(), v_cache.clone()
    for request in range(batch):
        for token in range(tokens):
            row = positions[request, token]
            k_cache[request, row] = k[request, :, token]
            v_cache[request, row] = v[request, :, token]

    def repeat_kv(cache: Tensor) -> Tensor:
        """[batch, context, kv_heads, dim] to [batch, heads, context, dim]."""

        per_head = cache.transpose(1, 2)
        context = per_head.shape[2]
        expanded = per_head[:, :, None].expand(
            batch, kv_heads, heads // kv_heads, context, dim
        )
        return expanded.reshape(batch, heads, context, dim)

    scores = (q @ repeat_kv(k_cache).transpose(-1, -2)) * dim**-0.5
    attention = torch.softmax(scores, dim=-1) @ repeat_kv(v_cache)
    attention = attention.transpose(1, 2).reshape(batch, tokens, heads * dim)
    hidden = x + attention @ weights["self_attn.o_proj.weight"].T

    normalized = rms_norm(hidden, weights["post_attention_layernorm.weight"])
    gate = F.silu(normalized @ weights["mlp.gate_proj.weight"].T)
    up = normalized @ weights["mlp.up_proj.weight"].T
    output = hidden + (gate * up) @ weights["mlp.down_proj.weight"].T
    return output, k_cache, v_cache


def _sharded_module(config: LlamaConfig, tp: int) -> Module:
    """Import, annotate and shard one decode step with context 2048.

    Args:
        config: Model config.
        tp: Number of ranks.

    Returns:
        The IR one rank runs.
    """

    program = export_model(build_meta_decoder_layer(config), example_inputs(config))
    # The example run spec's rules, with another number of ranks.
    plan = dataclasses.replace(LLAMA3_1_70B_RUN.parallel, tp=tp).build_plan()
    return PassManager(
        (
            AnnotateRolesAndAxes(semantic_annotations()),
            ApplyTensorParallel(plan, axis_sizes(config)),
        )
    ).run(import_exported_program(program, "m"))


class DecoderLayerNumericsTest(unittest.TestCase):
    """The custom-op layer computes what a plain-torch Llama layer computes."""

    def test_matches_reference(self) -> None:
        """Both RoPE kinds; new tokens at the last and at earlier positions."""

        torch.manual_seed(0)
        for base in (LLAMA2_70B, LLAMA3_1_70B):
            layer = LlamaDecoderLayer(_small_config(base))
            with torch.no_grad():
                for parameter in layer.parameters():
                    parameter.copy_(torch.randn_like(parameter) * 0.2)
            x = torch.randn(2, 1, 32)
            k_cache = torch.randn(2, 5, 2, 8)
            v_cache = torch.randn(2, 5, 2, 8)
            for positions in (torch.tensor([[4], [4]]), torch.tensor([[1], [3]])):
                with self.subTest(theta=base.rope_theta, positions=positions.tolist()):
                    got = layer(x, positions, k_cache, v_cache)
                    want = _reference_layer(layer, x, positions, k_cache, v_cache)
                    for got_tensor, want_tensor in zip(got, want, strict=True):
                        torch.testing.assert_close(got_tensor, want_tensor)


class MetaLayerTest(unittest.TestCase):
    """The 70B layer exports from meta tensors with checkpoint names."""

    def test_parameters(self) -> None:
        """Hugging Face names and nn.Linear [out, in] shapes, in bf16.

        q/o: 64 heads x 128 = 8192. k/v: 8 KV heads x 128 = 1024.
        gate/up/down: intermediate 28672.
        """

        layer = build_meta_decoder_layer(LLAMA3_1_70B)
        shapes = {name: tuple(p.shape) for name, p in layer.named_parameters()}
        self.assertEqual(
            shapes,
            {
                "input_layernorm.weight": (8192,),
                "self_attn.q_proj.weight": (8192, 8192),
                "self_attn.k_proj.weight": (1024, 8192),
                "self_attn.v_proj.weight": (1024, 8192),
                "self_attn.o_proj.weight": (8192, 8192),
                "post_attention_layernorm.weight": (8192,),
                "mlp.gate_proj.weight": (28672, 8192),
                "mlp.up_proj.weight": (28672, 8192),
                "mlp.down_proj.weight": (8192, 28672),
            },
        )
        for parameter in layer.parameters():
            self.assertEqual(parameter.dtype, torch.bfloat16)
            self.assertEqual(parameter.device.type, "meta")
        # The RoPE table is a buffer: one row of 64 cosines and 64 sines for
        # each of the 131072 positions Llama 3.1 supports.
        table = layer.self_attn.rotary_emb.cos_sin_cache
        self.assertEqual(
            (tuple(table.shape), table.dtype), ((131072, 128), torch.bfloat16)
        )

    def test_exported_op_sequence(self) -> None:
        """Custom ops stay whole; SiLU, * and + appear as aten ops."""

        program = export_model(
            build_meta_decoder_layer(LLAMA3_1_70B), example_inputs(LLAMA3_1_70B)
        )
        aloi, aten = torch.ops.aloi, torch.ops.aten
        targets = [n.target for n in program.graph.nodes if n.op == "call_function"]
        self.assertEqual(
            targets,
            [
                aloi.rms_norm.default,
                # q, k, v: each projection is split into heads right away.
                *(aloi.linear.default, aloi.split_heads.default) * 3,
                *(aloi.rope.default,) * 2,  # q, k
                *(aloi.kv_update.default,) * 2,  # k, v
                aloi.gqa_qk.default,
                aloi.softmax.default,
                aloi.gqa_sv.default,
                aloi.merge_heads.default,
                aloi.linear.default,  # o_proj
                aten.add.Tensor,  # attention residual
                aloi.rms_norm.default,
                *(aloi.linear.default,) * 2,  # gate, up
                aten.silu.default,
                aten.mul.Tensor,
                aloi.linear.default,  # down
                aten.add.Tensor,  # feed-forward residual
            ],
        )


class GoldenIrTest(unittest.TestCase):
    """Every stage of Llama 3.1 70B at tp=8 prints exactly as reviewed."""

    def test_llama3_1_stages(self) -> None:
        """Imported, semantic and sharded IR match the golden files."""

        texts = compile_stages(load_model(LLAMA3_1_70B_JSON), LLAMA3_1_70B_RUN)
        self.assertEqual(tuple(texts), STAGES)
        for stage in STAGES:
            with self.subTest(stage=stage):
                golden = (GOLDEN_DIR / f"{stage}.aloi").read_text()
                self.assertEqual(texts[stage], golden)

    def test_llama2_differs_only_in_rope_table_and_dtype(self) -> None:
        """Llama 2 70B is the same layer with fp16 and a 4096-row RoPE table.

        theta and the llama3 scaling only change the table's values, which
        the IR does not show; the rope op and its attributes are identical.
        """

        texts = compile_stages(load_model(LLAMA2_70B_JSON), LLAMA2_70B_RUN)
        for stage in STAGES:
            with self.subTest(stage=stage):
                golden = (GOLDEN_DIR / f"{stage}.aloi").read_text()
                expected = (
                    golden.replace("bf16", "fp16")
                    .replace("tensor<131072x128x", "tensor<4096x128x")
                    .replace("@llama3_1_70b", "@llama2_70b")
                )
                self.assertEqual(texts[stage], expected)


class TensorParallelStructureTest(unittest.TestCase):
    """Where the plan's three rules split tensors and add communication."""

    def test_all_reduces_follow_the_row_parallel_projections(self) -> None:
        """o_proj and down_proj sum over split axes; nothing else does."""

        function = _sharded_module(LLAMA3_1_70B, tp=8).get_function("forward")
        partial_producers = [
            op.name_hint
            for op in function.operations
            if any(isinstance(r.type.placement, Partial) for r in op.results)
        ]
        reduced = [
            function.defining_op(op.operands[0]).name_hint  # type: ignore[union-attr]
            for op in function.operations
            if isinstance(op, AllReduceOp)
        ]
        self.assertEqual(partial_producers, ["self_attn.o_proj", "mlp.down_proj"])
        self.assertEqual(reduced, partial_producers)

    def test_outputs(self) -> None:
        """The hidden state is whole; each rank returns its own KV heads."""

        function = _sharded_module(LLAMA3_1_70B, tp=8).get_function("forward")
        placements = [result.type.placement for result in function.results]
        self.assertEqual(placements, [Replicate(), Shard("kv_head"), Shard("kv_head")])
        # 8 KV heads over 8 ranks: one [2048, 128] head per rank.
        self.assertEqual(function.results[1].type.shape, (1, 2048, 1, 128))

    def test_tp_degrees(self) -> None:
        """Local widths shrink with tp; tp=1 needs no all_reduce."""

        for tp in (1, 2, 4, 8):
            with self.subTest(tp=tp):
                function = _sharded_module(LLAMA3_1_70B, tp).get_function("forward")
                by_name = {op.name_hint: op for op in function.operations}
                q_width = by_name["self_attn.q_proj"].results[0].type.shape[-1]
                gate_width = by_name["mlp.gate_proj"].results[0].type.shape[-1]
                all_reduces = sum(
                    isinstance(op, AllReduceOp) for op in function.operations
                )
                self.assertEqual(q_width, 8192 // tp)
                self.assertEqual(gate_width, 28672 // tp)
                self.assertEqual(all_reduces, 0 if tp == 1 else 2)

    def test_uneven_heads_are_rejected(self) -> None:
        """tp=16 would split 8 KV heads in halves; tp=3 does not divide 64."""

        for tp, message in ((3, "q_head of size 64"), (16, "kv_head of size 8")):
            with self.subTest(tp=tp):
                with self.assertRaisesRegex(ValueError, message):
                    _sharded_module(LLAMA3_1_70B, tp)


class ConfigTest(unittest.TestCase):
    """LlamaConfig.from_hf reads Hugging Face config.json files."""

    def test_derived_and_default_values(self) -> None:
        """head_dim = 8192 / 64; Llama 2's file omits rope_theta (HF default)."""

        self.assertEqual((LLAMA2_70B.head_dim, LLAMA3_1_70B.head_dim), (128, 128))
        self.assertEqual(LLAMA2_70B.rope_theta, 10000.0)
        self.assertEqual(LLAMA3_1_70B.rope_theta, 500000.0)
        self.assertIsNone(LLAMA2_70B.rope_scaling)
        self.assertEqual(LLAMA3_1_70B.rope_scaling["rope_type"], "llama3")  # type: ignore[index]
        self.assertEqual(
            (LLAMA2_70B.dtype, LLAMA3_1_70B.dtype), (torch.float16, torch.bfloat16)
        )
        self.assertEqual(
            (LLAMA2_70B.max_position_embeddings, LLAMA3_1_70B.max_position_embeddings),
            (4096, 131072),
        )

    def test_rejects_what_the_layer_does_not_model(self) -> None:
        """Biases, another activation, or an unknown dtype."""

        base = json.loads(LLAMA3_1_70B_JSON.read_text())
        cases: dict[str, tuple[dict[str, object], type[Exception]]] = {
            "attention bias": ({"attention_bias": True}, NotImplementedError),
            "mlp bias": ({"mlp_bias": True}, NotImplementedError),
            "gelu": ({"hidden_act": "gelu"}, ValueError),
            "float64": ({"torch_dtype": "float64"}, ValueError),
        }
        for name, (change, error) in cases.items():
            with self.subTest(case=name), self.assertRaises(error):
                LlamaConfig.from_hf({**base, **change})


if __name__ == "__main__":
    unittest.main()
