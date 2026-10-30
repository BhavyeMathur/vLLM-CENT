"""Tests for the Llama 4 Scout decoder layer example, end to end.

Run from the repository root (``python -m unittest tests.test_llama4_scout``)
so that the ``examples`` package is importable.
"""

from __future__ import annotations

import dataclasses
import json
import unittest
from pathlib import Path

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor

from examples.llama4_scout import (
    Llama4Config,
    Llama4DecoderLayer,
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

# The example's model file (HF config.json) and run specs.
EXAMPLE_DIR = Path(__file__).resolve().parents[1] / "examples" / "llama4_scout"
SCOUT_JSON = EXAMPLE_DIR / "llama4_scout.json"
SCOUT = Llama4Config.from_hf(json.loads(SCOUT_JSON.read_text()))
RUNS = {
    "rope_layer": load_run_spec(
        EXAMPLE_DIR / "runs" / "llama4_scout_rope_tp8_decode.yaml"
    ),
    "nope_layer": load_run_spec(
        EXAMPLE_DIR / "runs" / "llama4_scout_nope_tp8_decode.yaml"
    ),
}

# Golden IR of both layer types, tp=8, context 2048, written by
# ``aloi-compile --model examples/llama4_scout/llama4_scout.json
# --run examples/llama4_scout/runs/llama4_scout_<kind>_tp8_decode.yaml
# --out-dir tests/golden/llama4_scout_tp8/<kind>_layer`` and reviewed by hand.
GOLDEN_DIR = Path(__file__).parent / "golden" / "llama4_scout_tp8"

# Layer 0 is a RoPE layer and layer 3 the first NoPE layer
# (no_rope_layers = [1, 1, 1, 0, ...]).
ROPE_LAYER = 0
NOPE_LAYER = 3

# A model small enough to run on CPU in fp32, with Scout's structure: GQA
# with 2 query heads per KV head, 4 experts with top-1 routing, and a NoPE
# layer every fourth layer. floor_scale 2 makes temperature tuning change
# the queries already at position 1 (Scout's 8192 would leave them at 1).
SMALL = dataclasses.replace(
    SCOUT,
    hidden_size=16,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=4,
    num_hidden_layers=4,
    intermediate_size=8,
    num_local_experts=4,
    moe_layers=(0, 1, 2, 3),
    no_rope_layers=(1, 1, 1, 0),
    floor_scale=2.0,
    attn_scale=0.5,
    attention_chunk_size=64,
    max_position_embeddings=16,
    dtype=torch.float32,
)


def _rms(x: Tensor, eps: float) -> Tensor:
    """Hugging Face's Llama4TextL2Norm: ``x * rsqrt(mean(x**2) + eps)``, no weight.

    Args:
        x: Tensor normalized over its last dimension.
        eps: Constant added to the mean square.

    Returns:
        The normalized tensor.
    """
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps)


def _reference_layer(
    layer: Llama4DecoderLayer,
    layer_index: int,
    x: Tensor,
    positions: Tensor,
    k_cache: Tensor,
    v_cache: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    """Compute the layer the way Hugging Face's modeling_llama4.py does.

    Written independently of the ALOI ops: RoPE rotates adjacent pairs by
    complex multiplication (``view_as_complex``), the router scatters the
    top-k logits into -inf before the sigmoid, and the experts use one fused
    ``gate_up_proj [experts, hidden, 2 * intermediate]`` with ``bmm``, as the
    checkpoint stores it. Only the cos/sin values come from the layer's table.

    Args:
        layer: The ALOI layer whose parameters are used.
        layer_index: Its decoder layer index; selects RoPE or NoPE.
        x: ``[batch, 1, hidden]``.
        positions: ``[batch, 1]``.
        k_cache: ``[batch, context, kv_heads, head_dim]``.
        v_cache: Same shape as k_cache.

    Returns:
        ``(hidden states, new k_cache, new v_cache)``.
    """
    config = SMALL
    attn = layer.self_attn
    batch = x.shape[0]
    h = F.rms_norm(
        x, (config.hidden_size,), layer.input_layernorm.weight, config.rms_norm_eps
    )
    q = (h @ attn.q_proj.weight.T).view(
        batch, 1, config.num_attention_heads, config.head_dim
    )
    k = (h @ attn.k_proj.weight.T).view(
        batch, 1, config.num_key_value_heads, config.head_dim
    )
    v = (h @ attn.v_proj.weight.T).view(
        batch, 1, config.num_key_value_heads, config.head_dim
    )
    if config.uses_rope(layer_index):
        assert attn.rotary_emb is not None
        cos, sin = attn.rotary_emb.cos_sin_cache[positions].chunk(2, dim=-1)
        freqs = torch.complex(cos, sin)[:, :, None, :]

        def rotate(t: Tensor) -> Tensor:
            """Rotate adjacent pairs by the position's angles (HF apply_rotary_emb)."""
            pairs = torch.view_as_complex(t.float().reshape(*t.shape[:-1], -1, 2))
            return torch.view_as_real(pairs * freqs).flatten(3)

        q, k = (
            _rms(rotate(q), config.rms_norm_eps),
            _rms(rotate(k), config.rms_norm_eps),
        )
    else:
        scale = torch.log1p(torch.floor((positions.float() + 1.0) / config.floor_scale))
        q = q * (scale * config.attn_scale + 1.0)[:, :, None, None]
    rows = torch.arange(batch)[:, None]
    k_cache, v_cache = k_cache.clone(), v_cache.clone()
    k_cache[rows, positions] = k
    v_cache[rows, positions] = v
    group = config.num_attention_heads // config.num_key_value_heads
    keys = k_cache.repeat_interleave(group, dim=2)
    values = v_cache.repeat_interleave(group, dim=2)
    scores = torch.einsum("bthd,bchd->bthc", q, keys) * config.head_dim**-0.5
    context = torch.einsum("bthc,bchd->bthd", scores.softmax(-1), values)
    hidden = x + context.reshape(batch, 1, -1) @ attn.o_proj.weight.T

    moe = layer.feed_forward
    tokens = F.rms_norm(
        hidden,
        (config.hidden_size,),
        layer.post_attention_layernorm.weight,
        config.rms_norm_eps,
    ).reshape(-1, config.hidden_size)
    logits = tokens @ moe.router.weight.T
    top_value, top_index = torch.topk(logits, config.num_experts_per_tok, dim=1)
    router_scores = torch.sigmoid(
        torch.full_like(logits, float("-inf")).scatter_(1, top_index, top_value)
    )
    experts = config.num_local_experts
    routed_in = tokens.repeat(experts, 1) * router_scores.transpose(0, 1).reshape(-1, 1)
    gate_up_proj = torch.cat(
        (moe.experts.gate_proj.transpose(1, 2), moe.experts.up_proj.transpose(1, 2)),
        dim=-1,
    )
    gate_up = torch.bmm(routed_in.view(experts, -1, config.hidden_size), gate_up_proj)
    gate, up = gate_up.chunk(2, dim=-1)
    routed = torch.bmm(up * F.silu(gate), moe.experts.down_proj.transpose(1, 2)).sum(0)
    shared = moe.shared_expert
    shared_out = (
        F.silu(tokens @ shared.gate_proj.weight.T) * (tokens @ shared.up_proj.weight.T)
    ) @ shared.down_proj.weight.T
    return hidden + (routed + shared_out).view_as(hidden), k_cache, v_cache


def _sharded_module(layer: int, tp: int) -> Module:
    """Import, annotate and shard Scout's ``layer`` for one decode step.

    Args:
        layer: Decoder layer index.
        tp: Number of ranks.

    Returns:
        The IR one rank runs.
    """
    program = export_model(
        build_meta_decoder_layer(SCOUT, layer), example_inputs(SCOUT)
    )
    plan = dataclasses.replace(RUNS["rope_layer"].parallel, tp=tp).build_plan()
    return PassManager(
        (
            AnnotateRolesAndAxes(semantic_annotations(SCOUT, layer)),
            ApplyTensorParallel(plan, axis_sizes(SCOUT)),
        )
    ).run(import_exported_program(program, "m"))


class DecoderLayerNumericsTest(unittest.TestCase):
    """The ALOI layer computes what Hugging Face's Llama 4 layer computes."""

    def test_rope_and_nope_layers_match_reference(self) -> None:
        """Two requests whose new tokens sit at different cache positions.

        Positions 5 and 2 of a 6-position cache: with floor_scale 2 the NoPE
        query scales are ``log1p(3) * 0.5 + 1`` and ``log1p(1) * 0.5 + 1``.
        """
        torch.manual_seed(0)
        x = torch.randn(2, 1, SMALL.hidden_size)
        positions = torch.tensor([[5], [2]])
        k_cache = torch.randn(2, 6, SMALL.num_key_value_heads, SMALL.head_dim)
        v_cache = torch.randn(2, 6, SMALL.num_key_value_heads, SMALL.head_dim)
        for layer_index in (ROPE_LAYER, NOPE_LAYER):
            with self.subTest(layer=layer_index):
                layer = Llama4DecoderLayer(SMALL, layer_index)
                with torch.no_grad():
                    for parameter in layer.parameters():
                        parameter.copy_(torch.randn_like(parameter) * 0.3)
                    got = layer(x, positions, k_cache, v_cache)
                    want = _reference_layer(
                        layer, layer_index, x, positions, k_cache, v_cache
                    )
                for got_tensor, want_tensor in zip(got, want, strict=True):
                    torch.testing.assert_close(got_tensor, want_tensor)


class MetaLayerTest(unittest.TestCase):
    """The full-size meta layer has Scout's checkpoint shapes."""

    def test_parameters_follow_the_checkpoint(self) -> None:
        """Names and shapes match the safetensors index, except the experts.

        The checkpoint stores the routed experts as one fused
        ``gate_up_proj [16, 5120, 16384]`` and ``down_proj [16, 8192, 5120]``;
        the example keeps nn.Linear-layout matrices per expert instead.
        """
        layer = build_meta_decoder_layer(SCOUT, ROPE_LAYER)
        shapes = {name: tuple(p.shape) for name, p in layer.named_parameters()}
        self.assertEqual(
            shapes,
            {
                "input_layernorm.weight": (5120,),
                "self_attn.q_proj.weight": (5120, 5120),
                "self_attn.k_proj.weight": (1024, 5120),
                "self_attn.v_proj.weight": (1024, 5120),
                "self_attn.o_proj.weight": (5120, 5120),
                "post_attention_layernorm.weight": (5120,),
                "feed_forward.router.weight": (16, 5120),
                "feed_forward.experts.gate_proj": (16, 8192, 5120),
                "feed_forward.experts.up_proj": (16, 8192, 5120),
                "feed_forward.experts.down_proj": (16, 5120, 8192),
                "feed_forward.shared_expert.gate_proj.weight": (8192, 5120),
                "feed_forward.shared_expert.up_proj.weight": (8192, 5120),
                "feed_forward.shared_expert.down_proj.weight": (5120, 8192),
            },
        )
        self.assertTrue(all(p.dtype == torch.bfloat16 for p in layer.parameters()))

    def test_only_rope_layers_have_a_rotary_table(self) -> None:
        """The NoPE layer has the same parameters and no cos/sin buffer."""
        buffers = {
            layer: [
                name
                for name, _ in build_meta_decoder_layer(SCOUT, layer).named_buffers()
            ]
            for layer in (ROPE_LAYER, NOPE_LAYER)
        }
        self.assertEqual(buffers[ROPE_LAYER], ["self_attn.rotary_emb.cos_sin_cache"])
        self.assertEqual(buffers[NOPE_LAYER], [])


class GoldenIrTest(unittest.TestCase):
    """Every stage of both layer types at tp=8 prints exactly as reviewed."""

    def test_stages(self) -> None:
        """Imported, semantic and sharded IR match the golden files."""
        for kind, run in RUNS.items():
            texts = compile_stages(load_model(SCOUT_JSON), run)
            self.assertEqual(tuple(texts), STAGES)
            for stage in STAGES:
                with self.subTest(layer=kind, stage=stage):
                    golden = (GOLDEN_DIR / kind / f"{stage}.aloi").read_text()
                    self.assertEqual(texts[stage], golden)

    def test_layer_types_differ_only_in_position_handling(self) -> None:
        """RoPE + QK-norm (four ops) in layer 0 become one attn_temperature in layer 3."""
        names = {}
        for kind, layer in (("rope_layer", ROPE_LAYER), ("nope_layer", NOPE_LAYER)):
            function = _sharded_module(layer, tp=8).get_function("forward")
            names[kind] = [op.NAME for op in function.operations]
        rope_ops = ["aloi.rope", "aloi.rope", "aloi.rms_norm", "aloi.rms_norm"]
        start = names["rope_layer"].index("aloi.rope")
        self.assertEqual(names["rope_layer"][start : start + 4], rope_ops)
        expected = list(names["rope_layer"])
        expected[start : start + 4] = ["aloi.attn_temperature"]
        self.assertEqual(names["nope_layer"], expected)


class TensorParallelStructureTest(unittest.TestCase):
    """Where the plan's rules split tensors and add communication."""

    def test_all_reduces_follow_the_contractions_over_split_axes(self) -> None:
        """o_proj, the routed experts and the shared expert's down_proj."""
        function = _sharded_module(ROPE_LAYER, tp=8).get_function("forward")
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
        self.assertEqual(
            partial_producers,
            [
                "self_attn.o_proj",
                "feed_forward.experts",
                "feed_forward.shared_expert.down_proj",
            ],
        )
        self.assertEqual(reduced, partial_producers)

    def test_local_shapes(self) -> None:
        """40/8 = 5 query heads, 8/8 = 1 KV head, 8192/8 = 1024 expert width.

        The router has no split axis, so every rank scores all 16 experts.
        """
        function = _sharded_module(ROPE_LAYER, tp=8).get_function("forward")
        arguments = {a.name: a.type for a in function.operands}
        self.assertEqual(arguments["self_attn.q_proj.weight"].shape, (640, 5120))
        self.assertEqual(arguments["self_attn.k_proj.weight"].shape, (128, 5120))
        self.assertEqual(arguments["feed_forward.router.weight"].placement, Replicate())
        self.assertEqual(
            arguments["feed_forward.experts.gate_proj"].shape, (16, 1024, 5120)
        )
        self.assertEqual(
            arguments["feed_forward.experts.down_proj"].placement, Shard("intermediate")
        )
        self.assertEqual(function.results[1].type.shape, (1, 2048, 1, 128))

    def test_uneven_heads_are_rejected(self) -> None:
        """tp=3 does not divide 40 query heads; tp=5 does, but not 8 KV heads."""
        for tp, message in ((3, "q_head of size 40"), (5, "kv_head of size 8")):
            with self.subTest(tp=tp), self.assertRaisesRegex(ValueError, message):
                _sharded_module(ROPE_LAYER, tp)


class ConfigTest(unittest.TestCase):
    """Parsing Scout's config.json, its defaults, and the unmodeled parts."""

    def test_scout_values(self) -> None:
        """text_config is read; NoPE layers are 3, 7, ..., 47; every layer is MoE."""
        self.assertEqual(
            (SCOUT.hidden_size, SCOUT.num_attention_heads, SCOUT.num_key_value_heads),
            (5120, 40, 8),
        )
        self.assertEqual((SCOUT.num_local_experts, SCOUT.num_experts_per_tok), (16, 1))
        nope = [layer for layer in range(48) if not SCOUT.uses_rope(layer)]
        self.assertEqual(nope, list(range(3, 48, 4)))
        self.assertEqual(SCOUT.moe_layers, tuple(range(48)))
        self.assertEqual(SCOUT.dtype, torch.bfloat16)

    def test_defaults_when_entries_are_omitted(self) -> None:
        """An empty no_rope_layers means HF's default: NoPE every fourth layer."""
        text = json.loads(SCOUT_JSON.read_text())["text_config"]
        trimmed = {
            key: value
            for key, value in text.items()
            if key not in ("moe_layers", "floor_scale", "attn_scale")
        }
        trimmed["no_rope_layers"] = []
        config = Llama4Config.from_hf(trimmed)
        self.assertEqual(config.no_rope_layers, SCOUT.no_rope_layers)
        self.assertEqual(config.moe_layers, SCOUT.moe_layers)
        self.assertEqual((config.floor_scale, config.attn_scale), (8192.0, 0.1))

    def test_rejects_what_the_layer_does_not_model(self) -> None:
        """A long RoPE context needs chunk masks; a dense layer has no MoE."""
        long = example_inputs(SCOUT, context_len=SCOUT.attention_chunk_size + 1)
        with self.assertRaisesRegex(NotImplementedError, "attention_chunk_size"):
            export_model(build_meta_decoder_layer(SCOUT, ROPE_LAYER), long)
        dense = dataclasses.replace(SCOUT, moe_layers=(1,))
        with self.assertRaisesRegex(NotImplementedError, "dense"):
            build_meta_decoder_layer(dense, ROPE_LAYER)
        with self.assertRaisesRegex(ValueError, "outside"):
            SCOUT.uses_rope(48)


if __name__ == "__main__":
    unittest.main()
