"""Tests for the PyTorch frontend: custom op kernels and the importer.

The Llama decoder layer example is tested in test_llama_70b.py.
"""

from __future__ import annotations

import math
import unittest
from collections.abc import Callable
from unittest import mock

import torch
import torch.nn.functional as F  # noqa: N812
from torch import Tensor, nn
from torch.export import Dim

from vllm_cent.frontend import aloi, export_model, import_exported_program, importer
from vllm_cent.ir import print_module


# A test-only op whose fake kernel claims one more output feature than the
# real kernel produces. Routed to the linear converter, it makes PyTorch's and
# ALOI's shape inference disagree. Registered once at import: registering the
# same name twice raises.
# Its signature matches aloi::linear's so that the linear converter can read it.
@torch.library.custom_op("aloi_test::lying_linear", mutates_args=())
def _lying_linear(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Compute ``x @ weight.T + bias``, like ``aloi::linear``."""

    return F.linear(x, weight, bias)


@_lying_linear.register_fake
def _(x: Tensor, weight: Tensor, bias: Tensor | None = None) -> Tensor:
    """Return a result one column wider than the real kernel's."""

    return x.new_empty((*x.shape[:-1], weight.shape[0] + 1))


def _import(model: nn.Module, *inputs: Tensor) -> str:
    """Export, import and print ``model``.

    Args:
        model: Model to compile.
        inputs: Example inputs.

    Returns:
        The printed IR module.
    """

    program = export_model(model, inputs)
    return print_module(import_exported_program(program, "m"))


def _meta_input(*shape: int) -> Tensor:
    """Return an fp32 meta tensor of the given shape.

    Args:
        shape: Tensor shape.

    Returns:
        The tensor.
    """

    return torch.empty(shape, device="meta")


class _LinearWith(nn.Module):
    """Calls one linear-shaped op directly in the root forward."""

    def __init__(self, op: object, as_buffer: bool = False) -> None:
        """Create a ``[4, 16]`` weight.

        Args:
            op: Callable ``op(x, weight)``.
            as_buffer: Register the weight as a buffer, not a parameter.
        """

        super().__init__()
        self.op = op
        weight = torch.empty(4, 16)
        if as_buffer:
            self.register_buffer("w", weight)
        else:
            self.w = nn.Parameter(weight)

    def forward(self, x: Tensor) -> Tensor:
        """Apply the op.

        Args:
            x: Input of shape ``[*, 16]``.

        Returns:
            Output of shape ``[*, 4]``.
        """

        return self.op(x, self.w)  # type: ignore[operator, no-any-return]


class _RmsNormWithoutEps(nn.Module):
    """Calls ``aloi::rms_norm`` without passing eps."""

    def __init__(self) -> None:
        """Create a ``[16]`` scale."""

        super().__init__()
        self.g = nn.Parameter(torch.empty(16))

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x``.

        Args:
            x: Input of shape ``[*, 16]``.

        Returns:
            Output with the shape of ``x``.
        """

        result: Tensor = aloi.rms_norm(x, self.g)
        return result


class _Cosine(nn.Module):
    """Uses an aten op the importer has no converter for."""

    def forward(self, x: Tensor) -> Tensor:
        """Apply cosine.

        Args:
            x: Any tensor.

        Returns:
            ``cos(x)``.
        """

        return torch.cos(x)


class CustomOpTest(unittest.TestCase):
    """The custom ops compute what their F.* counterparts compute."""

    def test_real_kernels_match_functional(self) -> None:
        """On CPU, both ops equal F.rms_norm / F.linear bit for bit."""

        torch.manual_seed(0)
        x = torch.randn(2, 3, 16)
        gamma = torch.randn(16)
        weight = torch.randn(4, 16)
        self.assertTrue(
            torch.equal(
                aloi.rms_norm(x, gamma, 1e-5),
                F.rms_norm(x, (16,), gamma, 1e-5),
            )
        )
        self.assertTrue(torch.equal(aloi.linear(x, weight), F.linear(x, weight)))

    def test_meta_shapes(self) -> None:
        """Linear maps [*, in] to [*, out]; rms_norm keeps the shape."""

        x = _meta_input(2, 3, 16)
        self.assertEqual(aloi.linear(x, _meta_input(4, 16)).shape, (2, 3, 4))
        self.assertEqual(aloi.rms_norm(x, _meta_input(16), 1e-5).shape, (2, 3, 16))


def _meta_apply_llama3_scaling(
    frequencies: Tensor,
    factor: float,
    low_freq_factor: float,
    high_freq_factor: float,
    original_max_position: int,
) -> Tensor:
    """Meta's reference llama3 frequency scaling, written as a loop.

    This is a different formulation from rope_inverse_frequencies (torch.where
    masks), so agreement checks the arithmetic rather than a copy of it.

    Args:
        frequencies: Unscaled inverse frequencies.
        factor: Scaling factor.
        low_freq_factor: Low-frequency factor.
        high_freq_factor: High-frequency factor.
        original_max_position: Pretraining context length.

    Returns:
        The scaled frequencies.
    """

    low_freq_wavelength = original_max_position / low_freq_factor
    high_freq_wavelength = original_max_position / high_freq_factor
    scaled = []
    for frequency in frequencies.tolist():
        wavelength = 2 * math.pi / frequency
        if wavelength < high_freq_wavelength:
            scaled.append(frequency)
        elif wavelength > low_freq_wavelength:
            scaled.append(frequency / factor)
        else:
            smooth = (original_max_position / wavelength - low_freq_factor) / (
                high_freq_factor - low_freq_factor
            )
            scaled.append((1 - smooth) * frequency / factor + smooth * frequency)
    return torch.tensor(scaled, dtype=torch.float32)


class LayerTest(unittest.TestCase):
    """aloi.Linear and aloi.RMSNorm are used like their nn counterparts."""

    def test_linear_layer(self) -> None:
        """Same parameter name and [out, in] shape as nn.Linear(bias=False)."""

        torch.manual_seed(0)
        layer = aloi.Linear(16, 4)
        reference = nn.Linear(16, 4, bias=False)
        self.assertEqual(
            {name: tuple(p.shape) for name, p in layer.named_parameters()},
            {name: tuple(p.shape) for name, p in reference.named_parameters()},
        )
        with torch.no_grad():
            layer.weight.copy_(torch.randn(4, 16))
            reference.weight.copy_(layer.weight)
        x = torch.randn(2, 3, 16)
        self.assertTrue(torch.equal(layer(x), reference(x)))

    def test_linear_layer_with_bias(self) -> None:
        """bias=True matches nn.Linear: a [4] bias parameter, added to every row."""

        torch.manual_seed(0)
        layer = aloi.Linear(16, 4, bias=True)
        reference = nn.Linear(16, 4)
        self.assertEqual(
            {name: tuple(p.shape) for name, p in layer.named_parameters()},
            {"weight": (4, 16), "bias": (4,)},
        )
        with torch.no_grad():
            layer.weight.copy_(torch.randn(4, 16))
            layer.bias.copy_(torch.randn(4))
            reference.weight.copy_(layer.weight)
            reference.bias.copy_(layer.bias)
        x = torch.randn(2, 3, 16)
        torch.testing.assert_close(layer(x), reference(x))
        # Without a bias the attribute exists but holds no parameter.
        self.assertIsNone(aloi.Linear(16, 4).bias)

    def test_linear_fake_rejects_bad_bias(self) -> None:
        """The fake kernel checks the bias against out_features at export time."""

        with self.assertRaisesRegex(
            ValueError, r"Bias should have shape \[out_features\]"
        ):
            export_model(_LinearWithBias(4, 3), (torch.empty(1, 4, device="meta"),))

    def test_rms_norm_layer(self) -> None:
        """Same parameter name, initial weight and result as nn.RMSNorm."""

        layer = aloi.RMSNorm(16, 1e-5)
        reference = nn.RMSNorm(16, 1e-5)
        self.assertEqual(
            {name: tuple(p.shape) for name, p in layer.named_parameters()},
            {name: tuple(p.shape) for name, p in reference.named_parameters()},
        )
        self.assertTrue(torch.equal(layer.weight, reference.weight))
        x = torch.randn(2, 3, 16)
        torch.testing.assert_close(layer(x), reference(x))

    def test_rms_norm_layer_without_weight(self) -> None:
        """Without elementwise_affine there is no weight; the op has one operand."""

        layer = aloi.RMSNorm(4, 1e-5, elementwise_affine=False)
        reference = nn.RMSNorm(4, 1e-5, elementwise_affine=False)
        self.assertEqual(list(layer.named_parameters()), [])
        x = torch.randn(2, 4)
        torch.testing.assert_close(layer(x), reference(x))
        # _op_line keeps the layer as its wrapper's ``function`` submodule, so
        # the result is named after that module path.
        self.assertEqual(
            _op_line(layer, 1, 4),
            "%function = aloi.rms_norm(%x) {eps = 1e-05} : tensor<1x4xfp32>",
        )

    def test_layer_exports_as_its_op(self) -> None:
        """The exported graph holds the op node, not the module."""

        with torch.device("meta"):
            layer = aloi.Linear(16, 4)
        program = export_model(layer, (_meta_input(1, 16),))
        targets = [n.target for n in program.graph.nodes if n.op == "call_function"]
        self.assertEqual(targets, [torch.ops.aloi.linear.default])


class TransformerCustomOpTest(unittest.TestCase):
    """Real CPU kernels of the transformer custom ops."""

    def test_split_and_merge_heads_round_trip(self) -> None:
        """[2, 1, 12] is 3 heads of 4; merging restores the input."""

        x = torch.arange(24.0).reshape(2, 1, 12)
        heads = aloi.split_heads(x, 4)
        self.assertEqual(heads.shape, (2, 1, 3, 4))
        # Head 1 holds elements 4..7 of each token.
        self.assertTrue(torch.equal(heads[0, 0, 1], torch.tensor([4.0, 5, 6, 7])))
        self.assertTrue(torch.equal(aloi.merge_heads(heads), x))

    def test_split_heads_fake_rejects_bad_head_dim(self) -> None:
        """5 does not divide 12."""

        with self.assertRaises(ValueError):
            aloi.split_heads(_meta_input(1, 1, 12), 5)

    def test_rope_rotates_each_half_pair(self) -> None:
        """NeoX style: element i of a head pairs with element i + head_dim/2.

        head_dim 4, theta 10000: inverse frequencies are 10000^0 = 1 and
        10000^(-1/2) = 0.01. The unit vector [1, 0, 0, 0] at position 2 turns
        by angle 2 * 1 within the pair (0, 2): [cos 2, 0, sin 2, 0].
        """

        table = aloi.RotaryEmbedding(4, 4, 10000.0).cos_sin_cache
        x = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        rotated = aloi.rope(x, torch.tensor([[2]]), table)
        expected = torch.tensor([[[[math.cos(2.0), 0.0, math.sin(2.0), 0.0]]]])
        torch.testing.assert_close(rotated, expected)

    def test_rope_gptj_style_rotates_adjacent_pairs(self) -> None:
        """GPT-J style: elements 0 and 1 pair up.

        So [1, 0, 0, 0] gives [cos 2, sin 2, 0, 0].
        """

        table = aloi.RotaryEmbedding(4, 4, 10000.0).cos_sin_cache
        x = torch.tensor([[[[1.0, 0.0, 0.0, 0.0]]]])
        rotated = aloi.rope(x, torch.tensor([[2]]), table, False)
        expected = torch.tensor([[[[math.cos(2.0), math.sin(2.0), 0.0, 0.0]]]])
        torch.testing.assert_close(rotated, expected)

    def test_rope_styles_differ_by_a_permutation(self) -> None:
        """GPT-J style on x equals NeoX style on x with even elements first.

        This is why converting Meta's Llama weights to Hugging Face permutes
        the q/k projection rows.
        """

        table = aloi.RotaryEmbedding(8, 6, 10000.0).cos_sin_cache
        x = torch.randn(1, 1, 2, 8)
        positions = torch.tensor([[5]])
        evens_first = torch.tensor([0, 2, 4, 6, 1, 3, 5, 7])
        gptj = aloi.rope(x, positions, table, False)
        neox = aloi.rope(x[..., evens_first], positions, table, True)
        torch.testing.assert_close(gptj[..., evens_first], neox)

    def test_rope_partial_rotary_passes_the_rest_through(self) -> None:
        """rotary_dim 4 of head_dim 8: the last 4 elements are unchanged."""

        table = aloi.RotaryEmbedding(8, 6, 10000.0, rotary_dim=4).cos_sin_cache
        self.assertEqual(table.shape, (6, 4))
        x = torch.randn(1, 1, 2, 8)
        rotated = aloi.rope(x, torch.tensor([[3]]), table)
        self.assertTrue(torch.equal(rotated[..., 4:], x[..., 4:]))
        self.assertFalse(torch.equal(rotated[..., :4], x[..., :4]))

    def test_rope_position_zero_is_identity(self) -> None:
        """Angle 0 leaves every head unchanged."""

        table = aloi.RotaryEmbedding(8, 4, 10000.0).cos_sin_cache
        x = torch.randn(1, 1, 3, 8)
        torch.testing.assert_close(aloi.rope(x, torch.tensor([[0]]), table), x)

    def test_llama3_frequencies_match_meta_reference(self) -> None:
        """Llama 3.1 70B: head_dim 128, theta 500000, factor 8."""

        llama3: dict[str, float | int | str] = {
            "rope_type": "llama3",
            "factor": 8.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
        plain, plain_scale = aloi.rope_inverse_frequencies(128, 500000.0)
        scaled, scaled_scale = aloi.rope_inverse_frequencies(128, 500000.0, llama3)
        torch.testing.assert_close(
            scaled, _meta_apply_llama3_scaling(plain, 8.0, 1.0, 4.0, 8192)
        )
        self.assertEqual((plain_scale, scaled_scale), (1.0, 1.0))
        # High frequencies are kept, low ones slowed by the full factor 8.
        self.assertEqual(scaled[0].item(), plain[0].item())
        self.assertAlmostEqual(scaled[-1].item(), plain[-1].item() / 8, places=12)

    def test_unsupported_rope_type_is_rejected(self) -> None:
        """A variant without a frequency function fails loudly."""

        with self.assertRaisesRegex(NotImplementedError, "yarn"):
            aloi.rope_inverse_frequencies(128, 10000.0, {"rope_type": "yarn"})

    def test_rotary_embedding_layer(self) -> None:
        """The table is a non-persistent buffer; forward rotates q and k."""

        layer = aloi.RotaryEmbedding(8, 16, 10000.0)
        self.assertEqual(layer.cos_sin_cache.shape, (16, 8))
        # Derived from the config, so it is not part of a checkpoint.
        self.assertNotIn("cos_sin_cache", layer.state_dict())
        positions = torch.tensor([[7]])
        q, k = torch.randn(1, 1, 4, 8), torch.randn(1, 1, 2, 8)
        rotated_q, rotated_k = layer(positions, q, k)
        torch.testing.assert_close(
            rotated_q, aloi.rope(q, positions, layer.cos_sin_cache)
        )
        torch.testing.assert_close(
            rotated_k, aloi.rope(k, positions, layer.cos_sin_cache)
        )

    def test_kv_update_writes_each_request_at_its_position(self) -> None:
        """Request 0 writes row 4, request 1 row 2; the input is not modified."""

        cache = torch.zeros(2, 5, 1, 3)
        new = torch.ones(2, 1, 1, 3)
        updated = aloi.kv_update(cache, new, torch.tensor([[4], [2]]))
        written_rows = updated.any(-1).any(-1).nonzero().tolist()
        self.assertEqual(written_rows, [[0, 4], [1, 2]])
        self.assertEqual(cache.abs().sum().item(), 0.0)

    def test_gqa_matches_per_head_reference(self) -> None:
        """4 query heads over 2 KV heads: heads 0,1 read KV 0; heads 2,3 KV 1."""

        torch.manual_seed(0)
        q = torch.randn(1, 1, 4, 8)
        k_cache = torch.randn(1, 5, 2, 8)
        v_cache = torch.randn(1, 5, 2, 8)
        scale = 8**-0.5
        scores = aloi.gqa_qk(q, k_cache, scale)
        probs = aloi.softmax(scores)
        output = aloi.gqa_sv(probs, v_cache)

        for head in range(4):
            kv_head = head // 2
            want_scores = (k_cache[0, :, kv_head] @ q[0, 0, head]) * scale
            want_probs = torch.softmax(want_scores, dim=-1)
            torch.testing.assert_close(scores[0, 0, head], want_scores)
            torch.testing.assert_close(probs[0, 0, head], want_probs)
            torch.testing.assert_close(
                output[0, 0, head], want_probs @ v_cache[0, :, kv_head]
            )

    def test_gqa_fake_rejects_heads_that_do_not_group(self) -> None:
        """3 query heads cannot share 2 KV heads evenly."""

        with self.assertRaises(ValueError):
            aloi.gqa_qk(_meta_input(1, 1, 3, 8), _meta_input(1, 5, 2, 8), 1.0)


class ImporterTest(unittest.TestCase):
    """Exported programs become verified ALOI IR."""

    def test_omitted_eps_imports_as_none(self) -> None:
        """Without eps, the FX call has two args and the IR stores None."""

        text = _import(_RmsNormWithoutEps(), _meta_input(1, 16))
        self.assertEqual(
            text.splitlines()[3],
            "  %rms_norm = aloi.rms_norm(%x, %g) {eps = none} : tensor<1x16xfp32>",
        )

    def test_root_level_op_is_named_after_its_node(self) -> None:
        """An op called in the root forward has no module path.

        Its result takes FX's node name, here ``linear``.
        """

        with torch.device("meta"):
            model = _LinearWith(aloi.linear)
        text = _import(model, _meta_input(1, 16))
        self.assertEqual(
            text.splitlines()[3],
            "  %linear = aloi.linear(%x, %w) : tensor<1x4xfp32>",
        )

    def test_unknown_op_is_rejected(self) -> None:
        """An aten op without a converter names itself in the error."""

        with self.assertRaisesRegex(NotImplementedError, r"aten\.cos\.default"):
            _import(_Cosine(), _meta_input(1, 16))

    def test_dynamic_shape_is_rejected(self) -> None:
        """A symbolic dimension is not silently specialized.

        4 rows, because export specializes sizes 0 and 1 to constants.
        """

        with torch.device("meta"):
            model = _LinearWith(aloi.linear)
        program = torch.export.export(
            model, (_meta_input(4, 16),), dynamic_shapes={"x": {0: Dim("rows")}}
        )
        with self.assertRaisesRegex(NotImplementedError, "dynamic dimension"):
            import_exported_program(program, "m")

    def test_buffer_is_named_by_fqn(self) -> None:
        """A registered buffer becomes an argument named like a parameter."""

        with torch.device("meta"):
            model = _LinearWith(aloi.linear, as_buffer=True)
        text = _import(model, _meta_input(1, 16))
        self.assertIn(
            "func @forward(%w: tensor<4x16xfp32>, %x: tensor<1x16xfp32>)", text
        )

    def test_constant_tensor_is_rejected(self) -> None:
        """An unregistered tensor attribute has no name in the model's state."""

        with torch.device("meta"):
            model = _ConstantScale()
        with self.assertRaisesRegex(NotImplementedError, "CONSTANT_TENSOR"):
            _import(model, _meta_input(1, 16))

    def test_shape_disagreement_is_rejected(self) -> None:
        """IR and PyTorch shape inference must agree.

        The real result is [1, 4]; the lying fake kernel says [1, 5].
        """

        with torch.device("meta"):
            model = _LinearWith(_lying_linear)
        lying = torch.ops.aloi_test.lying_linear.default
        with mock.patch.dict(
            importer._CONVERTERS,
            {lying: importer._CONVERTERS[torch.ops.aloi.linear.default]},
        ):
            with self.assertRaisesRegex(
                ValueError,
                r"infers tensor<1x4xfp32>, but PyTorch says tensor<1x5xfp32>",
            ):
                _import(model, _meta_input(1, 16))


class _Gated(nn.Module):
    """Two ops inside one module and one op inside a submodule."""

    def __init__(self) -> None:
        """Create a ``[16, 16]`` projection."""

        super().__init__()
        self.proj = _LinearWith(aloi.linear)
        self.proj.w = nn.Parameter(torch.empty(16, 16))

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``silu(proj(x)) * x + x`` with plain PyTorch elementwise ops.

        Args:
            x: Input of shape ``[*, 16]``.

        Returns:
            Output with the shape of ``x``.
        """

        projected: Tensor = self.proj(x)
        return F.silu(projected) * x + x


class _Inner(nn.Module):
    """Wraps _Gated so its ops get a module path."""

    def __init__(self) -> None:
        """Create the wrapped module."""

        super().__init__()
        self.block = _Gated()

    def forward(self, x: Tensor) -> Tensor:
        """Run the block.

        Args:
            x: Input of shape ``[*, 16]``.

        Returns:
            Output with the shape of ``x``.
        """

        result: Tensor = self.block(x)
        return result


class _ScaledAdd(nn.Module):
    """``torch.add`` with ``alpha``, which has no IR op."""

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``x + 2 * x``.

        Args:
            x: Any tensor.

        Returns:
            ``3 * x``.
        """

        return torch.add(x, x, alpha=2)


class _ScalarMul(nn.Module):
    """Multiplies by a Python number, which is not a tensor operand."""

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``2 * x``.

        Args:
            x: Any tensor.

        Returns:
            ``2 * x``.
        """

        return x * 2.0


class _TableRope(nn.Module):
    """Calls rope without is_neox_style: the call has 3 of 4 arguments."""

    def __init__(self) -> None:
        """Create a table for head_dim 8 and 4 positions."""

        super().__init__()
        self.rotary_emb = aloi.RotaryEmbedding(8, 4, 10000.0)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        """Rotate x.

        Args:
            x: ``[batch, seq_len, heads, head_dim]``.
            positions: ``[batch, seq_len]``.

        Returns:
            The rotated heads.
        """

        result: Tensor = aloi.rope(x, positions, self.rotary_emb.cos_sin_cache)
        return result


class _ConstantScale(nn.Module):
    """Multiplies by a tensor attribute that is not a registered buffer."""

    def __init__(self) -> None:
        """Keep a plain tensor attribute."""

        super().__init__()
        self.scale = torch.ones(16)

    def forward(self, x: Tensor) -> Tensor:
        """Scale x.

        Args:
            x: ``[*, 16]``.

        Returns:
            ``x * scale``.
        """

        return x * self.scale


class _Unary(nn.Module):
    """Applies one plain PyTorch function to its input."""

    def __init__(self, function: Callable[[Tensor], Tensor]) -> None:
        """Remember the function to apply.

        Args:
            function: Called on the forward input, e.g. ``F.silu``.
        """

        super().__init__()
        self.function = function

    def forward(self, x: Tensor) -> Tensor:
        """Apply the function.

        Args:
            x: Any tensor.

        Returns:
            ``function(x)``.
        """

        return self.function(x)


def _op_line(function: Callable[[Tensor], Tensor], *shape: int) -> str:
    """Import ``function`` applied to an input of ``shape``; return its op line.

    Args:
        function: Plain PyTorch function of one tensor.
        shape: Input shape.

    Returns:
        The single op's printed line, without indentation.
    """

    return _import(_Unary(function), _meta_input(*shape)).splitlines()[3].strip()


class _Experts(nn.Module):
    """Wrap a MoEExperts layer so its parameters get the ``experts.`` prefix."""

    def __init__(self, experts: aloi.MoEExperts) -> None:
        """Keep the layer.

        Args:
            experts: The experts to run.
        """
        super().__init__()
        self.experts = experts

    def forward(self, x: Tensor, routing_weights: Tensor) -> Tensor:
        """Run the experts.

        Args:
            x: ``[tokens, hidden]``.
            routing_weights: ``[tokens, experts]``.

        Returns:
            ``[tokens, hidden]``.
        """
        output: Tensor = self.experts(x, routing_weights)
        return output


class Llama4KernelTest(unittest.TestCase):
    """The eager kernels of the Llama 4 ops compute their documented formulas."""

    def test_attn_temperature(self) -> None:
        """floor_scale 4: positions 0..2 keep scale 1; 3 gets log(2) * 0.5 + 1."""
        x = torch.ones(1, 2, 1, 2)
        positions = torch.tensor([[2, 3]])
        scaled = aloi.attn_temperature(x, positions, 4.0, 0.5)
        expected = torch.tensor([1.0, math.log(2.0) * 0.5 + 1.0])
        torch.testing.assert_close(scaled[0, :, 0, 0], expected)

    def test_topk_sigmoid(self) -> None:
        """Logits (2, -1, 0.5) with top_k 1 and 2 keep the sigmoid of the best ones."""
        logits = torch.tensor([[2.0, -1.0, 0.5]])
        sigmoid = torch.sigmoid(logits)[0]
        torch.testing.assert_close(
            aloi.topk_sigmoid(logits, 1), torch.tensor([[sigmoid[0].item(), 0.0, 0.0]])
        )
        torch.testing.assert_close(
            aloi.topk_sigmoid(logits, 2),
            torch.tensor([[sigmoid[0].item(), 0.0, sigmoid[2].item()]]),
        )

    def test_moe_experts_weight_on_input_or_output(self) -> None:
        """Two tokens, three experts; compare with each chosen expert run alone."""
        torch.manual_seed(0)
        gate, up = torch.randn(3, 5, 4), torch.randn(3, 5, 4)
        down = torch.randn(3, 4, 5)
        x = torch.randn(2, 4)
        # Token 0 uses expert 2 with weight 0.7, token 1 expert 0 with 0.4.
        routing = torch.tensor([[0.0, 0.0, 0.7], [0.4, 0.0, 0.0]])

        def expert(e: int, u: Tensor) -> Tensor:
            """Run expert ``e`` on input ``u``."""
            return (F.silu(u @ gate[e].T) * (u @ up[e].T)) @ down[e].T

        on_input = torch.stack((expert(2, 0.7 * x[0]), expert(0, 0.4 * x[1])))
        on_output = torch.stack((0.7 * expert(2, x[0]), 0.4 * expert(0, x[1])))
        for flag, expected in ((True, on_input), (False, on_output)):
            with self.subTest(apply_router_weight_on_input=flag):
                torch.testing.assert_close(
                    aloi.moe_experts(x, routing, gate, up, down, flag), expected
                )


class _LinearWithBias(nn.Module):
    """Calls aloi.linear with a bias of a chosen length."""

    def __init__(self, out_features: int, bias_width: int) -> None:
        """Create a [out_features, 4] weight and a [bias_width] bias.

        Args:
            out_features: Rows of the weight.
            bias_width: Length of the bias; differs from out_features to
                test the fake kernel's check.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_features, 4, device="meta"))
        self.bias = nn.Parameter(torch.empty(bias_width, device="meta"))

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``x @ weight.T + bias``.

        Args:
            x: ``[*, 4]``.

        Returns:
            ``[*, out_features]``.
        """
        projected: Tensor = aloi.linear(x, self.weight, self.bias)
        return projected


class ConverterTest(unittest.TestCase):
    """aten elementwise ops, argument defaults and result names."""

    def test_aten_elementwise_ops_and_name_hints(self) -> None:
        """F.silu, * and + import as aloi ops.

        Names: the projection is the only op in module ``block.proj``, so it
        is named after it. ``block`` runs three ops itself, so each gets the
        module path plus FX's node name.
        """

        with torch.device("meta"):
            model = _Inner()
        lines = _import(model, _meta_input(1, 16)).splitlines()[3:7]
        self.assertEqual(
            lines,
            [
                "  %block.proj = aloi.linear(%x, %block.proj.w) : tensor<1x16xfp32>",
                "  %block.silu = aloi.activation(%block.proj) {function = silu} "
                ": tensor<1x16xfp32>",
                "  %block.mul = aloi.mul(%block.silu, %x) : tensor<1x16xfp32>",
                "  %block.add = aloi.add(%block.mul, %x) : tensor<1x16xfp32>",
            ],
        )

    def test_llama4_ops_import_with_their_attributes(self) -> None:
        """attn_temperature, topk_sigmoid and moe_experts keep their constants."""

        positions = torch.empty(1, 2, dtype=torch.int64, device="meta")

        class _Temperature(nn.Module):
            """Scale queries by their positions."""

            def forward(self, q: Tensor, positions: Tensor) -> Tensor:
                """Apply attn_temperature with Scout's constants.

                Args:
                    q: ``[1, 2, 3, 4]`` queries.
                    positions: ``[1, 2]`` positions.

                Returns:
                    The scaled queries.
                """
                scaled: Tensor = aloi.attn_temperature(q, positions, 8192.0, 0.1)
                return scaled

        text = _import(_Temperature(), _meta_input(1, 2, 3, 4), positions)
        self.assertIn(
            "aloi.attn_temperature(%q, %positions) {floor_scale = 8192.0, attn_scale = 0.1}",
            text,
        )
        self.assertEqual(
            _op_line(lambda x: aloi.topk_sigmoid(x, 2), 1, 4),
            "%topk_sigmoid = aloi.topk_sigmoid(%x) {top_k = 2} : tensor<1x4xfp32>",
        )
        with torch.device("meta"):
            experts = aloi.MoEExperts(3, 4, 5, apply_router_weight_on_input=True)
        routing = torch.empty(2, 3, device="meta")
        text = _import(_Experts(experts), _meta_input(2, 4), routing)
        self.assertIn(
            "aloi.moe_experts(%x, %routing_weights, %experts.gate_proj, "
            "%experts.up_proj, %experts.down_proj) {apply_router_weight_on_input = true}",
            text,
        )

    def test_linear_bias_is_the_third_operand(self) -> None:
        """aloi.Linear(bias=True) imports as aloi.linear(%x, %w, %b)."""

        with torch.device("meta"):
            layer = aloi.Linear(4, 3, bias=True)
        self.assertEqual(
            _op_line(layer, 1, 4),
            "%function = aloi.linear(%x, %function.weight, %function.bias) "
            ": tensor<1x3xfp32>",
        )

    def test_aten_activations_become_one_op(self) -> None:
        """Every aten activation imports as aloi.activation with its kind."""

        cases = (
            (F.silu, "silu", "silu"),
            (torch.relu, "relu", "relu"),
            (torch.sigmoid, "sigmoid", "sigmoid"),
            (torch.tanh, "tanh", "tanh"),
            (F.gelu, "gelu", "gelu_erf"),
        )
        for function, node, kind in cases:
            with self.subTest(kind=kind):
                self.assertEqual(
                    _op_line(function, 1, 4),
                    f"%{node} = aloi.activation(%x) {{function = {kind}}} "
                    ": tensor<1x4xfp32>",
                )

    def test_gelu_approximation_picks_the_formula(self) -> None:
        """approximate="tanh" is a different formula from the default erf."""

        line = _op_line(lambda x: F.gelu(x, approximate="tanh"), 1, 4)
        self.assertIn("{function = gelu_tanh}", line)

    def test_clone_becomes_copy(self) -> None:
        """x.clone() keeps the shape and the row-major element order."""

        self.assertEqual(
            _op_line(torch.clone, 2, 3), "%clone = aloi.copy(%x) : tensor<2x3xfp32>"
        )

    def test_clone_that_reorders_memory_is_rejected(self) -> None:
        """channels_last stores a 4-d tensor in another element order."""

        with self.assertRaisesRegex(NotImplementedError, "memory_format"):
            _op_line(lambda x: x.clone(memory_format=torch.channels_last), 1, 2, 2, 2)

    def test_view_and_reshape_become_reshape(self) -> None:
        """view(-1, 3) on six elements resolves -1 to 2."""

        self.assertEqual(
            _op_line(lambda x: x.view(-1, 3), 6),
            "%view = aloi.reshape(%x) {shape = [2, 3]} : tensor<2x3xfp32>",
        )
        self.assertEqual(
            _op_line(lambda x: x.reshape(3, 2), 6),
            "%reshape = aloi.reshape(%x) {shape = [3, 2]} : tensor<3x2xfp32>",
        )

    def test_add_with_alpha_is_rejected(self) -> None:
        """``a + 2 * b`` is not an elementwise add of two tensors."""

        with self.assertRaisesRegex(NotImplementedError, "alpha=2"):
            _import(_ScaledAdd(), _meta_input(1, 16))

    def test_scalar_operand_is_rejected(self) -> None:
        """Only tensor operands are supported."""

        with self.assertRaisesRegex(TypeError, "expected a tensor operand"):
            _import(_ScalarMul(), _meta_input(1, 16))

    def test_omitted_optional_arguments_get_schema_defaults(self) -> None:
        """torch.export drops the unset is_neox_style; the IR shows its default."""

        positions = torch.empty(1, 1, dtype=torch.int64, device="meta")
        with torch.device("meta"):
            model = _TableRope()
        text = _import(model, _meta_input(1, 1, 2, 8), positions)
        self.assertEqual(
            text.splitlines()[3],
            "  %rope = aloi.rope(%x, %positions, %rotary_emb.cos_sin_cache) "
            "{is_neox_style = true} : tensor<1x1x2x8xfp32>",
        )


if __name__ == "__main__":
    unittest.main()
