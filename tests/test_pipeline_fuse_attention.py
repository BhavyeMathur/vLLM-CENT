"""Check the adaptor that fuses decomposed GQA attention for the CENT pipeline."""

import unittest

import torch
from torch import Tensor, nn

from vllm_cent.cent import CentHardwareSpec
from vllm_cent.frontend import aloi, export_model, import_exported_program
from vllm_cent.ir import (
    AttentionOp,
    DType,
    Function,
    GqaQKOp,
    GqaSVOp,
    ReshapeOp,
    Shard,
    SoftmaxOp,
    TensorType,
    Value,
    value_names,
)
from vllm_cent.ir.passes import FuseAttention
from vllm_cent.pipeline import CentTargetProfile, ExecutionContract, compile_graph
from vllm_cent.pipeline.ir import CapabilityProfile, legalize
from vllm_cent.runtime import CentNamedScalars

# Decode attention of one token: 4 query heads of width 2 over 5 cached
# positions, with values of width 3. GQA groups the 4 query heads over the
# cache's KV heads.
QUERY_HEADS = 4
CONTEXT = 5
KEY_WIDTH = 2
VALUE_WIDTH = 3
SCALE = 0.5


class _DecodeAttention(nn.Module):
    """Decomposed GQA attention of one decode step over a supplied cache."""

    def forward(self, q: Tensor, k_cache: Tensor, v_cache: Tensor) -> Tensor:
        """Attend from q to every cached position.

        Args:
            q: ``[batch, tokens, q_heads, key_width]``.
            k_cache: ``[batch, context, kv_heads, key_width]``.
            v_cache: ``[batch, context, kv_heads, value_width]``.

        Returns:
            ``[batch, tokens, q_heads, value_width]``.
        """
        scores = aloi.gqa_qk(q, k_cache, SCALE)
        context: Tensor = aloi.gqa_sv(aloi.softmax(scores), v_cache)
        return context


class _RotatedDecodeAttention(nn.Module):
    """Rotate the new token's query, then attend over a supplied key cache."""

    def forward(
        self,
        q: Tensor,
        k_cache: Tensor,
        v_cache: Tensor,
        positions: Tensor,
        table: Tensor,
    ) -> Tensor:
        """Rotate q by the new token's position, then attend to every cached key.

        Args:
            q: ``[1, 1, q_heads, key_width]``.
            k_cache: ``[1, context, kv_heads, key_width]``.
            v_cache: ``[1, context, kv_heads, value_width]``.
            positions: ``[1, 1]`` position of the new token.
            table: ``[positions, key_width]`` cosines then sines.

        Returns:
            ``[1, 1, q_heads, value_width]``.
        """
        q = aloi.rope(q, positions, table, True)
        scores = aloi.gqa_qk(q, k_cache, SCALE)
        context: Tensor = aloi.gqa_sv(aloi.softmax(scores), v_cache)
        return context


def _target() -> CentTargetProfile:
    """Return a small two-lane reference target with room for the tensors.

    Returns:
        A one-channel target with 512 Shared Buffer slots.
    """
    return CentTargetProfile(
        name="fuse-attention-reference",
        execution_contract=ExecutionContract.REFERENCE,
        max_accumulate_slots=2,
        hardware=CentHardwareSpec(
            num_channels=1,
            num_banks=4,
            dram_rows=1,
            dram_columns=4,
            burst_length=2,
            accumulator_slots_per_bank=2,
            sigmoid_activation_function_id=0,
            shared_buffer_slots=512,
        ),
    )


def _import(model: nn.Module, inputs: tuple[Tensor, ...]) -> Function:
    """Export ``model`` on meta tensors shaped like ``inputs`` and import it.

    Args:
        model: Module without parameters.
        inputs: Example inputs; only their shapes and dtypes are used.

    Returns:
        The imported ``forward`` function.
    """
    meta = tuple(torch.empty(t.shape, dtype=t.dtype, device="meta") for t in inputs)
    return import_exported_program(export_model(model, meta), "m").get_function(
        "forward"
    )


def _decode_inputs(kv_heads: int) -> tuple[Tensor, Tensor, Tensor]:
    """Return random q, k_cache, and v_cache for one decode token.

    Args:
        kv_heads: Heads in the cache; must divide ``QUERY_HEADS``.

    Returns:
        ``(q, k_cache, v_cache)`` in token-first layout.
    """
    return (
        torch.randn(1, 1, QUERY_HEADS, KEY_WIDTH),
        torch.randn(1, CONTEXT, kv_heads, KEY_WIDTH),
        torch.randn(1, CONTEXT, kv_heads, VALUE_WIDTH),
    )


def _run(function: Function, tensors: dict[str, Tensor]) -> Tensor:
    """Compile ``function``, bind tensors by input name, and return its output.

    Args:
        function: Function to compile with the reference target.
        tensors: A tensor for every input name; integers are bound as floats.

    Returns:
        The flattened output values.
    """
    artifact = compile_graph(function, _target())
    bindings = tuple(
        CentNamedScalars(
            name=value.name,
            values=tuple(float(e) for e in tensors[value.name].flatten().tolist()),
        )
        for value in artifact.inputs
    )
    (output,) = artifact.execute(bindings)
    return torch.tensor(output.values)


class FuseAttentionTests(unittest.TestCase):
    """The adaptor fuses chains a reshape can convert and leaves others alone."""

    def test_fuses_one_kv_head(self) -> None:
        """One KV head: [1, 5, 1, 2] and [1, 1, 5, 2] list the same elements.

        The chain becomes three layout reshapes, one AttentionOp with the
        same scale, and a reshape back that keeps gqa_sv's name.
        """
        source = _import(_DecodeAttention(), _decode_inputs(kv_heads=1))
        fused = FuseAttention().run_on_function(source)
        self.assertEqual(
            [type(op) for op in fused.operations],
            [ReshapeOp, ReshapeOp, ReshapeOp, AttentionOp, ReshapeOp],
        )
        attention = fused.operations[3]
        assert isinstance(attention, AttentionOp)
        self.assertEqual(
            attention.attributes(),
            {"causal": False, "query_position_offset": 0, "scale": SCALE},
        )
        self.assertEqual(
            [value_names(fused)[op.results[0]] for op in fused.operations],
            [
                "gqa_sv.q_bhsd",
                "gqa_sv.k_bhsd",
                "gqa_sv.v_bhsd",
                "gqa_sv.fused",
                "gqa_sv",
            ],
        )
        self.assertEqual(
            fused.results[0].type,
            TensorType(shape=(1, 1, QUERY_HEADS, VALUE_WIDTH), dtype=DType.FP32),
        )

    def test_keeps_chain_that_needs_a_transpose(self) -> None:
        """Two KV heads over five positions cannot change layout by a reshape.

        The chain stays decomposed, and legalization names the gqa_qk.
        """
        source = _import(_DecodeAttention(), _decode_inputs(kv_heads=2))
        fused = FuseAttention().run_on_function(source)
        self.assertEqual(
            [type(op) for op in fused.operations], [GqaQKOp, SoftmaxOp, GqaSVOp]
        )
        with self.assertRaisesRegex(NotImplementedError, r"aloi\.gqa_qk"):
            legalize(fused, CapabilityProfile())

    def test_keeps_chain_whose_probabilities_are_returned(self) -> None:
        """Fusion would hide the probabilities, so a returned softmax blocks it."""
        q = Value(type=TensorType(shape=(1, 1, 4, 2), dtype=DType.FP32), name="q")
        k = Value(type=TensorType(shape=(1, 5, 1, 2), dtype=DType.FP32), name="k")
        v = Value(type=TensorType(shape=(1, 5, 1, 3), dtype=DType.FP32), name="v")
        qk = GqaQKOp(operands=(q, k), scale=SCALE)
        softmax = SoftmaxOp(operands=qk.results)
        sv = GqaSVOp(operands=(softmax.results[0], v))
        source = Function(
            name="main",
            operands=(q, k, v),
            operations=(qk, softmax, sv),
            results=(sv.results[0], softmax.results[0]),
        )
        self.assertEqual(
            FuseAttention().run_on_function(source).operations, source.operations
        )

    def test_keeps_tensor_parallel_chain(self) -> None:
        """Heads split across ranks stay decomposed until the fused form has a TP rule.

        With tp=2, 8 query heads over 2 KV heads leave each rank 4 query heads
        and 1 KV head. One local KV head would pass the layout check, so only
        the placement check keeps the chain decomposed.
        """

        def split(
            name: str, shape: tuple[int, ...], axes: tuple[str, ...], axis: str
        ) -> Value:
            """Return a rank-local value that holds half of ``axis``.

            Args:
                name: Argument name.
                shape: Local shape; the split dimension is half the global one.
                axes: One axis name per dimension.
                axis: Axis split across the two ranks.

            Returns:
                A sharded fp32 value.
            """
            dim = axes.index(axis)
            whole = tuple(2 * d if i == dim else d for i, d in enumerate(shape))
            return Value(
                type=TensorType(
                    shape=shape,
                    dtype=DType.FP32,
                    axes=axes,
                    global_shape=whole,
                    placement=Shard(axis),
                ),
                name=name,
            )

        q = split(
            "q", (1, 1, 4, 2), ("batch", "seq_len", "q_head", "head_dim"), "q_head"
        )
        cache_axes = ("batch", "context", "kv_head", "head_dim")
        k = split("k", (1, 5, 1, 2), cache_axes, "kv_head")
        v = split("v", (1, 5, 1, 3), cache_axes, "kv_head")
        qk = GqaQKOp(operands=(q, k), scale=SCALE)
        softmax = SoftmaxOp(operands=qk.results)
        sv = GqaSVOp(operands=(softmax.results[0], v))
        source = Function(
            name="main",
            operands=(q, k, v),
            operations=(qk, softmax, sv),
            results=sv.results,
        )
        self.assertEqual(
            FuseAttention().run_on_function(source).operations, source.operations
        )

    def test_returns_the_same_function_without_chains(self) -> None:
        """Nothing to fuse leaves the input function itself, not a rebuilt copy."""
        x = Value(type=TensorType(shape=(2,), dtype=DType.FP32), name="x")
        source = Function(name="main", operands=(x,), operations=(), results=(x,))
        self.assertIs(FuseAttention().run_on_function(source), source)


class FusedAttentionExecutionTests(unittest.TestCase):
    """Decomposed attention written in PyTorch runs on CENT and matches PyTorch."""

    def test_decode_attention(self) -> None:
        """One token, four query heads sharing one KV head of five positions."""
        torch.manual_seed(1)
        q, k, v = _decode_inputs(kv_heads=1)
        model = _DecodeAttention()
        output = _run(_import(model, (q, k, v)), {"q": q, "k_cache": k, "v_cache": v})
        torch.testing.assert_close(output, model(q, k, v).flatten())

    def test_rotated_query_then_attention(self) -> None:
        """RoPE on the query, then fused attention, in one program.

        The new token is at position 4 of an 8-row table; its rotated query
        attends to all five cached keys.
        """
        torch.manual_seed(2)
        q, k, v = _decode_inputs(kv_heads=1)
        positions = torch.tensor([[4]])
        table = torch.randn(8, KEY_WIDTH)
        model = _RotatedDecodeAttention()
        inputs = (q, k, v, positions, table)
        names = ("q", "k_cache", "v_cache", "positions", "table")
        tensors = dict(zip(names, inputs, strict=True))
        output = _run(_import(model, inputs), tensors)
        torch.testing.assert_close(output, model(*inputs).flatten())


if __name__ == "__main__":
    unittest.main()
