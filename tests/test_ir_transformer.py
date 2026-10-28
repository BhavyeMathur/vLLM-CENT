"""Tests for the transformer semantic ops: heads, RoPE, KV cache, GQA, MoE."""

from __future__ import annotations

import unittest

from vllm_cent.ir.base import Value
from vllm_cent.ir.semantic.transformer import (
    AttentionOp,
    AttnTemperatureOp,
    GqaQKOp,
    GqaSVOp,
    KvUpdateOp,
    MergeHeadsOp,
    MoeExpertsOp,
    RopeOp,
    SplitHeadsOp,
    TopKSigmoidOp,
)
from vllm_cent.ir.types import (
    DType,
    Partial,
    Placement,
    Replicate,
    Shard,
    TensorRole,
    TensorType,
    packed_axis,
)

# Llama 70B attention under 8-way tensor parallelism: 64 query heads and 8 KV
# heads of 128, so each rank holds 8 query heads and 1 KV head. Decode: one
# new token; the cache holds 2048 positions.
Q_HEADS = 64
KV_HEADS = 8
HEAD_DIM = 128
TP = 8
CONTEXT = 2048
Q_PROJ_OUT = packed_axis("q_head", "head_dim")
TOKEN_AXES = ("batch", "seq_len")
CACHE_AXES = ("batch", "context", "kv_head", "head_dim")
REPLICATED = Replicate()


def _value(
    shape: tuple[int, ...],
    axes: tuple[str, ...] = (),
    placement: Placement = REPLICATED,
    global_shape: tuple[int, ...] | None = None,
    dtype: DType = DType.BF16,
    role: TensorRole = TensorRole.ACTIVATION,
) -> Value:
    """Create a value; every field but the shape is optional.

    Args:
        shape: Local shape.
        axes: Axis names, or () before annotation.
        placement: Which part each rank holds.
        global_shape: Whole shape, for shards.
        dtype: Element type.
        role: Tensor role.

    Returns:
        A new value.
    """

    return Value(
        type=TensorType(
            shape=shape,
            dtype=dtype,
            role=role,
            axes=axes,
            global_shape=global_shape,
            placement=placement,
        )
    )


def _positions() -> Value:
    """Return the replicated ``[batch=1, seq_len=1]`` int64 positions."""

    return _value((1, 1), TOKEN_AXES, dtype=DType.INT64)


def _q_heads(local_heads: int = Q_HEADS) -> Value:
    """Return query heads ``[1, 1, heads, 128]``, split if fewer than 64."""

    if local_heads == Q_HEADS:
        return _value((1, 1, Q_HEADS, HEAD_DIM), (*TOKEN_AXES, "q_head", "head_dim"))
    return _value(
        (1, 1, local_heads, HEAD_DIM),
        (*TOKEN_AXES, "q_head", "head_dim"),
        Shard("q_head"),
        (1, 1, Q_HEADS, HEAD_DIM),
    )


def _cache(local_kv_heads: int = KV_HEADS) -> Value:
    """Return a KV cache ``[1, 2048, kv_heads, 128]``, split if fewer than 8."""

    if local_kv_heads == KV_HEADS:
        return _value(
            (1, CONTEXT, KV_HEADS, HEAD_DIM), CACHE_AXES, role=TensorRole.KV_CACHE
        )
    return _value(
        (1, CONTEXT, local_kv_heads, HEAD_DIM),
        CACHE_AXES,
        Shard("kv_head"),
        (1, CONTEXT, KV_HEADS, HEAD_DIM),
        role=TensorRole.KV_CACHE,
    )


def _scores(local_heads: int = Q_HEADS) -> Value:
    """Return attention probabilities ``[1, 1, heads, 2048]``."""

    axes = (*TOKEN_AXES, "q_head", "context")
    if local_heads == Q_HEADS:
        return _value((1, 1, Q_HEADS, CONTEXT), axes)
    return _value(
        (1, 1, local_heads, CONTEXT), axes, Shard("q_head"), (1, 1, Q_HEADS, CONTEXT)
    )


class SplitMergeHeadsTest(unittest.TestCase):
    """Head reshapes turn a packed axis into two axes and back."""

    def test_split_unpacks_axis_and_keeps_shard(self) -> None:
        """q_proj's [1, 1, 1024] shard is 8 whole heads of 128 on this rank."""

        q_flat = _value(
            (1, 1, Q_HEADS * HEAD_DIM // TP),
            (*TOKEN_AXES, Q_PROJ_OUT),
            Shard("q_head"),
            (1, 1, Q_HEADS * HEAD_DIM),
        )
        op = SplitHeadsOp(operands=(q_flat,), head_dim=HEAD_DIM)
        self.assertEqual(op.results[0].type, _q_heads(Q_HEADS // TP).type)
        self.assertEqual(op.attributes(), {"head_dim": HEAD_DIM})

    def test_merge_is_the_inverse(self) -> None:
        """Merging the split heads restores the packed shard."""

        q_flat = _value(
            (1, 1, Q_HEADS * HEAD_DIM // TP),
            (*TOKEN_AXES, Q_PROJ_OUT),
            Shard("q_head"),
            (1, 1, Q_HEADS * HEAD_DIM),
        )
        merged = MergeHeadsOp(operands=(_q_heads(Q_HEADS // TP),))
        self.assertEqual(merged.results[0].type, q_flat.type)

    def test_unannotated_shapes(self) -> None:
        """Before annotation only shapes change: 8192 = 64 x 128."""

        split = SplitHeadsOp(operands=(_value((1, 1, 8192)),), head_dim=HEAD_DIM)
        self.assertEqual(split.results[0].type.shape, (1, 1, 64, 128))
        self.assertEqual(split.results[0].type.axes, ())

    def test_split_rejections(self) -> None:
        """Wrong head_dim, a plain axis, or a partial sum."""

        cases = {
            "head_dim does not divide": (_value((1, 1, 8192)), 100),
            "plain last axis": (_value((1, 1, 8192), (*TOKEN_AXES, "hidden")), 128),
            "partial sum": (
                _value((1, 1, 8192), (*TOKEN_AXES, Q_PROJ_OUT), Partial()),
                128,
            ),
        }
        for name, (x, head_dim) in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                SplitHeadsOp(operands=(x,), head_dim=head_dim)

    def test_merge_rejects_split_head_dim(self) -> None:
        """head_dim would become the inner factor of the merged axis."""

        split_head_dim = _value(
            (1, 1, Q_HEADS, HEAD_DIM // TP),
            (*TOKEN_AXES, "q_head", "head_dim"),
            Shard("head_dim"),
            (1, 1, Q_HEADS, HEAD_DIM),
        )
        with self.assertRaisesRegex(ValueError, "inner factor"):
            MergeHeadsOp(operands=(split_head_dim,))


def _table(
    max_position: int = 131072,
    rotary_dim: int = HEAD_DIM,
    placement: Placement = REPLICATED,
) -> Value:
    """Return a RoPE cos/sin table ``[max_position, rotary_dim]``.

    Llama 3.1 70B: 131072 positions, 64 cosines then 64 sines per row.
    """

    if isinstance(placement, Shard):
        return _value(
            (max_position, rotary_dim // TP),
            ("position", "rotary_dim"),
            placement,
            (max_position, rotary_dim),
            role=TensorRole.WEIGHT,
        )
    return _value(
        (max_position, rotary_dim), ("position", "rotary_dim"), role=TensorRole.WEIGHT
    )


class RopeOpTest(unittest.TestCase):
    """RoPE rotates whole head vectors; it never changes the type."""

    def test_keeps_type(self) -> None:
        """A sharded q stays sharded; only the pairing style is an attribute."""

        q = _q_heads(Q_HEADS // TP)
        op = RopeOp(operands=(q, _positions(), _table()))
        self.assertEqual(op.results[0].type, q.type)
        self.assertEqual(op.attributes(), {"is_neox_style": True})

    def test_partial_rotary_table(self) -> None:
        """A table narrower than head_dim rotates only the leading elements."""

        op = RopeOp(
            operands=(_q_heads(), _positions(), _table(rotary_dim=64)),
            is_neox_style=False,
        )
        self.assertEqual(op.results[0].type, _q_heads().type)

    def test_rejections(self) -> None:
        """Invalid shapes, positions, tables, or placements."""

        float_positions = _value((1, 1), TOKEN_AXES, dtype=DType.FP32)
        wrong_positions = _value((1, 2), TOKEN_AXES, dtype=DType.INT64)
        split_head_dim = _value(
            (1, 1, Q_HEADS, HEAD_DIM // TP),
            (*TOKEN_AXES, "q_head", "head_dim"),
            Shard("head_dim"),
            (1, 1, Q_HEADS, HEAD_DIM),
        )
        cases: dict[str, tuple[Value, Value, Value, str]] = {
            "odd rotary_dim": (
                _q_heads(),
                _positions(),
                _table(rotary_dim=127),
                "even rotary_dim",
            ),
            "table wider than head": (
                _q_heads(),
                _positions(),
                _table(rotary_dim=256),
                "even rotary_dim",
            ),
            "float positions": (_q_heads(), float_positions, _table(), "integers"),
            "one position per token": (
                _q_heads(),
                wrong_positions,
                _table(),
                "token dimensions",
            ),
            "split head_dim": (
                split_head_dim,
                _positions(),
                _table(),
                "split along head_dim",
            ),
            "split table": (
                _q_heads(),
                _positions(),
                _table(placement=Shard("rotary_dim")),
                "cos_sin_cache must be replicated",
            ),
        }
        for name, (x, positions, table, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                RopeOp(operands=(x, positions, table))


class KvUpdateOpTest(unittest.TestCase):
    """The updated cache has the cache's type, KV_CACHE role included."""

    def test_sharded_update(self) -> None:
        """Each rank writes its one KV head into its one cached KV head."""

        new_k = _value(
            (1, 1, 1, HEAD_DIM),
            (*TOKEN_AXES, "kv_head", "head_dim"),
            Shard("kv_head"),
            (1, 1, KV_HEADS, HEAD_DIM),
        )
        cache = _cache(KV_HEADS // TP)
        op = KvUpdateOp(operands=(cache, new_k, _positions()))
        self.assertEqual(op.results[0].type, cache.type)
        self.assertIs(op.results[0].type.role, TensorRole.KV_CACHE)

    def test_rejections(self) -> None:
        """Mismatched heads, axes, positions, or placements."""

        new_k = _value((1, 1, KV_HEADS, HEAD_DIM), (*TOKEN_AXES, "kv_head", "head_dim"))
        # One head, but replicated: shapes fit the split cache, placements do not.
        one_whole_head = _value(
            (1, 1, 1, HEAD_DIM), (*TOKEN_AXES, "kv_head", "head_dim")
        )
        cases = {
            "head count differs": (
                _cache(),
                _value((1, 1, 4, HEAD_DIM), (*TOKEN_AXES, "kv_head", "head_dim")),
                _positions(),
                "does not fit cache",
            ),
            "head axis differs": (
                _cache(),
                _value((1, 1, KV_HEADS, HEAD_DIM), (*TOKEN_AXES, "q_head", "head_dim")),
                _positions(),
                "do not match cache axes",
            ),
            "positions not one per token": (
                _cache(),
                new_k,
                _value((1, 2), TOKEN_AXES, dtype=DType.INT64),
                "positions must be integers",
            ),
            "cache split, new whole": (
                _cache(1),
                one_whole_head,
                _positions(),
                "unsupported placements",
            ),
        }
        for name, (cache, new, positions, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                KvUpdateOp(operands=(cache, new, positions))


class GqaOpsTest(unittest.TestCase):
    """gqa_qk and gqa_sv: per-head products with built-in head groups."""

    def test_qk_shapes_axes_and_scale(self) -> None:
        """[1, 1, 64, 128] x [1, 2048, 8, 128] gives [1, 1, 64, 2048]."""

        op = GqaQKOp(operands=(_q_heads(), _cache()), scale=HEAD_DIM**-0.5)
        self.assertEqual(op.results[0].type, _scores().type)
        self.assertEqual(op.attributes(), {"scale": HEAD_DIM**-0.5})

    def test_sv_shapes_and_axes(self) -> None:
        """[1, 1, 64, 2048] x [1, 2048, 8, 128] gives [1, 1, 64, 128]."""

        op = GqaSVOp(operands=(_scores(), _cache()))
        self.assertEqual(op.results[0].type, _q_heads().type)

    def test_tensor_parallel_heads(self) -> None:
        """8 query heads read this rank's 1 KV head: the group stays 8."""

        qk = GqaQKOp(operands=(_q_heads(8), _cache(1)), scale=1.0)
        sv = GqaSVOp(operands=(_scores(8), _cache(1)))
        self.assertEqual(qk.results[0].type, _scores(8).type)
        self.assertEqual(sv.results[0].type, _q_heads(8).type)

    def test_rejections(self) -> None:
        """Mismatched groups, contractions or placements."""

        # Both whole, but on a rank: 8 local query heads over 8 local KV heads
        # would be group 1 instead of 8.
        cases = {
            "rank group differs": (_q_heads(8), _cache(8), "group size must stay 8"),
            "query whole, cache split": (
                _q_heads(),
                _cache(1),
                "group size must stay 8",
            ),
            "head_dim differs": (
                _value((1, 1, Q_HEADS, 64), (*TOKEN_AXES, "q_head", "head_dim")),
                _cache(),
                "share batch and head_dim",
            ),
            "heads do not group": (
                _value((1, 1, 60, HEAD_DIM), (*TOKEN_AXES, "q_head", "head_dim")),
                _cache(),
                "cannot be grouped",
            ),
        }
        for name, (q, cache, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                GqaQKOp(operands=(q, cache), scale=1.0)

    def test_split_along_batch_is_unsupported(self) -> None:
        """Head groups fit, but only head splits have a placement rule.

        Splitting both operands along batch (2 requests over 2 ranks) leaves
        the head counts whole, so the group check passes; the placement rule
        then rejects it. When exactly one operand is split by heads, the
        group check above already fails, because its local group size
        differs from the whole one.
        """

        q_by_batch = _value(
            (1, 1, Q_HEADS, HEAD_DIM),
            (*TOKEN_AXES, "q_head", "head_dim"),
            Shard("batch"),
            (2, 1, Q_HEADS, HEAD_DIM),
        )
        cache_by_batch = _value(
            (1, CONTEXT, KV_HEADS, HEAD_DIM),
            CACHE_AXES,
            Shard("batch"),
            (2, CONTEXT, KV_HEADS, HEAD_DIM),
            role=TensorRole.KV_CACHE,
        )
        with self.assertRaisesRegex(ValueError, "unsupported placements"):
            GqaQKOp(operands=(q_by_batch, cache_by_batch), scale=1.0)

    def test_sv_rejects_wrong_contraction(self) -> None:
        """The probabilities must sum over the cache's context axis."""

        probs = _value(
            (1, 1, Q_HEADS, CONTEXT), (*TOKEN_AXES, "q_head", "seq_len_again")
        )
        with self.assertRaisesRegex(ValueError, "context"):
            GqaSVOp(operands=(probs, _cache()))


class AttentionOpTest(unittest.TestCase):
    """aloi.attention: the fused BHSD form the CENT pipeline lowers."""

    def _bhsd(self, *shape: int) -> Value:
        """Return a whole fp32 tensor of the given shape."""

        return _value(shape, dtype=DType.FP32)

    def test_shapes_with_and_without_mask(self) -> None:
        """Query [1, 4, 2, 3], key [1, 2, 5, 3], value [1, 2, 5, 7] give [1, 4, 2, 7]."""

        q, k, v = self._bhsd(1, 4, 2, 3), self._bhsd(1, 2, 5, 3), self._bhsd(1, 2, 5, 7)
        for mask in ((), (self._bhsd(2, 5),), (self._bhsd(1, 4, 2, 5),)):
            with self.subTest(mask=[m.type.shape for m in mask]):
                op = AttentionOp(operands=(q, k, v, *mask), causal=True, scale=0.5)
                self.assertEqual(op.results[0].type.shape, (1, 4, 2, 7))
        self.assertEqual(
            AttentionOp(operands=(q, k, v)).attributes(),
            {"causal": False, "query_position_offset": 0, "scale": None},
        )

    def test_rejections(self) -> None:
        """Bad attributes, shapes, groups, masks, or split tensors."""

        q, k, v = self._bhsd(1, 4, 2, 3), self._bhsd(1, 2, 5, 3), self._bhsd(1, 2, 5, 7)
        split_q = _value(
            (1, 2, 2, 3),
            ("batch", "q_head", "seq_len", "head_dim"),
            Shard("q_head"),
            (1, 4, 2, 3),
            dtype=DType.FP32,
        )
        cases: dict[str, tuple[tuple[Value, ...], dict[str, object], str]] = {
            "negative offset": ((q, k, v), {"query_position_offset": -1}, ">= 0"),
            "infinite scale": ((q, k, v), {"scale": float("inf")}, "finite"),
            "too few operands": ((q, k), {}, "expects"),
            "heads do not group": ((self._bhsd(1, 3, 2, 3), k, v), {}, "do not fit"),
            "key width differs": ((q, self._bhsd(1, 2, 5, 4), v), {}, "do not fit"),
            "bad mask": ((q, k, v, self._bhsd(5, 2)), {}, "mask must be"),
            "split query": ((split_q, k, v), {}, "only whole tensors"),
        }
        for name, (operands, attributes, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                AttentionOp(operands=operands, **attributes)  # type: ignore[arg-type]


class AttnTemperatureOpTest(unittest.TestCase):
    """aloi.attn_temperature: per-token query scale of Llama 4's NoPE layers."""

    def test_type_is_kept_for_split_and_partial_queries(self) -> None:
        """Scaling whole tokens is local, so heads and partial sums stay as they are.

        q is one rank's 5 of 40 query heads of Llama 4 Scout under tp=8.
        """
        positions = _value((1, 1), TOKEN_AXES, dtype=DType.INT64)
        split_q = _value(
            (1, 1, 5, HEAD_DIM),
            (*TOKEN_AXES, "q_head", "head_dim"),
            Shard("q_head"),
            (1, 1, 40, HEAD_DIM),
        )
        partial_q = _value(
            (1, 1, 5, HEAD_DIM), (*TOKEN_AXES, "q_head", "head_dim"), Partial()
        )
        for q in (split_q, partial_q):
            with self.subTest(placement=q.type.placement):
                op = AttnTemperatureOp(
                    operands=(q, positions), floor_scale=8192.0, attn_scale=0.1
                )
                self.assertEqual(op.results[0].type, q.type)
        self.assertEqual(op.attributes(), {"floor_scale": 8192.0, "attn_scale": 0.1})

    def test_rejections(self) -> None:
        """Bad constants, float or misshaped positions, or split positions."""
        q = _value((1, 1, 4, HEAD_DIM))
        positions = _value((1, 1), dtype=DType.INT64)
        split_positions = _value(
            (1, 1), ("batch", "seq_len"), Shard("batch"), (2, 1), dtype=DType.INT64
        )
        cases = {
            "zero floor_scale": ((q, positions), 0.0, 0.1, "floor_scale"),
            "infinite attn_scale": ((q, positions), 2.0, float("inf"), "attn_scale"),
            "float positions": ((q, _value((1, 1))), 2.0, 0.1, "integers"),
            "one position for two tokens": (
                (_value((1, 2, 4, HEAD_DIM)), positions),
                2.0,
                0.1,
                "token",
            ),
            "split positions": ((q, split_positions), 2.0, 0.1, "replicated"),
        }
        for name, (operands, floor_scale, attn_scale, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                AttnTemperatureOp(
                    operands=operands, floor_scale=floor_scale, attn_scale=attn_scale
                )


class TopKSigmoidOpTest(unittest.TestCase):
    """aloi.topk_sigmoid: dense routing weights over all experts."""

    def test_type_is_kept(self) -> None:
        """Scout's router logits [1, 1, 16] give routing weights [1, 1, 16]."""
        logits = _value((1, 1, 16), (*TOKEN_AXES, "expert"))
        op = TopKSigmoidOp(operands=(logits,), top_k=1)
        self.assertEqual(op.results[0].type, logits.type)

    def test_rejections(self) -> None:
        """top_k outside 1..experts, a partial sum, or split experts."""
        logits = _value((1, 16))
        split = _value((1, 2), ("seq_len", "expert"), Shard("expert"), (1, 16))
        partial = _value((1, 16), placement=Partial())
        cases = {
            "top_k 0": (logits, 0, "between 1 and 16"),
            "top_k above experts": (logits, 17, "between 1 and 16"),
            "partial logits": (partial, 1, "partial sum"),
            "split experts": (split, 1, "every expert"),
        }
        for name, (operand, top_k, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                TopKSigmoidOp(operands=(operand,), top_k=top_k)


class MoeExpertsOpTest(unittest.TestCase):
    """aloi.moe_experts: the weighted sum of SwiGLU experts and its TP rule."""

    # Scout's expert weights under tp=8: 16 experts, hidden 5120, and each
    # rank holding 8192 / 8 = 1024 of every expert's intermediate width.
    EXPERT_AXES = ("expert", "intermediate", "hidden")
    DOWN_AXES = ("expert", "hidden", "intermediate")

    def _operands(self, placement: Placement = REPLICATED) -> tuple[Value, ...]:
        """Return x, routing weights, and gate/up/down weights of Scout.

        Args:
            placement: Placement of the three expert weights; a shard splits
                the intermediate axis, so local widths are 1024 of 8192.

        Returns:
            The five operands.
        """
        width = 1024 if isinstance(placement, Shard) else 8192
        whole = {"gate": (16, 8192, 5120), "down": (16, 5120, 8192)}

        def weight(shape: tuple[int, ...], axes: tuple[str, ...], key: str) -> Value:
            """Return one expert weight with the requested placement."""
            sharded = isinstance(placement, Shard)
            return _value(
                shape,
                axes,
                placement,
                whole[key] if sharded else None,
                role=TensorRole.WEIGHT,
            )

        return (
            _value((1, 1, 5120), (*TOKEN_AXES, "hidden")),
            _value((1, 1, 16), (*TOKEN_AXES, "expert")),
            weight((16, width, 5120), self.EXPERT_AXES, "gate"),
            weight((16, width, 5120), self.EXPERT_AXES, "gate"),
            weight((16, 5120, width), self.DOWN_AXES, "down"),
        )

    def test_replicated_and_intermediate_split(self) -> None:
        """Whole experts give a whole result; a split intermediate a partial sum."""
        for placement, expected in (
            (REPLICATED, REPLICATED),
            (Shard("intermediate"), Partial()),
        ):
            with self.subTest(placement=placement):
                op = MoeExpertsOp(
                    operands=self._operands(placement),
                    apply_router_weight_on_input=True,
                )
                self.assertEqual(
                    op.results[0].type,
                    TensorType(
                        shape=(1, 1, 5120),
                        dtype=DType.BF16,
                        axes=(*TOKEN_AXES, "hidden"),
                        placement=expected,
                    ),
                )

    def test_rejections(self) -> None:
        """Mismatched shapes or axes, and an expert-axis split, are rejected."""
        x, weights, gate, up, down = self._operands()
        wrong_down = _value((16, 8192, 5120), self.EXPERT_AXES, role=TensorRole.WEIGHT)
        wrong_axes = _value((1, 1, 16), (*TOKEN_AXES, "router"))
        split_experts = tuple(
            _value(
                (2, *t.type.shape[1:]),
                t.type.axes,
                Shard("expert"),
                t.type.shape,
                role=TensorRole.WEIGHT,
            )
            for t in (gate, up, down)
        )
        cases = {
            "down not transposed": ((x, weights, gate, up, wrong_down), "do not fit"),
            "routing axis is not the expert axis": (
                (x, wrong_axes, gate, up, down),
                "axes",
            ),
            "split experts": ((x, weights, *split_experts), "intermediate axis"),
        }
        for name, (operands, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                MoeExpertsOp(operands=operands, apply_router_weight_on_input=True)


if __name__ == "__main__":
    unittest.main()
