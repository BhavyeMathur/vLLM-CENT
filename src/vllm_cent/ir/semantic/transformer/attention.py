"""Attention ops: the decomposed GQA steps and the fused form.

gqa_qk and gqa_sv use the token-first layout. The fused AttentionOp
uses batch/head/sequence/feature order, the form the CENT pipeline
lowers today.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from ...base import Operation, Value
from ...types import Placement, Replicate, Shard, TensorType
from ._common import _ATTENTION_RANK

__all__ = [
    "AttentionOp",
    "GqaQKOp",
    "GqaSVOp",
]


# Index of the heads dimension in [batch, tokens, heads, head_dim] tensors,
# in KV caches, and in [batch, seq_len, q_heads, context] scores.
_HEADS_DIM = 2


def _check_gqa_heads(name: str, query: TensorType, cache: TensorType) -> None:
    """Check that query heads split evenly into groups over the KV heads.

    Args:
        name: Op name for error messages.
        query: Tensor whose dimension 2 is query heads.
        cache: KV cache whose dimension 2 is KV heads.

    Raises:
        ValueError: If the local or whole head counts do not form whole
            groups, or a rank's query heads are not the groups of its KV
            heads.
    """

    q_heads, kv_heads = query.shape[_HEADS_DIM], cache.shape[_HEADS_DIM]
    whole_q_heads = query.full_shape[_HEADS_DIM]
    whole_kv_heads = cache.full_shape[_HEADS_DIM]
    if kv_heads == 0 or q_heads % kv_heads or whole_q_heads % whole_kv_heads:
        raise ValueError(
            f"{name}: {q_heads} query heads cannot be grouped over {kv_heads} KV heads"
        )
    # Query head h reads KV head h // group. With contiguous shards, a rank
    # holding query heads [r*q, (r+1)*q) needs exactly KV heads
    # [r*kv, (r+1)*kv), which holds when the local group size equals the
    # whole one.
    if q_heads // kv_heads != whole_q_heads // whole_kv_heads:
        raise ValueError(
            f"{name}: this rank holds {q_heads} query heads but {kv_heads} KV "
            f"heads; the group size must stay {whole_q_heads // whole_kv_heads}"
        )


def _gqa_placement(name: str, query: TensorType, cache: TensorType) -> Placement:
    """Derive the placement of a GQA product.

    Accepted: both replicated, or the query split along its heads and the
    cache split along its KV heads (each rank computes whole heads).

    Args:
        name: Op name for error messages.
        query: Queries or probabilities, heads on dimension 2.
        cache: K or V cache, KV heads on dimension 2.

    Returns:
        The result's placement.

    Raises:
        ValueError: For any other combination.
    """

    match query.placement, cache.placement:
        case Replicate(), Replicate():
            return Replicate()
        case Shard(q_axis), Shard() if (
            query.sharded_dim == _HEADS_DIM and cache.sharded_dim == _HEADS_DIM
        ):
            return Shard(q_axis)
    raise ValueError(
        f"{name}: unsupported placements query={query.placement}, "
        f"cache={cache.placement}"
    )


@dataclass(frozen=True, eq=False, slots=True)
class GqaQKOp(Operation):
    """Attention scores: every query head against every cached key.

    ``scores[b, t, h, c] = scale * dot(q[b, t, h, :], k[b, c, h // group, :])``

    GqaQKOp(operands=(q, k_cache), scale=1 / sqrt(head_dim))

    Attributes:
        scale: Factor applied to every dot product, ``head_dim ** -0.5``.
    """

    NAME = "aloi.gqa_qk"
    NUM_OPERANDS = 2

    scale: float

    @property
    def q(self) -> Value:
        return self.operands[0]

    @property
    def k_cache(self) -> Value:
        return self.operands[1]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Scores are ``[batch, seq_len, q_heads, context]``.

        Raises:
            ValueError: If the shapes do not fit, the batch or head_dim axes
                differ, the heads do not group, or the placements are
                unsupported.
        """

        q = self.q.type
        k = self.k_cache.type
        if self.scale <= 0:
            raise ValueError(f"{self.NAME}: scale must be positive")
        if q.rank != _ATTENTION_RANK or k.rank != _ATTENTION_RANK:
            raise ValueError(f"{self.NAME}: q and k_cache must be rank 4")
        if q.dtype != k.dtype:
            raise ValueError(f"{self.NAME}: q and k_cache dtypes differ")
        if q.shape[0] != k.shape[0] or q.shape[3] != k.shape[3]:
            raise ValueError(
                f"{self.NAME}: q {q.shape} and k_cache {k.shape} must share "
                "batch and head_dim"
            )

        axes: tuple[str, ...] = ()
        if q.axes and k.axes:
            # The dot product runs over head_dim; batch must line up.
            if q.axes[0] != k.axes[0] or q.axes[3] != k.axes[3]:
                raise ValueError(
                    f"{self.NAME}: q axes {q.axes} and k_cache axes {k.axes} "
                    "must share batch and head_dim"
                )
            axes = (q.axes[0], q.axes[1], q.axes[2], k.axes[1])

        _check_gqa_heads(self.NAME, q, k)
        placement = _gqa_placement(self.NAME, q, k)
        shape = (q.shape[0], q.shape[1], q.shape[2], k.shape[1])
        global_shape = None
        if isinstance(placement, Shard):
            global_shape = (shape[0], shape[1], q.full_shape[2], shape[3])
        return (
            TensorType(
                shape=shape,
                dtype=q.dtype,
                axes=axes,
                global_shape=global_shape,
                placement=placement,
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class GqaSVOp(Operation):
    """Attention output: probability-weighted sum of the cached values.

    ``out[b, t, h, :] = sum_c probs[b, t, h, c] * v[b, c, h // group, :]``

    GqaSVOp(operands=(probs, v_cache))
    """

    NAME = "aloi.gqa_sv"
    NUM_OPERANDS = 2

    @property
    def probs(self) -> Value:
        return self.operands[0]

    @property
    def v_cache(self) -> Value:
        return self.operands[1]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The output is ``[batch, seq_len, q_heads, head_dim]``.

        Raises:
            ValueError: If the shapes do not fit, the batch or context axes
                differ, the heads do not group, or the placements are
                unsupported.
        """

        probs = self.probs.type
        v = self.v_cache.type
        if probs.rank != _ATTENTION_RANK or v.rank != _ATTENTION_RANK:
            raise ValueError(f"{self.NAME}: probs and v_cache must be rank 4")
        if probs.dtype != v.dtype:
            raise ValueError(f"{self.NAME}: probs and v_cache dtypes differ")
        if probs.shape[0] != v.shape[0] or probs.shape[3] != v.shape[1]:
            raise ValueError(
                f"{self.NAME}: probs {probs.shape} and v_cache {v.shape} must "
                "share batch and context"
            )

        axes: tuple[str, ...] = ()
        if probs.axes and v.axes:
            # The weighted sum runs over the context; batch must line up.
            if probs.axes[0] != v.axes[0] or probs.axes[3] != v.axes[1]:
                raise ValueError(
                    f"{self.NAME}: probs axes {probs.axes} and v_cache axes "
                    f"{v.axes} must share batch and context"
                )
            axes = (probs.axes[0], probs.axes[1], probs.axes[2], v.axes[3])

        _check_gqa_heads(self.NAME, probs, v)
        placement = _gqa_placement(self.NAME, probs, v)
        shape = (probs.shape[0], probs.shape[1], probs.shape[2], v.shape[3])
        global_shape = None
        if isinstance(placement, Shard):
            global_shape = (shape[0], shape[1], probs.full_shape[2], shape[3])
        return (
            TensorType(
                shape=shape,
                dtype=probs.dtype,
                axes=axes,
                global_shape=global_shape,
                placement=placement,
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class AttentionOp(Operation):
    """Fused scaled dot-product attention in batch/head/sequence/feature order.

    ``q [B, Hq, L, D]``, ``k [B, Hkv, S, D]``, ``v [B, Hkv, S, Dv]`` and an
    optional additive ``mask`` (``[L, S]`` or ``[B, Hq, L, S]``) give
    ``[B, Hq, L, Dv]``. Query head h reads KV head ``h // (Hq // Hkv)``. With
    ``causal``, query row i sees keys up to ``query_position_offset + i``.

    This is the form the staged CENT pipeline lowers today. The decomposed
    gqa_qk -> softmax -> gqa_sv form stays the frontend's form; an adaptor
    fuses it into this op for that backend.

    AttentionOp(operands=(q, k, v), scale=0.125)

    Attributes:
        causal: Hide keys after each query's position.
        query_position_offset: Position of query row 0.
        scale: Score multiplier; None means ``1 / sqrt(D)``.
    """

    NAME = "aloi.attention"
    NUM_OPERANDS = (3, 4)

    causal: bool = False
    query_position_offset: int = 0
    scale: float | None = None

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result is ``[B, Hq, L, Dv]``.

        Raises:
            ValueError: For invalid attributes, shapes that do not fit, heads
                that do not group, a mask of another shape, mixed dtypes, or
                any tensor that is not whole on every rank.
        """

        if self.query_position_offset < 0:
            raise ValueError(f"{self.NAME}: query_position_offset must be >= 0")
        if self.scale is not None and not isfinite(self.scale):
            raise ValueError(f"{self.NAME}: scale must be finite")
        types = [operand.type for operand in self.operands]
        q, k, v = types[:3]
        if any(t.rank != _ATTENTION_RANK for t in (q, k, v)):
            raise ValueError(f"{self.NAME}: q, k and v must be rank 4")
        if any(t.dtype != q.dtype for t in types):
            raise ValueError(f"{self.NAME}: all operands need one dtype")
        batch, heads, queries, width = q.shape
        if not (
            k.shape[0] == v.shape[0] == batch
            and k.shape[1] == v.shape[1]
            and heads % k.shape[1] == 0
            and k.shape[3] == width
            and k.shape[2] == v.shape[2]
        ):
            raise ValueError(
                f"{self.NAME}: q {q.shape}, k {k.shape}, v {v.shape} do not fit"
            )
        if len(types) == 4 and types[3].shape not in (
            (queries, k.shape[2]),
            (batch, heads, queries, k.shape[2]),
        ):
            raise ValueError(f"{self.NAME}: mask must be [L, S] or [B, Hq, L, S]")
        # TODO(attention TP): Head-split q/k/v would follow the gqa ops' rule
        # (whole head groups per rank); not needed by the backend yet.
        if any(not isinstance(t.placement, Replicate) for t in types):
            raise ValueError(f"{self.NAME}: only whole tensors are supported")

        axes: tuple[str, ...] = ()
        if q.axes and v.axes:
            axes = (*q.axes[:3], v.axes[3])
        return (
            TensorType(
                shape=(batch, heads, queries, v.shape[3]), dtype=q.dtype, axes=axes
            ),
        )
