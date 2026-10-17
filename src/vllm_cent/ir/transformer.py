"""Semantic ops of decoder-only transformers: heads, RoPE, KV cache, GQA.

These ops describe one decode step of attention without a transpose or an
expand. All tensors keep the token-first layout:

- an activation of one head type is ``[batch, seq_len, heads, head_dim]``;
- a KV cache is ``[batch, context, kv_heads, head_dim]``;
- attention scores are ``[batch, seq_len, q_heads, context]``.

Grouped-query attention (GQA) is built into gqa_qk and gqa_sv: with
``group = q_heads // kv_heads``, query head ``h`` reads KV head
``h // group``. Multi-head attention is the special case ``group == 1``.

Every op checks shapes, then axes (once annotated), then placements (once
tensor parallelism is applied), like the ops in semantic.py.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import isfinite

from .base import Operation, Value
from .types import (
    DType,
    Partial,
    Placement,
    Replicate,
    Shard,
    TensorRole,
    TensorType,
    packed_axis,
    packed_axis_factors,
)

__all__ = [
    "AttentionOp",
    "GqaQKOp",
    "GqaSVOp",
    "KvUpdateOp",
    "MergeHeadsOp",
    "RopeOp",
    "SplitHeadsOp",
]

# Index of the heads dimension in [batch, tokens, heads, head_dim] tensors,
# in KV caches, and in [batch, seq_len, q_heads, context] scores.
_HEADS_DIM = 2
# Rank of every attention tensor above.
_ATTENTION_RANK = 4
# Integer dtypes accepted for token positions.
_POSITION_DTYPES = (DType.INT64, DType.INT32)


def _reshaped_placement(
    tensor_type: TensorType, whole_result_shape: tuple[int, ...]
) -> tuple[tuple[int, ...] | None, Placement]:
    """Return the global shape and placement of a head reshape's result.

    Splitting or merging heads moves no data between ranks, so the result
    keeps the operand's placement. Only a shard records its whole shape.

    Args:
        tensor_type: The operand type.
        whole_result_shape: Result shape of the whole, unsplit tensor.

    Returns:
        ``(global_shape, placement)``.
    """

    if isinstance(tensor_type.placement, Shard):
        return whole_result_shape, tensor_type.placement
    return None, tensor_type.placement


@dataclass(frozen=True, eq=False, slots=True)
class SplitHeadsOp(Operation):
    """Split the last dimension into ``[heads, head_dim]``.

    ``[batch, seq_len, q_head*head_dim]`` becomes
    ``[batch, seq_len, q_head, head_dim]``. The packed axis name says what the
    two new dimensions mean, so the result's axes follow from the operand's.

    SplitHeadsOp(operands=(x,), head_dim=128)

    Attributes:
        head_dim: Length of one head; the new last dimension.
    """

    NAME = "aloi.split_heads"
    NUM_OPERANDS = 1

    head_dim: int

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Reshape the last dimension; keep shard placements on the heads.

        A shard of q_head*head_dim holds whole heads, so after the split it
        is a shard of the new q_head dimension.

        Raises:
            ValueError: If x is a scalar or a partial sum, head_dim does not
                divide the last dimension, or the annotated last axis is not
                a packed pair such as q_head*head_dim.
        """

        x = self.x.type
        if x.rank < 1:
            raise ValueError(f"{self.NAME}: x needs a dimension to split")
        if self.head_dim < 1 or x.shape[-1] % self.head_dim != 0:
            raise ValueError(
                f"{self.NAME}: head_dim {self.head_dim} does not divide "
                f"x's last dimension {x.shape[-1]}"
            )
        if isinstance(x.placement, Partial):
            raise ValueError(f"{self.NAME}: x is a partial sum; all_reduce it first")

        axes: tuple[str, ...] = ()
        if x.axes:
            factors = packed_axis_factors(x.axes[-1])
            if len(factors) != 2:
                raise ValueError(
                    f"{self.NAME}: x's last axis {x.axes[-1]!r} must pack exactly "
                    "two axes, e.g. q_head*head_dim"
                )
            axes = (*x.axes[:-1], *factors)

        shape = (*x.shape[:-1], x.shape[-1] // self.head_dim, self.head_dim)
        whole = (*x.full_shape[:-1], x.full_shape[-1] // self.head_dim, self.head_dim)
        global_shape, placement = _reshaped_placement(x, whole)
        return (
            TensorType(
                shape=shape,
                dtype=x.dtype,
                axes=axes,
                global_shape=global_shape,
                placement=placement,
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class MergeHeadsOp(Operation):
    """Merge the last two dimensions ``[heads, head_dim]`` into one.

    The inverse of SplitHeadsOp: ``[batch, seq_len, q_head, head_dim]``
    becomes ``[batch, seq_len, q_head*head_dim]``.

    MergeHeadsOp(operands=(x,))
    """

    NAME = "aloi.merge_heads"
    NUM_OPERANDS = 1

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Flatten the last two dimensions, heads outermost.

        Raises:
            ValueError: If x has fewer than two dimensions, is a partial sum,
                or is split along head_dim. A head_dim shard would become an
                inner factor of the packed axis, which cannot be split
                contiguously.
        """

        x = self.x.type
        if x.rank < 2:
            raise ValueError(f"{self.NAME}: x needs [heads, head_dim] dimensions")
        if isinstance(x.placement, Partial):
            raise ValueError(f"{self.NAME}: x is a partial sum; all_reduce it first")
        if x.sharded_dim == x.rank - 1:
            raise ValueError(
                f"{self.NAME}: x is split along its last dimension, which would "
                "become the inner factor of the merged axis"
            )

        axes: tuple[str, ...] = ()
        if x.axes:
            axes = (*x.axes[:-2], packed_axis(x.axes[-2], x.axes[-1]))

        shape = (*x.shape[:-2], x.shape[-2] * x.shape[-1])
        whole = (*x.full_shape[:-2], x.full_shape[-2] * x.full_shape[-1])
        global_shape, placement = _reshaped_placement(x, whole)
        return (
            TensorType(
                shape=shape,
                dtype=x.dtype,
                axes=axes,
                global_shape=global_shape,
                placement=placement,
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class RopeOp(Operation):
    """Rotary position embedding of one head tensor, via a cos/sin table.

    ``cos_sin_cache[p]`` holds ``rotary_dim // 2`` cosines then as many sines
    for position p. Each head's first ``rotary_dim`` elements are rotated by
    the angles of its token's position; the rest pass through. The op is the
    same for every model: the RoPE variant (theta, llama3 or YaRN scaling)
    only changes the table's values, never this op.

    RopeOp(operands=(x, positions, cos_sin_cache), is_neox_style=True)

    Attributes:
        is_neox_style: True pairs element i with element i + rotary_dim/2
            (Hugging Face / vLLM ``rotate_half``); False pairs elements 2i
            and 2i+1 (Meta's original Llama code, GPT-J).
    """

    NAME = "aloi.rope"
    NUM_OPERANDS = 3

    is_neox_style: bool = True

    @property
    def x(self) -> Value:
        return self.operands[0]

    @property
    def positions(self) -> Value:
        return self.operands[1]

    @property
    def cos_sin_cache(self) -> Value:
        return self.operands[2]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result has x's shape, dtype, axes and placement.

        Raises:
            ValueError: If x is not ``[*, heads, head_dim]``, the table is not
                ``[positions, rotary_dim]`` with an even rotary_dim of at most
                head_dim, positions do not give one integer per token of x,
                the axes disagree, or the placements cannot be applied
                locally.
        """

        x = self.x.type
        positions = self.positions.type
        table = self.cos_sin_cache.type
        if x.rank < 2:
            raise ValueError(
                f"{self.NAME}: x must be [*, heads, head_dim], got {x.shape}"
            )
        # Compare with the whole head_dim: a split head_dim is a placement
        # problem, reported below with its own message.
        head_dim = x.full_shape[-1]
        if table.rank != 2 or table.shape[1] % 2 or table.shape[1] > head_dim:
            raise ValueError(
                f"{self.NAME}: cos_sin_cache {table.shape} must be [positions, "
                f"rotary_dim] with an even rotary_dim <= head_dim {head_dim}"
            )
        if positions.dtype not in _POSITION_DTYPES:
            raise ValueError(
                f"{self.NAME}: positions must be integers, got {positions.dtype}"
            )
        # One position per token: x's dimensions before [heads, head_dim].
        if positions.shape != x.shape[:-2]:
            raise ValueError(
                f"{self.NAME}: positions shape {positions.shape} must equal "
                f"x's token dimensions {x.shape[:-2]}"
            )
        if x.axes and positions.axes and positions.axes != x.axes[:-2]:
            raise ValueError(
                f"{self.NAME}: positions axes {positions.axes} must equal "
                f"x's token axes {x.axes[:-2]}"
            )

        # A rotation pairs elements within one head, so a rank needs whole
        # head vectors; every rank needs every position and the whole table.
        if not isinstance(positions.placement, Replicate):
            raise ValueError(f"{self.NAME}: positions must be replicated")
        if not isinstance(table.placement, Replicate):
            raise ValueError(f"{self.NAME}: cos_sin_cache must be replicated")
        if isinstance(x.placement, Partial):
            raise ValueError(f"{self.NAME}: x is a partial sum; all_reduce it first")
        if x.sharded_dim == x.rank - 1:
            raise ValueError(f"{self.NAME}: x is split along head_dim")
        return (replace(x, role=TensorRole.ACTIVATION),)


@dataclass(frozen=True, eq=False, slots=True)
class KvUpdateOp(Operation):
    """Write this step's keys or values into a cache; return the new cache.

    Functional: the input cache is not modified. ``new[b, t]`` lands at
    ``cache[b, positions[b, t]]``.

    KvUpdateOp(operands=(cache, new, positions))
    """

    NAME = "aloi.kv_update"
    NUM_OPERANDS = 3

    @property
    def cache(self) -> Value:
        return self.operands[0]

    @property
    def new(self) -> Value:
        return self.operands[1]

    @property
    def positions(self) -> Value:
        return self.operands[2]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result has the cache's type, including its KV_CACHE role.

        Raises:
            ValueError: If the cache is not ``[batch, context, kv_heads,
                head_dim]``, new does not match it apart from its token
                count, positions do not give one integer per new token, the
                axes disagree, or the placements differ.
        """

        cache = self.cache.type
        new = self.new.type
        positions = self.positions.type
        if cache.rank != _ATTENTION_RANK or new.rank != _ATTENTION_RANK:
            raise ValueError(
                f"{self.NAME}: cache and new must be [batch, tokens, heads, "
                f"head_dim], got {cache.shape} and {new.shape}"
            )
        if cache.dtype != new.dtype:
            raise ValueError(f"{self.NAME}: cache and new dtypes differ")
        # Same batch, heads and head_dim; only the token count differs.
        if (cache.shape[0], *cache.shape[2:]) != (new.shape[0], *new.shape[2:]):
            raise ValueError(
                f"{self.NAME}: new {new.shape} does not fit cache {cache.shape}"
            )
        if positions.dtype not in _POSITION_DTYPES or positions.shape != new.shape[:2]:
            raise ValueError(
                f"{self.NAME}: positions must be integers of shape {new.shape[:2]}"
            )

        if cache.axes and new.axes:
            if (cache.axes[0], *cache.axes[2:]) != (new.axes[0], *new.axes[2:]):
                raise ValueError(
                    f"{self.NAME}: new axes {new.axes} do not match cache axes "
                    f"{cache.axes}"
                )
            if positions.axes and positions.axes != new.axes[:2]:
                raise ValueError(
                    f"{self.NAME}: positions axes {positions.axes} must equal "
                    f"{new.axes[:2]}"
                )

        # Each rank writes its own heads of the new token into its own heads
        # of the cache. Splitting the context dimension is rejected: the
        # position being written could belong to another rank.
        if not isinstance(positions.placement, Replicate):
            raise ValueError(f"{self.NAME}: positions must be replicated")
        if cache.placement != new.placement or isinstance(cache.placement, Partial):
            raise ValueError(
                f"{self.NAME}: unsupported placements cache={cache.placement}, "
                f"new={new.placement}"
            )
        if cache.sharded_dim == 1:
            raise ValueError(f"{self.NAME}: the cache's context dimension is split")
        return (cache,)


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
