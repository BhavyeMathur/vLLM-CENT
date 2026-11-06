"""CENT's attention kernels for KV-head tensor parallelism, in the Tile DSL.

They follow CENT's KV-head TP layout (``../ALOI/cent_simulation/
tp_mapping.py``, ``KVHeadTPLayout``) and trace generator
(``TransformerBlock.py:1884-2068``) for one KV head per rank:

- K cache: position ``s`` sits in flat bank ``s % banks`` (all channels'
  banks), column block ``(s // banks) % 8`` of row ``s // (8 * banks)``;
  each key's 128 values fill one column block (``k_location``).
- V cache: the context is split into ``shards`` shards of
  ``max_seq_len / shards`` positions; shard ``j`` lives on channels
  ``j * 8`` to ``j * 8 + 7``, value dimension ``d`` in channel ``d // 16``
  of the shard and bank ``d % 16``, the positions along the row
  (``v_location``).
- QK: per query head, one ``WR_GB`` of the query; per 512-position
  segment, ``WR_BIAS`` / ``MAC_ABK`` / ``RD_MAC`` on the channels that hold
  positions of the segment.
- SV: per query head, ``WR_BIAS`` on the active shards' channels, one
  ``WR_GB`` of each shard's probabilities, one ``MAC_ABK`` per operation
  size, one ``RD_MAC``; the shards' partial sums are added in PNM
  (analytical).
"""

from __future__ import annotations

from math import ceil

from vllm_cent.device import MemorySpace
from vllm_cent.ir import DType
from vllm_cent.tile import (
    Bank,
    Channel,
    Column,
    ColumnBlock,
    Lane,
    Row,
    Serial,
    Slot,
    split,
)
from vllm_cent.tile.dsl import TileKernel, TileValue

from .analytical import sv_reduction_cycles

__all__ = ["V_SHARDS", "cache_write", "k_cache_layout", "qk", "sv", "v_cache_layout"]

# KV-head TP splits one KV head's context over this many channel groups
# (``v_context_shards = banks_per_kv_head / head_dim`` = 512 / 128).
V_SHARDS = 4


def _k_cache(
    k: TileKernel, name: str, context: int, head_dim: int, max_context: int, row: int
) -> TileValue:
    """The K cache of one KV head: ``(context, head_dim)``."""

    arch = k.arch
    banks = arch.channels * arch.banks_per_channel
    row_values = arch.row_size * 8 // DType.BF16.bits
    blocks = row_values // head_dim
    waves = ceil(max_context / (banks * blocks))
    cache = k.tensor(name, (context, head_dim), DType.BF16, MemorySpace.PIM_DRAM)
    k.bind_layout(
        cache,
        split(
            s=(Row("wave", waves), ColumnBlock("segment", blocks), Channel(), Bank()),
            d=(Column(head_dim),),
        ),
        base=row,
    )
    return cache


def _v_cache(
    k: TileKernel, name: str, context: int, head_dim: int, max_context: int, row: int
) -> TileValue:
    """The V cache of one KV head: ``(context, head_dim)``, context in shards."""

    arch = k.arch
    capacity = ceil(max_context / V_SHARDS)
    row_values = arch.row_size * 8 // DType.BF16.bits
    rows = ceil(capacity / row_values)
    cache = k.tensor(name, (context, head_dim), DType.BF16, MemorySpace.PIM_DRAM)
    along = (
        (Row("v_row", rows), Column(row_values)) if rows > 1 else (Column(capacity),)
    )
    k.bind_layout(
        cache,
        split(
            s=(Channel(V_SHARDS), *along),
            d=(Channel(), Bank()),
        ),
        base=row,
    )
    return cache


def k_cache_layout(
    k: TileKernel, name: str, context: int, head_dim: int, max_context: int, row: int
) -> TileValue:
    """Declare and lay out a K cache in kernel ``k``."""
    return _k_cache(k, name, context, head_dim, max_context, row)


def v_cache_layout(
    k: TileKernel, name: str, context: int, head_dim: int, max_context: int, row: int
) -> TileValue:
    """Declare and lay out a V cache in kernel ``k``."""
    return _v_cache(k, name, context, head_dim, max_context, row)


def cache_write(
    k: TileKernel,
    *,
    cache: TileValue,
    new_name: str,
    head_dim: int,
    position: int,
) -> None:
    """Record the write of one position's key or value into its cache.

    Ordinary DRAM writes, one per burst: a key's 128 values fill one column
    block (8 bursts in one bank); a value's 128 dimensions sit in 128 banks
    (one burst each). The kernel returns the cache's new version.
    """

    new = k.tensor(new_name, (head_dim,), DType.BF16, MemorySpace.PNM_SB)
    k.bind_layout(new, split(d=(Slot(), Lane())))
    k.returns(k.wr_mem(new, into=cache, start=(position, 0)))


def qk(
    k: TileKernel,
    *,
    names: tuple[str, str, str],
    heads: int,
    head_dim: int,
    context: int,
    max_context: int,
    k_row: int,
) -> None:
    """Record CENT's scores ``q_h . K`` of one KV head's query heads.

    Args:
        k: The kernel to record into.
        names: Names of the queries, the K cache and the scores.
        heads: Query heads that share the KV head.
        head_dim: Values per head.
        context: Positions in the cache.
        max_context: The window the cache is laid out for.
        k_row: The K cache's first row.
    """

    arch = k.arch
    banks_per_channel = arch.banks_per_channel
    segment = arch.channels * banks_per_channel
    burst_values = arch.burst_size * 8 // DType.BF16.bits
    q = k.tensor(names[0], (heads, head_dim), DType.BF16, MemorySpace.PNM_SB)
    k.bind_layout(
        q, split(h=(Serial("h", heads),), d=(Slot(head_dim // burst_values), Lane()))
    )
    cache = _k_cache(k, names[1], context, head_dim, max_context, k_row)
    segments = ceil(context / segment)
    scores = k.tensor(names[2], (heads, context), DType.BF16, MemorySpace.PNM_SB)
    k.bind_layout(
        scores,
        split(
            h=(Serial("h", heads),),
            s=(Serial("segment", segments), Slot(of=Channel()), Lane(of=Bank())),
        ),
    )
    row_values = arch.row_size * 8 // DType.BF16.bits
    blocks = row_values // head_dim
    chans = k.all_channels()

    with k.serial("h", heads, carry=(scores,)) as loop:
        h, (s,) = loop.index, loop.carried
        gb = k.wr_gb(q.tile(h=h), channels=chans)
        for g in range(segments):
            filled = min(segment, context - g * segment)
            valid = tuple(range(ceil(filled / banks_per_channel)))
            acc = k.wr_bias_zero(channels=valid)
            acc = k.mac_abk(
                gb,
                cache.tile(wave=g // blocks, segment=g % blocks),
                acc,
                channels=valid,
            )
            s = k.rd_mac(acc, into=s.tile(h=h, segment=g), channels=valid)
        loop.yield_(s)
    k.returns(*loop.results)


def sv(
    k: TileKernel,
    *,
    names: tuple[str, str, str],
    heads: int,
    head_dim: int,
    context: int,
    max_context: int,
    v_row: int,
) -> None:
    """Record CENT's ``p_h . V`` of one KV head's query heads, by context shard.

    Args:
        k: The kernel to record into.
        names: Names of the probabilities, the V cache and the output.
        heads: Query heads that share the KV head.
        head_dim: Values per head.
        context: Positions in the cache.
        max_context: The window the cache is laid out for.
        v_row: The V cache's first row.
    """

    arch = k.arch
    burst_values = arch.burst_size * 8 // DType.BF16.bits
    capacity = ceil(max_context / V_SHARDS)
    per_shard = arch.channels // V_SHARDS
    active = ceil(context / capacity)
    p = k.tensor(names[0], (heads, context), DType.BF16, MemorySpace.PNM_SB)
    k.bind_layout(
        p,
        split(
            h=(Serial("h", heads),),
            s=(Serial("shard", active), Slot(capacity // burst_values), Lane()),
        ),
    )
    cache = _v_cache(k, names[1], context, head_dim, max_context, v_row)
    partial = k.tensor(
        "partial", (heads, V_SHARDS, head_dim), DType.BF16, MemorySpace.PNM_SB
    )
    k.bind_layout(
        partial,
        split(
            h=(Serial("h", heads),),
            j=(Slot(of=Channel(V_SHARDS)),),
            d=(Slot(of=Channel()), Lane(of=Bank())),
        ),
    )
    out = k.tensor(names[2], (heads, head_dim), DType.BF16, MemorySpace.PNM_SB)
    k.bind_layout(out, split(h=(Serial("h", heads),), d=(Slot(), Lane())))

    used = tuple(range(active * per_shard))
    # One MAC per operation size: the last shard may hold fewer positions.
    opsizes = cache_layout_opsizes(k, cache)
    sizes: dict[int, list[int]] = {}
    for channel in used:
        sizes.setdefault(opsizes[channel], []).append(channel)

    with k.serial("h", heads, carry=(partial,)) as loop:
        h, (part,) = loop.index, loop.carried
        acc = k.wr_bias_zero(channels=used)
        gbs = [
            k.wr_gb(
                p.tile(h=h, shard=j),
                channels=tuple(range(j * per_shard, (j + 1) * per_shard)),
            )
            for j in range(active)
        ]
        gb = k.gb_concat(*gbs) if len(gbs) > 1 else gbs[0]
        for channels in sorted(sizes.values()):
            acc = k.mac_abk(gb, cache.tile(), acc, channels=tuple(channels))
        part = k.rd_mac(acc, into=part.tile(h=h), channels=used)
        loop.yield_(part)
    (part,) = loop.results
    out = k.pnm(
        "SV partial-sum reduction",
        part,
        into=out,
        cycles=sv_reduction_cycles(
            context, heads, head_dim, V_SHARDS, capacity, arch.channels
        ),
    )
    k.returns(out)


def cache_layout_opsizes(k: TileKernel, cache: TileValue) -> dict[int, int]:
    """The bursts each channel holds of a cache laid out without indices."""

    return k.layout_of(cache).channel_opsizes({})
