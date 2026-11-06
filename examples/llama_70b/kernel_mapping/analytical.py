"""CENT's analytical latency model for the work its traces leave out.

CENT simulates the AiM commands and charges the rest analytically: the PNM
work of RMSNorm, softmax, RoPE and the reduction of SV partial sums
(``../ALOI/cent_simulation/run_sim.py:754-865``, ``calculate_acc_latency``),
and the CXL collectives between tensor-parallel devices
(``cxl_latency.py``). This module ports those formulas for one block of
KV-head tensor parallelism, so the layer trace can state them in comments.

Constants (``../ALOI/aim_simulator/scripts/cellar_power_calculator.py``,
``cent_simulation/cent_power_calculator.py``): the PNM units run at 2 GHz;
a Shared Buffer read or write takes 1 cycle; the accelerators' SRAM serves
32 channels in parallel; an exponent lane takes 11 cycles; RMSNorm's RISC-V
step 26 cycles; RoPE 3/8 cycle per element and softmax 17.25 cycles per
head on the RISC-V cores.
"""

from __future__ import annotations

from math import ceil

__all__ = [
    "PNM_CYCLES_PER_MS",
    "cxl_all_reduce_ms",
    "pipeline_handoff_ms",
    "rms_norm_cycles",
    "rope_cycles",
    "softmax_cycles",
    "sv_reduction_cycles",
]

PNM_CYCLES_PER_MS = 2.0e9 / 1000  # FREQ / KILO

_SB_RD = 1.0
_SB_WR = 1.0
_SRAM_IO_PARALLEL = 32.0
_EXP_LANE = 11.0
_RV_RMSNORM = 26.0
_RV_COUNT = 8
_RV_ROPE_PER_ELEMENT = 3.0 / _RV_COUNT
_RV_SOFTMAX_PER_HEAD = 16.0 * _SB_WR + 2.0 / _RV_COUNT + 1.0 * _SB_RD


def _accel(channels: int) -> dict[str, float]:
    """``ACCEL_CYCLE`` for a device of ``channels`` channels."""

    share = channels / _SRAM_IO_PARALLEL
    return {
        "EXP": share * _SB_RD + _EXP_LANE + _SB_WR,
        "VEC_ADD": share * 2.0 * _SB_RD + 1.0 + _SB_WR,
        "VEC_MUL": share * 2.0 * _SB_RD + 1.0 + _SB_WR,
    }


def rms_norm_cycles(hidden: int, channels: int) -> float:
    """PNM cycles of one RMSNorm: the sum of squares' reduction and the rsqrt.

    CENT charges two norms per block together; this is one of them.
    """

    return (
        hidden / 16.0 / 16.0 / channels * _accel(channels)["VEC_ADD"]
        + _SB_RD
        + _SB_WR
        + 1.0
        + _RV_RMSNORM
    )


def softmax_cycles(seqlen: int, local_heads: int, channels: int) -> float:
    """PNM cycles of softmax over ``seqlen`` scores of each local head."""

    accel = _accel(channels)
    per_score = accel["EXP"] + accel["VEC_ADD"] + 2.0 * accel["VEC_MUL"]
    return (
        seqlen * local_heads / 16.0 / channels * per_score
        + local_heads * _SB_RD
        + local_heads * _RV_SOFTMAX_PER_HEAD
    )


def rope_cycles(elements: int) -> float:
    """RISC-V cycles of RoPE over ``elements`` values (Q and K counted apart)."""

    return elements * _RV_ROPE_PER_ELEMENT


def sv_reduction_cycles(
    seqlen: int,
    local_heads: int,
    head_dim: int,
    shards: int,
    capacity: int,
    channels: int,
) -> float:
    """PNM cycles adding the SV partial sums of the active context shards.

    ``tp_mapping.KVHeadTPLayout.v_reduction_adds``: each active shard but
    one adds ``local_heads * head_dim`` values.
    """

    active = min(shards, ceil(seqlen / capacity))
    adds = local_heads * head_dim * (active - 1)
    return ceil(adds / 16.0 / channels) * _accel(channels)["VEC_ADD"]


# CXL (cxl_latency.py). The one-hop round trip is computed before MULTICAST
# doubles the PCIe latency, so it stays 25 * 4 + 30 * 2 + 20 = 180 ns, while
# MULTICAST halves the bandwidth per lane to 4.
_ONE_HOP_NS = 25 * 4 + (5 + 20 + 5) * 2 + 20
_BANDWIDTH_PER_LANE = 8 * 0.5


def _transfer_ms(flits: int, lanes: int) -> float:
    bandwidth = lanes * _BANDWIDTH_PER_LANE
    return (_ONE_HOP_NS + flits * 256 / bandwidth * 1e9 / 1024 / 1024 / 1024) / 1e6


def cxl_all_reduce_ms(hidden: int, lanes: int, devices: int) -> float:
    """One all-reduce of a ``hidden`` vector across ``devices`` TP devices.

    ``kv_head_tp_latency`` charges two per block (after o_proj and
    down_proj): a gather of ``devices - 1`` partial vectors, then a
    broadcast of the sum. This is one of them.
    """

    if devices == 1:
        return 0.0
    flits = ceil(hidden * 2 / 64 / 3)
    return _transfer_ms(flits * (devices - 1), lanes) + _transfer_ms(flits, lanes)


def pipeline_handoff_ms(hidden: int, lanes: int) -> float:
    """Passing a ``hidden`` vector to the next pipeline stage (``vector_latency``)."""

    return _transfer_ms(ceil(hidden * 2 / 64 / 3), lanes)
