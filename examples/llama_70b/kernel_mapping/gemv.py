"""CENT's GEMV mappings, written in the Tile DSL.

Both kernels follow CENT's trace generator (``../ALOI/cent_simulation``),
so their traces match its records but for the GPR field (CENT writes 0):

- ``reuse_gb_gemv``: ``TransformerBlock.Vector_Matrix_Mul_weight_pim_only_trace``
  in its "reuse-GB" order. Output ``n`` of a weight ``(N, K)`` sits in bank
  ``n % banks`` of channel ``(n // banks) % channels``; each bank holds
  ``cols = N / (channels * banks)`` outputs, and output column ``c`` of a
  bank reads DRAM row ``base + c * rows + k_row`` (``rows = ceil(K / 1024)``
  for bf16). For each K row: one ``WR_GB``; then, per group of up to
  ``group`` outputs, the group's ``WR_BIAS``, then its ``MAC_ABK``, then its
  ``RD_MAC``. The partial sums cross K rows through the output tensor, so it
  must hold zeros on entry (TODO(runtime)).
- ``kv_proj``: ``Vector_Matrix_Mul_kv_projection_pim_only_trace``: the K and
  V projections on disjoint channel groups, their records interleaved so
  the two streams overlap.

``group`` is CENT's ``reuse_size`` (32) capped by the device's MAC registers
per bank: on AiM (one register) every output is its own group, which is
CENT's "no-reuse" order.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import ceil

from vllm_cent.device import CentArchitecture, MemorySpace
from vllm_cent.ir import DType
from vllm_cent.tile import Bank, Channel, Column, Lane, Row, Serial, Slot, split
from vllm_cent.tile.dsl import TileKernel, TileValue

__all__ = ["CENT_REUSE_SIZE", "GemvTensors", "kv_proj", "reuse_gb_gemv"]

# CENT's GEMV group size: run_sim.py passes --reuse-size 32 to every trace.
CENT_REUSE_SIZE = 32


@dataclass(frozen=True)
class GemvTensors:
    """One GEMV's tensors in a kernel: input, weight and output."""

    x: TileValue
    w: TileValue
    y: TileValue


def _shape(arch: CentArchitecture, n: int, k: int, channels: range) -> tuple[int, int]:
    """Return ``(cols, rows)``: outputs per bank and K rows per output.

    Raises:
        ValueError: If ``n`` is not a whole number of outputs per bank.
    """

    banks = len(channels) * arch.banks_per_channel
    if n % banks:
        raise ValueError(f"N = {n} is not a multiple of the {banks} banks used")
    row_values = arch.row_size * 8 // DType.BF16.bits
    return n // banks, ceil(k / row_values)


def _declare(
    k: TileKernel,
    names: tuple[str, str, str],
    n: int,
    kdim: int,
    channels: range,
    w_row: int | None,
) -> GemvTensors:
    """Declare and lay out one GEMV's x, w and y."""

    arch = k.arch
    cols, rows = _shape(arch, n, kdim, channels)
    row_values = arch.row_size * 8 // DType.BF16.bits
    burst_values = arch.burst_size * 8 // DType.BF16.bits
    on = None if len(channels) == arch.channels else channels
    x = k.tensor(names[0], (kdim,), DType.BF16, MemorySpace.PNM_SB)
    w = k.tensor(names[1], (n, kdim), DType.BF16, MemorySpace.PIM_DRAM)
    y = k.tensor(names[2], (n,), DType.BF16, MemorySpace.PNM_SB)
    # x: one K row's values per Global Buffer write; the last may be partial.
    k.bind_layout(
        x,
        split(k=(Serial("k_row", rows), Slot(row_values // burst_values), Lane())),
    )
    col = (Serial("col", cols),) if cols > 1 else ()
    k.bind_layout(
        w,
        split(
            channels=on,
            n=(*col, Channel(), Bank()),
            k=(Row("k_row", rows), Column(row_values)),
        ),
        base=w_row,
    )
    k.bind_layout(
        y,
        split(channels=on, n=(*col, Slot(of=Channel()), Lane(of=Bank()))),
    )
    return GemvTensors(x, w, y)


def _col(cols: int, c: int) -> dict[str, int]:
    """The tile index of output column ``c``: none when a bank holds one."""
    return {"col": c} if cols > 1 else {}


def reuse_gb_gemv(
    k: TileKernel,
    *,
    n: int,
    kdim: int,
    names: tuple[str, str, str] = ("x", "w", "y"),
    w_row: int | None = None,
    group: int = CENT_REUSE_SIZE,
    activation: str | None = None,
) -> GemvTensors:
    """Record ``y = w @ x`` on every channel, in CENT's reuse-GB order.

    Args:
        k: The kernel to record into.
        n: Outputs (weight rows).
        kdim: Inputs (weight columns).
        names: Names of x, w and y, as the Device IR op names its operands.
        w_row: The weight's first DRAM row, if the model's mapping fixes it.
        group: Outputs per group; capped by the MAC registers per bank.
        activation: The name of a second output, the activation (sigmoid)
            of ``y``: after the last K row's MACs, each output's ``AF`` and
            ``RD_AF`` come before its ``RD_MAC``, in groups of half the size
            (``Vector_Matrix_Mul_weight_af_pim_only_trace``).

    Returns:
        The tensors; the kernel returns ``y`` (and the activation).
    """

    arch = k.arch
    chans = k.all_channels()
    t = _declare(k, names, n, kdim, range(arch.channels), w_row)
    cols, rows = _shape(arch, n, kdim, range(arch.channels))
    size = group // 2 if activation is not None else group
    groups = _groups(cols, min(size, arch.accumulators_per_bank))
    act = None
    if activation is not None:
        act = k.tensor(activation, (n,), DType.BF16, MemorySpace.PNM_SB)
        col = (Serial("col", cols),) if cols > 1 else ()
        k.bind_layout(act, split(n=(*col, Slot(of=Channel()), Lane(of=Bank()))))

    def k_row(
        r: TileValue | int, y: TileValue, act: TileValue | None
    ) -> tuple[TileValue, TileValue | None]:
        """One K row: its Global Buffer write, then every group of outputs."""

        gb = k.wr_gb(t.x.tile(k_row=r), channels=chans)
        for members in groups:
            accs = [
                k.wr_bias(y.tile(**_col(cols, c)), stride=1, channels=chans)
                for c in members
            ]
            accs = [
                k.mac_abk(t.w.tile(**_col(cols, c), k_row=r), gb, acc, channels=chans)
                for c, acc in zip(members, accs, strict=True)
            ]
            if act is not None:
                for c, acc in zip(members, accs, strict=True):
                    act = k.rd_af(acc, into=act.tile(**_col(cols, c)), channels=chans)
            for c, acc in zip(members, accs, strict=True):
                y = k.rd_mac(acc, into=y.tile(**_col(cols, c)), channels=chans)
        return y, act

    # With an activation, the last K row differs: it is recorded after the
    # loop over the others, with a constant index.
    looped = rows - 1 if act is not None else rows
    y = t.y
    if looped:
        with k.serial("k_row", looped, carry=(y,)) as loop:
            r, (y_now,) = loop.index, loop.carried
            loop.yield_(k_row(r, y_now, None)[0])
        (y,) = loop.results
    if act is not None:
        y, act = k_row(rows - 1, y, act)
        assert act is not None
        k.returns(y, act)
    else:
        k.returns(y)
    return t


def kv_proj(
    k: TileKernel,
    *,
    n: int,
    kdim: int,
    k_names: tuple[str, str, str],
    v_names: tuple[str, str, str],
    k_row: int | None = None,
    v_row: int | None = None,
    group: int = CENT_REUSE_SIZE,
) -> tuple[GemvTensors, GemvTensors]:
    """Record the K and V projections on disjoint channel groups, interleaved.

    CENT puts K on the first channels of the lower half and V on the first
    channels of the upper half, each on as few whole channels as hold its
    ``n`` outputs (``kv_projection_channels_per_operand``, at most half). The
    two GEMVs share x: each half gets its own Global Buffer write.

    Returns:
        The K and V tensors; the kernel returns ``(k_y, v_y)``.
    """

    arch = k.arch
    half = arch.channels // 2
    active = min(half, ceil(n / arch.banks_per_channel))
    k_ch, v_ch = range(active), range(half, half + active)
    kt = _declare(k, k_names, n, kdim, k_ch, k_row)
    # V reads the same x: declare only its weight and output.
    cols, rows = _shape(arch, n, kdim, k_ch)
    vt = _declare_weight_and_output(k, v_names, n, kdim, v_ch, v_row, kt.x)
    groups = _groups(cols, min(group, arch.accumulators_per_bank))
    k_mask, v_mask = tuple(k_ch), tuple(v_ch)

    with k.serial("k_row", rows, carry=(kt.y, vt.y)) as loop:
        r, (ky, vy) = loop.index, loop.carried
        k_gb = k.wr_gb(kt.x.tile(k_row=r), channels=k_mask)
        v_gb = k.wr_gb(kt.x.tile(k_row=r), channels=v_mask)
        for members in groups:
            accs = []
            for c in members:
                accs.append(
                    k.wr_bias(ky.tile(**_col(cols, c)), stride=1, channels=k_mask)
                )
                accs.append(
                    k.wr_bias(vy.tile(**_col(cols, c)), stride=1, channels=v_mask)
                )
            macs = []
            for i, c in enumerate(members):
                macs.append(
                    k.mac_abk(
                        kt.w.tile(**_col(cols, c), k_row=r),
                        k_gb,
                        accs[2 * i],
                        channels=k_mask,
                    )
                )
                macs.append(
                    k.mac_abk(
                        vt.w.tile(**_col(cols, c), k_row=r),
                        v_gb,
                        accs[2 * i + 1],
                        channels=v_mask,
                    )
                )
            for i, c in enumerate(members):
                ky = k.rd_mac(
                    macs[2 * i], into=ky.tile(**_col(cols, c)), channels=k_mask
                )
                vy = k.rd_mac(
                    macs[2 * i + 1], into=vy.tile(**_col(cols, c)), channels=v_mask
                )
        loop.yield_(ky, vy)
    k.returns(*loop.results)
    return kt, vt


def _declare_weight_and_output(
    k: TileKernel,
    names: tuple[str, str, str],
    n: int,
    kdim: int,
    channels: range,
    w_row: int | None,
    x: TileValue,
) -> GemvTensors:
    """Declare w and y of a GEMV whose x is already declared."""

    arch = k.arch
    cols, rows = _shape(arch, n, kdim, channels)
    row_values = arch.row_size * 8 // DType.BF16.bits
    w = k.tensor(names[1], (n, kdim), DType.BF16, MemorySpace.PIM_DRAM)
    y = k.tensor(names[2], (n,), DType.BF16, MemorySpace.PNM_SB)
    col = (Serial("col", cols),) if cols > 1 else ()
    k.bind_layout(
        w,
        split(
            channels=channels,
            n=(*col, Channel(), Bank()),
            k=(Row("k_row", rows), Column(row_values)),
        ),
        base=w_row,
    )
    k.bind_layout(
        y, split(channels=channels, n=(*col, Slot(of=Channel()), Lane(of=Bank())))
    )
    return GemvTensors(x, w, y)


def _groups(cols: int, size: int) -> list[range]:
    """CENT's groups: ``ceil(cols / size)`` groups of ``ceil(cols / groups)``."""

    count = ceil(cols / size)
    per = ceil(cols / count)
    return [range(g * per, min(cols, (g + 1) * per)) for g in range(count)]
