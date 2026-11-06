"""CENT's vector kernels, written in the Tile DSL: RMSNorm, SiLU * up, residual.

They follow CENT's trace generator (``../ALOI/cent_simulation/Llama.py``,
``trace_only``), which keeps elementwise operands in the DRAM banks:

- A vector for a MAC between neighbor banks is stored in bank pairs: its
  ``b``-th chunk in bank ``2b + g`` (``store_for_neighbor_bank_input``),
  i.e. ``Bank(every=2, offset=g)``.
- A vector for ``EWMUL`` is stored in groups of four banks: chunk ``b`` in
  bank ``4b + g`` (``store_for_EWMUL_input``), ``Bank(every=4, offset=g)``;
  ``EWMUL`` multiplies members 0 and 1 into member 2 of each group.
- A result feeds the next ``EWMUL`` through ``COPY_BKGB`` / ``COPY_GBBK``
  from member 2 to member 1 of each group (banks 2, 6, 10, 14 to 1, 5, 9,
  13).

Stores and loads are ordinary DRAM writes and reads (``W MEM`` / ``R MEM``),
one per burst, burst by burst across the banks.
"""

from __future__ import annotations

from vllm_cent.device import MemorySpace
from vllm_cent.ir import DType
from vllm_cent.tile import Bank, Channel, Column, Lane, Slot, split
from vllm_cent.tile.dsl import TileKernel, TileValue

from .analytical import rms_norm_cycles

__all__ = ["residual_add", "rms_norm", "silu_mul"]


def _sb_vector(k: TileKernel, name: str, n: int) -> TileValue:
    """A Shared Buffer vector, in slots of one burst."""

    t = k.tensor(name, (n,), DType.BF16, MemorySpace.PNM_SB)
    k.bind_layout(t, split(n=(Slot(), Lane())))
    return t


def _dram_vector(
    k: TileKernel, name: str, n: int, row: int, every: int, g: int
) -> TileValue:
    """A DRAM vector in member ``g`` of each group of ``every`` banks of ``row``."""

    t = k.tensor(name, (n,), DType.BF16, MemorySpace.PIM_DRAM)
    k.bind_layout(
        t, split(n=(Channel(), Bank(every=every, offset=g), Column())), base=row
    )
    return t


def _copy_result(
    k: TileKernel, src: TileValue, dst: TileValue, group: int
) -> TileValue:
    """Copy member 2 of each bank group of ``src`` to member 1 of ``dst``."""

    chans = k.all_channels()
    for first in range(0, k.arch.banks_per_channel, group):
        gb = k.copy_bkgb(src, bank=first + 2, channels=chans)
        dst = k.copy_gbbk(gb, into=dst, bank=first + 1, channels=chans)
    return dst


def rms_norm(
    k: TileKernel,
    *,
    hidden: int,
    names: tuple[str, str, str],
    rows: tuple[int, int, int],
    scale_first: bool = False,
) -> None:
    """Record CENT's RMSNorm of ``x``: squares in the banks, the rest by EWMUL.

    1. ``x`` into bank pairs of ``rows[0]``; ``MAC_ABK`` between neighbors
       sums its squares per bank pair; ``RD_MAC`` reads the sums.
    2. PNM (analytical): add the sums and take the reciprocal square root,
       giving the scale vector.
    3. ``x`` and the scale into banks 4b and 4b + 1 of ``rows[1]``; ``EWMUL``.
    4. The product into banks 4b + 1 of ``rows[2]``, where the weight sits in
       banks 4b; ``EWMUL``; ``R MEM`` reads the normalized vector; ``SYNC``.

    Args:
        k: The kernel to record into.
        hidden: The vector length.
        names: Names of x, the weight and the output, as the Device IR op
            names its operands.
        rows: CENT's rows for x, its copy and the normalized vector.
        scale_first: Store the scale before ``x`` in step 3; CENT does so
            for the FFN norm (``Llama.py:1016-1017``).
    """

    arch = k.arch
    chans = k.all_channels()
    group = arch.banks_per_group
    x = _sb_vector(k, names[0], hidden)
    weight = _dram_vector(k, names[1], hidden, rows[2], group, 0)
    y = _sb_vector(k, names[2], hidden)
    sums = k.tensor(
        "sums",
        (arch.channels * arch.banks_per_channel,),
        DType.BF16,
        MemorySpace.PNM_SB,
    )
    k.bind_layout(sums, split(n=(Slot(of=Channel()), Lane(of=Bank()))))
    scale = _sb_vector(k, "scale", hidden)

    pair = [_dram_vector(k, f"x_pair{g}", hidden, rows[0], 2, g) for g in range(2)]
    pair = [k.wr_mem(x, into=p) for p in pair]
    acc = k.wr_bias_zero(channels=chans)
    acc = k.mac_bk_bk(pair[0], pair[1], acc, channels=chans)
    sums = k.rd_mac(acc, into=sums, channels=chans)
    scale = k.pnm(
        "RMSNorm sum of squares and rsqrt",
        sums,
        into=scale,
        cycles=rms_norm_cycles(hidden, arch.channels),
    )

    x_g0 = _dram_vector(k, "x_g0", hidden, rows[1], group, 0)
    scale_g1 = _dram_vector(k, "scale_g1", hidden, rows[1], group, 1)
    product = _dram_vector(k, "x_scaled", hidden, rows[1], group, 2)
    if scale_first:
        scale_g1 = k.wr_mem(scale, into=scale_g1)
        x_g0 = k.wr_mem(x, into=x_g0)
    else:
        x_g0 = k.wr_mem(x, into=x_g0)
        scale_g1 = k.wr_mem(scale, into=scale_g1)
    product = k.ewmul(x_g0, scale_g1, into=product, channels=chans)

    scaled = _dram_vector(k, "x_scaled_g1", hidden, rows[2], group, 1)
    normed = _dram_vector(k, "normed", hidden, rows[2], group, 2)
    scaled = _copy_result(k, product, scaled, group)
    normed = k.ewmul(weight, scaled, into=normed, channels=chans)
    y = k.rd_mem(normed, into=y)
    k.sync()
    k.returns(y)


def silu_mul(
    k: TileKernel,
    *,
    n: int,
    names: tuple[str, str, str, str],
    row: int,
) -> None:
    """Record CENT's ``silu(gate) * up``: two EWMULs in one row's bank groups.

    ``gate`` and ``sigmoid(gate)`` into banks 4b and 4b + 1, ``EWMUL`` gives
    the SiLU in 4b + 2; it is copied to 4b + 1, ``up`` stored in 4b, and a
    second ``EWMUL`` gives the product, read back with ``R MEM``; ``SYNC``.

    Args:
        k: The kernel to record into.
        n: The vector length.
        names: Names of gate, sigmoid(gate), up and the output.
        row: CENT's row (``x1_sigmoid``) for all of it.
    """

    chans = k.all_channels()
    group = k.arch.banks_per_group
    gate, sigmoid, up, y = (_sb_vector(k, name, n) for name in names)
    a = _dram_vector(k, "ew_a", n, row, group, 0)
    b = _dram_vector(k, "ew_b", n, row, group, 1)
    c = _dram_vector(k, "ew_c", n, row, group, 2)
    a = k.wr_mem(gate, into=a)
    b = k.wr_mem(sigmoid, into=b)
    c = k.ewmul(a, b, into=c, channels=chans)
    b = _copy_result(k, c, b, group)
    a = k.wr_mem(up, into=a)
    c = k.ewmul(a, b, into=c, channels=chans)
    y = k.rd_mem(c, into=y)
    k.sync()
    k.returns(y)


def residual_add(k: TileKernel, *, n: int, names: tuple[str, str]) -> None:
    """Record CENT's residual add: one ``EWADD`` (a no-op in the AiM simulator)."""

    a, b = (_sb_vector(k, name, n) for name in names)
    k.returns(k.ewadd(a, b))
