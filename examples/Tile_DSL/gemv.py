"""Hand-written CENT GEMV kernels in the Tile DSL: two schedules, one layout.

Run from the repository root: ``python examples/Tile_DSL/gemv.py``. Each
kernel is built with ``vllm_cent.tile.dsl``, verified for the device, and
printed, then printed again as an AiM simulator trace
(``vllm_cent.tile.codegen``); schedule 2's trace is CENT's own GEMV trace
(``../ALOI/cent_simulation``) but for the GPR field. Each DSL construct maps
to one IR structure:

- ``with k.serial(name, extent, carry=...) as loop`` builds one ``tile.for``.
  ``loop.index`` and ``loop.carried`` are its block arguments,
  ``loop.yield_(...)`` sets the block's results, and ``loop.results`` are
  the op's results. The Python body runs once, while the builder records it.
- ``k.wr_bias``, ``k.wr_gb``, ``k.mac_abk`` and ``k.rd_mac`` each build one
  ``cent.*`` op, named after the AiM instruction it becomes (see the AiM
  simulator's README, ``../ALOI/aim_simulator/README.md``):

  - ``WR_BIAS GPR_0 channel_mask``: one 256-bit burst from the Shared
    Buffer into the 16-bit MAC register of each of a channel's 16 banks;
  - ``WR_GB opsize GPR_0 channel_mask``: Shared Buffer into the Global
    Buffer of every selected channel;
  - ``MAC_ABK opsize channel_mask row``: every bank multiplies its row with
    the Global Buffer and adds to its MAC register;
  - ``RD_MAC GPR_0 channel_mask``: the 16 MAC registers of a channel back
    into one 256-bit Shared Buffer slot.

- ``channels=chans`` is the instruction's ``channel_mask``: one ``MAC_ABK``
  drives every bank of the selected channels, and one ``WR_GB`` broadcasts
  to every selected channel's Global Buffer. Banks and channels are
  therefore layout dimensions of the data, not loops.
- ``x.tile(k_tile=kt)`` names a logical tile by the serial dimensions of the
  tensor's layout; the layout maps it to channels, banks, rows and columns.

AiM has one 16-bit MAC register per bank (``accumulators_per_bank: 1`` in
``cent.yaml``; Hot Chips 2022 GDDR6-AiM, slide 6), so a schedule can keep
only one partial sum per bank in the accumulators. A schedule that needs
more keeps the others in the Shared Buffer: ``RD_MAC`` saves one,
``WR_BIAS`` restores it. (The CENT paper's ISA addresses 32 registers by
``Regid``; AiM and its simulator have no such operand.)

``WR_BIAS`` and ``RD_MAC`` move one channel's 16 registers as one 256-bit
Shared Buffer slot, lane = bank, but name only one slot (``GPR_0``).
Neither the AiM simulator nor CENT's functional model defines where the
data of several channels goes (the simulator sends every channel the same
``GPR_0``), so these kernels follow our convention, not a documented
hardware rule:

- ``RD_MAC`` names a base slot ``Rd`` aligned to the device's channel
  count; channel ``c`` writes slot ``Rd + c``. The offset is the physical
  channel number, not its position in ``channel_mask``, so a channel's slot
  does not depend on which other channels are selected. ``Rd`` comes from
  the layout of the tile written (``Slot(of=Channel())``), not from the
  kernel author.
- ``WR_BIAS`` takes ``stride``: with ``stride=1`` channel ``c`` reads slot
  ``Rs + c`` (aligned, as for ``RD_MAC``); with ``stride=0`` every channel
  reads slot ``Rs``.

A sum that starts at zero uses ``wr_bias_zero``: a ``WR_BIAS`` of zeros that
reads no Shared Buffer slot, like a zero register.

How these transfers reach many channels is the run spec's
``codegen.channel_issue``: ``broadcast`` issues one full-mask instruction
naming the base slot, as CENT's own traces do; ``serialize`` issues one
instruction per channel, each naming its own slot (see
``vllm_cent.tile.expand``).

Kernel: ``y = w @ x`` (``nn.Linear``: ``w`` is ``[N, K]``), ``N = K = 4096``,
bf16. Every size below is derived from the device spec; nothing is assumed
about the paper-scale numbers.
"""

from vllm_cent.device import CentArchitecture, MemorySpace, load_device_spec
from vllm_cent.ir import DType
from vllm_cent.ir import print_function
from vllm_cent.tile import Bank, Channel, Column, Lane, Row, Serial, Slot, split
from vllm_cent.tile.codegen import aim_trace
from vllm_cent.tile.dsl import TileKernel

device = load_device_spec("examples/cent/cent.yaml")
arch = device.architecture
# These kernels use CENT instructions, so they need a CENT device.
assert isinstance(arch, CentArchitecture)
BF16 = DType.BF16
N = K = 4096

# Each output element is computed by one bank, so one wave of all banks
# produces CHANNELS * BANKS outputs. One Global Buffer holds one reduction
# tile, so K is reduced in GB-sized pieces; with the paper spec a GB holds
# one DRAM row (1024 bf16), so a tile is also one row of every weight.
CHANNELS, BANKS = arch.channels, arch.banks_per_channel
GB_VALUES = device.memory_space(MemorySpace.PIM_GB).capacity_values(BF16)
N_WAVES, K_TILES = N // (CHANNELS * BANKS), K // GB_VALUES
assert N_WAVES * CHANNELS * BANKS == N and K_TILES * GB_VALUES == K
# WR_BIAS and RD_MAC move one burst per channel: one 16-bit value per bank.
# The MAC registers are 16 bits (AiM simulator README) and the PUs compute
# in BF16 (Hot Chips 2022 GDDR6-AiM, slide 5; CENT paper, Section 4.2).
assert BANKS * BF16.bits == arch.burst_size * 8


# ==============================================================================
# Schedule 1: output-stationary (n_wave outside, k_tile inside).
# The partial sums of one wave stay in the MAC registers for the whole
# reduction; x's tile is written to the GB once per (n_wave, k_tile), i.e.
# N_WAVES * K_TILES times.
# ==============================================================================
with TileKernel("q_proj_gemv_output_stationary", device) as k:
    x = k.tensor("x", (K,), BF16, MemorySpace.PNM_SB)
    w = k.tensor("w", (N, K), BF16, MemorySpace.PIM_DRAM)
    y = k.tensor("y", (N,), BF16, MemorySpace.PNM_SB)

    # x: one reduction tile after another; a tile fills Shared Buffer slots
    # of one burst (lanes) each. Slot() and Lane() sizes follow the device.
    k.bind_layout(x, split(k=(Serial("k_tile", K_TILES), Slot(), Lane())))
    # w: output n = (n_wave, channel, bank), outermost first, so one wave
    # covers CHANNELS * BANKS consecutive outputs; the reduction axis runs
    # along DRAM rows, one row per k_tile.
    k.bind_layout(
        w,
        split(
            n=(Serial("n_wave", N_WAVES), Channel(), Bank()),
            k=(Row("k_row", K_TILES), Column()),
        ),
    )
    # y: one wave of results after another; a wave is one aligned group of
    # slots, one per channel, each holding that channel's banks. This is the
    # shape RD_MAC writes: the wave's first slot is its Rd.
    k.bind_layout(
        y, split(n=(Serial("n_wave", N_WAVES), Slot(of=Channel()), Lane(of=Bank())))
    )

    chans = k.all_channels()  # channel_mask selecting every channel

    with k.serial("n_wave", N_WAVES, carry=(y,)) as waves:
        n, (y_now,) = waves.index, waves.carried
        # A WR_BIAS of zeros: it reads no Shared Buffer slot.
        acc0 = k.wr_bias_zero(channels=chans)

        with k.serial("k_tile", K_TILES, carry=(acc0,)) as reduce:
            kt, (acc,) = reduce.index, reduce.carried
            gb = k.wr_gb(x.tile(k_tile=kt), channels=chans)
            reduce.yield_(
                k.mac_abk(w.tile(n_wave=n, k_row=kt), gb, acc, channels=chans)
            )

        (acc_done,) = reduce.results
        waves.yield_(k.rd_mac(acc_done, into=y_now.tile(n_wave=n), channels=chans))

    k.returns(*waves.results)

# build() allocates the layouts and runs verify_kernel: the kernel is legal.
output_stationary = k.build()  # a CentKernel: the function and its layouts
# TODO(Step 6): print(evaluate(output_stationary, device).summary())
print(print_function(output_stationary.function))
print(aim_trace(output_stationary, arch))  # the AiM simulator trace


# ==============================================================================
# Schedule 2: GB-stationary (k_tile outside, every n_wave inside).
# Each WR_GB serves all N_WAVES waves, so x is written to the GB only
# K_TILES times. But every wave's partial sum must survive the whole k_tile
# loop, and AiM has one MAC register per bank: each wave saves its partial
# sum to y with RD_MAC and restores it with WR_BIAS on the next k_tile. The
# GB writes saved are paid for in WR_BIAS/RD_MAC round trips; the model
# decides which costs more.
#
# y holds the partial sums between k_tiles, so it must start at zero.
# ==============================================================================
with TileKernel("q_proj_gemv_gb_stationary", device) as k:
    # The same tensors and layout as schedule 1: only the loops differ.
    x = k.tensor("x", (K,), BF16, MemorySpace.PNM_SB)
    w = k.tensor("w", (N, K), BF16, MemorySpace.PIM_DRAM)
    # TODO(runtime): y must be zero on entry: the first WR_BIAS reads it.
    y = k.tensor("y", (N,), BF16, MemorySpace.PNM_SB)
    k.bind_layout(x, split(k=(Serial("k_tile", K_TILES), Slot(), Lane())))
    k.bind_layout(
        w,
        split(
            n=(Serial("n_wave", N_WAVES), Channel(), Bank()),
            k=(Row("k_row", K_TILES), Column()),
        ),
    )
    # The layout of schedule 1. RD_MAC saves and WR_BIAS (stride=1)
    # restores each channel's partial sums in that channel's slot.
    k.bind_layout(
        y, split(n=(Serial("n_wave", N_WAVES), Slot(of=Channel()), Lane(of=Bank())))
    )

    chans = k.all_channels()

    with k.serial("k_tile", K_TILES, carry=(y,)) as reduce:
        kt, (partial,) = reduce.index, reduce.carried
        gb = k.wr_gb(x.tile(k_tile=kt), channels=chans)  # once per k_tile

        with k.serial("n_wave", N_WAVES, carry=(partial,)) as waves:
            n, (partial_now,) = waves.index, waves.carried
            acc = k.wr_bias(partial_now.tile(n_wave=n), stride=1, channels=chans)
            acc = k.mac_abk(w.tile(n_wave=n, k_row=kt), gb, acc, channels=chans)
            waves.yield_(
                k.rd_mac(acc, into=partial_now.tile(n_wave=n), channels=chans)
            )

        reduce.yield_(*waves.results)

    k.returns(*reduce.results)

gb_stationary = k.build()
# TODO(Step 6): print(evaluate(gb_stationary, device).summary())
print(print_function(gb_stationary.function))
print(aim_trace(gb_stationary, arch))  # the AiM simulator trace

# Which schedule is faster is the model's answer, not an assumption: the
# reports give GB writes, WR_BIAS/RD_MAC round trips, row switches and the
# makespan under the timing spec, with its assumptions.
