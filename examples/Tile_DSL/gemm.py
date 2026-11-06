"""Hand-written CENT GEMM kernels in the Tile DSL: two schedules, one layout.

Run from the repository root: ``python examples/Tile_DSL/gemm.py``. The
constructs are those of ``gemv.py`` (one ``with
k.serial`` is one ``tile.for``; ``wr_bias``/``wr_gb``/``mac_abk``/``rd_mac``
are the AiM instructions; ``channels=`` is the channel_mask), and so is
the Shared Buffer convention for ``WR_BIAS``/``RD_MAC``; see there.

Kernel: ``y = x @ w.T`` (``nn.Linear``) for ``M = 4`` tokens of decode,
``N = K = 4096``, bf16. Each token row of x is its own GEMV against the same
weights, so the two schedules differ in what stays in place between MACs:

1. token-stationary: finish one token's output before the next one; its
   partial sums stay in the MAC registers, but every weight row is opened
   again for each token;
2. weight-row-stationary: while one weight row (n_wave, k_tile) is open,
   multiply it with every token. AiM has one MAC register per bank, so the
   M partial sums take turns in it: each token's sum is restored with
   WR_BIAS before its MAC and saved with RD_MAC after it.

Sizes come from the device spec, as in ``gemv.py``.
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
# x and y take M * 256 Shared Buffer slots each: M = 4 fills all 2048 slots
# (64 KiB). Sums start with wr_bias_zero, which needs no slot of zeros.
M, N, K = 4, 4096, 4096

CHANNELS, BANKS = arch.channels, arch.banks_per_channel
GB_VALUES = device.memory_space(MemorySpace.PIM_GB).capacity_values(BF16)
N_WAVES, K_TILES = N // (CHANNELS * BANKS), K // GB_VALUES
assert N_WAVES * CHANNELS * BANKS == N and K_TILES * GB_VALUES == K
# WR_BIAS and RD_MAC move one burst per channel: one 16-bit value per bank.
# The MAC registers are 16-bit BF16; see gemv.py for the sources.
assert BANKS * BF16.bits == arch.burst_size * 8


# ==============================================================================
# Schedule 1: token-stationary (n_wave -> m -> k_tile).
# A token's partial sums stay in the MAC registers for the whole reduction;
# between two tokens the k_tile loop opens all weight rows again.
# ==============================================================================
with TileKernel("q_proj_gemm_token_stationary", device) as k:
    x = k.tensor("x", (M, K), BF16, MemorySpace.PNM_SB)
    w = k.tensor("w", (N, K), BF16, MemorySpace.PIM_DRAM)
    y = k.tensor("y", (M, N), BF16, MemorySpace.PNM_SB)

    # x: token rows one after another, each laid out like the GEMV input.
    k.bind_layout(
        x, split(m=(Serial("m", M),), k=(Serial("k_tile", K_TILES), Slot(), Lane()))
    )
    # w: the same layout as the GEMV weight.
    k.bind_layout(
        w,
        split(
            n=(Serial("n_wave", N_WAVES), Channel(), Bank()),
            k=(Row("k_row", K_TILES), Column()),
        ),
    )
    # y: each (token, wave) is one aligned group of slots, one per channel,
    # as RD_MAC writes it (the convention in gemv.py).
    k.bind_layout(
        y,
        split(
            m=(Serial("m", M),),
            n=(Serial("n_wave", N_WAVES), Slot(of=Channel()), Lane(of=Bank())),
        ),
    )

    chans = k.all_channels()

    with k.serial("n_wave", N_WAVES, carry=(y,)) as waves:
        n, (y_wave,) = waves.index, waves.carried

        with k.serial("m", M, carry=(y_wave,)) as tokens:
            m, (y_token,) = tokens.index, tokens.carried
            acc0 = k.wr_bias_zero(channels=chans)

            with k.serial("k_tile", K_TILES, carry=(acc0,)) as reduce:
                kt, (acc,) = reduce.index, reduce.carried
                gb = k.wr_gb(x.tile(m=m, k_tile=kt), channels=chans)
                reduce.yield_(
                    k.mac_abk(w.tile(n_wave=n, k_row=kt), gb, acc, channels=chans)
                )

            (acc_done,) = reduce.results
            tokens.yield_(
                k.rd_mac(acc_done, into=y_token.tile(m=m, n_wave=n), channels=chans)
            )

        waves.yield_(*tokens.results)

    k.returns(*waves.results)

token_stationary = k.build()
# TODO(Step 6): print(evaluate(token_stationary, device).summary())
print(print_function(token_stationary.function))
print(aim_trace(token_stationary, arch))  # the AiM simulator trace


# ==============================================================================
# Schedule 2: weight-row-stationary (n_wave -> k_tile -> m).
# The weight row (n_wave, k_tile) stays open across the M MACs. The GB still
# holds one tile, so each token's tile is written right before its MAC, and
# the single MAC register holds one token's partial sum at a time: y keeps
# the others between k_tiles (RD_MAC saves, WR_BIAS restores).
#
# y holds the partial sums between k_tiles, so it must start at zero.
# ==============================================================================
with TileKernel("q_proj_gemm_weight_row_stationary", device) as k:
    # The same tensors and layout as schedule 1: only the loops differ.
    x = k.tensor("x", (M, K), BF16, MemorySpace.PNM_SB)
    w = k.tensor("w", (N, K), BF16, MemorySpace.PIM_DRAM)
    # TODO(runtime): y must be zero on entry; see gemv.py.
    y = k.tensor("y", (M, N), BF16, MemorySpace.PNM_SB)
    k.bind_layout(
        x, split(m=(Serial("m", M),), k=(Serial("k_tile", K_TILES), Slot(), Lane()))
    )
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
        y,
        split(
            m=(Serial("m", M),),
            n=(Serial("n_wave", N_WAVES), Slot(of=Channel()), Lane(of=Bank())),
        ),
    )

    chans = k.all_channels()

    with k.serial("n_wave", N_WAVES, carry=(y,)) as waves:
        n, (y_wave,) = waves.index, waves.carried

        with k.serial("k_tile", K_TILES, carry=(y_wave,)) as reduce:
            kt, (partial,) = reduce.index, reduce.carried
            w_row = w.tile(n_wave=n, k_row=kt)

            with k.serial("m", M, carry=(partial,)) as tokens:
                m, (partial_now,) = tokens.index, tokens.carried
                acc = k.wr_bias(
                    partial_now.tile(m=m, n_wave=n), stride=1, channels=chans
                )
                gb = k.wr_gb(x.tile(m=m, k_tile=kt), channels=chans)
                acc = k.mac_abk(w_row, gb, acc, channels=chans)
                tokens.yield_(
                    k.rd_mac(
                        acc, into=partial_now.tile(m=m, n_wave=n), channels=chans
                    )
                )

            reduce.yield_(*tokens.results)

        waves.yield_(*reduce.results)

    k.returns(*waves.results)

weight_row_stationary = k.build()
# TODO(Step 6): print(evaluate(weight_row_stationary, device).summary())
print(print_function(weight_row_stationary.function))
print(aim_trace(weight_row_stationary, arch))  # the AiM simulator trace

# Both schedules issue the same MACs and GB writes; they differ in row
# switches and WR_BIAS/RD_MAC round trips. The reports give both, and the
# makespan under the timing spec; no schedule is assumed faster.
