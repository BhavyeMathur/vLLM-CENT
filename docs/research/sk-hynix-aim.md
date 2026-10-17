# SK hynix AiM and CENT: hardware, software, and unresolved contracts

This note records what three SK hynix sources establish for vLLM-CENT. It is
research evidence, **not** an amendment to the accepted [source hierarchy](../decisions/README.md):
the CENT paper defines our paper-oriented ISA, the public AiM simulator defines
its timing-trace target, and a future execution target must define its own ABI.
The documents describe a fabricated GDDR6-AiM device and evolving SK hynix
systems, not an implementation specification for every CENT extension.

## Sources and how to read them

| ID | Primary source | Most useful locations |
| --- | --- | --- |
| JSSC | D. Kwon et al., [*A 1ynm 1.25V 8Gb 16Gb/s/Pin GDDR6-Based Accelerator-in-Memory...*](https://doi.org/10.1109/JSSC.2022.3200718), IEEE JSSC 58(1), 2023 | pp. 292–297, especially Figs. 1–3, 9–14 and Table I; pp. 298–299 for evaluation |
| HC22 | D. Kwon et al., [*System Architecture and Software Stack for GDDR6-AiM*](https://doi.org/10.1109/HCS55958.2022.9895629), Hot Chips 2022 | slides 5–7 (chip/commands), 9–12 (subsystem/ISR), 16–19 (software/tiling), 21–23 (evaluation) |
| HC23 | D. Kwon et al., [*Memory-Centric Computing with SK Hynix's Domain-Specific Memory*](https://doi.org/10.1109/HCS59251.2023.10254717), Hot Chips 2023 | slides 7–9 (chip/commands), 13–18 (tiling/scale-out), 20–22 (prototype/software/evaluation) |
| CENT | [CENT paper](https://arxiv.org/pdf/2502.07578), §§4.2–4.3, Fig. 7, Tables 2–3 | Defines the architectural additions and typed instruction surface that our compiler targets |
| AiM timing | [Public AiM simulator](https://github.com/arkhadem/aim_simulator), especially [`request.h`](../../../CENT/aim_simulator/src/base/request.h), [`aim_DRAM_system.cpp`](../../../CENT/aim_simulator/src/memory_system/impl/aim_DRAM_system.cpp) | Trace grammar and timing behavior, **not** tensor-value semantics |

Page references for JSSC use printed journal pages; slide references use the
slide numbers printed on the decks. The locally supplied PDFs live in
`artifacts/AiM/` beside the two repositories. These sources can have different
names for different layers of the same operation: a physical DRAM command, a
controller-level ISR instruction, a CENT ISA instruction, and an AiM trace
record must not be assumed to have identical operands.

## The smallest useful hardware picture

One fabricated GDDR6-AiM package has two 4-Gb dies (8 Gb/1 GB total). Each die
is one x16 channel with 16 DRAM banks and **one near-bank processing unit (PU)
per bank**. The die has a 2-KB Global Buffer (GB), physically split into two
1-KB regions. A PU operates at 1 GHz and processes a 32-byte DRAM column
access: 16 BF16 weight/vector pairs, reduced to one MAC contribution. Thus the
published 32 GFLOPS/PU, 512 GFLOPS/die, and approximately 1 TFLOPS/package
are **peak** rates, not measured whole-model throughput. The 2-KB row holds
64 such 32-byte column accesses, or 1,024 BF16 elements. [JSSC pp. 292–293,
295–296, Figs. 1–3, 9; HC22 slides 5–7; HC23 slides 7–9.]

In CENT, a channel still has 16 banks and a near-bank PU at each bank, but
**four banks make one bank group**, not one PU. CENT adds CXL-attached devices,
PNM units, and a 64-KB Shared Buffer. The GB and Shared Buffer are different
storage: the GB feeds bank MACs; CENT's Shared Buffer stages data between PIM,
PNM, and CXL. The HC22 controller's 512-KB general-purpose register (GPR)
SRAM is a third structure; its bias/result/debug uses do not make it identical
to CENT's Shared Buffer. [CENT §4.2; HC22 slides 10–11.]

**Project naming:** [`BANKS_PER_GROUP = 4`](../../src/vllm_cent/cent/hardware.py)
counts banks per bank group in placement and lowering. It replaces the misleading
name `BANKS_PER_PU`; all grouping values and layouts are unchanged. Each bank
has its own processing unit, so this constant does not count PUs per channel. [JSSC p. 292, Fig. 1; CENT §4.2, Fig. 7.]

## What the instructions actually operate on

The fabricated chip has physical ACT/ACTAF, MAC, AF, elementwise multiply,
copy, GB/bank/register read-write commands, and special mode-register behavior
for certain reads/writes. HC22's AiM DMA accepts higher-level **ISR** commands;
the ISR FSM expands them into channel packets and the controllers turn those
into physical DRAM commands. Its illustrated command format includes a 6-bit
column, 14-bit row/GPR address, operation size, channel mask, and bank/AF
selection fields. A 2023 scale-out proposal further shows a matrix-vector
accumulate instruction containing channel mask, start row/column, iteration
count, and MAC register ID. Neither format automatically defines the CENT ISA
encoding. [JSSC p. 292 Table I; HC22 slides 7, 10–12, 18; HC23 slides 9, 17.]

`CO` is therefore best understood **physically as a burst-column address**:
one access is 32 bytes/16 BF16 values; the 2-KB row has 64 columns. Both the
HC22 ISR format and the public AiM simulator expand an operation over
successive column addresses. Our current [`CentMemoryAddress`](../../src/vllm_cent/cent/instructions/address.py)
calls `column` a *scalar position* and the
[`validator`](../../src/vllm_cent/cent/instructions/validation.py) advances by
`OPsize * burst_length`; that is a likely unit mismatch if the typed field is
meant to be the paper's `CO`. It needs a whole-code audit and migration rather
than an isolated documentation edit. The paper says OPsize targets subsequent
DRAM column addresses but does not provide every byte/slot conversion rule.
[JSSC pp. 293, 296; HC22 slides 6, 11; CENT §4.3; AiM
`aim_DRAM_system.cpp`.]

The chip can copy a DRAM row to a different row in the **same or another
bank**, using GB as an intermediate. Thus a source/destination bank choice is
physically real; the public AiM trace's explicit bank for `COPY_BKGB` and
`COPY_GBBK` is not gratuitous. The CENT paper's Table 3 omits `BK` for those
instructions, while its `CO` identifies a column/start segment for an
operation-sized transfer. Whether the omission is shorthand, an error, or a
different implicit bank-selection rule remains unresolved. In the public
simulator the copy trace does not expose `CO`: the memory system defaults the
start column to zero and increments it per generated operation. **Do not claim
that `CO` is meaningless in CENT just because this timing trace cannot encode
it.** [JSSC p. 293 Fig. 2(d–e); HC22 slide 11; CENT Table 3; AiM `request.h`
and `aim_DRAM_system.cpp`.]

GB-to-bank MAC and neighboring-bank MAC are different physical data paths.
The former can activate 16 PUs per channel; the even/odd neighboring-bank mode
uses eight PUs because one bank provides the vector to its partner. The public
AiM simulator chooses the vector source using persistent **CFR0** state (0 =
GB, 1 = adjacent bank), written as a separate control-register transaction;
it is not intrinsically an operand of its `MAC_ABK` trace line. Our typed
`MacOperandSource` is a compiler-level declaration of the needed behavior, and
the renderer writes CFR0 before MAC. That is a reasonable target adapter, but
the numeric CFR encoding is simulator evidence, not established by JSSC.
[JSSC p. 293 Fig. 2(a–b); HC22 slide 6; CENT §5.4; AiM
`aim_DRAM_system.cpp`.]

The chip command set also has one-, four-, and sixteen-bank MAC variants. That
is a possible hardware mechanism for avoiding work in unused banks, but CENT's
published `MAC_ABK` is an *all-bank* instruction and its `CHmask` selects
channels, not individual banks. Therefore these Hynix commands do not by
themselves solve our inactive-PU/group dataflow TODO. Similarly, CENT
`WR_ABK` explicitly splits a 256-bit Shared Buffer slot into 16 BF16 values
and writes one to each bank; this establishes a write-side bank-lane mapping,
not the reverse `RD_MAC` result order. [JSSC p. 293 Fig. 2(a–b), Table I;
HC22 slide 7; CENT §4.2 and Tables 2–3.]

Elementwise multiplication reads two banks and stores to another bank in a
group. JSSC establishes the dataflow, not a universal fixed “bank 0 and 1 in,
bank 2 out” ABI; that precise mapping is behavior of the original CENT
functional reference, not a conclusion from the fabricated chip. [JSSC p. 293
Fig. 2(c); CENT §5.4; original CENT [`aim_sim.py`](../../../CENT/cent_simulation/aim_sim.py).]

## Accumulators, bias, and `RD_MAC`

The chip has a distinct `MAC_REG` result path and `AF_REG` activation-result
path; `RDMAC` and `RDAF` are separate physical reads. JSSC Fig. 10 describes
**two selectable result latches per accumulator**, selected through a mode
register to reduce GB rewrites when scheduling tiles. The alternating EV/OD
accumulator *pipes* in Figs. 7–8 are a timing technique, not an additional
software-visible register count. HC23 slide 14 uses the two MAC result IDs in a
column-first tiling example. The public AiM request structure separately
describes a `thread_index` of 0 or 1; no cited source proves that it maps to
the physical latch selector, and the trace renderer must still respect
which fields its actual grammar exposes. [JSSC pp. 295–296, Figs. 7–10; HC22
slide 6; HC23 slides 8, 14; AiM `request.h`.]

**This conflicts with CENT's explicit statement of 32 accumulation registers
per PU**, addressed by `Regid`. It may be a deliberate CENT extension, a
different abstraction, or a paper error; these documents cannot decide. Do
not silently clamp CENT `Regid` to 0–1 or pretend the public AiM timing trace
faithfully represents all 32 registers. [CENT §4.2, Table 2.]

HC22 says `WRBIAS` writes the MAC result register and its GPR SRAM can retain
bias values for reuse. This explains **what** bias initialization does and why
the controller has a bias source, but not **which CENT `Regid`** is selected by
`WR_BIAS CHmask Rs`, since that CENT instruction omits `Regid`. It also does
not specify how one CENT Shared Buffer slot is fanned out over several
channels/banks. The existing [`WR_BIAS` TODOs](../../src/vllm_cent/cent/instructions/data_movement.py)
remain valid. [HC22 slides 7, 10, 18; CENT Tables 2–3.]

One MAC produces one accumulated scalar *per active bank/PU*, whereas a
channel-wide `RDMAC` reads multiple banks. The HC22 controller collects results
in its GPR, but gives no lane/slot order for CENT's `RD_MAC CHmask Rd Regid`
when several channels are selected. A 256-bit CENT Shared Buffer slot is real
(16 BF16 lanes), and 16 banks' **BF16-sized** outputs could fit one slot as an
inference. That does **not** prove the output format, precision, ordering,
multi-channel stride, or whether a particular `RD_MAC` writes only one slot.
Treat “one result slot per register” in the current
[`Llama compiler`](../../src/vllm_cent/models/llama/compiler.py) as a local
mapping assumption, not a verified hardware rule. [HC22 slides 6, 10–11;
CENT §§4.2–4.3; `request.h`.]

## Numerical execution and activation functions

The chip multiplies BF16 inputs in groups of 16, retains a 16-bit product
mantissa, then uses a bank-wide mantissa-shift adder tree with 23 fractional
bits; the accumulator performs normalization/rounding and special-value
handling. The authors describe its accuracy as operating like an FP32 system.
This is **not proof of bitwise IEEE FP32 dot-product or PyTorch equivalence**:
input special cases, alignment, accumulation order, overflow/underflow, and
rounding need a specified numeric model and tests. [JSSC pp. 293–295,
§III and Table III.]

ReLU and leaky ReLU use computation circuits; sigmoid, tanh, and GELU use LUT
rows in DRAM and interpolation. A mode-register selection precedes AF, and
LUT-based AF also needs the appropriate `ACTAF` row. Results go to `AF_REG`,
not `MAC_REG`. The sources show a sigmoid LUT example with 512 addresses and a
specific range; that is not a universal CENT `AFid` encoding or generic range
for all functions. User-programmed LUT functions have accuracy limits. The
public AiM simulator's persistent CFR2 selects its AF mode, but these papers
do not settle the numeric IDs used by a future CENT target. [JSSC pp. 296–297,
§IV and Figs. 11–14; HC22 slides 6–7; AiM `aim_DRAM_system.cpp`.]

HC22 mentions an **SK hynix functional software emulator** and performance
model. This is not the same artifact as the publicly available
`arkhadem/aim_simulator`, which is Ramulator-based, trace-driven timing
simulation. The latter cannot validate tensor values, BF16 arithmetic, LUT
outputs, `RD_MAC` lane packing, PNM instructions, or CXL dataflow. Our own
functional executor must implement an explicitly chosen semantic contract and
test it independently. [HC22 slide 17; AiM README; CENT §4.2.]

## Compiler/runtime and performance implications

- The Hynix software pipeline includes model/runtime placement and weight
  reshaping, ISR generation, device-driver allocation/dispatch, DMA,
  multicast, and controller scheduling. AiM requests are scheduled in order;
  ordinary DRAM requests can be FR-FCFS reordered. The distinction supports
  keeping compiler layout, runtime tensor binding, target instruction lowering,
  and timing scheduling as separate layers. It does not implement our missing
  vLLM worker, KV-cache ownership, or CXL execution. [HC22 slides 9–12,
  16–18; HC23 slides 17, 21.]
- Column-major tiling reuses vectors in GB; row-major tiling keeps partial sums
  in `MAC_REG`. Two selectable latches enable an additional column-first
  schedule. These are useful candidates for the future layout/cost model and
  make GB traffic and register occupancy first-class costs, but do not change
  the current correctness-first milestone. [JSSC p. 296 Fig. 10; HC22 slide
  19; HC23 slide 14.]
- The HC23 scale-out board adds local memory, ALUs for layer normalization,
  softmax and residual addition, an inter-device router, and a multi-device
  scheduler. It supports CENT's broad PIM-plus-other-compute motivation, but
  its ALUs/router are **not** evidence for the specific CENT `EXP`, `RED`,
  `ACC`, RISC-V, Shared Buffer, or CXL ABI. [HC23 slides 16–18, 21; CENT §4.2.]
- Keep peak, measured, and modeled numbers distinct. JSSC tested the chip on
  ATE up to 16 Gb/s/pin, while its FPGA-board system experiments were limited
  by the board interface and several larger-system speedups use an analytical
  model. HC23's scale-out PoC describes 2.67 Gb/s/pin and also projects a
  higher-rate design. None is a measured vLLM-CENT end-to-end latency. [JSSC
  pp. 297–299; HC22 slides 21–23; HC23 slides 20–22.]

## Questions answered, changed, and still open

| Question | Evidence-backed answer / change to our understanding | Remaining contract |
| --- | --- | --- |
| Is AiM the ground truth for all DRAM behavior? | It is the target for **its timing trace**. The fabricated chip and Hynix controller explain many physical operations, while CENT defines a broader paper ISA. | Which concrete CENT hardware/runtime ABI will execute instructions, including features the timing trace omits? |
| Is a PU a four-bank unit? | No: one bank has one PU; four banks form a group. | The compiler now uses `BANKS_PER_GROUP`; the four-bank grouping and layouts are unchanged. |
| What is a DRAM `CO`? | Strong physical evidence for a 32-byte burst-column index, advanced by `OPsize`; 64 such columns fit a 2-KB row. | Audit/convert scalar-indexed addresses, validation, builders, traces, and tests before changing the typed API. |
| Does copy need `BK`; is `CO` useless? | A bank choice is physically necessary for cross-bank copy. CENT Table 3 omits it; AiM trace requires it and defaults its hidden start column to 0. | Determine paper's intended bank selection and whether nonzero copy columns are executable on a real target. |
| How does `MAC_ABK` choose its second operand? | GB versus neighboring bank are chip modes; public AiM selects them via a preceding persistent CFR0 write. | Define register/control-state behavior for any non-AiM execution target. |
| Can we suppress unused banks? | Physical AiM has narrower one-/four-bank MAC commands, but CENT exposes all-bank `MAC_ABK`. | Determine whether the CENT decoder exposes bank-group masks or whether inactive banks require safe initialized data and output filtering. |
| How many accumulation registers exist? | Fabricated AiM demonstrates two selectable latches; CENT explicitly claims 32 per PU. | Reconcile hardware version/abstraction, or keep target-specific limits and reject incompatible traces. |
| What do `WR_BIAS` and `RD_MAC` mean? | Bias initializes MAC state; `RDMAC` reads bank MAC results, distinct from AF results. | Bias destination register and broadcast source; result precision, bank/channel order, stride, and Shared Buffer slot count. |
| Can AiM validate numerical correctness? | No, the public trace simulator is a timing model; Hynix's separate functional emulator is not this public repository. | Establish and test a functional contract for MAC, AF, EWMUL, copy, PNM, and CXL. |
| Can Hynix's stack fill our vLLM gaps? | It provides a useful decomposition of placement, ISR generation, dispatch, DMA, and scheduling. | Define our own weight/KV loading, runtime binding, vLLM worker/runner, PNM/CXL executor, and device interface. |

### Reference implementation and physical export priorities

1. Ask the CENT/AiM authors or obtain an executable ABI/spec for the 32-vs-2
   `Regid` discrepancy, `WR_BIAS` target register, and multi-channel `RD_MAC`
   layout. These block claims of physical compatibility, not explicit-register
   reference GEMV/RMSNorm execution. Define and test the reference contract
   while these export details remain open.
2. Decide the typed `CO` unit and make an end-to-end migration with tiny
   two-column/one-row tests. This is the clearest likely modeling error with a
   tractable source-backed direction.
3. Clarify copy's implicit/explicit bank and nonzero-column semantics per
   target. Continue to reject operands that the AiM timing trace cannot
   preserve instead of silently dropping them.
4. Specify numerical precision, special values, accumulator initialization,
   AF LUT/mode IDs, and result formats for the functional executor; validate
   small hand-calculated cases before whole-model claims.
5. After a numerically correct vertical slice, measure GB reloads, MAC-result
   register occupancy, DRAM traffic, and inter-device transfers to guide
   tiling; do not extrapolate peak chip FLOPS into serving throughput.


## Selection controls: what is actually established

JSSC printed p296, Fig10(a–c), explicitly selects latch0 and latch1 through a
mode-register-set (MRS) command and later revisits each partial sum. Its circuit
shows a `LAT_SEL` mux choosing the result and accumulation feedback. This is
strong evidence for explicit register intent in compiler IR. HC23 printed
slide17 (PDF18) independently gives the matrix-vector-accumulate macroinstruction
an explicit `MAC regID`; printed slide14 (PDF15) schedules two `MAC_REG`s.
The controller can lower a richer operation to several commands. An omitted
operand in CENT's summary table therefore does not require a register-zero-only
compiler. [JSSC](https://doi.org/10.1109/JSSC.2022.3200718),
[official HC23 slides](https://hc2023.hotchips.org/assets/program/conference/day1/PIM/HC35_SKhynix_DSM_YongkeeKwon_rev3.pdf).

Keep these controls distinct:

| Control | Established role | What it does not establish |
| --- | --- | --- |
| AiM DL Mode (`ADM`), JSSC p292 | Enables recognition of special input/output commands, including WRBIAS/RDMAC. | A numbered latch destination. |
| MRS latch selection, JSSC p296 Fig10 | Selects which partial sum MAC continues; circuit shows selected result/feedback. | Public bit encoding, scope, or complete WRBIAS/RDMAC/AF selection rules. |
| EV/OD pipes, JSSC p295 Figs7–8 | Alternate internal accumulation work to meet timing. | Additional software-visible register IDs. |
| MRS activation settings, JSSC pp296–297 §IV | Select activation-function type and leaky-ReLU slope. | The MAC latch selector or a CENT AFid encoding. |
| `T`, HC22 slide11, bit32 | A one-bit field exists in the illustrated controller format. | Its meaning: the deck does not define it. Do not identify it with latch/thread selection by name alone. |

The [ASPLOS 2023 Hynix tutorial](https://events.safari.ethz.ch/asplos-pim-tutorial/lib/exe/fetch.php?media=system_architecture_and_software_stack_for_gddr6-aim_sk_hynix_pim_tutorial_asplos23_sent.pdf),
printed slide16 (PDF15), confirms macroinstruction-to-microcommand expansion.
Its WRMAC footnote concerns recognition, not latch numbering. It does not add
a latch-selection bit specification.

**TODO — physical latch coupling:** obtain how bias writes, MAC, result reads,
and activation inputs select the same intended result on the chosen device.
MRS-controlled accumulation itself is established. What remains missing is the
precise per-command relationship and export encoding. Internal typed operations
should retain explicit register IDs; a target adapter must preserve them or
reject the export, rather than silently discard the selector.

## RMSNorm and scalar operations

CENT §5.4 and Figure 10(b) place dot products and elementwise multiplication
on PIM and scalar square-root/division work on PNM. Section 4.2 provides
RISC-V cores and Shared Buffer access; it does not define an RMSNorm routine's
arguments, completion, or scalar broadcast. Original CENT `Llama.py:83–102`
computes the sum through PIM, then uses host `torch.rsqrt(sum/dim+1e-5)` and
`torch.full` to create the repeated scale vector. This is an executable
reference formula, not evidence that copy or multiply broadcasts within a slot.

Reference lowering may explicitly implement per-row sum, mean plus epsilon,
reciprocal square root, broadcast, and multiplication by input and weights.
Divide by the logical row length, excluding zero padding. CENT `RED` only sums
one slot at a time and writes its first BF16 lane; unused lanes and a multi-slot
sum sequence require an explicit policy. Document such reference operations as
software contracts; implement an actual scalar routine before claiming hardware
RISC-V execution. See [execution contracts](execution-contracts.md#rmsnorm-complete-reference-contract-and-physical-export-questions)
for examples and bounded export TODOs.

## Activation reference profiles are not LUT emulation

The fabricated chip's supported default functions are sigmoid, tanh, GELU,
ReLU, and leaky ReLU (JSSC pp296–297, §IV). The first three use lookup tables;
ReLU variants use circuits. The example sigmoid table covers
`[-8,7.96875]` with 512 addresses, with interpolation after 24-bit fixed-point
conversion. Do not generalize this example into a universal GELU table or
out-of-range behavior. Native SiLU is not established by that list; CENT
Figure 10(c) composes sigmoid and multiplication.

A mathematical reference may instead provide stable SiLU, Gaussian-CDF GELU,
and explicitly selected tanh-approximate GELU. These implement model formulas,
not the chip's LUT error, AFid encoding, or timing. Likewise, a max-subtracted
reference softmax is not a physical `EXP` instruction: CENT describes a
polynomial exponent accelerator and leaves scalar division and data movement
to other components. See [activation execution contracts](execution-contracts.md#activations-and-softmax-mathematical-reference-versus-chip-functions)
for official formula sources, masking policy, and original-code limitations.

**TODO — activation hardware interface:** define which MAC result is consumed,
which mode/LUT row selects the function, numeric conversion/interpolation and
out-of-range rules, and how `RD_AF` packs results into Shared Buffer. Existing
MRS/ACTAF evidence answers the mechanism; it does not answer these exact
operands and formats. These export questions do not block separately named
and tested mathematical reference functions.
