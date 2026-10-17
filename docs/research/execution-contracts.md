# Execution contracts: what the original CENT reference resolves

The original CENT Python simulator supplies enough detail to build a small,
explicit **reference execution target** for matrix-vector multiplication (GEMV).
It does not supply the missing physical Shared Buffer interface. Treat these
as different questions: reference arithmetic can proceed under a documented
software contract while hardware compatibility remains unproven.

This note supplements [SK hynix AiM](sk-hynix-aim.md) and
[AiM timing compatibility](aim-simulator.md). Source findings are separated from
local reference rules below. The implemented contract is recorded in the
[decision register](../decisions/README.md#explicit-reference-computation-and-physical-export).

## Sources and scope

Inspected local original CENT revision
`4c2aac6ce6aa3ddfe1944fc9c5d656b90d3fc7d2` and its AiM timing submodule revision
`2eb1ee04f4e8f3c255c9926e7721081726aa4fe4`. References below use repository paths
and line numbers from that snapshot. The local repositories contain unrelated
untracked artifacts; this investigation did not change them.

- [CENT paper, version 3](https://arxiv.org/html/2502.07578v3), §§4.2–4.3,
  §5.6, Tables 2–3.
- [Original functional primitives](../../../CENT/cent_simulation/aim_sim.py),
  especially lines 4–21, 140–183, and 313–315.
- [Original GEMV compiler/executor](../../../CENT/cent_simulation/TransformerBlock.py),
  lines 869–928.
- [Original Llama execution](../../../CENT/cent_simulation/Llama.py), lines
  83–101 and 348–352.
- [AiM request grammar](../../../CENT/aim_simulator/src/base/request.h), lines
  89–115 and 253–279; [request expansion](../../../CENT/aim_simulator/src/memory_system/impl/aim_DRAM_system.cpp),
  lines 273–318.

The functional Python API receives tensors, channels, and latch indices
explicitly. Its emitted trace omits several of those arguments. The timing
simulator consumes that trace and schedules requests; it is not a numeric
implementation of the Python API or of every paper instruction.

## Facts established by the paper

The paper describes 32 accumulation registers per near-bank processing unit.
`MAC_ABK` and `RD_MAC` name `Regid`, but `WR_BIAS` does not. Its compilation
example initializes accumulators, computes, and reads results. It does not
resolve the omitted bias selector or result packing.

`RED` sums 16 BF16 inputs from one 256-bit slot into one BF16 result in
that destination slot's first element. Figure 7(b), PDF page 7, explicitly
shows `Rd = sum(Rs[i], i=0..15)` and an adder tree; sum is not an inference
from the word “reduction.” The remaining elements are unspecified.
`EXP` acts per lane using a tenth-order Taylor approximation. `RISCV` starts
execution at `PC`; cores access the Shared Buffer with byte addresses and
16-bit loads/stores. Calling convention, completion, and numeric details are
not supplied. `OPsize` expands instructions across consecutive slots/columns.

These statements establish more than “PNM is unsupported,” but not a complete
executable interface. In particular, a sum across several source slots needs
another reduction stage; one `RED` must not silently mean a whole-vector sum.
[Source: CENT §§4.2–4.3, Figure 7(b), and §5.6.](https://arxiv.org/pdf/2502.07578)

## What the original functional code answers

| Question | Verified behavior in the original Python reference | What remains unresolved |
| --- | --- | --- |
| Which register receives bias? | `WR_BIAS(..., latch_index, bias, ...)` assigns `bias[bank]` to that latch in every bank of the explicitly supplied channel (`aim_sim.py:140–147`). | Paper `WR_BIAS` has no latch argument. Keep the destination explicit in the compiler and reference executor. A physical adapter must implement its selection; JSSC documents MRS-selected accumulation, but not the complete CENT bias-write encoding. |
| Is bias scalar broadcast? | It is a bank-indexed list, so banks can receive different initial values. The function updates one channel; other channels require additional calls. | There is no Shared Buffer source read. Reusing one bank vector across a channel mask, channel-specific bias arrays, source footprint, and physical lane order are not specified by this API. |
| Does MAC reset automatically? | No. Global Buffer mode performs `latch[latch_index] += MAC(A, B)` for each burst and each bank (`aim_sim.py:149–160`). A later call continues from that value unless `WR_BIAS` overwrites it. | Exact physical rounding and accumulation order need their own target definition. |
| Does reading clear the register? | `RD_MAC` only appends latch values to a list (`aim_sim.py:174–183`); there is no clearing assignment. | A physical read-side policy still needs confirmation. The returned Python tensors may alias latch state; see the probe below. |
| What is readback order? | Within one channel, list index equals bank index, including banks not used for logical output. | No `Rd`, Shared Buffer write, cross-channel packing, output byte width, or slot footprint appears in `RD_MAC`. Host-side output assembly is not evidence of hardware packing. |
| How does neighboring-bank MAC behave? | Bank pairs `(0,1)`, `(2,3)`, etc. multiply; only each even bank's selected latch is updated (`aim_sim.py:162–172`). Odd latches retain their prior values and are still returned by `RD_MAC`. | The target needs an explicit rule for which returned results are meaningful. Summing every latch is correct only when the irrelevant latches were initialized appropriately. |
| Are inactive banks skipped? | Global Buffer MAC visits every bank, irrespective of logical matrix size. GEMV pads bias entries for unused banks with zero and copies only `left_banks` results into its output accumulator (`TransformerBlock.py:877–899, 919–928`). | This does not establish a bank mask. The reference's initially zero-filled DRAM hides missing writes to unused banks; our strict executor must initialize every consumed scalar explicitly. |
| Are values BF16 accurate? | DRAM and Global Buffer storage use `torch.zeros` with no explicit dtype (`aim_sim.py:9,17`). `MAC` computes elementwise multiplication followed by `sum()` (`313–315`). The inspected runtime uses `torch.float32` by default. | This is not a BF16 arithmetic specification or bitwise hardware oracle. Default dtype, tensor reduction order, and in-place tensor updates matter. |
| How many latches exist? | `args.reuse_size` creates the latch list (`aim_sim.py:10`). The interactive parser defaults to 2 (`utils.py:22`); the trace runner defaults to 32 (`run_sim.py:16`). | These are configurations, not evidence resolving the fabricated-chip versus CENT register-count discrepancy. |

### GEMV output assembly and continuation

`Vector_Matrix_Mul_weight_pim` keeps a Python `accumulator` array. Its output
index is

`(channel * num_banks + bank) * matrix_col_per_bank + vector_index_per_bank`.

Thus each bank's assigned output columns are adjacent in the final array,
even when execution visits one column from many banks at a time. Readback bank
order alone is not the logical tensor order for this mapping.

For a vector spanning multiple DRAM rows, the loop reads a partial result
back into this array and supplies it as bias for the next row. This proves a
working reference strategy for carrying partial sums; it does not require
that every partial sum stay in a device register across all rows. The reuse
variant batches several latch initializations, MACs, and reads while reusing
the same Global Buffer input (`TransformerBlock.py:883–928`).

The reference rounds `op_size` up to full bursts but passes `vector_tmp`
directly to `WR_GB` (`889–892`, `909–911`). The latter assigns it to a slice
whose length is `op_size * burst_length` (`aim_sim.py:204`). This is not a
complete arbitrary-tail padding algorithm: a short final vector can cause a
shape mismatch rather than automatically fill zeros. Do not claim arbitrary
shapes are covered merely because the reference computes `ceil(length/burst)`.

### Small execution probe

A read-only experiment imported the unchanged `PIM` class using
`CENT/.venv/bin/python` (PyTorch 2.14.0) and wrote its empty trace to a temporary
file. Test geometry was two channels, four banks, two latches, two scalars per
burst, and one four-scalar row. This is a test setup, not physical geometry.

For channel 0, bank vectors were `[1,2]`, `[3,4]`, `[5,6]`, `[7,8]`; the Global
Buffer held `[10,20]`. Biases `[100,200,300,400]` were written to latch 1.

- First MAC/read returned `[150,310,470,630]` in bank order.
- Channel 0 latch 0 and channel 1 latch 1 remained `[0,0,0,0]`.
- A second MAC without bias initialization returned `[200,420,640,860]`.
- The *earlier returned list* then also showed `[200,420,640,860]`: its tensor
  objects aliased the mutable latches. A new executor should capture scalar
  values when preparing readback rather than copy this incidental aliasing.
- Reinitializing latch 1 to `[1,2,3,4]`, then neighboring-bank MAC, returned
  `[12,2,86,4]`: `1 + (1*3 + 2*4) = 12` and
  `3 + (5*7 + 6*8) = 86`; odd-bank values stayed unchanged.

These results verify this local Python implementation. They do not test
physical hardware, Shared Buffer result writes, or complete Llama inference.

## What timing and host math do not answer

The AiM timing request has a commented register/thread index, but the actual
`WR_BIAS` and `RD_MAC` grammar lists only a GPR address and channel mask
(`request.h:110–111, 253–279`). The request expansion reads the persistent
vector-source control setting, advances burst columns, and distributes
requests to selected controllers (`aim_DRAM_system.cpp:273–318`). Neither
request dispatch order nor its GPR address establishes numerical output
packing in CENT Shared Buffer.

The original functional trace emitter also writes a potentially much wider
channel mask while its Python method updates just the explicit `channel`
(`aim_sim.py:140–147,149–160,174–183`). Trace traffic and the individual Python
call cannot be treated as the same operation without accounting for the
higher-level channel loops.

The original Llama path performs reductions and nonlinear work directly on
host tensors: it sums `RD_MAC` results in Python and calls `torch.rsqrt`
(`Llama.py:83–101`); softmax calls `torch.exp`, `torch.sum`, and division,
then writes tensors back to DRAM (`348–352`). No `RED`, `EXP`, or `RISCV`
primitive implementation was found in `cent_simulation/*.py`. These expressions
provide mathematical intent, not executable instruction buffer layouts,
RISC-V binaries, completion events, or approximation/rounding rules.

## Implemented reference profile

`ExecutionContract.REFERENCE` selects these local software rules. They are
implemented in `cent/reference.py`, `simulator/reference.py`, and
`pipeline/reference_codegen.py`; they are not physical encoding claims:

1. **Explicit accumulator selection.** Give initialization, MAC, and readback
   the same channel/bank/register identity. The original Python API already
   has this selector. Validate register IDs against the chosen target's
   capacity; do not force every internal operation to register zero. A physical
   adapter may emit a mode-register write followed by a command, but must
   reject selections it cannot preserve. Starting with one channel is a
   scheduling simplification, not a restriction imposed by missing bit encodings.
2. **Declare initialization and readback layout.** `InitializeAccumulator`
   supplies one scalar to every bank in selected channels; unbiased Linear uses
   zero. Per-bank bias arrays from the original reference are not exposed by
   this first service. `ReadAccumulator` packs channels in channel-set order,
   then increasing bank order, and zeroes final slot padding. The pipeline uses
   one channel. These are local packing rules because the original reference
   returns a list, not Shared Buffer bytes.
3. **Use a named reference numeric policy.** Continue existing Python `float`
   arithmetic for dataflow checks, with declared sequential reduction order;
   compare against tiny integer-valued reference cases where rounding does
   not obscure errors. Label any PyTorch comparison tolerance explicitly.
   Do not call the profile BF16 accurate or equate simulated scalar capacity
   with physical 256-bit packing of Python floats.
4. **Initialize all consumed state.** Every selected bank receives bias and
   complete input data, including dummy zero weights for inactive banks and
   zero-filled burst tails. Global Buffer tails must also be written. Only
   the planned logical output rows are returned to the caller. Retain strict
   unwritten-memory faults; do not copy reference-wide zero initialization.
5. **Preserve register continuation and read snapshots.** MAC adds to the
   existing selected register; only bias writes reset it. Readback does not
   clear it and copies scalar values. A multi-row input may retain register
   state or round-trip partial sums through explicit bias initialization;
   either must be represented in the lowering plan.
6. **Name reference scalar operations honestly.** `ReferenceReduceSum` sums
   the specified logical prefix, not one paper `RED` slot. `ReferenceRmsScale`
   defines Python division and reciprocal square root under the reference math
   policy; `ReferenceBroadcast` repeats its scalar and zeroes padding. None is
   rendered as `RED` or a guessed `RISCV PC`. Physical scalar execution remains
   a separate target task. Mathematical activation/softmax services use a
   separate reference contract; see the activation section below.

Before claiming the GEMV path complete, test two consecutive MACs, bias reset,
readback snapshot independence, nonzero source columns, multi-row reduction,
output reordering, inactive banks, final-burst padding, and faults without
partial writes. The smallest exact oracle is two matrix rows dotted with a
short vector, followed by a case whose vector crosses a DRAM row boundary.

## Scalar scale expansion is a separate operation

The original RMSNorm implementation computes a scalar `norm`, then executes
`torch.full(self.x.shape, norm)` and stores the resulting repeated vector
(`Llama.py:100–102`; repeated at `450–452`). This is explicit **host scalar
expansion**, not evidence that an existing CENT copy or multiply instruction
broadcasts a scalar across lanes.

A concrete example: a computed slot `[s, 0, 0, 0]` is not yet a four-element
scale vector `[s, s, s, s]`. Copying the slot preserves its lane positions.
Elementwise multiplication by `[1, 1, 1, 1]` also leaves it `[s, 0, 0, 0]`.
Changing the tensor's shape does not create three additional copies of `s`.
A channel broadcast copies a complete vector to other channels; that is a
different operation from duplicating a scalar within the vector.

These conclusions follow from the existing copy and multiply kernels in
[`simulator/kernels.py`](../../src/vllm_cent/simulator/kernels.py): they copy
whole source tuples or combine matching positions. Consequently, `Multiply`,
`Copy`, and `Reshape` can support exact-shape scale multiplication, and
`Multiply(x, x)` can compute squares, without claiming to implement the scalar
expansion needed by complete RMSNorm.

Options for a later expansion stage:

- **Already repeated input:** require an exact-shape scale tensor as an explicit
  runtime input. This is honest for a scale-multiply suboperation, but does not
  make internally computed RMSNorm end to end: another component still must
  produce that tensor.
- **Explicit reference broadcast operation:** add a semantic scalar-to-shape
  operation and a declared reference-only executor. Record its host execution
  and transfers if that is its implementation. Do not hide it inside tensor
  binding or present it as physical instruction lowering.
- **Device fill routine:** implement a specified RISC-V routine that loads the
  scalar and writes all requested lanes. This needs a real entry point,
  calling convention, completion rule, and padding behavior before export.
- **Future GEMV-based expansion:** after GEMV/readback are numerically defined,
  a matrix with one coefficient per output can multiply the scalar into many
  repeated results. This is an explicit alternative lowering requiring input
  packing, complete dummy-bank data, result packing, and a cost comparison;
  it is not available simply by enabling elementwise multiplication.

Recommendation: represent reduction, scalar computation, and broadcast as
explicit operations with declared reference semantics. This permits an
end-to-end RMSNorm reference implementation now. Copy/reshape must still not
silently broadcast. Physical export must either lower each operation to a
specified device routine or reject it; a working reference operation is not
proof of an executable RISC-V binary.

## Questions for the physical target owner

- What selects the `WR_BIAS` destination register when the instruction lacks
  `Regid`? Is there persistent control state, an implicit register, or a missing
  operand? What is the bias source footprint and channel broadcast rule?
- What scalar format does `RD_MAC` store, in what bank/channel order, with what
  alignment, stride, and total number of slots? Are reads destructive?
- Which accumulator count and numeric pipeline does this target implement?
  Specify product/accumulator precision, rounding points, special values,
  and continuation across bursts/rows.
- Are there bank masks or narrower MAC instructions? If not, which dummy-bank
  initialization is required and how are inactive results suppressed?
- What happens to unused `RED` output lanes? How are multiple slots reduced to
  one scalar? What approximation range, evaluation order, and special-value
  rules define `EXP`?
- Which RISC-V binaries/PCs exist, how are `Rs`, `Rd`, and `OPsize` passed,
  how is completion observed, and which Shared Buffer bytes may be modified?

The reference simulator removes several *software-design* uncertainties. It
cannot remove physical target questions by itself. Keeping those categories
separate permits real numerical progress without presenting a chosen reference
contract as discovered hardware behavior.

## Compiler interface versus final instruction encoding

An omitted operand in the published `WR_BIAS` assembly does **not** prevent the
compiler from representing an explicit accumulator destination. Keep that
meaning in a typed initialization operation, alongside explicitly targeted
MAC and readback operations. The semantic `LinearOperation` remains a tensor
calculation; it should not acquire register or mode-register fields.

The recommended architecture is logical accumulator ownership in device
lowering, target-aware assignment of bank/channel/register resources during
mapping, and explicit initialize/accumulate/read commands before final target
encoding. A target adapter can realize selection through an operand, a mode
write followed by a write command, or a verified restricted-register schedule.
These are alternatives, not assertions that a particular target supports them.
It must reject a destination it cannot realize rather than drop the selector.

Our existing `MacAllBanks.operand_source` provides a local example of this
separation: the typed field becomes an AiM CFR0 write followed by `MAC_ABK`
(`src/vllm_cent/cent/aim.py:225–236`). A compiler operand need not correspond
one-for-one to a printed instruction operand.

A register-zero-only schedule is an optional small test, not an internal
representation requirement. An explicit-register reference executor can match
the original Python latch API now. Physical export still needs verified selection, initialization, and
readback rules. Resource limits must inform mapping before final serialization:
a target with two latches cannot directly realize a schedule requiring 32
simultaneously live physical accumulators. Serialization, rescheduling, or
spilling is possible only when its required read/restore operations are known.

### Audit of the public timing simulator's apparent selectors

The local AiM request declares `thread_index` (0 or 1) at
`src/base/request.h:110–111`, but no opcode lists it among its legal fields.
The parser checks the exact field count (`aim_trace.cpp:177–207`), so appending
an index to `WR_BIAS` is not supported. The accepted bias fields are only a GPR
address and channel mask (`request.h:253–260`). The same loss of register
selection affects MAC/readback trace records.

Its only implemented configuration registers are broadcast operand source,
elementwise bank group, and activation mode (`aim_DRAM_system.cpp:36–40,
120–130`). The controller's `TMOD` toggles register-access mode
(`AiM_dram_controller.cpp:198–214`); it does not select a numbered latch.
`WR_BIAS` becomes `WRMAC16` in the timing model, without numerical accumulator
state (`GDDR6.cpp:364–379,998–1015`). The existing adapter's register-zero
restriction is therefore a local compatibility boundary, not proof that the
timing model physically selects latch zero.

### Additional hardware evidence and its limits

[Hynix Hot Chips 2022 slides 6–7 and 11](https://www.hc34.hotchips.org/assets/program/posters/hc34.SKhynix.YongkeeKwon.v03.pdf)
identify `WRMAC` (also called `WRBIAS`) as a MAC-register write and mark it as
requiring a special mode-register setting for command recognition. Command
recognition and selection among result latches are different mechanisms; the
footnote alone does not define a latch selector. Slide 11 shows a one-bit `T`
field in the controller instruction format. Its role must not be inferred
solely from the similarly named unused timing-simulator field.

A separate [SK hynix patent publication US20250173122A1](https://patents.google.com/patent/US20250173122A1/en)
describes an accumulation-latch selection signal `ALS[1:0]`, generated by a
command/address decoder, selecting among four latch circuits (Figures 35–37
and the selected-result readback discussion). This establishes a disclosed
hardware design with explicit selection. A patent embodiment is **not** a
GDDR6-AiM command bit specification and does not establish the meaning of
CENT's `WR_BIAS` or connect its bias write to a public selector encoding.

The strongest product evidence is **JSSC 2023, printed page 296, Figure 10 and
Section III**: it describes two result latches per accumulator, selects them
with a mode-register-set command, and revisits each latch when continuing its
partial sum across tiles. This is actual documented selectable accumulation
state, not merely an inference from a command name. The EV/OD accumulator
pipes on page 295 are a separate timing mechanism. [JSSC source](https://doi.org/10.1109/JSSC.2022.3200718).

The [Hot Chips 2023 deck](https://hc2023.hotchips.org/assets/program/conference/day1/PIM/HC35_SKhynix_DSM_YongkeeKwon_rev3.pdf),
printed slides 14 and 17 (PDF pages 15 and 18), also shows two `MAC_REG`s in column-first scheduling and
a matrix-vector-accumulate macroinstruction with `MAC regID` among its operands.
These support explicit register selection above physical command generation.
They do not provide the MRS address/bit encoding, its full synchronization rules,
or an explicit statement that the same selection governs bias writes, result
reads, and activation operations. Thus the narrow unresolved question is how
**each** operation accesses the selected latch on the chosen target, not
whether Hynix accumulation hardware has selectable latches at all.


## RMSNorm: complete reference contract and physical export questions

For a nonempty row `x` of logical length `n`, scale weights `gamma`, and the
operation's epsilon `eps`, the intended computation is
`sum_sq = sum(x[i] * x[i])`, `scale = 1 / sqrt(sum_sq / n + eps)`, and
`y[i] = (x[i] * scale) * gamma[i]`. Epsilon is added **after** dividing by `n`.
The original Llama reference uses `1e-5`, not a model-independent hardware
constant (`Llama.py:100–102,450–452`). A generic operation must carry its own
epsilon. Each row has its own sum and scale; flattening a batch into one sum
changes the operation.

**Established architecture:** CENT Figure 10(b) and §5.4 (PDF page 8) assign
the dot product and elementwise multiplication to PIM; square root and division
are PNM work. Section 4.2 (PDF page 6) provides RISC-V cores for square root
and inversion, byte-addressable Shared Buffer access, and 16-bit loads/stores.
It supplies no normalization calling convention or scalar-fill binary.
[Primary paper](https://arxiv.org/pdf/2502.07578).

**Established reference code:** `Llama.py:83–102` zeroes selected latches,
computes neighboring-bank dot products, reads all bank results, sums them on
the host, computes reciprocal square root, and materializes a repeated scale
tensor with `torch.full`. The second norm repeats this at `437–452`.
The reference's host execution proves the formula and dataflow, not device
instructions for the scalar steps. Odd-bank latches are included in its sum;
initializing them to zero matters because neighboring-bank MAC updates only
even banks (`aim_sim.py:162–183`).

**Implemented reference lowering:** square the input, reduce squares to one scalar
per row, compute the scalar scale, broadcast it to that row's logical shape,
then multiply by the input and gamma. Each stage owns an explicit output.
A named scalar-scale operation may combine division, epsilon addition and
reciprocal square root, provided its inputs and arithmetic order are stated.
It must not pretend to execute `RISCV PC` unless an instruction-set interpreter
or actual device routine executes that program. Define finite-value and
positive-radicand checks according to the selected reference numerical policy.

A concrete padded example uses five logical values in two four-lane slots:
`[1,2,3,4]` and `[5,0,0,0]`. Squaring and slot reductions give `30` and `25`.
The final sum is `55`; the mean is `55/5`, never `55/8`. With positive epsilon `eps=1`,
the scale is `1/sqrt(12)`. Broadcast writes five scale values and zeros the
three padding lanes. This four-lane example explains the software layout;
the CENT paper's physical slot contains sixteen BF16 lanes.

`RED OPsize=2` reduces two slots independently; it does not combine their
results. Under the reference policy, outputs might be `[30,0,0,0]` and
`[25,0,0,0]`. An explicit second stage must combine those two scalars, through
slotwise accumulation followed by reduction, scalar collection, or a declared
whole-row reduction operation. Do not silently equate that operation with one
paper `RED`. Padding must be initialized before reads, and unused result lanes
must be zeroed if later stages rely on zero padding. Zeroing those lanes is a
software choice: the paper specifies only the first result lane.

### Bounded TODOs for physical export

- **Bias/MAC/read coupling:** selectable accumulation is established by JSSC
  p296 Fig10. Obtain the actual mode-register address/bits, selection scope,
  settling/order rules, and reset value. Confirm separately which latch
  `WRMAC`, `RDMAC`, and `AF` access. Do not infer bias write-enable behavior
  solely from Fig10's selected output/feedback mux.
- **Capacity:** declare two latches for the cited fabricated AiM generation
  and 32 registers for a CENT-paper profile. Mapping must respect the chosen
  capacity and live partial sums. Restoring spilled values needs a working
  selected-register initialization path; changing an integer limit alone
  does not provide one.
- **Result layout:** choose bank/channel order, precision, slots per channel,
  padding, and whether reads clear state. The original Python reference only
  proves bank-order, nonclearing list readback within one channel. A reference
  profile can define the rest explicitly without waiting for a physical ABI.
- **Scalar routine:** identify actual code and an entry point for mean,
  epsilon addition, reciprocal square root, and fill. Specify argument locations,
  logical row length, epsilon representation, output location, completion,
  touched bytes, precision, and failure behavior. Square-root capability alone
  does not define that interface.
- **Reduction:** confirm rounding/order and unused physical output lanes;
  select a real multi-slot reduction sequence. The mathematical sum and first
  output lane are already documented and are not open questions.

These TODOs constrain hardware claims and export. They do not prevent typed,
fully tested reference execution of GEMV or RMSNorm.

## Activations and softmax: mathematical reference versus chip functions

### Primary definitions

[PyTorch GELU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.GELU.html)
defines the default function as `x * Phi(x)`, where `Phi` is the standard
normal cumulative distribution function. Its separate `tanh` approximation is
`0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))`. These are distinct functions:
the approximation choice must survive every lowering stage. Calling either
one “GELU” without recording the choice loses model meaning.

[PyTorch SiLU](https://docs.pytorch.org/docs/2.14/generated/torch.nn.SiLU.html)
defines `x * sigmoid(x)`, applied elementwise. A stable equivalent for finite
inputs uses `x/(1+exp(-x))` when `x >= 0` and
`x*exp(x)/(1+exp(x))` otherwise. This branch is a reference implementation
choice, derived algebraically; it avoids computing `exp(1000)` for `x=-1000`.
For exact GELU, `0.5*x*erfc(-x/sqrt(2))` avoids subtracting nearly equal numbers
in the negative tail. “Exact” selects the Gaussian-CDF definition; host
transcendental functions still have finite precision and are not a bitwise
hardware or PyTorch oracle. Tanh GELU must handle large finite inputs without
an avoidable `x**3` overflow exception; saturation or an overflow-safe expression
must retain the intended positive/negative tail.

[PyTorch softmax](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.softmax.html)
defines exponentiation followed by normalization along the chosen dimension.
For the reference last-dimension operation, compute each logical row separately:
`m=max(row)`, `e[i]=exp(row[i]-m)`, `y[i]=e[i]/sum(e)`. Subtracting the row
maximum is an algebraically equivalent stability choice for finite rows.
Padding is outside the logical row and must never enter the maximum or sum.
For example `[1000,1000]` gives `[0.5,0.5]`; evaluating `exp(1000)` first is
unnecessary. Producing zero padding is separate from applying softmax to an
extra zero-valued input.

### Implemented reference numerical policy

Use declared host reference math for these nonlinear functions, then convert
the result once according to the supported storage policy. Do not silently
combine a custom arithmetic policy with an unspecified transcendental routine.
The elementwise reference profile propagates NaN and defines infinity limits
explicitly: sigmoid tends to zero/one, tanh to minus/plus one, and ReLU,
SiLU and both GELU variants tend to zero at negative infinity and positive
infinity at positive infinity. These chosen limits avoid indeterminate
intermediate expressions such as infinity times zero; they are not a promise
of bitwise PyTorch special-value behavior. Test large finite inputs, zeros,
and transition points as well as these cases.

For softmax, accept finite logits and negative infinity as an explicit mask.
A masked entry contributes zero probability. Require at least one finite logit
in every row; reject an all-negative-infinity row, positive infinity, or NaN.
This is a **chosen reference policy**, not a claim about every PyTorch attention
kernel. An all-masked row has no probability distribution under the ordinary
formula: returning zeros or NaNs would require a different stated contract.
For `[0,-inf,0]`, output is `[0.5,0,0.5]`; for `[-inf,-inf]`, report failure.
Validate all rows before committing an instruction's outputs so a malformed
later row cannot leave an earlier row partially written. This guarantee applies
to one instruction; earlier completed instructions need not roll back.

### What the Hynix device and original CENT code establish

JSSC printed pp296–297, §IV and Figs11–14, document sigmoid, tanh, GELU, ReLU,
and leaky ReLU. ReLU variants use circuits; the other three use programmed
lookup-table rows and interpolation. The sigmoid example uses 512 column
addresses over `[-8,7.96875]` after conversion to 24-bit fixed point. It does
not specify the exact/tanh GELU API above. MRS selects the activation type;
LUT functions also require `ACTAF` to activate the appropriate reserved row.
Results are on the separate activation-result path. These facts do not
establish native SiLU or a rowwise softmax instruction.
[JSSC source](https://doi.org/10.1109/JSSC.2022.3200718).

Original CENT `aim_sim.py:57–58,223–249` implements a sigmoid table from
`torch.linspace(-10,10,512)`, appends an endpoint, indexes with `20/512`, and
uses its own interpolation expression. This differs from the JSSC example's
range and is not a chip-accuracy oracle. It selects a MAC latch explicitly and
returns activation results in bank order, but supplies no CENT Shared Buffer
readback format. More importantly, `Llama.py:493–510` obtains this approximate
sigmoid alongside GEMV results but then computes `torch.sigmoid(x1_aim)` for
the subsequent SiLU path. Do not cite whole-reference SiLU agreement as
validation of the lookup-table implementation.

CENT Figure 10(c–d), §5.4, shows SiLU as sigmoid times input and softmax as
exponentiation and normalization across PIM/PNM work. Original
`Llama.py:348–352` performs exp, sum, reciprocal, and repeated-vector creation
on host tensors. Neither is an executable scalar/broadcast ABI. The paper's
`EXP` uses a tenth-order approximation; a reference `math.exp` service must
not be serialized as that accelerator without an explicit approximation
contract. [CENT paper](https://arxiv.org/pdf/2502.07578).

**TODO — physical activation export:** obtain the AFid/mode encoding, selected
MAC-latch relationship, LUT contents/range and out-of-range handling,
fixed-point conversion and interpolation rounding, activation-result capacity,
and `RD_AF` precision/lane/channel packing. For softmax, additionally specify
max reduction, exponential approximation, division, masks, and the scalar/vector
transfer sequence. Until those are supplied, explicit mathematical reference
services can execute and be tested, while physical serializers reject them.
Activation support alone does not implement attention, KV-cache storage, or an
entire transformer model.

## Supplied-tensor scaled dot-product attention

### Source-backed shape and grouping rules

The [official PyTorch scaled-dot-product-attention specification](https://docs.pytorch.org/docs/2.14/generated/torch.nn.functional.scaled_dot_product_attention.html)
uses queries with head count `Hq`, keys/values with head count `Hkv`, query
length `L`, key/value length `S`, query/key width `D`, and value width `Dv`.
The output keeps the query heads and length but has width `Dv`. Default scale
is `1/sqrt(D)`. Grouped-query attention repeats each key/value head consecutively:
with four query heads and two KV heads, assignments are `[0,0,1,1]`, not
`[0,1,0,1]`. Its non-square causal mask is upper-left aligned. PyTorch accepts
broadcastable boolean or additive masks; its documented interface disallows
combining `attn_mask` and `is_causal`.

### Chosen local reference boundary

The local operation takes already-computed tensors `Q[B,Hq,L,D]`,
`K[B,Hkv,S,D]`, and `V[B,Hkv,S,Dv]`, with positive dimensions,
`Hq % Hkv == 0`, and matching batch/key/value dimensions. It returns
`O[B,Hq,L,Dv]`. This deliberately supports one batch dimension and two
additive-mask shapes: shared `[L,S]` or full `[B,Hq,L,S]`. It does not accept
arbitrary broadcast shapes or boolean-mask conventions. Query head `h` uses
KV head `h // (Hq/Hkv)`; a full mask is indexed by `h`, not that KV head.

For each batch and query head, compute `scores = Q_head @ K_head.T`, apply
scale, add the supplied mask, impose the causal restriction if requested,
normalize each score row with the declared reference softmax, and compute
`probabilities @ V_head`. Scaling happens before mask addition. Both matrix
products are explicit GEMV sequences; transposing V prepares its columns as
weight rows for the second product. `Dv` need not equal `D` or `S`.

The causal rule is `key_index <= query_position_offset + query_index`.
Default offset zero is upper-left alignment: with `L=1,S=5`, only key zero
is visible. A caller supplying the last query for a five-position sequence
must explicitly choose offset four to make all five keys visible. The runtime
must not infer that offset from unequal lengths, pretend to append cache entries,
or choose bottom-right alignment silently. Combining this causal rule with an
additive mask is a local extension, not an assertion of identical PyTorch API
behavior; compare against an explicitly combined mask when building an oracle.

Negative infinity marks excluded scores. The existing reference softmax rejects
NaN, positive infinity, and rows with no unmasked position; this remains the
local policy even if another attention implementation returns zeros for a fully
masked row. The score primitive validates every supplied mask entry, including causally
hidden entries: NaN and positive infinity are rejected. An explicitly excluded
score is replaced by negative infinity before arithmetic, so a hidden infinite
score does not become NaN through infinity plus negative infinity. Unmasked
scores follow the numeric policy and subsequent softmax checks. Zero and
negative finite custom scales are accepted and multiply scores before mask
addition; nonfinite scales are rejected.

### Lowering and review requirements

- Preserve Q, K, V, and masks until their last head/query use. Do not allow an
  output initialization to overwrite an input merely because its final graph
  use is this attention operation; its internal schedule reads repeatedly.
- Use scalar offsets for packed rows and heads, because odd widths need not
  start at a slot boundary. Zero staging tails, preserve logical output order,
  and keep padding out of softmax. Transpose must move values, not just reshape.
- Account for simultaneous score, probability, transposed-value, and GEMV
  scratch lifetimes. Reuse only after the last internal read; include all
  required slots in capacity checks before execution.
- Test `Hq>Hkv>1`, distinct batch/head masks, `L!=S`, `Dv!=D`, an offset-bearing
  single-query case, multiple query rows, masked tails, invalid rows, nonzero
  accumulator selection, and preservation of returned inputs.
- Physical export still requires implementations of packing, transpose,
  score scaling/masking, nonlinear arithmetic, and selected-register readback.
  Named reference services must continue to reject physical serialization.

This scope excludes dropout, cache ownership/append/eviction, rotary position
encoding, and Q/K/V/output projections. Supplying K and V tensors is not a
persistent KV-cache implementation. This is one attention calculation, not
whole-model inference or a complete transformer layer.
