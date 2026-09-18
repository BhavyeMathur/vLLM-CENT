# Project state

Last synchronized: 2026-09-17  
Code snapshot: `d76b8c1` (`Added tensor/vector layout planning API`)

## Goal and ownership boundary

The long-term goal is to run vLLM-supported models on CENT hardware. This
repository owns the compiler-side software interface: it turns model
operations, tensor bindings, and hardware-aware layout plans into a typed and
validated `CentProgram`.

The intended serving path is:

```text
vLLM request and decode scheduler
              |
     future CENT worker/runner
              |
      model-family frontend
              |
     reusable operation lowering
              |
       typed CENT program
              |
  runtime bindings and target adapter
          /                 \
AiM timing simulation   functional execution
```

The current integration plan keeps prefill on a GPU and sends supported decode
work to CENT. The vLLM worker/runner, tensor loader, execution runtime, and
functional emulator are not implemented here yet. A PyTorch backend may become
one useful entry point, but it would not make vLLM compatibility automatic;
vLLM still needs explicit scheduling, KV-cache, worker, and result-transfer
integration.

## Implemented and verified

- Typed instructions cover the paper ISA, use descriptive Python operands, and
  retain the paper names in documentation and rendering.
- Hardware, addresses, instructions, programs, and renderers perform structural
  validation without depending on Llama.
- The AiM adapter renders the DRAM-side subset accepted by the timing simulator
  and rejects operands or instructions that its trace format cannot represent.
- Reusable lowering exists for data movement, GEMV, elementwise operations,
  normalization stages, and transformer attention stages.
- Model-specific Llama code plans storage and orchestrates one batch-one decode
  block. Tests cover small examples and the Llama 3 8B and 70B dimensions.
- Layout choice is represented by immutable plans supplied to lowerers. The
  current planner uses a simple baseline policy; an optimizer can replace that
  policy without rewriting instruction emission.
- Logical Shared Buffer vectors have an explicit physical layout and a
  zero-padding contract. Raw output spans remain raw when the ISA does not yet
  establish their lane order or padding.

As of this snapshot, Ruff and strict mypy pass. All 117 unit tests pass on
Python 3.14, with 98% branch coverage. These checks establish typed structural
behavior, not end-to-end numerical correctness.

## Not implemented yet

- The compiler does not load weights, activations, KV-cache data, or request
  values into their planned addresses.
- Several public lowering functions emit only structural stages because result
  packing, reductions, or target runtime calls are not defined.
- One Llama block is therefore not numerically executable even though it
  produces a structurally valid program.
- No runtime executes the mixed near-bank, PNM, and CXL instruction stream.
- No vLLM platform, worker, model runner, or KV-cache connector calls this
  compiler.
- No cost model or search procedure chooses layouts for latency, bandwidth,
  capacity, or parallelism.

## Immediate milestone

Make one RMSNorm invocation numerically complete on a small, hand-checkable
vector. That requires the compiler and runtime contract to account for every
input value, partial sum, final reduction, scale, output lane, and padding lane.
Validate the result against a simple host implementation or an agreed
functional executor. AiM alone cannot provide this check because it models
timing rather than tensor values.

This milestone is intentionally below a whole transformer block. It tests the
same boundaries that later operations need: data loading, accumulator
initialization, `RD_MAC` packing, reduction, PNM calls, zero padding, and result
retrieval.

## Blocking questions

The affected source also contains nearby `TODO` comments. This list includes
only questions that block several operations or an end-to-end result.

1. **`WR_BIAS` accumulator selection and source.** Which accumulator does it
   initialize when the instruction has no `Regid`, and how is the zero value
   placed at its Shared Buffer source?
2. **`RD_MAC` result layout.** When several channels, banks, and registers
   produce results, which Shared Buffer lanes and slots receive each value?
3. **PNM operation contracts.** What exact operations and output layouts do
   `RED`, `ACC`, and `EXP` implement, and which RISC-V program counters provide
   reciprocal, square root, and other required scalar functions?
4. **Inactive PU groups.** How does a program disable bank groups that have no
   logical work so they cannot contribute stale values?
5. **Runtime tensor manifest.** Which component owns weights and request data,
   performs physical packing, establishes zero padding, launches a program,
   and interprets its outputs?

The paper/AiM copy-instruction operand mismatch is documented in the AiM
research note. The current policy keeps a paper-oriented typed instruction and
handles the simulator's explicit bank operand in its target adapter.

## Planned order of work

1. Resolve the blocking ISA and runtime contracts with the CENT and simulator
   owners; update code-level `TODO` comments and tests with each answer.
2. Complete and numerically validate RMSNorm.
3. Apply the same result-packing and runtime-binding contracts to GEMV,
   attention, softmax, residuals, and the feed-forward network.
4. Make one Llama block numerically match a trusted host implementation.
5. Define the vLLM worker/runner boundary for GPU prefill and CENT decode.
6. Add a cost model and layout search only after correctness measurements exist.
