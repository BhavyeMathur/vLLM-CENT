# Project state

Last synchronized: 2026-09-18
Code snapshot: working tree based on `1aa7d1b`

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
work to CENT. A reusable raw runtime manifest and the first functional-executor
slice now exist. The vLLM worker/runner, model tensor loader, persistent
KV-cache runtime, and complete model executor are not implemented. A PyTorch
backend may become one useful entry point, but it would not make vLLM
compatibility automatic; vLLM still needs explicit scheduling, KV-cache,
worker, and result-transfer integration.

## Implemented and verified

- Typed instructions cover the paper ISA, use descriptive Python operands, and
  retain the paper names in documentation and rendering.
- Hardware, addresses, instructions, programs, and renderers perform structural
  validation without depending on Llama.
- Instruction dataclasses remain pure data. Generic validation and renderers
  use independent external single-dispatch functions, with completeness tests
  against the typed instruction catalog.
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
- `CentExecutable` pairs a typed program with reusable raw DRAM, Shared Buffer,
  and Global Buffer regions plus separate named input/output bindings. Request
  values stay separate so one executable can be reused; mutable values may keep
  one name across both directions.
- Global Buffer capacity is explicit in `CentHardwareSpec`; generic Global
  Buffer and bank-register addresses live in the model-independent `cent/`
  layer.
- The functional simulator has componentized initialization-aware state,
  Python-float reference math, structured failures, whole-program preflight,
  immutable prepared writes, and atomic cross-region commits. Host conversion,
  stored-value transport, and arithmetic-result rounding are separate numeric
  boundaries.
- Simulator instruction semantics live in a separate registry that binds each
  supported type's preflight rule to its effect kernel. Kernels, semantic
  resolution, and sequential execution are separate modules.
- Optional summary events are emitted only for committed instructions and name
  their exact typed physical read/write regions. Low-level state and execution
  APIs remain private to the simulator package.
- Functional kernels are implemented for `WR_SBK`, `RD_SBK`, `WR_GB`,
  `COPY_BKGB`, `COPY_GBBK`, `EW_MUL`, and `ACC`. Every other opcode fails
  preflight instead of receiving guessed semantics.

As of this snapshot, all 205 unit tests pass on Python 3.14. Ruff and strict
mypy pass, overall branch coverage is 98%, and the simulator package has full
statement and branch coverage. These checks establish typed structural and
numerical behavior for the supported simulator slice, not for a complete
lowering or model block.

## Not implemented yet

- The compiler does not yet produce runtime manifests that load weights,
  activations, KV-cache data, or request values into planned model addresses.
- Several public lowering functions emit only structural stages because result
  packing, reductions, or target runtime calls are not defined.
- One Llama block is therefore not numerically executable even though it
  produces a structurally valid program.
- The functional runtime does not execute the full mixed near-bank, PNM, and
  CXL instruction stream.
- No vLLM platform, worker, model runner, or KV-cache connector calls this
  compiler.
- No cost model or search procedure chooses layouts for latency, bandwidth,
  capacity, or parallelism.

## Immediate milestone

Make one RMSNorm invocation numerically complete on a small, hand-checkable
vector using the functional executor. The reviewed machine foundation and raw
materialization path are complete. The compiler and future logical runtime
contract must next account for every input value, partial sum, final reduction,
scale, output lane, and padding lane.

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
5. **Logical tensor bindings.** Raw runtime spans now exist. Which reusable
   typed layouts own model weights and request data, perform physical packing,
   establish zero padding, and interpret raw accumulator results?

The paper/AiM copy-instruction operand mismatch is documented in the AiM
research note. The current policy keeps a paper-oriented typed instruction and
handles the simulator's explicit bank operand in its target adapter.

## Planned order of work

1. Resolve the blocking ISA and logical-layout contracts with the CENT and
   simulator owners; update code-level `TODO` comments and tests with each
   answer.
2. Complete and numerically validate RMSNorm.
3. Apply the same result-packing and runtime-binding contracts to GEMV,
   attention, softmax, residuals, and the feed-forward network.
4. Make one Llama block numerically match a trusted host implementation.
5. Define the vLLM worker/runner boundary for GPU prefill and CENT decode.
6. Add a cost model and layout search only after correctness measurements exist.
