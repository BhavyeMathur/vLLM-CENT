# Accepted design decisions

This file records decisions that affect multiple modules or tasks. Local
implementation details belong in code and tests. Change an entry when the team
revises the decision; do not preserve obsolete behavior merely for compatibility
with an early prototype.

## Sources and target differences

- The [CENT paper](https://arxiv.org/abs/2502.07578) is the primary description
  of the architecture and ISA.
- The [original CENT repository](https://github.com/Yufeng98/CENT) and the AiM
  simulator are evidence about their implementations, not automatic overrides
  of the paper.
- The compiler keeps a typed, paper-oriented instruction model. A target adapter
  translates it to a simulator or runtime ABI and must reject meaning it cannot
  encode. Target-specific fields must not leak into every generic instruction.
- Compatibility with the original prototype is not a goal. Keep an old command
  only when the current paper ISA or an actual execution target requires it.

## Language and data representation

- Compiler orchestration and the current backend are Python. Native MLIR is an
  optional frontend/toolchain during incremental migration. Registered target
  dialects and performance-sensitive passes may use C++ when needed; the
  simulator does not require a rewrite.
- Use typed, immutable dataclasses and enums for in-process structures. Protocol
  Buffers are unnecessary until a real process, language, or storage boundary
  requires a versioned wire format.
- A `CentProgram` is typed intermediate representation, not a string trace.
  Rendering is a target-specific final step.

## Package responsibilities

- `vllm_cent/cent/` owns model-independent hardware, addresses, instructions,
  validation, programs, builders, and target rendering.
- `vllm_cent/lowering/` owns reusable operation lowering. Transformer-specific
  operations live in its `transformer/` subpackage rather than being presented
  as universal neural-network operations.
- `vllm_cent/models/<family>/` owns model dimensions, tensor planning, and the
  order in which reusable operations form a model block.
- The top-level compiler API dispatches to model-family frontends. A registry is
  deferred until a second family makes its requirements concrete.

## Planning before lowering

- Layout selection and instruction emission are separate responsibilities.
- An immutable operation plan names partitions, channels, banks, addresses, and
  result regions. A lowerer validates and obeys that plan; it must not silently
  choose a different layout.
- The baseline planner may choose a simple even partition. Future cost models
  may provide different valid plans without changing operation lowering.
- Performance optimization is deferred until a numerically correct path can be
  measured. The plan interface is the extension point for that future work.

## Logical vector contract

- Every producer of a logical Shared Buffer vector writes every lane in every
  occupied slot.
- Within each physical partition, logical values come first and all remaining
  lanes are zero. Padding is per partition, so zero lanes may appear between
  groups of logical values.
- A consumer may rely on this contract. An operation that changes padding—for
  example, applying `EXP` to zero lanes—must restore zero padding before a
  later operation can observe those lanes.
- A result remains a raw capacity span when its lane order or padding is not
  proven. In particular, unresolved `RD_MAC`, activation, reduction, RISC-V,
  and CXL outputs must not be mislabeled as logical vectors.

## Correctness boundary

- Structural validation proves that instructions, operands, layouts, and
  capacities are internally valid. It does not prove that the program computes
  the requested tensor operation.
- A public operation should eventually emit the complete mathematical result.
  Until it does, the missing numerical stage must be explicit in its API or in
  a nearby source `TODO`.
- AiM is a DRAM timing target, not a functional oracle. Numerical validation
  requires a functional executor or a trusted reference implementation.

## Serving integration

- The intended first serving split keeps prefill on a GPU and offloads supported
  decode work to CENT.
- vLLM integration will require an explicit CENT platform/worker/model-runner
  boundary plus tensor, KV-cache, launch, and result-transfer contracts.
- A future PyTorch backend could make individual tensor operations callable
  through PyTorch. It would not automatically implement vLLM's scheduler,
  workers, cache ownership, or heterogeneous GPU/CENT execution.

## Multi-level compilation and mapping ownership

- One IR (`vllm_cent.ir`) serves every frontend. Each operation is a concrete
  immutable type that owns its parameters and infers its result type. The
  shared `Operation` base only holds operands, results, and a name hint.
  Adding a new operation does not extend a central enum or add optional fields
  to unrelated operations. Values are compared by identity; `legalize()` gives
  them the IR printer's names at the device boundary, where schedules and
  storage plans refer to tensors by name.
- The compilation scope is a shape-specialized decode-step graph with retained
  model-block provenance. Supported operations are listed in
  [project state](../PROJECT_STATE.md); full decode remains a roadmap gate,
  not an implemented capability.
- Semantic and sharded IR contain no physical addresses. Capability facts enter
  Device IR; geometry and physical placement enter mapping. Target IR references
  symbolic storage objects before address allocation and instruction emission.
- Manual plans and mapping providers use the same validation contract. This
  repository owns that interface and a deterministic baseline. The external
  mapping team owns cost models, metrics, scoring, and search.
- Storage reuse must preserve every input or intermediate result still needed
  by a later calculation or requested as an output. Destructive CENT ACC
  requires an explicit preservation copy when its left operand remains live.
- Logical artifacts wrap the raw runtime manifest with versioned shape and
  padding contracts. Binding fresh request data must not recompile instructions.
- Native MLIR is adopted incrementally using registered dialects. Current
  func/arith import and optimization are native; Device/Target passes remain
  typed Python until registered conversion passes are implemented.
- Keep the existing structural compiler until its replacement has equivalent
  verification. See [the architecture RFC](../MULTI_LEVEL_COMPILER.md) for the
  precise supported subset and remaining gates.

## Reference arithmetic and shape operations

- Multiplication follows the existing functional-reference bank roles. Its
  selected channel has every bank group's operands explicitly initialized;
  no correctness claim depends on globally zero-initialized simulator memory.
- Graph tensors remain in Shared Buffer storage. The first multiplication
  lowering uses sequential, row-sized DRAM scratch and returns results to
  Shared Buffer. Persistent DRAM allocations remain unsupported.
- Copy and reshape describe immutable values, not mandatory allocation or
  transfer. A same-storage result can emit no commands; empty instruction
  programs are valid when runtime bindings implement the graph.
- Reshape preserves flat element order and count. It does not imply transpose,
  strides, scalar broadcasting, or multi-rank ownership transformations.
- The native MLIR bridge supports addition and multiplication; other newly
  supported graph operations use the direct Python API until their native
  dialect mappings are implemented and verified.

## Explicit reference computation and physical export

- Preserve accumulator selection in compiler IR even when a final command table
  omits that operand. Hynix documents MRS-selected accumulation latches, and its
  higher-level matrix-vector command includes a register ID. Selection can lower
  into control-state changes rather than a one-to-one printed operand.
- `ExecutionContract.REFERENCE` explicitly enables the local accumulator,
  scalar, and packing services needed for numerical Linear and RMSNorm. The
  default contract does not silently opt a caller into those extensions.
- `AccumulatorPlacement` belongs in the mapping plan and its register must fit
  the target. Distinct Hynix and CENT register capacities remain target facts,
  not limits hard-coded into generic semantic operations.
- Reference services use distinct instruction types and opcode names. They must
  not impersonate paper `WR_BIAS`, `RD_MAC`, `RED`, or `RISCV` semantics. Paper
  and AiM serializers reject services without a verified physical translation.
- Linear means bias-free `x @ weights.T`. Leading input dimensions are executed
  as independent vectors. The baseline stages current weight tiles in exclusive
  DRAM scratch; weights are request inputs, not persistent DRAM objects.
- RMSNorm computes each row's square sum, divides by its logical width, adds the
  operation's epsilon, takes a reciprocal square root, broadcasts that scalar,
  and multiplies by input and learned scale. Padding is excluded from the mean
  and output padding is explicitly zero. Reference scalar math does not claim a
  RISC-V binary or calling convention.
- The complete physical export still requires selected-register write/read
  controls, result layout, numerical rules, and scalar routine/completion
  contracts. These questions do not block numerical reference implementation.
  [Execution contracts](../research/execution-contracts.md) records the primary
  sources and remaining questions in detail.


## Activation mathematics and model graph composition

- `ActivationKind` is a hardware-independent mathematical choice, shared by
  semantic operations and reference services. `ActivationOperation` owns that
  required choice; it adds no optional activation fields to unrelated operations.
- Gaussian-CDF GELU and tanh-approximate GELU remain separate choices through
  every stage. Reference transcendental functions do not claim Hynix LUT
  accuracy or the paper's polynomial EXP behavior.
- Pointwise reference activations propagate NaN and use documented infinity
  limits. Last-axis softmax accepts finite logits and negative-infinity masks;
  it rejects NaN, positive infinity, and rows with no finite entry. Padding
  never contributes to a row's normalization and output padding stays zero.
- Activation and softmax commands read their inputs before committing writes,
  allowing exact in-place execution after the input's last use. Requested
  inputs and intermediate outputs remain protected by the mapping verifier.
- Llama's executable feed-forward graph lives in `models/llama/graphs.py`.
  It describes tensor operations and explicit RMS epsilon without hardware
  addresses. Its weights remain ordinary runtime inputs; the older structural
  compiler is unchanged and does not gain these numerical guarantees implicitly.


## Attention over supplied tensors

- `AttentionOperation` computes scaled dot-product attention from Q/K/V inputs;
  it performs no projection, RoPE, dropout, cache mutation, or output projection.
  Shapes are explicit: Q `[B,Hq,L,D]`, K `[B,Hkv,S,D]`, V `[B,Hkv,S,Dv]`, and
  output `[B,Hq,L,Dv]`. Batches must match; `Hq` must be divisible by `Hkv`.
  Consecutive query-head groups share one KV head. Heads stay separate on output.
- Score scaling defaults to `1/sqrt(D)`. A caller may supply any finite scale,
  including zero or negative values. Mask input is either shared `[L,S]` or full
  `[B,Hq,L,S]`, with finite additive bias or negative infinity for exclusion.
  Boolean masks and arbitrary implicit broadcasting are not part of this API.
- Causal masking permits key index `k <= query_position_offset + q`. Offset zero
  is upper-left alignment, including non-square matrices. A decoding suffix
  must explicitly supply its position. Causality intersects an additive mask;
  all-masked rows fail under the existing softmax reference policy.
- Code generation reuses existing GEMV/MAC lowering twice per query. It reuses
  packed keys and transposed values within a KV-head group and streams score
  rows through a shared workspace. It does not replace attention with one host
  function, reserve all query/key scores at once, or claim FlashAttention.
- Attention conservatively requires output storage separate from its operands.
  A mapping-selected accumulator is reused sequentially for the two products;
  initialization and scratch capacity remain explicit and checked. Persistent
  KV allocation and hardware export are separate milestones.
