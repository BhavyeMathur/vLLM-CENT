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

- The compiler is Python. Compilation is control-plane work, while tensor
  arithmetic belongs on CENT or in an executor; C++ would add integration cost
  before compilation speed is shown to matter.
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
