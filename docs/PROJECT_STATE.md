# Project state

Last synchronized: 2026-10-17

## Goal

Compile one model decode step for PIM hardware, while keeping model blocks
identifiable. vLLM will own request scheduling and sampling. GPU prefill followed
by CENT decode is the intended serving path; that integration is not implemented.
See [MULTI_LEVEL_COMPILER.md](MULTI_LEVEL_COMPILER.md) for the stages and API examples.

## What works

- One IR (`vllm_cent.ir`) serves every frontend. The compiler separates tensor
  math, tensor-parallel splitting, selecting supported device operations,
  assigning storage, and emitting CENT instructions. Each stage checks its
  input. Manual mappings and the default mapper use the same checks.
- A PyTorch frontend exports models written with `aloi` layers and ops through
  `torch.export` and imports them into the IR. Models annotate semantic axes;
  a per-model parallel plan splits them into one SPMD program with derived
  all-reduces. `aloi-compile --model <config.json> --run <run.yaml>` prints the
  IR of each stage for the example Llama 2 / 3.1 70B decoder layer.
- Addition, elementwise multiplication, copy, and contiguous reshape compile and
  execute on logical device zero. Shapes are static, and addition/multiplication
  require equal shapes. Storage reuse preserves requested inputs and intermediate
  outputs. Padding lanes are initialized to zero. Compiled programs accept new
  tensor values without recompilation.
- With `ExecutionContract.REFERENCE`, bias-free Linear (`x @ weights.T`), RMSNorm,
  ReLU, sigmoid, tanh, SiLU, both GELU formulas, and last-axis softmax execute
  numerically. Linear processes each input vector through bounded matrix tiles.
  RMSNorm computes its own mean square and scale. Softmax rejects NaN, positive
  infinity, and rows whose entries are all negative infinity.
- Rotary position embedding (`aloi.rope`: NeoX and GPT-J pairing, partial
  rotary) executes through one reference service, `ReferenceRotaryEmbedding`,
  which looks up each token's table row and rotates its heads in software.
  Positions may be int32/int64 tensors; they are bound as exact scalar values.
- `compile_graph` fuses decomposed `gqa_qk → softmax → gqa_sv` chains into the
  supported attention when a reshape can change their layout (one decode token,
  or one KV head) and every tensor is whole, i.e. without tensor parallelism.
  PyTorch-written decode attention, with or without a rotated query, matches
  PyTorch.
- `models.llama.build_feed_forward_graph` composes RMSNorm, gate/up projections,
  SiLU and multiplication, a down projection, and the residual addition.
- `graphs.build_attention_graph` computes attention over supplied rank-four
  Q/K/V tensors. It supports batches, grouped query heads, separate value widths,
  additive masks, and causal masks with an explicit query-position offset.
  It returns separate heads. Keys and transposed values are reused per KV head;
  one row of attention scores and matrix scratch are reused across queries.
- Reference services explicitly initialize/read selected accumulators and perform
  packing and nonlinear math. They have distinct instruction types. Physical
  CENT and AiM serializers reject them instead of pretending they have hardware
  encodings. All graph inputs and scratch must fit in the Shared Buffer.
- The functional simulator checks supported data movement and arithmetic, faults
  on unwritten inputs, and commits each instruction's writes together. Disabling
  tracing discards completed write records rather than retaining every temporary
  value. Its Python-float results validate data flow, not hardware precision or
  timing.
- Optional native MLIR parsing, verification, and optimization support addition
  and multiplication through registered `func`/`arith` dialects. Imported
  functions use the same compilation stages as PyTorch-imported and directly
  constructed ones.
- The older typed CENT instructions, structural Llama compiler, and AiM timing
  serializer remain available. A structurally valid Llama trace is not evidence
  of an executable numerical model.

Validation: **616 tests pass**, including independent scalar oracles, native
MLIR, and the external PyTorch SDPA oracle (the MLIR and SDPA tests skip when
IREE or the torch interpreter is unavailable). Strict mypy passes. Coverage and
format checks were not re-run after the IR merge. These checks cover the
supported reference calculations, not physical hardware or complete model decode.

## What does not work yet

- There is no physical export for the reference services. Register selection,
  result packing, scalar-program arguments, completion rules, and hardware
  precision still need verified target definitions. Hynix MRS latch selection
  is established; its complete command encoding is not. See
  [execution contracts](research/execution-contracts.md).
- No persistent weights, KV-cache updates, dynamic shapes, worker integration,
  or complete model decode are implemented. Current scratch owns DRAM row zero;
  it must not be mixed with long-lived DRAM data without a shared allocator.
- Tensor-parallel IR is produced, but execution is single-device: `legalize()`
  lowers one rank's program, and all-reduce has no device lowering. Distributed
  execution is not implemented.
- There are no general semantic broadcasting, reduction, transpose, or strided
  view operations. Internal reference helpers provide only the specific steps
  needed by the supported operations. Batched Linear repeats GEMV; it is not
  an optimized matrix-matrix kernel.
- Attention does not include Q/K/V projections, RoPE, cache management, head
  merging, or the output projection. The feed-forward and attention graphs do
  not yet form a complete transformer block.
- PyTorch-imported functions reach CENT instructions only when every operation
  has a device lowering and every tensor is fp32 (RoPE positions excepted).
  KV-cache update has no device lowering yet. Decomposed attention runs only
  through the fusion adaptor: other layouts and tensor-parallel heads stay
  unsupported. RoPE has no physical form: the paper does not say which of its
  steps run on PIM, PNM, or RISC-V.
- Native device/target MLIR dialects are not implemented. Device and target
  passes remain typed Python.

## Next work

1. Compose model-specific projections, positional encoding, head merging, and
   output projection around attention; compare a complete block with an
   independent numerical implementation.
2. Introduce persistent weights and a shared DRAM allocator, then define KV-cache
   updates and repeated decode before connecting vLLM workers.
3. Resolve physical encoding, precision, reduction, and scalar-runtime questions
   before exporting reference operations to hardware. Keep the column-address
   units audit from the
   [SK hynix research](research/sk-hynix-aim.md) explicit.
4. Lower KV-cache update, then the decomposed attention steps one by one
   (replacing the fusion adaptor), so the imported Llama decoder layer
   compiles end to end; then add native device/target MLIR passes.
5. Connect the external mapper through `MappingProvider` once a numerically
   correct baseline can be measured. Cost models and search belong to that team.
