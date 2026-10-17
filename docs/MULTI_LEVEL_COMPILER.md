# Multi-level PIM compiler architecture

Status: architecture accepted for incremental migration; executable addition,
multiplication, copy, reshape, and opt-in reference Linear/RMSNorm, activations,
softmax, and supplied-tensor attention implemented. A complete Llama
feed-forward subgraph is tested. This document distinguishes the implemented interfaces from
the remaining roadmap. It does not claim a complete model compiler or a
production hardware runtime.

## Compilation and ownership

The intended compilation unit is a shape-specialized **decode-step graph**.
Model blocks remain named provenance boundaries inside that graph. Compilation
may inspect data dependencies and storage lifetimes across blocks. Request
scheduling, sampling, and admission control remain with vLLM.

```text
Model frontend                          Native MLIR func/arith frontend
      |                                          |
      +------------ SemanticGraph <--------------+
                           | caller ParallelPlan
                     ShardedGraph
                           | CapabilityProfile
                       DeviceGraph
                           | MappingRequest + physical CentTargetProfile
                manual MappingPlan / MappingProvider
                           | common verifier
                  symbolic TargetProgram
                           | allocation + instruction selection
             CentExecutable + versioned logical bindings
                           | fresh per-request values
                     functional simulator
```

`pipeline/` is independent of model families. `ir_mlir/` is an optional native
MLIR adapter; `cent/`, existing `lowering/`, and the structural Llama frontend
remain available. No mapping policy or hardware geometry leaks into the model
or native semantic frontend.

## Stage contracts and verification

| Stage | Allowed information | Verifier obligations |
| --- | --- | --- |
| Semantic | SSA names, static shapes, F32 type, mathematical operation, block provenance | Unique definitions, dominance, arity, dtype/shape agreement, unique observable results, explicit RMSNorm epsilon |
| Sharded semantic | Caller-selected logical rank IDs and partition axis, global/local types, ownership intervals | Exact coverage, equal uniform shards, preserved operation semantics, no physical addresses |
| Device | PNM_ADD, PIM_MULTIPLY, COPY, RESHAPE, Shared Buffer boundary, dependencies and logical rank | Capability support, preserved lineage, legal component operation and dependency graph |
| Mapping | Physical target, operation schedule, symbolic storage assignments and exact optional constraints | Complete placements, topological order, capacity, equal alias sizes, last-use safety, no input-entry aliases |
| Target | Symbolic COPY, ACCUMULATE, and MULTIPLY, storage IDs, slot counts, provenance | Commands must implement the validated mapping; missing preservation transfers are rejected |
| Allocation/codegen | Concrete Shared Buffer slots, private scratch DRAM location, command tiling | Disjoint object intervals, target bounds, row-limited transfers, existing instruction verification |
| Artifact/runtime | Version, logical signature, physical manifest and padded binding rules | Exact names/counts, artifact consistency, complete input initialization and zero padding |

These are immutable typed Python pass boundaries. The frontend uses actual
native MLIR parsing, builders, registered `func`/`arith` verification, and
canonicalize/CSE passes through IREE's packaged bindings. **There is no native
CENT dialect or MLIR dialect-conversion pass yet.** Device and Target stages
currently execute as Python passes. An MLIR text dump is not presented as a
registered device dialect. The executable bridge rejects unknown operations,
fast-math assumptions, dynamic types, and non-F32 types.

## Operation types

`Operation` contains only the graph connections shared by calculations: name,
input names, output, and source-block label. It is abstract; each concrete type
implements its own shape check. `AddOperation`, `MultiplyOperation`, `CopyOperation`, `ReshapeOperation`, and
`LinearOperation` live in
`pipeline/operations/tensor.py`. `RmsNormOperation` lives in
`pipeline/operations/transformer.py` and alone requires an `epsilon` value.

The graph verifier checks names and dependencies, resolves input types, then
calls the operation's verifier. It has no RMSNorm branch or optional epsilon
field to police. Target legalization remains a separate decision: a valid new
semantic operation is rejected until the backend has an implementation for it.

## Capability and physical target separation

`CapabilityProfile` lists legal component operations without channel counts,
capacities, or addresses. The executable set includes addition, multiplication,
copy, reshape, Linear, RMSNorm, activations, softmax, and attention. An empty profile rejects operations rather
than creating an incomplete program. Linear, RMSNorm, activations, softmax, and attention require
`ExecutionContract.REFERENCE` on the physical profile. This explicit opt-in
permits documented software rules for accumulator access, packing, and scalar
math; it does not assert that a hardware encoder exists for those services.

`CentTargetProfile` supplies detailed `CentHardwareSpec` geometry and a
caller-provided maximum ACC tile size. Two profiles can compile the same graph
into different tilings without changing the frontend. These profiles are
reference CENT configurations, not evidence for two different physical chips.
A future target backend must add its own legal decompositions, memory spaces,
precision contracts, topology, and encoding; it must not overload this profile
with unsupported PIM behavior.

## Parallel-plan boundary

`ParallelPlan` supports identity or uniform addition/multiplication/copy partitioning along
one logical tensor axis. Every rank retains full global and local shapes plus
its exact ownership interval. These pointwise operations need no collective. Scalar identity
sharding is legal; nontrivial scalar partitioning is not.

The API does not pretend that this implements full pipeline/tensor/sequence
parallelism. Multi-rank reshape is rejected because reshaping can change which
flat elements belong to a rank; no ownership transformation is implemented. CENT artifact
emission currently requires one rank on logical device zero. Full PP/TP/SP
needs logical collective/transfer operations and topology-aware execution;
CXL routing and runtime receive matching are unresolved.

## Shared manual and provider mapping boundary

`MappingRequest(graph, target, constraints)` contains Device IR (which retains
the caller's parallel plan), physical target, and optional exact value-to-object
constraints. A `MappingProvider` returns a `MappingPlan`. A complete manual plan
bypasses the provider but passes through exactly the same verifier.

The initial plan contains an operation schedule and assignments of every value
to symbolic Shared Buffer objects. Linear operations also have explicit
`AccumulatorPlacement` entries, checked against the target register capacity.
Logical dependencies determine inclusive
lifetimes: inputs exist at entry, outputs remain live through extraction, and
an intermediate dies after its final consumer. Alias legality is checked
against these derived lifetimes, not trusted provider annotations.

The deterministic baseline counts uses once, schedules in definition order,
and reuses a dead left operand or an expired same-size object. Expiration
buckets and free lists avoid repeatedly scanning the graph. It reserves manual
storage names and generates collision-free defaults. This is a deterministic
baseline, not a complete constraint solver: a rejected set of partial
constraints may admit a different manual schedule/plan.

Graph values currently live in Shared Buffer storage. Multiplication stages
them through private DRAM scratch and gathers results back into Shared Buffer;
its workspace is temporary, not a persistent weight allocation. Required copies
are materialized in Target IR. Broader mapping must explicitly add partitioning/tiling, selected
components, transfer paths, lifetimes, and fusion proofs alongside their first
consumers and verifiers. The external team owns the analytical model, scoring,
metric definitions, and search. No placeholder costs are defined here.

## Worked executable example

For `z = x + y; out = z + y` with three-element vectors and two scalar lanes
per slot:

1. Semantic values each have `tensor<3xf32>`; block labels can differ.
2. Identity sharding preserves the types and records the one-rank ownership.
3. Capability legalization creates two PNM_ADD operations, with the second
   depending on the first.
4. If only `out` is observable, `x` dies at the first operation and `z` dies at
   the second. Both results can share `x`'s symbolic object. `y` remains live.
5. Allocation gives `x` two slots and `y` two slots. The loader writes logical
   lanes followed by one explicit zero to each object.
6. The resulting program is two ACC instructions (or four when the target
   permits only one slot per ACC). No redundant transfer is emitted.
7. Given `x=(1,2,3)` and `y=(10,20,30)`, extraction returns `(21,42,63)`.

If `x` is also observable, it cannot be overwritten. Target IR first copies it
to a separate object, then accumulates. CENT has no assumed Shared Buffer copy
instruction: codegen uses WR_SBK followed by RD_SBK through privately owned
channel 0/bank 0/row 0 scratch. Large copies are split into row-sized chunks;
each chunk is read back before scratch reuse. Scratch is safe only because
this slice has no DRAM weights or KV objects. Any extension introducing those
must put scratch into the general DRAM allocator first.

## Multiplication, copy, and reshape

`MultiplyOperation` requires equal input/output shapes and uses the same input
name twice for a square. It does not implement scalar broadcasting. A caller
can provide an explicitly repeated vector to multiply by a uniform scale;
there is no hidden host expansion. The RMSNorm reference path has a separate
explicit broadcast service; this does not change the equal-shape multiply API.

Target `MULTIPLY` means `destination *= source`. If the left input must survive,
a preceding copy preserves it. For each row-sized chunk, code generation writes
both operands into banks zero and one of **every** four-bank group on channel
zero, emits `EW_MUL`, and reads bank two of the first group into the destination.
Every selected group is initialized, so the strict simulator never reads dummy
unwritten banks. Other groups calculate redundant results; this conservative
reference baseline is not a performance-optimized placement. The bank roles
come from the existing functional-reference contract, not a claim about every
physical target. Zero padding is transferred and multiplied too.

Copy and reshape preserve flattened element order. Their output may reuse the
input's storage after its final use, emitting no commands. If storage differs,
a real row-chunked copy is emitted. Reshape is not transpose or an arbitrary
strided view. Its new shape is the operation output's `TensorType`.

An identity graph and a same-storage reshape can have an empty `CentProgram`;
the manifest still loads inputs and extracts named outputs. Returning the old
input as well keeps it live, so the conservative mapper allocates separate
storage for a new result rather than tracking long-lived view aliases.

Native MLIR import/export supports `arith.addf` and `arith.mulf`, including
mixed graphs. Copy and reshape currently use the direct Python graph API;
no tensor-dialect bridge is claimed.

Copy and multiplication scratch share row zero safely because each transfer
or product completes before the next command reuses it. The artifact owns that
row in channel zero's banks exclusively. No persistent weight/KV buffers are
supported; adding those requires a shared DRAM allocator first.

## RMSNorm-plus-Linear example and remaining gates

For `x: [H]`, `scale: [H]`, and `weight: [O,H]`:

```text
n = RMS_NORM(x, scale, epsilon) : [H]
    n[i] = x[i] * scale[i] / sqrt(sum_j x[j]^2 / H + epsilon)
y = LINEAR(n, weight) : [O]
    y[o] = sum_i weight[o,i] * n[i]
```

This graph executes through the Python pipeline when the target explicitly
selects `ExecutionContract.REFERENCE`:

```python
from dataclasses import replace
from vllm_cent.pipeline import ExecutionContract, compile_graph

reference_target = replace(target, execution_contract=ExecutionContract.REFERENCE)
artifact = compile_graph(graph, reference_target)
```

Here `graph` is the SemanticGraph described above and `target` supplies the
hardware geometry. Linear keeps the meaning `x @ weight.T`;
leading dimensions become independent GEMVs, not an optimized GEMM kernel.
RMSNorm normalizes each last-dimension row and uses its own positive epsilon.
Both currently require one logical device.

Mapping records the selected Linear accumulator and prevents the output from
overwriting an input that the multi-stage calculation still needs. Target IR
uses separate command types for Linear and RMSNorm instead of adding optional
normalization fields to every command. Final lowering exposes operand staging,
MAC continuation, register initialization/readback, scalar reduction, scale
calculation, broadcasting, and result packing as instructions.

The reference services are distinct from paper instructions. They define their
own packing and scalar math and are rejected by paper/AiM serializers. The
hardware TODOs are narrow: selected-register bias/read encoding, physical result
packing, reduction precision, and the scalar routine's arguments, entry point,
and completion rules. Hynix MRS-selected latches are documented; their existence
is no longer an unanswered question. See [execution contracts](research/execution-contracts.md).

Weights are ordinary request inputs held in Shared Buffer. Tiles are staged in
exclusive DRAM scratch and consumed before reuse; no persistent weight or KV
allocation is claimed. The older model-specific structural lowerers remain
separate and do not automatically gain this executable implementation.

## Activations and a complete feed-forward subgraph

`ActivationOperation(function=ActivationKind.SILU, ...)` applies a selected
function to every logical element without changing the tensor shape. Other
choices are `RELU`, `SIGMOID`, `TANH`, `GELU_ERF`, and `GELU_TANH`. The last two
select different formulas; lowering never silently exchanges them. Function
names are mathematical choices, not numeric hardware activation IDs.

`SoftmaxOperation` normalizes each last-dimension row. Its reference service
subtracts the row maximum before exponentiation, sums the exponentials, and
divides. Negative infinity masks an entry; a fully masked row, positive infinity,
or NaN raises an execution error. This is an explicit reference service, not a
claim that the physical ISA has a softmax instruction. Both nonlinear services
exclude padding and write zero to unused output lanes. They permit exact
in-place execution only when mapping proves the old input is no longer needed.

The model frontend now supplies a complete feed-forward subgraph:

```python
from vllm_cent.models.llama import LlamaModelSpec, build_feed_forward_graph
from vllm_cent.pipeline import compile_graph

model = LlamaModelSpec(
    hidden_size=3, intermediate_size=5,
    num_attention_heads=1, num_kv_heads=1,
)
graph = build_feed_forward_graph(model, epsilon=1e-5)
artifact = compile_graph(graph, reference_target)
```

Here `reference_target` is the explicit reference profile from the preceding
example. Supply `x`, `norm_weight`, `gate_weight`, `up_weight`, and `down_weight`
when executing the artifact. The builder records the following calculation:

```text
normalized = RMSNorm(x, norm_weight)
gate = Linear(normalized, gate_weight)
up = Linear(normalized, up_weight)
gated = SiLU(gate) * up
output = x + Linear(gated, down_weight)
```

`batch_shape` adds leading input dimensions; the same weights apply to each
vector. The normalized input must survive both branches, and `x` must survive
until the residual addition. This composition exercises those lifetimes and
supports both generated and caller-written mappings. Supplied-tensor attention
is available separately, as described below. Combining both into a complete
transformer block still requires model-specific projections, RoPE, head
reordering, output projection, and persistent KV updates.

## Attention over supplied query, key, and value tensors

The public graph builder accepts four-dimensional inputs in
batch/head/sequence/feature order:

```python
from vllm_cent.graphs import build_attention_graph

attention = build_attention_graph(
    query_shape=(1, 4, 2, 3),
    key_shape=(1, 2, 5, 3),
    value_shape=(1, 2, 5, 7),
    causal=True,
    query_position_offset=3,
)
artifact = compile_graph(attention, reference_target)
```

Bind flattened `query`, `key`, and `value` tensors to execute. The output shape
is `(1, 4, 2, 7)`: query heads zero/one share KV head zero, and query heads
two/three share KV head one. Query rows represent positions three and four,
so their causal masks expose four and five keys respectively. Nothing appends
to a cache or performs a hidden projection. Q/K already contain any positional
rotation the caller requires; heads remain separate in the output.

`mask_shape=(2, 5)` adds a shared `mask` input. Full per-batch/per-query-head
masks may instead use `(1, 4, 2, 5)`. Masks add finite biases to scaled scores,
while negative infinity removes an entry. Causal and explicit masks intersect.
The scale defaults to `1/sqrt(query_width)` and may be overridden with a finite
constant. Fully masked rows raise an error rather than yielding arbitrary data.

The semantic operation retains these mathematical choices without addresses.
Mapping assigns a register and protects every input from output overwrites.
Target IR records dimensions, head groups, masks, scale, tile sizes, and the
selected register. Final code generation emits, for each KV-head group:

1. Pack its keys and transpose its values once into reusable working buffers.
2. Pack one query and run the existing tiled GEMV to calculate dot products.
3. Apply score scaling/masks and run stable row softmax.
4. Run a second tiled GEMV against transposed values and gather that query's result.
5. Reuse the score/result/matrix scratch for the next query.

Every multiplication uses the existing DRAM/Global Buffer/MAC path. Packing,
transpose, masking, and softmax retain their declared reference services; there
is no opaque host attention kernel. Scratch size depends on one key/value head
and one score row, not all query/key pairs. All supplied tensors and the current
KV-head copies must still fit Shared Buffer; this is not a persistent-cache or
streaming-large-context memory design. Physical export remains unsupported.

## Runtime and numerical contract

A version-one `CompiledArtifact` includes logical input/output names, shapes,
dtypes, row-major flattening, contiguous slot padding, Target IR, and the raw
`CentExecutable`. `bind` verifies exact logical scalar counts and writes all
occupied input lanes, including zero padding. `execute` runs the
functional executor, then strips only the known logical tail. The simulator
does not repair missing copies or initialize undeclared temporaries.

Execution uses Python-float reference math to check dataflow. The semantic F32
type is retained for frontends, but execution is **not bit-exact F32 or BF16**.
Tests use exact arithmetic cases and tolerance-based independent host oracles
for normalization and mixed graphs. A production target needs a supported
precision/rounding ABI before this artifact can claim hardware execution.

Instructions are reused across calls with changed input values. Current
bindings are all request inputs; they do not yet describe persistent weights,
KV pointers, token metadata, or GPU handoff. Those need an extended versioned
contract, separate from recompiling the graph per request.

## Acceptance status and roadmap

| Work package | Current status | Next gate |
| --- | --- | --- |
| Architecture RFC and refreshed state | Implemented here | Revisit contracts when adding the next operation |
| Interface/verifier design | Executable arithmetic, shape operations, and reference nonlinear operations implemented | Stateful effects, general PP/TP/SP, transfer/topology/fusion contracts |
| MLIR proof | Native semantic bridge through typed Python stages to numerical CENT execution | Registered out-of-tree CENT dialects and native conversion passes |
| Numerical growth | Feed-forward and supplied-tensor attention validated numerically | Projected/rotated attention, complete block, full decode; physical ABI export |
| Frontends/targets | Native MLIR and direct SemanticGraph; two physical profiles tested | Remaining Llama semantic migration; maintained torch.export-to-MLIR subgraph path; second target backend |
| Serving/performance | Interfaces leave scheduler ownership with vLLM | GPU-prefill/CENT-decode worker, KV lifecycle, logits parity, external mapper integration |

Do not remove the direct Llama compiler until the replacement has equivalent
structural and numerical tests. Do not write a general ad hoc torch.export
importer when the maintained conversion path rejects a graph. Do not attach
performance claims to instruction count reductions alone.

## Development commands

```sh
python -m pip install -e '.[dev,mlir]'
python -m unittest discover -s tests
coverage run -m unittest discover -s tests
coverage report
```

`mlir` is optional for users of the Python compiler, but CI installs it for the
numerical/coverage job so native tests are exercised, not silently skipped.
The backend continues to work without it; attempting native import gives an
actionable dependency diagnostic.

Native API references: [MLIR Python bindings](https://mlir.llvm.org/docs/Bindings/Python/),
[MLIR dialect conversion](https://mlir.llvm.org/docs/DialectConversion/), and
[IREE Python bindings](https://iree.dev/reference/bindings/python/).
