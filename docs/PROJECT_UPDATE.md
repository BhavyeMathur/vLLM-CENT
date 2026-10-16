# vLLM-CENT: Project Update

**October 15, 2026 · Current working tree, including uncommitted changes**

## Executive summary

We now have a working compiler from a tensor graph to instructions that can be
executed in a functional simulator. It supports basic tensor operations,
Linear/matrix-vector multiplication, RMSNorm, activations, softmax, a gated
feed-forward block, and attention over supplied query/key/value tensors.

The main architectural result is a sequence of checked stages that separates
**what to compute, where to place data, and how to execute it**. Compiled programs
can run again with new inputs without recompilation.

Complex operations currently use an explicit **reference execution mode**:
CENT arithmetic and transfers are combined with software-defined services for
packing and nonlinear math. This validates complete calculations, but physical
hardware export and full model-serving integration remain unfinished.

**Validation:** 334 tests pass, including native MLIR and PyTorch attention
comparisons; 98% branch-enabled coverage. Lint, formatting, and strict type checks
also pass.

## 1. Input: a graph of tensor calculations

The main input is a `SemanticGraph`: named tensors with static shapes, an ordered
list of operations, and requested outputs. Actual tensor values arrive later.
For example, this computes `z = x + y`, then `out = z + y`:

```python
from vllm_cent.pipeline.ir import SemanticGraph, TensorType, Value
from vllm_cent.pipeline.operations import AddOperation

vector = TensorType((3,))  # Three elements; the current semantic dtype is F32.
graph = SemanticGraph(
    inputs=(Value("x", vector), Value("y", vector)),
    operations=(
        AddOperation(name="add0", inputs=("x", "y"), output=Value("z", vector)),
        AddOperation(name="add1", inputs=("z", "y"), output=Value("out", vector)),
    ),
    outputs=("out",),
)
```

Each operation has a name, input names, one output, and an optional model-block
label. Specialized parameters belong to their operation: RMSNorm owns `epsilon`,
while attention owns its scale and masking choices. The graph contains no
hardware addresses.

The compilation entry point is:

```text
compile_graph(graph, target, *, parallel=None, capabilities=None,
              mapping=None, mapper=baseline_mapping, constraints=())
    -> CompiledArtifact
```

`target` supplies hardware capacities and instruction limits. Callers may provide
a complete mapping or a mapper that chooses one. Both use the same safety checks.
For Linear, RMSNorm, activations, softmax, and attention, construct the target
profile with `execution_contract=ExecutionContract.REFERENCE`.

An optional `compile_mlir(text, target)` frontend accepts native MLIR tensor
addition and multiplication. Native parsing, verification, and optimization work;
the later IRs and passes are currently typed Python, not native MLIR dialects.

## 2. Middle IRs: progressively make the graph concrete

An intermediate representation (IR) records the calculation at one level of
detail. Each pass introduces the information needed by the next stage:

```text
SemanticGraph → ShardedGraph → DeviceGraph
    → MappingPlan → TargetProgram → allocated CENT instructions
```

The table follows the same two-addition example. Assume one device and a small
test target with **two scalar values per Shared Buffer slot**. This geometry is
for illustration, not a chip specification. Entries summarize actual records;
they are not a separate IR syntax.

| Stage | What it records | Example |
| --- | --- | --- |
| **Semantic** | Tensor shapes and mathematical operations | `add0: (x, y) → z`; `add1: (z, y) → out`; all shapes `(3,)` |
| **Sharded** | A local graph per logical device, plus global/local shapes and owned intervals | Device 0 owns all three elements of each tensor: `axis=-1, start=0, stop=3` |
| **Device** | Supported operation kinds, storage space, and dependencies | Both operations become `PNM_ADD` in `SHARED_BUFFER`; `add1.dependencies=("add0",)` |
| **Mapping** | Operation schedule, named storage objects, and required accumulator registers | Schedule `("add0", "add1")`; `x`, `z`, and `out` share storage `"x"`; `y` uses `"y"` |
| **Target** | Implementation commands, transfer sizes, and temporary-space requirements | Two `ACCUMULATE` commands: source `"y"`, destination `"x"`, `slot_count=2` |
| **Allocation** | Concrete addresses for named storage | `"x"` occupies slots 0–1; `"y"` occupies slots 2–3 |

Storage reuse is safe because the original `x` is no longer needed after the
first addition, and `z` is no longer needed after the second. If the caller also
requested `x` as an output, the compiler would preserve it instead of overwriting
it. Three elements require two slots; the unused fourth lane is zero padding.

The actual Target IR record for the first addition is:

```text
TargetCommand(
    kind=TargetCommandKind.ACCUMULATE,
    source="y", destination="x", slot_count=2, operation="add0",
)
```

Larger calculations use dedicated records such as `LinearCommand`,
`RmsNormCommand`, and `AttentionCommand`. These retain dimensions, selected
registers, tile sizes, and mathematical parameters. Emission expands them into
transfers and arithmetic; attention becomes two matrix-vector products per query,
with scaling, masking, and softmax between them.

The compiler checks shapes, dependencies, target support, memory capacity, and
safe reuse at the relevant boundaries. The default mapper reuses available
storage deterministically; it is not an optimizing search algorithm. Multi-device
partitioning can be verified for addition, multiplication, and copy, but execution
currently supports only logical device zero.

## 3. Final output: a reusable program and its input/output locations

The result is a `CompiledArtifact` containing:

- Logical input/output names and shapes.
- The retained Target IR and mapping.
- A `CentProgram`: ordered typed instructions and its hardware target.
- An execution manifest: where to load inputs and read outputs.

For the example above, the program contains two `Accumulate` instructions. Its
readable assembly is:

```text
ACC 2 0 2
ACC 2 0 2
```

Each line adds two slots starting at source slot 2 into destination slot 0.
This is assembly text, not a binary hardware image.

Once `artifact = compile_graph(graph, target)` has been created, execution is:

```python
from vllm_cent.runtime import CentNamedScalars

outputs = artifact.execute((
    CentNamedScalars(name="x", values=(1.0, 2.0, 3.0)),
    CentNamedScalars(name="y", values=(10.0, 20.0, 30.0)),
))
# (CentNamedScalars(name="out", values=(21.0, 42.0, 63.0)),)
```

Inputs are flat, row-major values. Binding validates names and counts and adds
padding; execution removes that padding from the returned results.

Reference-mode programs have the same artifact structure but also contain
explicit services such as `ReferenceSoftmax` and accumulator initialization and
readback. These remain inspectable instructions. Physical CENT and AiM exporters
reject services without a verified hardware translation.

## 4. Current capabilities

| Capability | Supported behavior |
| --- | --- |
| Basic tensors | Equal-shape addition/multiplication, copy, contiguous reshape; no implicit broadcasting |
| Linear / GEMV | Bias-free `x @ weights.T`; multiple input rows run as repeated matrix-vector products |
| RMSNorm | Per-row mean square, epsilon, reciprocal square root, and learned scale |
| Nonlinear functions | ReLU, sigmoid, tanh, SiLU, exact-formula and tanh-approximate GELU; stable last-axis softmax |
| Feed-forward block | RMSNorm → gate/up projections → SiLU/product → down projection → residual addition |
| Attention | Supplied Q/K/V, batches, grouped query heads, additive masks, and causal position offsets |

Attention uses `[batch, heads, sequence, features]` tensors. For example:

```python
from vllm_cent.graphs import build_attention_graph

attention_graph = build_attention_graph(
    query_shape=(1, 4, 2, 3),
    key_shape=(1, 2, 5, 3),
    value_shape=(1, 2, 5, 7),
    causal=True,
    query_position_offset=3,
)
# Inputs: query, key, value. Output shape: (1, 4, 2, 7).
```

Here pairs of query heads share one key/value head. The two queries represent
positions 3 and 4, so they may attend to four and five keys respectively.
The operation computes `softmax(Q @ K.T / sqrt(key_width) + mask) @ V`.
Projections, positional rotation, cache updates, head merging, and output
projection remain outside this operation.

All graph tensors—including weights—and temporary buffers must currently fit
in Shared Buffer memory. There is no persistent weight allocation, streaming
large-context execution, or optimized general matrix-matrix kernel.

## 5. What the simulator establishes

The functional simulator executes transfers, multiplication, accumulation, MAC,
and the declared reference services against explicit memory and register state.
It detects unwritten reads and invalid accesses, applies each instruction's
writes together, and optionally traces the regions read and written.

It checks **numerical results and data movement** using Python floats. It does
not model cycle timing, DRAM scheduling, exact F32/BF16 rounding, or the complete
hardware state machine. The separate AiM adapter supports timing traces for a
restricted instruction subset; those traces do not establish numerical correctness.

## 6. Next milestones

1. Combine attention and feed-forward computation with projections and positional
   encoding into a complete, numerically validated transformer block.
2. Add persistent weights, a shared DRAM allocator, KV-cache updates, and repeated
   decode execution.
3. Resolve physical register controls, result packing, precision, and scalar-runtime
   interfaces so reference services can be translated to hardware.
4. Extend native MLIR/PyTorch integration and connect vLLM workers before measuring
   end-to-end serving performance.

For implementation details, see the [compiler architecture](MULTI_LEVEL_COMPILER.md),
[IR definitions](../src/vllm_cent/pipeline/ir.py), and
[hardware execution questions](research/execution-contracts.md).
