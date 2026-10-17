# vLLM-CENT

vLLM-CENT compiles model-specific operations into a typed CENT program. The
first frontend lowers one batch-one Llama decode block; the generic `cent/`
package contains no Llama or transformer definitions.

The project includes a functional simulator for supported instruction dataflow
and an incremental multi-stage compiler for static tensor addition, elementwise
multiplication, copy, contiguous reshape, bias-free Linear (GEMV), RMSNorm,
activations, stable softmax, and supplied-tensor attention.
Linear, RMSNorm, activations, softmax, and attention require `ExecutionContract.REFERENCE` on the target profile;
their explicit accumulator, packing, and scalar services have no claimed
physical instruction encoding. Squaring uses multiplication with the same input
twice. These execute under reference arithmetic; the compiler does
not yet execute a complete Llama block or integrate with a vLLM worker.

The [multi-level compiler architecture](docs/MULTI_LEVEL_COMPILER.md) describes
Semantic IR → sharding → Device IR → mapping → symbolic Target IR → CENT
commands, the native MLIR frontend, logical runtime bindings, and remaining
milestones. [Project state](docs/PROJECT_STATE.md) distinguishes numerical
support from structural lowering. The new
`models.llama.build_feed_forward_graph` composes RMSNorm, three projections,
SiLU gating, and a residual addition into a numerically tested subgraph.
`graphs.build_attention_graph` executes attention over supplied Q/K/V, with
batching, grouped heads, additive masks, and explicit causal alignment. Model
projections, RoPE, persistent KV updates, and a complete transformer block are
still unfinished.

## Paper-faithful instruction layer

The instruction classes model all commands in Tables 2 and 3 with their full
operand lists:

- Near-bank PUs: `MAC_ABK`, `EW_MUL`, and `AF`.
- PNM units: `EXP`, `RED`, `ACC`, and `RISCV`.
- CXL movement: `SEND_CXL`, `RECV_CXL`, and `BCAST_CXL`.
- Shared Buffer and DRAM movement: `WR_SBK`, `RD_SBK`, and `WR_ABK`.
- Global Buffer and PU movement: `COPY_BKGB`, `COPY_GBBK`, `WR_BIAS`,
  `RD_MAC`, and `WR_GB`.

Python fields use readable names while docstrings retain the paper terms. For
example, `CentMemoryAddress.row` is the paper's `RO`, `column` is `CO`, and
`CentSharedBufferAddress.slot` is used as `Rs` or `Rd` according to whether it
is a source or destination. `accumulation_register` is `Regid`.

`OPsize` is represented by `operation_size`. One instruction causes the CENT
decoder to generate that many micro-operations over consecutive 256-bit Shared
Buffer slots and DRAM column regions. A five-burst transfer that fits in one
row is therefore one instruction with `operation_size=5`, not five duplicate
instructions.

`render_text_program` emits typed assembly. The separate AiM timing adapter
handles its own target-specific trace protocol and rejects operands it cannot
encode. Both serializers reject functional-only reference services. Timing
simulation does not verify tensor values. See the [execution contracts](docs/research/execution-contracts.md)
for Hynix latch-selection evidence and the remaining hardware questions.

## Architecture boundary

```text
models/llama/                  cent/
--------------------------    --------------------------
Llama dimensions              CENT hardware geometry
Llama tensor layout           complete typed CENT ISA
Llama lowering stages         hardware-aware validation
Llama block orchestration     generic program builder
                               assembly renderers
             \                 /
              CompileRequest
                    |
          compile_transformer_block
                    |
               CentProgram
```

A future model family belongs in its own `models/<family>/` package and reuses
the target types. `compiler.py` remains only a small family dispatcher.

## Current public interface

- `LlamaModelSpec` describes one Llama block.
- `CentHardwareSpec` describes DRAM geometry, the 64KB Shared Buffer as 2,048
  256-bit slots by default, PU accumulation-register capacity, and the target
  ABI's sigmoid `AFid`.
- `CentBlockPlacementSpec` assigns physical channels to one block.
- `DecodeStepSpec` provides current and reserved context lengths.
- `CompileRequest` combines the source model, target, placement, and decode
  step.
- `compile_transformer_block` returns a validated `CentProgram`.
- `render_text_program` emits paper-ISA assembly.

The Llama frontend currently lowers RMSNorm, Q/K/V projections, rotary
embedding data movement and multiplication, KV-cache updates, attention-score
GEMV, the existing two-pass softmax data path, value-cache GEMV, output
projection, residual additions, and the gated SiLU feed-forward network.

Two paper-to-implementation gaps remain explicit rather than guessed:

- The paper names `AFid` but does not publish numeric function IDs. Rather than
  guessing, `CentHardwareSpec.sigmoid_activation_function_id` requires the
  caller to provide the target ABI's value. Tests use zero only as a documented
  fake target value.
- The paper explains that uncommon operations such as reciprocal and square
  root run through `RISCV`, but does not publish their program-counter entry
  addresses. Consequently the older structural softmax/RMSNorm lowering is not yet a
  complete executable paper algorithm. The new pipeline executes RMSNorm through
  explicitly named reference services; physical export still requires a
  target runtime ABI that supplies the correct `PC` values.

## Run the tests

From this directory, either install the package in editable mode or expose its
`src` directory:

```bash
python3 -m pip install -e ".[dev,mlir]"
python3 -m unittest discover -s tests -v
```

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The tests render all 18 paper instructions with distinct operands, validate
DRAM/Shared Buffer/register boundaries, verify immutable static opcodes, test
every nontrivial lowering helper directly, preserve exact small-model golden
counts, and compile Llama 3 8B and 70B block dimensions.

The `mlir` extra installs native IREE-packaged MLIR bindings. It is optional for
the existing compiler and Python graph API; native tests require it. The staged
entry points are `vllm_cent.pipeline.compile_graph` and `compile_mlir`, returning
reusable artifacts with `bind` and reference-math `execute` methods.

### Optional attention comparison with PyTorch

The normal suite has an independent scalar attention oracle and needs no torch
installation. To also compare against PyTorch SDPA, point this environment
variable at a Python interpreter where torch is installed:

```bash
VLLM_CENT_TORCH_PYTHON=/path/to/python-with-torch python3 -m unittest discover -s tests -p test_attention_torch.py -v
```

The oracle runs in a separate process so its Python version need not match the
compiler's. It uses float64 reference math, zero dropout, and an explicit mask
for our causal offsets. Passing this check does not establish accelerator
rounding or timing accuracy.
