# vLLM-CENT

vLLM-CENT compiles model-specific operations into a typed CENT program. The
first frontend lowers one batch-one Llama decode block; the generic `cent/`
package contains no Llama or transformer definitions. A small functional
simulator executes the instruction subset whose value semantics are defined.

This project does not yet execute a complete model. Its compiler output is an
ordered sequence of instructions for the architecture described in the
[CENT paper](https://arxiv.org/abs/2502.07578); the functional simulator is a
correctness target, while the AiM adapter remains a separate timing target.

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

`render_text_program` emits the Table 2/3 assembly. The package deliberately
does not model `SYNC`, `EOC`, or `RD_AF`: a full-text search of the paper found
no such instructions, and they belong to the old reference simulator's trace
protocol rather than CENT's published ISA. `CentProgram` is therefore just an
immutable, nonempty ordered tuple of real ISA instructions; finishing a builder
does not append an artificial terminator.

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
- `CentHardwareSpec` describes DRAM geometry, explicit per-channel Global
  Buffer capacity, the 64KB Shared Buffer as 2,048 256-bit slots by default,
  PU accumulation-register capacity, and the target ABI's sigmoid `AFid`.
- `CentBlockPlacementSpec` assigns physical channels to one block.
- `DecodeStepSpec` provides current and reserved context lengths.
- `CompileRequest` combines the source model, target, placement, and decode
  step.
- `compile_transformer_block` returns a validated `CentProgram`.
- `render_text_program` emits paper-ISA assembly.
- `CentExecutable` attaches named raw scalars to reusable DRAM, Shared Buffer,
  and Global Buffer regions. Inputs cannot overlap; read-only outputs may.
- `execute_functionally` runs supported instructions with strict uninitialized
  reads, transactional instruction commits, reference Python-float arithmetic,
  and optional summary events containing physical read/write regions.

The functional slice currently supports `WR_SBK`, `RD_SBK`, `WR_GB`,
`COPY_BKGB`, `COPY_GBBK`, `EW_MUL`, and `ACC`. It rejects all other opcodes
during whole-program preflight. This is intentionally not a claim that the
current RMSNorm or Llama block lowering is numerically executable.

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
  addresses. Consequently the current softmax/RMSNorm lowering is not yet a
  complete executable paper algorithm; wiring those operations requires a
  target runtime ABI that supplies the correct `PC` values.

## Run the tests

From this directory, either install the package in editable mode or expose its
`src` directory:

```bash
python3 -m pip install -e .
python3 -m unittest discover -s tests -v
```

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

The tests render all 18 paper instructions with distinct operands, validate
DRAM/Shared Buffer/register boundaries, verify immutable static opcodes, test
every nontrivial lowering helper directly, preserve exact small-model golden
counts, and compile Llama 3 8B and 70B block dimensions.
