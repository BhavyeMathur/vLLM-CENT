# vLLM-CENT

vLLM-CENT compiles PyTorch models for CENT, a CXL-attached accelerator built
from SK hynix GDDR6-AiM processing-in-memory devices. A model is exported
with `torch.export`, lowered through three IR levels, and mapped onto CENT
as Tile IR kernels, which expand into AiM simulator traces.

```text
torch model ──frontend──▶ Semantic IR ──parallel──▶ sharded Semantic IR
            ──legalize──▶ Device IR (PIM / PNM / CXL ops of one device)
            ──mapping───▶ Tile IR kernels (loops, cent.* ops, physical layouts)
            ──tile──────▶ CENT instructions ──▶ AiM trace
```

## Packages

- `frontend`: `torch.export` graphs to Semantic IR.
- `ir`: the SSA IR and its three levels: Semantic IR (`aloi.*`), Device IR
  (`pim.*`, `pnm.*`, `cxl.*`, `device.*`) and Tile IR (`tile.for`,
  `cent.*`), with the printer and passes.
- `parallel`, `legalize`: tensor parallelism; lowering to a device's ops.
- `device`: device specs in YAML (`examples/cent/cent.yaml`): geometry and
  supported ops.
- `tile`: physical layouts (`split`, `resolve`, `allocate`), the Tile DSL
  (`tile.dsl.TileKernel`), the kernel verifier, expansion into CENT
  instructions and AiM trace generation (`tile.codegen`).
- `cent`: CENT's hardware description and typed instructions.
- `driver`: `aloi-compile`, from a model and a run spec to every IR stage.

## Run it

```bash
python3 -m pip install -e ".[dev]"

# Every IR stage of one Llama 3.1 70B decoder layer, one tensor-parallel rank.
aloi-compile --model examples/llama_70b/llama3_1_70b.json \
             --run examples/llama_70b/runs/llama3_1_70b_tp8_decode.yaml \
             --device examples/cent/cent.yaml

# The layer as CENT's simulator maps it: Tile IR and the AiM trace.
python -m examples.llama_70b.kernel_mapping.llama3_1_70b \
       examples/llama_70b/llama3_1_70b_ir
```

`examples/llama_70b/llama3_1_70b_ir` holds every stage of the second
command: `00_imported` to `03_device`, `04_tile.aloi` (each kernel's Tile
IR and layouts) and `05_aim.trace`. The trace runs on the AiM simulator
(`../ALOI/aim_simulator`); it is CENT's own trace of that configuration but
for one row (see `examples/llama_70b/kernel_mapping/llama3_1_70b.py`).

`examples/Tile_DSL` has hand-written GEMV and GEMM kernels in the Tile DSL;
each prints its Tile IR and AiM trace.

## Run the tests

```bash
python3 -m unittest discover -s tests
```

Tests that run the AiM simulator are skipped unless it is built; set
`AIM_SIMULATOR` to its directory if it is not `../ALOI/aim_simulator`.

## Documentation

`docs/research` collects the hardware evidence (SK hynix AiM, the AiM
simulator, execution contracts). `docs/FUNCTIONAL_SIMULATOR_DESIGN.md`,
`docs/MULTI_LEVEL_COMPILER.md`, `docs/PROJECT_STATE.md` and
`docs/PROJECT_UPDATE.md` describe the earlier backend (a functional
simulator, a graph pipeline and a direct Llama lowering to CENT programs),
which has been removed; they remain for its design rationale.
