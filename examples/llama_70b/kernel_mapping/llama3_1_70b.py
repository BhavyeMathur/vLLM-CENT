"""Map one Llama 3.1 70B decoder layer onto CENT, as CENT's simulator does.

The run spec ``runs/llama3_1_70b_kvtp8_decode.yaml`` names the evaluated
configuration (KV-head TP = 8, PP = 4 on 32 devices, GDDR6, 2304 of a
4096-position window). This module takes the Device IR of one rank for that
run, gives each op the Tile DSL kernel of CENT's mapping
(``../ALOI/cent_simulation``, ``Llama.trace_only``), checks that each
kernel's tensors have the shapes of the op's operands, and writes the
layer's AiM trace:

- PIM work: the kernels' records, CENT's but for the GPR fields (CENT
  writes 0) and configuration-register writes (CENT writes none).
- PNM work CENT charges analytically (the RMSNorm reduction, RoPE,
  softmax, the SV partial-sum reduction) and the CXL all-reduces: comments
  with CENT's cycles or milliseconds (``analytical``).
- Ops that move no data in CENT's mapping (transfers between PIM and PNM,
  head splits and merges): comments.

One known difference: CENT stores the FFN norm's squared input in bank
pairs of row ``sa_copy`` (61), the row its EWMUL operands then overwrite,
and leaves row ``sa`` (60) unused (``Llama.py:1005-1007``). The verifier
does not let two tensors overlap, so this mapping uses row 60, as the
attention norm uses its own row 0.

Run from the repository root::

    python -m examples.llama_70b.kernel_mapping.llama3_1_70b [OUT_DIR]

With ``OUT_DIR`` (``examples/llama_70b/llama3_1_70b_ir``), it writes every
stage of the run: ``00``-``03`` as ``aloi-compile`` does, ``04_tile.aloi``
(the kernels' Tile IR, with their layouts) and ``05_aim.trace``; without,
it prints the trace.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from math import prod
from pathlib import Path

from vllm_cent.device import CentArchitecture, DeviceSpec, load_device_spec
from vllm_cent.driver import RunSpec, compile_module, load_model, load_run_spec
from vllm_cent.ir import Function, Operation, Value, print_function
from vllm_cent.ir.device import PIMGemvOp
from vllm_cent.tile import CentKernel
from vllm_cent.tile.codegen import MAC_FROM_GLOBAL_BUFFER, iter_aim_records
from vllm_cent.tile.dsl import TileKernel
from vllm_cent.tile.expand import ChannelIssue

from . import analytical
from .attention import cache_write, k_cache_layout, qk, sv, v_cache_layout
from .gemv import kv_proj, reuse_gb_gemv
from .vector import residual_add, rms_norm, silu_mul

__all__ = ["DRAM_ROWS", "WEIGHT_ROWS", "LayerMapping", "layer_trace", "map_layer"]

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
MODEL = EXAMPLE_DIR / "llama3_1_70b.json"
RUN = EXAMPLE_DIR / "runs" / "llama3_1_70b_kvtp8_decode.yaml"
# CENT's MAC register count, so the GEMV order is CENT's (see cent_paper.yaml).
DEVICE = EXAMPLE_DIR.parents[0] / "cent" / "cent_paper.yaml"

# The evaluated configuration's window and system (run_cent_memory_cases.py):
# a 4096-position window, 32 devices with 144 PCIe lanes, PP = 4.
WINDOW = 4096
DEVICES = 32
PCIE_LANES = 144
PIPELINE_STAGES = 4

# First DRAM row of each weight, as CENT's Llama.memory_mapping places them
# for KV-head TP = 8 with a 4096 window (Llama.py:1494-1748).
WEIGHT_ROWS = {
    "self_attn.q_proj.weight": 3,
    "self_attn.k_proj.weight": 19,
    "self_attn.v_proj.weight": 27,
    "self_attn.o_proj.weight": 44,
    "mlp.gate_proj.weight": 63,
    "mlp.up_proj.weight": 119,
    "mlp.down_proj.weight": 179,
}
# The rows of activations and the KV cache in the same map. "sa" holds the
# FFN norm's bank pairs here; see the module docstring.
DRAM_ROWS = {
    "x": 0,
    "x_copy": 1,
    "sa_norm": 2,
    "cache_k": 37,
    "cache_v": 42,
    "sa": 60,
    "sa_copy": 61,
    "ffn_norm": 62,
    "x1_sigmoid": 177,
}

_NO_RECORDS = "no records: CENT's kernels read and write the data in place"


@dataclass
class LayerMapping:
    """One layer's ops, in order, with the kernel or comment each gets.

    Attributes:
        ops: The Device IR function's ops.
        kernels: The kernel that runs an op.
        covered_by: Ops run by another op's kernel, e.g. ``v_proj`` by the
            fused K/V kernel of ``k_proj``.
        comments: The trace comment of an op, alone or before its kernel.
    """

    ops: tuple[Operation, ...]
    name: str = "layer"
    kernels: dict[Operation, CentKernel] = field(default_factory=dict)
    covered_by: dict[Operation, Operation] = field(default_factory=dict)
    comments: dict[Operation, str] = field(default_factory=dict)


def _squeeze(shape: Sequence[int]) -> tuple[int, ...]:
    """Drop size-1 dimensions: ``[1, 2304, 1, 128]`` is ``(2304, 128)``."""

    return tuple(d for d in shape if d != 1)


def _check(
    op: Operation, kernel: CentKernel, pairs: Sequence[tuple[str, Value]]
) -> None:
    """Check that each named kernel tensor has its Device IR value's shape.

    Raises:
        ValueError: If a shape or dtype differs.
    """

    by_name = {a.name: a for a in kernel.function.arguments}
    for name, value in pairs:
        arg = by_name[name]
        ours = (_squeeze(arg.type.shape), arg.type.dtype)
        theirs = (_squeeze(value.type.shape), value.type.dtype)
        if ours != theirs:
            raise ValueError(
                f"{op.name_hint}: kernel tensor {name!r} is {ours[0]} {ours[1]}, "
                f"but the Device IR value is {theirs[0]} {theirs[1]}"
            )


def _check_gemv(op: Operation, kernel: CentKernel, names: tuple[str, str, str]) -> None:
    """Check a GEMV kernel's x, w and y against the op."""

    values = (op.operands[0], op.operands[1], op.results[0])
    _check(op, kernel, list(zip(names, values, strict=True)))


def _weight(op: Operation) -> str:
    name = op.operands[1].name
    assert name is not None
    return name


def _build(
    name: str, arch: CentArchitecture, record: Callable[[TileKernel], object]
) -> CentKernel:
    with TileKernel(name, arch) as k:
        record(k)
    return k.build()


def map_layer(function: Function, arch: CentArchitecture, context: int) -> LayerMapping:
    """Give each op of one rank's layer its CENT kernel or trace comment.

    Args:
        function: One rank's Device IR of the layer.
        arch: The device.
        context: KV cache positions, including the new one.

    Raises:
        ValueError: If a kernel's tensors do not match its op's shapes.
    """

    mapping = LayerMapping(function.operations)
    ops = {op.name_hint: op for op in function.operations if op.name_hint}
    users: dict[Value, list[Operation]] = {}
    for user in function.operations:
        for value in user.operands:
            users.setdefault(value, []).append(user)

    for op in function.operations:
        if op in mapping.covered_by:
            continue
        if op.NAME == "pim.gemv":
            _map_gemv(op, ops, users, arch, mapping)
        elif op.NAME == "pnm.rms_norm":
            mapping.kernels[op] = _map_rms_norm(op, arch)
        elif op.NAME == "pim.cache_write":
            mapping.kernels[op] = _map_cache_write(op, arch, context)
        elif op.NAME in ("pim.qk", "pim.sv"):
            mapping.kernels[op] = _map_attention(op, arch, context)
        elif op.NAME == "pim.ewmul" and op.name_hint == "mlp.silu":
            mul = ops["mlp.mul"]
            mapping.kernels[op] = _map_silu_mul(op, mul, arch)
            mapping.covered_by[mul] = op
        elif op.NAME == "pnm.add":
            mapping.kernels[op] = _map_residual_add(op, arch)
            mapping.comments[op] = (
                "residual add: EWADD, as CENT issues it; CENT charges no time for it"
            )
        else:
            mapping.comments[op] = _comment(op, arch, context)
    return mapping


def _map_residual_add(op: Operation, arch: CentArchitecture) -> CentKernel:
    """CENT's residual add: one EWADD of the two vectors."""

    n = _squeeze(op.results[0].type.shape)[-1]
    names = ("residual", "update")

    def record(k: TileKernel) -> None:
        residual_add(k, n=n, names=names)

    kernel = _build(op.name_hint or "add", arch, record)
    _check(op, kernel, list(zip(names, op.operands, strict=True)))
    return kernel


def _map_gemv(
    op: Operation,
    ops: dict[str, Operation],
    users: dict[Value, list[Operation]],
    arch: CentArchitecture,
    mapping: LayerMapping,
) -> None:
    """Map one GEMV: K with V, the gate with its activation, the others alone."""

    assert isinstance(op, PIMGemvOp)
    name = op.name_hint or "gemv"
    n, kdim = op.operands[1].type.shape
    if name == "self_attn.k_proj":
        v_op = ops["self_attn.v_proj"]
        k_names = ("x", _weight(op), "k")
        v_names = ("x", _weight(v_op), "v")
        kernel = _build(
            "kv_proj",
            arch,
            lambda k: kv_proj(
                k,
                n=n,
                kdim=kdim,
                k_names=k_names,
                v_names=v_names,
                k_row=WEIGHT_ROWS[k_names[1]],
                v_row=WEIGHT_ROWS[v_names[1]],
            ),
        )
        _check_gemv(op, kernel, k_names)
        _check_gemv(v_op, kernel, v_names)
        mapping.covered_by[v_op] = op
    elif name == "mlp.gate_proj":
        act = next(u for u in users[op.results[0]] if u.NAME == "pim.activation")
        names = ("x", _weight(op), "y")
        kernel = _build(
            "gate_proj",
            arch,
            lambda k: reuse_gb_gemv(
                k,
                n=n,
                kdim=kdim,
                names=names,
                w_row=WEIGHT_ROWS[names[1]],
                activation="sigmoid",
            ),
        )
        _check_gemv(op, kernel, names)
        _check(act, kernel, [("sigmoid", act.results[0])])
        mapping.covered_by[act] = op
    else:
        names = ("x", _weight(op), "y")
        kernel = _build(
            name.replace(".", "_"),
            arch,
            lambda k: reuse_gb_gemv(
                k, n=n, kdim=kdim, names=names, w_row=WEIGHT_ROWS[names[1]]
            ),
        )
        _check_gemv(op, kernel, names)
    mapping.kernels[op] = kernel


def _map_rms_norm(op: Operation, arch: CentArchitecture) -> CentKernel:
    """CENT's RMSNorm; the attention norm and the FFN norm use their own rows."""

    hint = op.name_hint or "rms_norm"
    first = hint == "input_layernorm"
    rows = (
        (DRAM_ROWS["x"], DRAM_ROWS["x_copy"], DRAM_ROWS["sa_norm"])
        if first
        else (DRAM_ROWS["sa"], DRAM_ROWS["sa_copy"], DRAM_ROWS["ffn_norm"])
    )
    names = ("x", f"{hint}.weight", "y")
    hidden = op.results[0].type.shape[-1]
    kernel = _build(
        hint,
        arch,
        lambda k: rms_norm(
            k, hidden=hidden, names=names, rows=rows, scale_first=not first
        ),
    )
    values = (op.operands[0], op.operands[1], op.results[0])
    _check(op, kernel, list(zip(names, values, strict=True)))
    return kernel


def _map_cache_write(op: Operation, arch: CentArchitecture, context: int) -> CentKernel:
    """Write the new position (the last of the context) into a KV cache."""

    cache_name = op.operands[0].name or "cache"
    head_dim = _squeeze(op.operands[1].type.shape)[-1]
    is_k = cache_name.startswith("k")
    layout = k_cache_layout if is_k else v_cache_layout
    row = DRAM_ROWS["cache_k"] if is_k else DRAM_ROWS["cache_v"]

    def record(k: TileKernel) -> None:
        cache = layout(k, cache_name, context, head_dim, WINDOW, row)
        cache_write(
            k, cache=cache, new_name="new", head_dim=head_dim, position=context - 1
        )

    kernel = _build(op.name_hint or "cache_write", arch, record)
    _check(
        op,
        kernel,
        [
            (cache_name, op.operands[0]),
            ("new", op.operands[1]),
            (cache_name, op.results[0]),
        ],
    )
    return kernel


def _map_attention(op: Operation, arch: CentArchitecture, context: int) -> CentKernel:
    """QK or SV of one KV head's query heads."""

    heads, head_dim = _squeeze(op.operands[0].type.shape)
    if op.NAME == "pim.qk":
        names = ("q", "k_cache", "scores")
        kernel = _build(
            "qk",
            arch,
            lambda k: qk(
                k,
                names=names,
                heads=heads,
                head_dim=head_dim,
                context=context,
                max_context=WINDOW,
                k_row=DRAM_ROWS["cache_k"],
            ),
        )
    else:
        head_dim = _squeeze(op.results[0].type.shape)[-1]
        names = ("p", "v_cache", "out")
        kernel = _build(
            "sv",
            arch,
            lambda k: sv(
                k,
                names=names,
                heads=heads,
                head_dim=head_dim,
                context=context,
                max_context=WINDOW,
                v_row=DRAM_ROWS["cache_v"],
            ),
        )
    _check(op, kernel, list(zip(names, (*op.operands, op.results[0]), strict=True)))
    return kernel


def _map_silu_mul(
    silu: Operation, mul: Operation, arch: CentArchitecture
) -> CentKernel:
    """CENT's ``silu(gate) * up``: the two ``pim.ewmul`` ops in one kernel."""

    n = _squeeze(silu.results[0].type.shape)[-1]
    names = ("gate", "sigmoid", "up", "y")
    kernel = _build(
        "silu_mul",
        arch,
        lambda k: silu_mul(k, n=n, names=names, row=DRAM_ROWS["x1_sigmoid"]),
    )
    values = (silu.operands[0], silu.operands[1], mul.operands[1], mul.results[0])
    _check(silu, kernel, list(zip(names, values, strict=True)))
    return kernel


def _comment(op: Operation, arch: CentArchitecture, context: int) -> str:
    """The trace comment of an op CENT models analytically or not at all."""

    hint = op.name_hint or ""
    if op.NAME == "pnm.rope":
        cycles = analytical.rope_cycles(prod(_squeeze(op.results[0].type.shape)))
        return f"PNM RoPE: {cycles:g} cycles (analytical, as CENT)"
    if hint.startswith("self_attn.softmax"):
        if op.NAME == "pnm.reduce_max":
            heads = _squeeze(op.operands[0].type.shape)[0]
            cycles = analytical.softmax_cycles(context, heads, arch.channels)
            return (
                f"PNM softmax (max, exp, sum, div): {cycles:g} cycles "
                "(analytical, as CENT)"
            )
        return "part of the softmax above"
    if op.NAME == "cxl.all_reduce":
        hidden = _squeeze(op.results[0].type.shape)[-1]
        tp = DEVICES // PIPELINE_STAGES
        ms = analytical.cxl_all_reduce_ms(hidden, PCIE_LANES // DEVICES, tp)
        return f"CXL all-reduce over {tp} devices: {ms:.7f} ms (analytical, as CENT)"
    return _NO_RECORDS


def layer_trace(
    mapping: LayerMapping,
    arch: CentArchitecture,
    issue: ChannelIssue = ChannelIssue.BROADCAST,
) -> Iterator[str]:
    """Yield the layer's AiM trace, op by op, ending with EOC."""

    yield MAC_FROM_GLOBAL_BUFFER
    for op in mapping.ops:
        label = f"{op.NAME} ({op.name_hint})" if op.name_hint else op.NAME
        if op in mapping.covered_by:
            yield f"# {label}: in the kernel of {mapping.covered_by[op].name_hint}"
            continue
        comment = mapping.comments.get(op)
        yield f"# {label}: {comment}" if comment else f"# {label}"
        if op in mapping.kernels:
            yield from iter_aim_records(mapping.kernels[op], arch, issue=issue)
    if PIPELINE_STAGES > 1:
        hidden = _squeeze(mapping.ops[-1].results[0].type.shape)[-1]
        ms = analytical.pipeline_handoff_ms(hidden, PCIE_LANES // DEVICES)
        yield (
            f"# CXL hand-off to the next pipeline stage: {ms:.7f} ms "
            "(analytical, as CENT)"
        )
    yield "AiM EOC"


def device_ir(run: RunSpec, device: DeviceSpec) -> Function:
    """One rank's Device IR for ``run``: the function every rank runs."""

    module, _ = compile_module(load_model(MODEL), run, device)
    return module.functions[0]


def tile_text(mapping: LayerMapping) -> str:
    """The layer's kernels as Tile IR, in op order, each with its layouts.

    ``//`` lines name the Device IR ops a kernel runs and give each kernel
    tensor's physical layout; the functions are printed as the IR printer
    prints them.
    """

    sections = [f"module @{mapping.name}_tile"]
    for op in mapping.ops:
        kernel = mapping.kernels.get(op)
        if kernel is None:
            continue
        runs = [op, *(o for o, by in mapping.covered_by.items() if by is op)]
        header = [f"// {o.NAME} ({o.name_hint})" for o in runs]
        header += [
            f"// layout %{arg.name}: {kernel.layouts[arg].describe()}"
            for arg in kernel.function.arguments
        ]
        sections.append("\n".join([*header, print_function(kernel.function)]))
    return "\n\n".join(sections) + "\n"


def write_stages(out_dir: Path) -> list[Path]:
    """Write every stage of the evaluated run to ``out_dir``.

    Returns:
        The files written.
    """

    run = load_run_spec(RUN)
    device = load_device_spec(DEVICE)
    arch = device.architecture
    assert isinstance(arch, CentArchitecture)
    module, texts = compile_module(load_model(MODEL), run, device)
    mapping = map_layer(module.functions[0], arch, run.workload.context_len)
    mapping.name = module.name
    out_dir.mkdir(parents=True, exist_ok=True)
    files = {f"{stage}.aloi": text for stage, text in texts.items()}
    files["04_tile.aloi"] = tile_text(mapping)
    trace = layer_trace(mapping, arch, run.codegen.channel_issue)
    files["05_aim.trace"] = "".join(line + "\n" for line in trace)
    paths = []
    for name, text in files.items():
        path = out_dir / name
        path.write_text(text)
        paths.append(path)
    return paths


def main(argv: list[str]) -> None:
    """Write every stage to ``argv[0]``, or print the layer's trace."""

    if argv:
        for path in write_stages(Path(argv[0])):
            print(path)
        return
    run = load_run_spec(RUN)
    device = load_device_spec(DEVICE)
    arch = device.architecture
    assert isinstance(arch, CentArchitecture)
    mapping = map_layer(device_ir(run, device), arch, run.workload.context_len)
    for line in layer_trace(mapping, arch, run.codegen.channel_issue):
        print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
