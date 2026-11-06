"""Map one Llama 3.1 70B decoder layer onto CENT, as CENT's simulator does.

The run spec ``runs/llama3_1_70b_kvtp8_decode.yaml`` names the evaluated
configuration (KV-head TP = 8, GDDR6, 2304 positions). This module takes
the Device IR of one rank for that run and gives each op the Tile DSL kernel
of CENT's mapping (``../ALOI/cent_simulation``), checks that the kernel's
tensors have the shapes of the op's operands, and writes the layer's AiM
trace.

Mapped so far: the GEMVs (``pim.gemv``) but for the gate projection, whose
trace fuses the activation (``AF``/``RD_AF``) and comes with
``pim.activation``. Every other op is written to the trace as a comment
naming it; they are mapped next (PIM side) or modeled analytically, as CENT
does (PNM and CXL).

Run from the repository root::

    python -m examples.llama_70b.kernel_mapping.llama3_1_70b [trace.txt]
"""

from __future__ import annotations

import sys
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from vllm_cent.device import CentArchitecture, DeviceSpec, load_device_spec
from vllm_cent.driver import RunSpec, compile_module, load_model, load_run_spec
from vllm_cent.ir import Function, Operation
from vllm_cent.ir.device import PIMGemvOp
from vllm_cent.tile import CentKernel
from vllm_cent.tile.codegen import MAC_FROM_GLOBAL_BUFFER, iter_aim_records
from vllm_cent.tile.dsl import TileKernel
from vllm_cent.tile.expand import ChannelIssue

from .gemv import kv_proj, reuse_gb_gemv

__all__ = ["WEIGHT_ROWS", "LayerMapping", "layer_trace", "map_layer"]

EXAMPLE_DIR = Path(__file__).resolve().parents[1]
MODEL = EXAMPLE_DIR / "llama3_1_70b.json"
RUN = EXAMPLE_DIR / "runs" / "llama3_1_70b_kvtp8_decode.yaml"
# CENT's MAC register count, so the GEMV order is CENT's (see cent_paper.yaml).
DEVICE = EXAMPLE_DIR.parents[0] / "cent" / "cent_paper.yaml"

# First DRAM row of each weight, as CENT's Llama.memory_mapping places them
# for KV-head TP = 8 with a 4096 window (Llama.py:1494-1748). The rows in
# between hold activations and the KV cache: x 0, x_copy 1, SANorm 2, xq 35,
# xk 36, cache_k 37, scores 38, cache_v 42, output 43, sa 60, sa_copy 61,
# FFNNorm 62, x1 175, x3 176, x1_sigmoid 177, ffn_vector 178.
WEIGHT_ROWS = {
    "self_attn.q_proj.weight": 3,
    "self_attn.k_proj.weight": 19,
    "self_attn.v_proj.weight": 27,
    "self_attn.o_proj.weight": 44,
    "mlp.gate_proj.weight": 63,
    "mlp.up_proj.weight": 119,
    "mlp.down_proj.weight": 179,
}

# Device IR ops whose kernel is not written yet: the trace names them.
_PENDING = "not mapped yet"


@dataclass(frozen=True)
class LayerMapping:
    """One layer's ops, in order, each with its kernel or None if not mapped.

    Attributes:
        ops: The Device IR function's ops.
        kernels: For each op, the kernel that runs it; None where the op is
            not mapped, or is run by another op's kernel (``covered_by``).
        covered_by: Ops run by an earlier op's kernel, e.g. ``v_proj`` by
            the fused K/V kernel of ``k_proj``.
    """

    ops: tuple[Operation, ...]
    kernels: dict[Operation, CentKernel]
    covered_by: dict[Operation, Operation]


def _squeeze(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Drop leading size-1 dimensions: ``[1, 1, 8192]`` is the vector 8192."""

    dims = list(shape)
    while len(dims) > 1 and dims[0] == 1:
        dims.pop(0)
    return tuple(dims)


def _check_gemv(op: PIMGemvOp, kernel: CentKernel, names: tuple[str, str, str]) -> None:
    """Check that the kernel's x, w and y have the op's operand shapes.

    Raises:
        ValueError: If a shape or dtype differs.
    """

    by_name = {a.name: a for a in kernel.function.arguments}
    expected = {
        names[0]: (_squeeze(op.operands[0].type.shape), op.operands[0].type.dtype),
        names[1]: (op.operands[1].type.shape, op.operands[1].type.dtype),
        names[2]: (_squeeze(op.results[0].type.shape), op.results[0].type.dtype),
    }
    for name, (shape, dtype) in expected.items():
        arg = by_name[name]
        if (arg.type.shape, arg.type.dtype) != (shape, dtype):
            raise ValueError(
                f"{op.name_hint}: kernel tensor {name!r} is {arg.type.shape} "
                f"{arg.type.dtype}, but the Device IR operand is {shape} {dtype}"
            )


def _gemv_dims(op: PIMGemvOp) -> tuple[int, int]:
    """``(N, K)`` of a GEMV: its weight is ``(N, K)``."""

    n, kdim = op.operands[1].type.shape
    return n, kdim


def _weight(op: PIMGemvOp) -> str:
    name = op.operands[1].name
    assert name is not None
    return name


def map_layer(function: Function, arch: CentArchitecture) -> LayerMapping:
    """Give each op of one rank's layer its CENT kernel, where one exists.

    Raises:
        ValueError: If a kernel's tensors do not match its op's shapes.
    """

    ops = function.operations
    kernels: dict[Operation, CentKernel] = {}
    covered: dict[Operation, Operation] = {}
    gemvs = {op.name_hint: op for op in ops if isinstance(op, PIMGemvOp)}

    for op in ops:
        if not isinstance(op, PIMGemvOp) or op in covered:
            continue
        name = op.name_hint or "gemv"
        if name == "mlp.gate_proj":
            continue  # fused with its activation: mapped with pim.activation
        n, kdim = _gemv_dims(op)
        if name == "self_attn.k_proj":
            v_op = gemvs["self_attn.v_proj"]
            k_names = ("x", _weight(op), "k")
            v_names = ("x", _weight(v_op), "v")
            with TileKernel("kv_proj", arch) as k:
                kv_proj(
                    k,
                    n=n,
                    kdim=kdim,
                    k_names=k_names,
                    v_names=v_names,
                    k_row=WEIGHT_ROWS[_weight(op)],
                    v_row=WEIGHT_ROWS[_weight(v_op)],
                )
            kernel = k.build()
            _check_gemv(op, kernel, k_names)
            _check_gemv(v_op, kernel, v_names)
            covered[v_op] = op
        else:
            names = ("x", _weight(op), "y")
            with TileKernel(name.replace(".", "_"), arch) as k:
                reuse_gb_gemv(
                    k, n=n, kdim=kdim, names=names, w_row=WEIGHT_ROWS[names[1]]
                )
            kernel = k.build()
            _check_gemv(op, kernel, names)
        kernels[op] = kernel
    return LayerMapping(ops, kernels, covered)


def layer_trace(
    mapping: LayerMapping,
    arch: CentArchitecture,
    issue: ChannelIssue = ChannelIssue.BROADCAST,
) -> Iterator[str]:
    """Yield the layer's AiM trace, op by op; unmapped ops as comments."""

    yield MAC_FROM_GLOBAL_BUFFER
    for op in mapping.ops:
        label = f"{op.NAME} ({op.name_hint})" if op.name_hint else op.NAME
        if op in mapping.kernels:
            yield f"# {label}"
            yield from iter_aim_records(mapping.kernels[op], arch, issue=issue)
        elif op in mapping.covered_by:
            yield f"# {label}: in the kernel of {mapping.covered_by[op].name_hint}"
        else:
            yield f"# {label}: {_PENDING}"
    yield "AiM EOC"


def device_ir(run: RunSpec, device: DeviceSpec) -> Function:
    """One rank's Device IR for ``run``: the function every rank runs."""

    module, _ = compile_module(load_model(MODEL), run, device)
    return module.functions[0]


def main(argv: list[str]) -> None:
    """Write the layer's trace to ``argv[0]``, or print it."""

    run = load_run_spec(RUN)
    device = load_device_spec(DEVICE)
    arch = device.architecture
    assert isinstance(arch, CentArchitecture)
    mapping = map_layer(device_ir(run, device), arch)
    lines = layer_trace(mapping, arch, run.codegen.channel_issue)
    if argv:
        with Path(argv[0]).open("w") as out:
            out.writelines(line + "\n" for line in lines)
    else:
        for line in lines:
            print(line)


if __name__ == "__main__":
    main(sys.argv[1:])
