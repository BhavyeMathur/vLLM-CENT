"""Run spec: how a model runs, read from YAML.

The model files say what the model computes; the run spec says how it runs:
tensor parallelism, the workload of one step, and how much of the model to
compile. Together they define the SPMD program the compiler produces::

    parallel:
      tp: 8
      split: [q_head, kv_head, intermediate]   # rules, by semantic axis
      replicate: []                            # exceptions: keep these whole
      split_only: {}                           # exceptions: axis -> [tensors]
    workload:
      phase: decode
      batch: 4
      context_len: 2048
    scope:
      layers: 1
      layer: 0          # which decoder layer: layer types may differ
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path
from types import MappingProxyType
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..parallel.plan import ParallelPlan
from ..utils import config

__all__ = [
    "ParallelSpec",
    "Phase",
    "RunSpec",
    "Scope",
    "Workload",
    "load_run_spec",
    "parse_run_spec",
]

# Names this kind of document in configuration error messages.
_RUN_SPEC = "run spec"


class Phase(StrEnum):
    """Which inference step is compiled.

    Members:
        DECODE: One new token per request, attending to a KV cache.
        PREFILL: The whole prompt at once (not supported yet).
    """

    DECODE = "decode"
    PREFILL = "prefill"


@dataclass(frozen=True, slots=True)
class ParallelSpec:
    """Tensor parallelism of the run.

    Attributes:
        tp: Number of tensor-parallel ranks.
        split: Axes split on every tensor that has them.
        replicate: Tensors kept whole despite the rules.
        split_only: Axis to the tensors split along it as an exception.
    """

    tp: int = 1
    split: tuple[str, ...] = ()
    replicate: tuple[str, ...] = ()
    split_only: Mapping[str, tuple[str, ...]] = field(
        default_factory=lambda: MappingProxyType({})
    )

    def build_plan(self) -> ParallelPlan:
        """Return the ParallelPlan these settings describe.

        Returns:
            A new plan.

        Raises:
            ValueError: If the settings conflict; see ParallelPlan.
        """

        plan = ParallelPlan(tp=self.tp)
        for axis in self.split:
            plan.split(axis)
        if self.replicate:
            plan.replicate(*self.replicate)
        for axis, names in self.split_only.items():
            plan.split(axis, only=names)
        return plan


@dataclass(frozen=True, slots=True)
class Workload:
    """The shapes of one compiled step.

    Attributes:
        phase: Decode or prefill.
        batch: Number of requests.
        context_len: KV cache positions, including the new token.
    """

    phase: Phase = Phase.DECODE
    batch: int = 1
    context_len: int = 2048


@dataclass(frozen=True, slots=True)
class Scope:
    """How much of the model is compiled.

    Attributes:
        layers: Number of decoder layers.
        layer: Index of the first compiled layer. It matters when layers
            differ, e.g. Llama 4 interleaves layers with and without RoPE.
    """

    layers: int = 1
    layer: int = 0


@dataclass(frozen=True, slots=True)
class RunSpec:
    """Everything about how a model runs.

    Attributes:
        parallel: Tensor parallelism.
        workload: Shapes of the compiled step.
        scope: How much of the model is compiled.
    """

    parallel: ParallelSpec = ParallelSpec()
    workload: Workload = Workload()
    scope: Scope = Scope()


def parse_run_spec(data: Mapping[str, Any]) -> RunSpec:
    """Build a RunSpec from a parsed YAML document; omitted keys get defaults.

    Args:
        data: The document.

    Returns:
        The run spec.

    Raises:
        ValueError: For unknown sections or keys, or invalid values.
    """

    unknown = sorted(set(data) - {"parallel", "workload", "scope"})
    if unknown:
        raise ValueError(f"unknown run spec sections: {unknown}")

    parallel = config.section(
        _RUN_SPEC, data, "parallel", {"tp", "split", "replicate", "split_only"}
    )
    split_only = parallel.get("split_only") or {}
    if not isinstance(split_only, Mapping):
        raise ValueError("run spec parallel.split_only must map axes to tensor names")
    parallel_spec = ParallelSpec(
        tp=config.positive_int(_RUN_SPEC, "parallel.tp", parallel.get("tp", 1)),
        split=config.names(_RUN_SPEC, "parallel.split", parallel.get("split", [])),
        replicate=config.names(
            _RUN_SPEC, "parallel.replicate", parallel.get("replicate", [])
        ),
        split_only=MappingProxyType(
            {
                str(axis): config.names(_RUN_SPEC, f"parallel.split_only.{axis}", names)
                for axis, names in split_only.items()
            }
        ),
    )

    workload = config.section(
        _RUN_SPEC, data, "workload", {"phase", "batch", "context_len"}
    )
    phase = workload.get("phase", Phase.DECODE.value)
    try:
        phase = Phase(phase)
    except ValueError as error:
        raise ValueError(
            f"run spec workload.phase must be one of {[p.value for p in Phase]}, "
            f"got {phase!r}"
        ) from error
    workload_spec = Workload(
        phase=phase,
        batch=config.positive_int(
            _RUN_SPEC, "workload.batch", workload.get("batch", 1)
        ),
        context_len=config.positive_int(
            _RUN_SPEC, "workload.context_len", workload.get("context_len", 2048)
        ),
    )

    scope = config.section(_RUN_SPEC, data, "scope", {"layers", "layer"})
    scope_spec = Scope(
        layers=config.positive_int(_RUN_SPEC, "scope.layers", scope.get("layers", 1)),
        layer=config.nonnegative_int(_RUN_SPEC, "scope.layer", scope.get("layer", 0)),
    )
    return RunSpec(parallel=parallel_spec, workload=workload_spec, scope=scope_spec)


def load_run_spec(path: Path) -> RunSpec:
    """Read a run spec YAML file.

    Args:
        path: The file.

    Returns:
        The run spec.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the document is not a mapping or is invalid.
    """

    data = yaml.safe_load(path.read_text()) or {}
    if not isinstance(data, Mapping):
        raise ValueError(f"run spec {path} must be a YAML mapping")
    return parse_run_spec(data)
