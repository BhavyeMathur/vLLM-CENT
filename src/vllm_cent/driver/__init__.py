"""Compiler driver: a model (HF config + registered code) and a run spec in,
the IR of every stage out.

    aloi-compile --model path/to/config.json --run path/to/run.yaml
"""

from .loader import LoadedModel, load_model
from .registry import ModelRegistration, get_registration, register_model
from .run_spec import (
    ParallelSpec,
    Phase,
    RunSpec,
    Scope,
    Workload,
    load_run_spec,
    parse_run_spec,
)
from .stages import STAGES, compile_stages

__all__ = [
    "STAGES",
    "LoadedModel",
    "ModelRegistration",
    "ParallelSpec",
    "Phase",
    "RunSpec",
    "Scope",
    "Workload",
    "compile_stages",
    "get_registration",
    "load_model",
    "load_run_spec",
    "parse_run_spec",
    "register_model",
]
