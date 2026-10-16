"""Run the compiler pipeline on a model and a run spec; keep every stage."""

from __future__ import annotations

from ..frontend import export_model, import_exported_program
from ..ir import AnnotateRolesAndAxes, Module, PassManager, print_module
from ..parallel import ApplyTensorParallel
from .loader import LoadedModel
from .run_spec import Phase, RunSpec

__all__ = ["STAGES", "compile_stages"]

# Stage names, in pipeline order. They name the dump files: 00_imported.aloi.
STAGES = ("00_imported", "01_semantic", "02_sharded")

# Stage produced by each pass, keyed by pass name.
_PASS_STAGES = {
    AnnotateRolesAndAxes.NAME: "01_semantic",
    ApplyTensorParallel.NAME: "02_sharded",
}


def compile_stages(model: LoadedModel, run: RunSpec) -> dict[str, str]:
    """Compile one step of ``model`` as ``run`` describes; print every stage.

    Stages:

    - ``00_imported``: torch.export graph translated to IR (shapes, dtypes).
    - ``01_semantic``: after AnnotateRolesAndAxes (roles and axes).
    - ``02_sharded``: after ApplyTensorParallel (local shapes, placements,
      all_reduces); the SPMD IR every tensor-parallel rank runs.

    Args:
        model: Model family and config.
        run: How the model runs.

    Returns:
        Stage name to IR text, in STAGES order.

    Raises:
        NotImplementedError: For prefill, or more than one layer.
    """

    if run.workload.phase is not Phase.DECODE:
        raise NotImplementedError(f"phase {run.workload.phase.value!r}; only decode")
    # TODO(scope): Models are single decoder layers today; compiling N layers
    # needs models that stack them.
    if run.scope.layers != 1:
        raise NotImplementedError(f"scope.layers={run.scope.layers}; only 1")

    registration, config = model.registration, model.config
    program = export_model(
        registration.build_model(config),
        registration.example_inputs(
            config, run.workload.batch, run.workload.context_len
        ),
    )
    module = import_exported_program(program, model.name)
    texts = {STAGES[0]: print_module(module)}

    def record(pass_name: str, stage: Module) -> None:
        """Keep the text of the IR after each pass."""

        texts[_PASS_STAGES[pass_name]] = print_module(stage)

    PassManager(
        (
            AnnotateRolesAndAxes(registration.annotations(config)),
            ApplyTensorParallel(
                run.parallel.build_plan(), registration.axis_sizes(config)
            ),
        ),
        dump=record,
    ).run(module)
    return texts
