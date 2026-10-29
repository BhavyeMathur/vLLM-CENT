"""Run the compiler pipeline on a model and a run spec; keep every stage."""

from __future__ import annotations

from ..device import DeviceSpec
from ..frontend import export_model, import_exported_program
from ..ir import AnnotateRolesAndAxes, Module, Pass, PassManager, print_module
from ..ir.passes import ApplyTensorParallel
from ..legalize import LowerToDevice
from .loader import LoadedModel
from .run_spec import Phase, RunSpec

__all__ = ["DEVICE_STAGE", "STAGES", "compile_stages"]

# Stage names, in pipeline order. They name the dump files: 00_imported.aloi.
STAGES = ("00_imported", "01_semantic", "02_sharded")

# The stage after STAGES, produced only when a device spec is given.
DEVICE_STAGE = "03_device"

# Stage produced by each pass, keyed by pass name.
_PASS_STAGES = {
    AnnotateRolesAndAxes.NAME: "01_semantic",
    ApplyTensorParallel.NAME: "02_sharded",
    LowerToDevice.NAME: DEVICE_STAGE,
}


def compile_stages(
    model: LoadedModel, run: RunSpec, device: DeviceSpec | None = None
) -> dict[str, str]:
    """Compile one step of ``model`` as ``run`` describes; print every stage.

    Stages:

    - ``00_imported``: torch.export graph translated to IR (shapes, dtypes).
    - ``01_semantic``: after AnnotateRolesAndAxes (roles and axes).
    - ``02_sharded``: after ApplyTensorParallel (local shapes, placements,
      all_reduces); the SPMD IR every tensor-parallel rank runs.
    - ``03_device``: after LowerToDevice, only when ``device`` is given; the
      device ops of that target, with the transfers between domains.

    Args:
        model: Model family and config.
        run: How the model runs.
        device: Target to lower to; None stops at ``02_sharded``.

    Returns:
        Stage name to IR text, in STAGES order, then DEVICE_STAGE if
        ``device`` is given.

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
        registration.build_model(config, run.scope.layer),
        registration.example_inputs(
            config, run.workload.batch, run.workload.context_len
        ),
    )
    module = import_exported_program(program, model.name)
    texts = {STAGES[0]: print_module(module)}

    def record(pass_name: str, stage: Module) -> None:
        """Keep the text of the IR after each pass."""

        texts[_PASS_STAGES[pass_name]] = print_module(stage)

    passes: list[Pass] = [
        AnnotateRolesAndAxes(registration.annotations(config, run.scope.layer)),
        ApplyTensorParallel(run.parallel.build_plan(), registration.axis_sizes(config)),
    ]
    if device is not None:
        passes.append(LowerToDevice(device))
    PassManager(tuple(passes), dump=record).run(module)
    return texts
