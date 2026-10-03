"""ALOI's policy for running ``torch.export`` on a model."""

from __future__ import annotations

import torch
from torch import Tensor
from torch.export import ExportedProgram

# Imported for its side effect: the model's forward calls torch.ops.aloi.*,
# which exist only after this module has registered them.
from . import custom_ops  # noqa: F401

__all__ = ["export_model"]


def export_model(
    model: torch.nn.Module, example_inputs: tuple[Tensor, ...]
) -> ExportedProgram:
    """Export ``model`` into an FX graph that the importer can read.

    The policy is deliberately fixed:

    - ``strict=False`` runs forward() as plain Python instead of tracing it
      with TorchDynamo.
    - No ``dynamic_shapes``: every dimension is specialized to the example
      inputs, because ALOI handles static shapes only.
    - No ``run_decompositions()``: it would break high-level aten ops into
      primitives. Semantic IR keeps ops whole; later stages decompose them.

    Args:
        model: Model to export; meta-device parameters are fine.
        example_inputs: Positional ``forward`` arguments that fix the shapes.

    Returns:
        The exported program.
    """

    return torch.export.export(model, example_inputs, strict=False)
