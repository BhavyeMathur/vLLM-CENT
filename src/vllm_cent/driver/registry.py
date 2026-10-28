"""Model registry: model families register themselves by HF model_type.

A model directory holds HF config files, the model code, and a
``registry.py`` that calls ``register_model``. The compiler imports that
``registry.py`` and finds the family by the config's ``model_type``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from torch import Tensor, nn

from ..ir import TensorAnnotation

__all__ = ["ModelRegistration", "get_registration", "register_model"]


@dataclass(frozen=True, slots=True)
class ModelRegistration:
    """How the compiler builds and describes one model family.

    The typed config is whatever ``parse_config`` returns; the other
    callables receive it.

    Attributes:
        model_type: HF ``model_type`` this family handles, e.g. ``"llama"``.
        parse_config: Turns a parsed HF config.json into the typed config.
        build_model: Creates the model for ``(config, layer)``, normally with
            parameters on the meta device so that full-size models export
            without memory. ``layer`` is the decoder layer index, for models
            whose layers differ.
        example_inputs: Sample ``forward`` arguments for
            ``(config, batch, context_len)`` of one decode step.
        annotations: Role and axes of every ``forward`` argument, parameter
            and buffer of the model for ``(config, layer)``, keyed by name.
        axis_sizes: Size of each axis that is the outer factor of a packed
            dimension, e.g. ``{"q_head": 64}``; TP needs it to split whole
            heads.
    """

    model_type: str
    parse_config: Callable[[Mapping[str, Any]], Any]
    build_model: Callable[[Any, int], nn.Module]
    example_inputs: Callable[[Any, int, int], tuple[Tensor, ...]]
    annotations: Callable[[Any, int], Mapping[str, TensorAnnotation]]
    axis_sizes: Callable[[Any], Mapping[str, int]]


# model_type -> registration. Filled by the registry.py of each model directory.
_REGISTRY: dict[str, ModelRegistration] = {}


def register_model(
    model_type: str,
    *,
    parse_config: Callable[[Mapping[str, Any]], Any],
    build_model: Callable[[Any, int], nn.Module],
    example_inputs: Callable[[Any, int, int], tuple[Tensor, ...]],
    annotations: Callable[[Any, int], Mapping[str, TensorAnnotation]],
    axis_sizes: Callable[[Any], Mapping[str, int]],
) -> ModelRegistration:
    """Register a model family; called from a model directory's registry.py.

    Args:
        model_type: HF ``model_type`` the family handles.
        parse_config: See ModelRegistration.
        build_model: See ModelRegistration.
        example_inputs: See ModelRegistration.
        annotations: See ModelRegistration.
        axis_sizes: See ModelRegistration.

    Returns:
        The registration.

    Raises:
        ValueError: If another family already registered ``model_type``.
    """

    if model_type in _REGISTRY:
        raise ValueError(f"model_type {model_type!r} is already registered")
    registration = ModelRegistration(
        model_type=model_type,
        parse_config=parse_config,
        build_model=build_model,
        example_inputs=example_inputs,
        annotations=annotations,
        axis_sizes=axis_sizes,
    )
    _REGISTRY[model_type] = registration
    return registration


def get_registration(model_type: str) -> ModelRegistration:
    """Return the family registered for ``model_type``.

    Args:
        model_type: HF ``model_type``.

    Returns:
        The registration.

    Raises:
        KeyError: If no family registered it.
    """

    registration = _REGISTRY.get(model_type)
    if registration is None:
        raise KeyError(
            f"no model registered for model_type {model_type!r}; "
            f"registered: {sorted(_REGISTRY)}"
        )
    return registration
