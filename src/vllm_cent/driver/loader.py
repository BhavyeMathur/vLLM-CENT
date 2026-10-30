"""Load a model from an HF config.json and its directory's registry.py."""

from __future__ import annotations

import importlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from .registry import ModelRegistration, get_registration

__all__ = ["LoadedModel", "load_model"]

# File in a model directory that registers the model family.
_REGISTRY_FILE = "registry.py"


@dataclass(frozen=True, slots=True)
class LoadedModel:
    """A model family with one parsed config.

    Attributes:
        name: Name for the IR module: the config file's stem, e.g.
            ``llama3_1_70b`` for ``llama3_1_70b.json``.
        registration: The family's registration.
        config: The typed config ``registration.parse_config`` returned.
    """

    name: str
    registration: ModelRegistration
    config: Any


def load_model(config_path: Path) -> LoadedModel:
    """Load an HF config.json and the model family that handles it.

    The ``registry.py`` next to the config is imported, which registers the
    family; the config's ``model_type`` then selects it.

    Args:
        config_path: The HF config.json.

    Returns:
        The loaded model.

    Raises:
        FileNotFoundError: If the config or the registry.py is missing.
        ValueError: If the config has no ``model_type``.
        KeyError: If no family registered the config's ``model_type``.
    """

    config_path = config_path.resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"model config not found: {config_path}")
    registry_path = config_path.parent / _REGISTRY_FILE
    if not registry_path.is_file():
        raise FileNotFoundError(
            f"no {_REGISTRY_FILE} next to {config_path.name} in {config_path.parent}"
        )
    hf_config = json.loads(config_path.read_text())
    model_type = hf_config.get("model_type")
    if not isinstance(model_type, str):
        raise ValueError(f"{config_path} has no model_type")

    _import_file(registry_path)
    registration = get_registration(model_type)
    return LoadedModel(
        name=config_path.stem,
        registration=registration,
        config=registration.parse_config(hf_config),
    )


def _import_file(path: Path) -> ModuleType:
    """Import a Python file as a module, keeping its package.

    A file inside a package (directories with ``__init__.py``) is imported
    under its dotted name, e.g. ``examples.llama_70b.registry``, so relative
    imports work and importing it twice registers once. The directory above
    the top package is added to ``sys.path``.

    Args:
        path: The ``.py`` file.

    Returns:
        The imported module.
    """

    parts = [path.stem]
    root = path.parent
    while (root / "__init__.py").is_file():
        parts.insert(0, root.name)
        root = root.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return importlib.import_module(".".join(parts))
