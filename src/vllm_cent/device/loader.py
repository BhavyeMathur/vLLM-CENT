"""Read a DeviceSpec from YAML, and write it back as a resolved snapshot.

A device spec document has this shape::

    schema_version: "0.1"
    name: cent_paper
    target: {target_id: cent_aim, trace_format: aim_text_trace}
    architecture:
      type: cent                 # selects the architecture class
      channels: 32               # ... its fields; every *_size is in bytes
    capabilities:
      ops:                       # supported Device IR ops and their dtypes,
        pim: {gemv: [bf16]}      # grouped by the op name's prefix: pim.gemv
      pim_activations: {sigmoid: 0}    # function -> AFid
      pnm_activations: [silu]
      pim_gemv_bias: false

Reading is strict: an unknown key, op name, dtype or activation is an error
that names its dotted path, rather than being ignored. ``snapshot()`` writes
the same shape with every value resolved and every collection sorted, so its
output can be read back and compared.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml  # type: ignore[import-untyped]

from ..activations import ActivationKind
from ..ir.base import Operation, operation_class, registered_operations

# Importing ir.device also defines every Device IR op class, which registers
# its name for the operation_class() lookups below.
from ..ir.device import DeviceOp, ViewOp
from ..ir.types import DType
from ..utils import config
from .cent import CentArchitecture
from .spec import (
    Architecture,
    DeviceCapability,
    DeviceSpec,
    TargetDescriptor,
    TraceFormat,
)

__all__ = ["load_device_spec", "parse_device_spec", "snapshot"]

# The only device spec format this module reads and writes.
SCHEMA_VERSION = "0.1"

# Names this kind of document in configuration error messages.
_DEVICE_SPEC = "device spec"

# architecture.type -> the class whose fields the section lists.
_ARCHITECTURES: dict[str, type[CentArchitecture]] = {"cent": CentArchitecture}


def load_device_spec(path: str | Path) -> DeviceSpec:
    """Read a device spec YAML file.

    Args:
        path: The YAML file.

    Returns:
        The device spec it describes.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the document does not describe a valid device spec;
            see ``parse_device_spec``.
    """

    with open(path, encoding="utf-8") as file:
        data = yaml.safe_load(file)
    if not isinstance(data, Mapping):
        raise ValueError(f"{_DEVICE_SPEC} {path} must be a mapping")
    return parse_device_spec(data)


def parse_device_spec(data: Mapping[str, Any]) -> DeviceSpec:
    """Build a device spec from a parsed YAML or JSON document.

    Args:
        data: The whole document.

    Returns:
        The device spec.

    Raises:
        ValueError: If a section or key is unknown or missing, the schema
            version is not supported, or a value is invalid (including the
            checks of ``CentArchitecture`` and ``DeviceCapability``).
    """

    unknown = sorted(
        set(data) - {"schema_version", "name", "target", "architecture", "capabilities"}
    )
    if unknown:
        raise ValueError(f"unknown {_DEVICE_SPEC} sections: {unknown}")
    if data.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{_DEVICE_SPEC} schema_version must be {SCHEMA_VERSION!r}, "
            f"got {data.get('schema_version')!r}"
        )
    name = data.get("name")
    if not isinstance(name, str) or not name:
        raise ValueError(f"{_DEVICE_SPEC} name must be a non-empty string")
    return DeviceSpec(
        name=name,
        target=_target(data),
        architecture=_architecture(data),
        capabilities=_capabilities(data),
    )


def snapshot(spec: DeviceSpec) -> dict[str, Any]:
    """Write a device spec as a plain document that ``parse_device_spec`` reads.

    Args:
        spec: The device spec.

    Returns:
        A mapping of strings, numbers and lists only (JSON- and
        YAML-serializable), with the ops grouped by name prefix and every
        collection sorted, so equal specs give equal snapshots.

    Raises:
        TypeError: If the architecture class has no ``architecture.type``.
    """

    architecture = spec.architecture
    architecture_type = next(
        (key for key, cls in _ARCHITECTURES.items() if isinstance(architecture, cls)),
        None,
    )
    if architecture_type is None or not isinstance(architecture, CentArchitecture):
        raise TypeError(f"no architecture.type for {type(architecture).__name__}")
    capabilities = spec.capabilities
    ops: dict[str, dict[str, list[str]]] = {}
    for op, dtypes in sorted(capabilities.ops.items(), key=lambda item: item[0].NAME):
        namespace, _, short_name = op.NAME.partition(".")
        ops.setdefault(namespace, {})[short_name] = sorted(d.value for d in dtypes)
    return {
        "schema_version": SCHEMA_VERSION,
        "name": spec.name,
        "target": {
            "target_id": spec.target.target_id,
            "trace_format": spec.target.trace_format.value,
        },
        "architecture": {
            "type": architecture_type,
            **dataclasses.asdict(architecture),
        },
        "capabilities": {
            "ops": ops,
            "pim_activations": {
                function.value: function_id
                for function, function_id in sorted(
                    capabilities.pim_activations.items()
                )
            },
            "pnm_activations": sorted(f.value for f in capabilities.pnm_activations),
            "pim_gemv_bias": capabilities.pim_gemv_bias,
        },
    }


def _target(data: Mapping[str, Any]) -> TargetDescriptor:
    """Read the ``target`` section.

    Args:
        data: The whole document.

    Returns:
        The target descriptor.

    Raises:
        ValueError: If a key is unknown, ``target_id`` is missing or empty, or
            ``trace_format`` is not a ``TraceFormat`` value.
    """

    target = config.section(_DEVICE_SPEC, data, "target", {"target_id", "trace_format"})
    target_id = target.get("target_id")
    if not isinstance(target_id, str) or not target_id:
        raise ValueError(f"{_DEVICE_SPEC} target.target_id must be a non-empty string")
    trace_format = target.get("trace_format")
    try:
        return TargetDescriptor(
            target_id=target_id, trace_format=TraceFormat(str(trace_format))
        )
    except ValueError:
        raise ValueError(
            f"{_DEVICE_SPEC} target.trace_format must be one of "
            f"{[f.value for f in TraceFormat]}, got {trace_format!r}"
        ) from None


def _architecture(data: Mapping[str, Any]) -> Architecture:
    """Read the ``architecture`` section into its architecture class.

    Args:
        data: The whole document.

    Returns:
        The architecture; ``type`` selects the class and the other keys are
        its fields, each a positive int.

    Raises:
        ValueError: If the type is unknown, a field is unknown, missing or not
            a positive int, or the class rejects the geometry.
    """

    # The type decides which keys are allowed, so read it before checking
    # the section's keys.
    raw = data.get("architecture")
    architecture_type = raw.get("type") if isinstance(raw, Mapping) else None
    architecture_class = _ARCHITECTURES.get(str(architecture_type))
    if architecture_class is None:
        raise ValueError(
            f"{_DEVICE_SPEC} architecture.type must be one of "
            f"{sorted(_ARCHITECTURES)}, got {architecture_type!r}"
        )
    field_names = {field.name for field in dataclasses.fields(architecture_class)}
    section = config.section(_DEVICE_SPEC, data, "architecture", {"type", *field_names})
    missing = sorted(field_names - set(section))
    if missing:
        raise ValueError(f"{_DEVICE_SPEC} architecture is missing {missing}")
    values = {
        key: config.positive_int(_DEVICE_SPEC, f"architecture.{key}", section[key])
        for key in sorted(field_names)
    }
    return architecture_class(**values)


def _capabilities(data: Mapping[str, Any]) -> DeviceCapability:
    """Read the ``capabilities`` section.

    Args:
        data: The whole document.

    Returns:
        The capabilities.

    Raises:
        ValueError: If a key, op name, dtype or activation is unknown, a value
            has the wrong type, or ``DeviceCapability`` rejects the result.
    """

    section = config.section(
        _DEVICE_SPEC,
        data,
        "capabilities",
        {"ops", "pim_activations", "pnm_activations", "pim_gemv_bias"},
    )
    pim_activations = section.get("pim_activations") or {}
    if not isinstance(pim_activations, Mapping):
        raise ValueError(
            f"{_DEVICE_SPEC} capabilities.pim_activations must be a mapping"
        )
    bias = section.get("pim_gemv_bias", False)
    if not isinstance(bias, bool):
        raise ValueError(
            f"{_DEVICE_SPEC} capabilities.pim_gemv_bias must be true or false, got {bias!r}"
        )
    return DeviceCapability(
        ops=_ops(section),
        pim_activations={
            _activation(f"capabilities.pim_activations.{name}", name): (
                config.nonnegative_int(
                    _DEVICE_SPEC, f"capabilities.pim_activations.{name}", function_id
                )
            )
            for name, function_id in pim_activations.items()
        },
        pnm_activations=frozenset(
            _activation("capabilities.pnm_activations", name)
            for name in config.names(
                _DEVICE_SPEC,
                "capabilities.pnm_activations",
                section.get("pnm_activations", []),
            )
        ),
        pim_gemv_bias=bias,
    )


def _ops(capabilities: Mapping[str, Any]) -> dict[type[Operation], frozenset[DType]]:
    """Read ``capabilities.ops``: device op names, grouped by prefix, to dtypes.

    Args:
        capabilities: The ``capabilities`` section.

    Returns:
        Each listed op class with its dtypes.

    Raises:
        ValueError: If a prefix is not a device op namespace, an op name is not
            registered, an op is a view (always supported, so never listed),
            or a dtype is unknown.
    """

    groups = config.section(
        _DEVICE_SPEC, capabilities, "capabilities.ops", _device_namespaces()
    )
    ops: dict[type[Operation], frozenset[DType]] = {}
    for namespace, entries in groups.items():
        if not isinstance(entries, Mapping):
            raise ValueError(
                f"{_DEVICE_SPEC} capabilities.ops.{namespace} must be a mapping"
            )
        for short_name, dtype_names in entries.items():
            path = f"capabilities.ops.{namespace}.{short_name}"
            try:
                op = operation_class(f"{namespace}.{short_name}")
            except KeyError as error:
                raise ValueError(f"{_DEVICE_SPEC} {path}: {error.args[0]}") from error
            if issubclass(op, ViewOp):
                raise ValueError(
                    f"{_DEVICE_SPEC} {path}: views are always supported; remove it"
                )
            ops[op] = frozenset(
                _dtype(path, name)
                for name in config.names(_DEVICE_SPEC, path, dtype_names)
            )
    return ops


def _device_namespaces() -> set[str]:
    """Return the name prefixes of the Device IR ops, e.g. ``pim`` of ``pim.gemv``.

    These are the groups allowed under ``capabilities.ops``. They are read
    from the registered ``DeviceOp`` classes, so a new kind of device op needs
    no change here.

    Returns:
        The prefixes before the dot in every device op's ``NAME``.
    """

    return {
        op.NAME.partition(".")[0]
        for op in registered_operations().values()
        if issubclass(op, DeviceOp)
    }


def _dtype(path: str, name: str) -> DType:
    """Read one dtype name.

    Args:
        path: Dotted path of the setting, for error messages.
        name: A ``DType`` value, e.g. ``"bf16"``.

    Returns:
        The dtype.

    Raises:
        ValueError: If ``name`` is not a ``DType`` value.
    """

    try:
        return DType(name)
    except ValueError:
        raise ValueError(
            f"{_DEVICE_SPEC} {path}: unknown dtype {name!r}; "
            f"use one of {[d.value for d in DType]}"
        ) from None


def _activation(path: str, name: str) -> ActivationKind:
    """Read one activation function name.

    Args:
        path: Dotted path of the setting, for error messages.
        name: An ``ActivationKind`` value, e.g. ``"silu"``.

    Returns:
        The activation function.

    Raises:
        ValueError: If ``name`` is not an ``ActivationKind`` value.
    """

    try:
        return ActivationKind(name)
    except ValueError:
        raise ValueError(
            f"{_DEVICE_SPEC} {path}: unknown activation {name!r}; "
            f"use one of {[k.value for k in ActivationKind]}"
        ) from None
