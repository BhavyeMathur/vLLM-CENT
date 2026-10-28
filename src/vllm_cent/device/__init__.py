"""Target device descriptions: geometry, storage and supported Device IR ops.

``DeviceSpec`` is what lowering and mapping know about a target. It is
usually read from YAML with ``load_device_spec``.
"""

from .cent import CentArchitecture
from .loader import load_device_spec, parse_device_spec, snapshot
from .spec import (
    Architecture,
    DeviceCapability,
    DeviceSpec,
    MemorySpace,
    MemorySpaceSpec,
    TargetDescriptor,
    TraceFormat,
)

__all__ = [
    "Architecture",
    "CentArchitecture",
    "DeviceCapability",
    "DeviceSpec",
    "MemorySpace",
    "MemorySpaceSpec",
    "TargetDescriptor",
    "TraceFormat",
    "load_device_spec",
    "parse_device_spec",
    "snapshot",
]
