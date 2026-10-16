"""Connect program input and output names to locations in CENT memory."""

from .bindings import (
    CentAccumulatorRegion,
    CentDramRegion,
    CentGlobalBufferRegion,
    CentInputBinding,
    CentNamedScalars,
    CentOutputBinding,
    CentPhysicalRegion,
    CentSharedBufferRegion,
)
from .executable import CentExecutable, CentExecutionManifest

__all__ = [
    "CentAccumulatorRegion",
    "CentDramRegion",
    "CentExecutable",
    "CentExecutionManifest",
    "CentGlobalBufferRegion",
    "CentInputBinding",
    "CentNamedScalars",
    "CentOutputBinding",
    "CentPhysicalRegion",
    "CentSharedBufferRegion",
]
