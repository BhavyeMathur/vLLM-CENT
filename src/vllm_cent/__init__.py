"""Expose the structural model compiler, CENT types, and functional executor.

The model compiler builds instruction sequences for a Llama block. The executor
runs only instructions with implemented numerical behavior. The newer graph
compiler is available separately from ``vllm_cent.pipeline``; importing this package
does not initialize a vLLM worker or run a model.
"""

from .cent import (
    CentBlockPlacementSpec,
    CentHardwareSpec,
    CentInstruction,
    CentOpcode,
    CentProgram,
    render_aim_trace,
    render_text_program,
)
from .compiler import compile_transformer_block
from .models import LlamaModelSpec
from .request import CompileRequest, DecodeStepSpec

from .runtime import (
    CentDramRegion,
    CentExecutable,
    CentExecutionManifest,
    CentGlobalBufferRegion,
    CentInputBinding,
    CentNamedScalars,
    CentOutputBinding,
    CentPhysicalRegion,
    CentSharedBufferRegion,
)
from .simulator import (
    CentExecutionEvent,
    CentExecutionFault,
    CentExecutionRequest,
    CentExecutionResult,
    CentManifestError,
    CentNumericProfile,
    CentSimulationError,
    CentSimulationLocation,
    CentSimulatorConfiguration,
    CentTraceLevel,
    CentUninitializedReadError,
    CentUnsupportedSemanticsError,
    execute_functionally,
)

__all__ = [
    "CentBlockPlacementSpec",
    "CentDramRegion",
    "CentExecutable",
    "CentExecutionEvent",
    "CentExecutionFault",
    "CentExecutionManifest",
    "CentExecutionRequest",
    "CentExecutionResult",
    "CentGlobalBufferRegion",
    "CentHardwareSpec",
    "CentInputBinding",
    "CentInstruction",
    "CentManifestError",
    "CentNamedScalars",
    "CentNumericProfile",
    "CentOpcode",
    "CentOutputBinding",
    "CentPhysicalRegion",
    "CentProgram",
    "CentSharedBufferRegion",
    "CentSimulationError",
    "CentSimulationLocation",
    "CentSimulatorConfiguration",
    "CentTraceLevel",
    "CentUninitializedReadError",
    "CentUnsupportedSemanticsError",
    "CompileRequest",
    "DecodeStepSpec",
    "LlamaModelSpec",
    "compile_transformer_block",
    "execute_functionally",
    "render_aim_trace",
    "render_text_program",
]
