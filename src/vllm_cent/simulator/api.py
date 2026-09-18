"""Public manifest-facing API for functional CENT execution."""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import assert_never

from vllm_cent.cent import CentInstruction
from vllm_cent.runtime import (
    CentDramRegion,
    CentExecutable,
    CentGlobalBufferRegion,
    CentInputBinding,
    CentNamedScalars,
    CentOutputBinding,
    CentPhysicalRegion,
    CentSharedBufferRegion,
)
from .errors import CentManifestError
from .execution import (
    CentExecutionSummary,
    _execute_preflighted_program,
    preflight_program,
)
from .numeric import CentNumericSemantics, ReferenceMathSemantics
from .state import CentDeviceState

__all__ = [
    "CentExecutionEvent",
    "CentExecutionRequest",
    "CentExecutionResult",
    "CentNumericProfile",
    "CentSimulatorConfiguration",
    "CentTraceLevel",
    "execute_functionally",
]


class CentNumericProfile(Enum):
    """Select the numeric contract used by functional execution.

    Members:
        REFERENCE_MATH: Python float arithmetic for compiler dataflow checks.
    """

    REFERENCE_MATH = auto()


class CentTraceLevel(Enum):
    """Select how much successful instruction tracing a request returns.

    Members:
        NONE: Return outputs and an instruction count without events.
        SUMMARY: Return one typed event for every committed instruction.
    """

    NONE = auto()
    SUMMARY = auto()


@dataclass(frozen=True, slots=True, kw_only=True)
class CentSimulatorConfiguration:
    """Configure one deterministic functional execution.

    Attributes:
        numeric_profile: Arithmetic contract used for state and instructions.
        trace_level: Amount of typed execution tracing returned to the caller.
    """

    numeric_profile: CentNumericProfile = CentNumericProfile.REFERENCE_MATH
    trace_level: CentTraceLevel = CentTraceLevel.NONE


@dataclass(frozen=True, slots=True, kw_only=True)
class CentExecutionRequest:
    """Combine a reusable executable with named values for one run.

    Attributes:
        executable: Program and raw physical runtime manifest to execute.
        inputs: Raw values matched to manifest input bindings by name.
        configuration: Numeric and trace configuration for this run.
    """

    executable: CentExecutable
    inputs: tuple[CentNamedScalars, ...] = ()
    configuration: CentSimulatorConfiguration = field(
        default_factory=CentSimulatorConfiguration
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class CentExecutionEvent:
    """Describe one instruction committed by a successful execution.

    Attributes:
        device_id: Zero-based simulated device identifier.
        instruction_index: Zero-based position in the program.
        instruction: Typed instruction that committed at this position.
        reads: Physical source regions consumed by the instruction.
        writes: Physical destination regions changed by its committed effects.
    """

    device_id: int
    instruction_index: int
    instruction: CentInstruction
    reads: tuple[CentPhysicalRegion, ...]
    writes: tuple[CentPhysicalRegion, ...]

    def __post_init__(self) -> None:
        """Reject negative event coordinates.

        Raises:
            ValueError: If an event coordinate is negative.
        """

        if self.device_id < 0:
            raise ValueError("device_id cannot be negative")
        if self.instruction_index < 0:
            raise ValueError("instruction_index cannot be negative")


@dataclass(frozen=True, slots=True, kw_only=True)
class CentExecutionResult:
    """Return named raw outputs and deterministic execution metadata.

    Attributes:
        outputs: Values extracted in manifest output-binding order.
        events: Optional instruction summaries selected by the trace level.
        executed_instruction_count: Number of committed instructions.
    """

    outputs: tuple[CentNamedScalars, ...]
    events: tuple[CentExecutionEvent, ...]
    executed_instruction_count: int


def execute_functionally(request: CentExecutionRequest) -> CentExecutionResult:
    """Materialize, execute, and extract one CENT functional request.

    The entire program is preflighted before request values mutate fresh state.
    Inputs and outputs use raw physical spans only; this function infers no
    logical layout, packing, or zero-padding rule.

    Args:
        request: Executable, named input values, and simulator configuration.

    Returns:
        Named outputs plus the configured trace and instruction count.

    Raises:
        CentManifestError: If request inputs disagree with the manifest.
        CentSimulationError: If preflight or instruction execution fails.
    """

    executable = request.executable
    state = CentDeviceState(
        hardware=executable.program.hardware,
        numeric=_numeric_semantics(request.configuration.numeric_profile),
    )

    # Capability checks deliberately precede instruction zero and all runtime
    # writes, so a valid prefix cannot hide unsupported meaning later on.
    preflight_program(executable.program, state)
    try:
        ordered_inputs = executable.manifest.order_input_values(request.inputs)
    except ValueError as error:
        raise CentManifestError(str(error)) from error

    for binding, value in zip(
            executable.manifest.inputs,
            ordered_inputs,
            strict=True,
    ):
        _materialize_input(state, binding, value)

    summary = _execute_preflighted_program(executable.program, state)
    outputs = tuple(
        _extract_output(state, binding) for binding in executable.manifest.outputs
    )
    events = _execution_events(request, state, summary)
    return CentExecutionResult(
        outputs=outputs,
        events=events,
        executed_instruction_count=summary.executed_instruction_count,
    )


def _numeric_semantics(profile: CentNumericProfile) -> CentNumericSemantics:
    """Construct the arithmetic implementation for one public profile.

    Args:
        profile: Numeric profile requested by the caller.

    Returns:
        Numeric semantics used by state and instruction kernels.
    """

    if profile is CentNumericProfile.REFERENCE_MATH:
        return ReferenceMathSemantics()
    return assert_never(profile)  # pragma: no cover - closed enum guard


def _materialize_input(
        state: CentDeviceState,
        binding: CentInputBinding,
        value: CentNamedScalars,
) -> None:
    """Write one ordered host value to its typed physical region.

    Args:
        state: Fresh device state receiving the input.
        binding: Named raw physical destination declared by the manifest.
        value: Name- and size-validated host scalars.
    """

    region = binding.region
    if isinstance(region, CentDramRegion):
        state.write_dram(region.address, value.values)
        return
    if isinstance(region, CentSharedBufferRegion):
        state.write_shared_buffer(region.address, value.values)
        return
    if isinstance(region, CentGlobalBufferRegion):
        state.write_global_buffer(region.address, value.values)
        return
    assert_never(region)  # pragma: no cover - closed union guard


def _extract_output(
        state: CentDeviceState,
        binding: CentOutputBinding,
) -> CentNamedScalars:
    """Read one named raw output from its typed physical region.

    Args:
        state: Successfully executed device state.
        binding: Named raw physical source declared by the manifest.

    Returns:
        Named host values in increasing physical-lane order.
    """

    region = binding.region
    if isinstance(region, CentDramRegion):
        values = state.read_dram(region.address, value_count=region.scalar_count)
    elif isinstance(region, CentSharedBufferRegion):
        values = state.read_shared_buffer_values(
            region.address,
            value_count=region.scalar_count,
        )
    elif isinstance(region, CentGlobalBufferRegion):
        values = state.read_global_buffer(
            region.address,
            value_count=region.scalar_count,
        )
    else:
        assert_never(region)  # pragma: no cover - closed union guard
    return CentNamedScalars(name=binding.name, values=values)


def _execution_events(
        request: CentExecutionRequest,
        state: CentDeviceState,
        summary: CentExecutionSummary,
) -> tuple[CentExecutionEvent, ...]:
    """Build the selected successful instruction trace.

    Args:
        request: Completed request whose trace level selects the result.
        state: Device state that supplied the event device identifier.
        summary: Exact physical effects committed by the executor.

    Returns:
        Empty tuple for no tracing, otherwise one event per instruction.
    """

    if request.configuration.trace_level is CentTraceLevel.NONE:
        return ()
    return tuple(
        CentExecutionEvent(
            device_id=state.device_id,
            instruction_index=index,
            instruction=instruction,
            reads=effects.reads,
            writes=tuple(effect.region for effect in effects.writes),
        )
        for index, (instruction, effects) in enumerate(
            zip(
                request.executable.program.instructions,
                summary.committed_effects,
                strict=True,
            )
        )
    )
