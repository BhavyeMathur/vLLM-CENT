"""Run a CENT program using named inputs and output memory bindings.

Functional execution computes values; it does not predict hardware timing.
Each request gets fresh device state. The runtime manifest maps names to
memory ranges, and tracing reports the ranges each instruction reads or writes.
"""

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import assert_never

from vllm_cent.cent import CentInstruction
from vllm_cent.runtime import (
    CentAccumulatorRegion,
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
    """Select how simulator values are stored and arithmetic is computed.

    Members:
        REFERENCE_MATH: Python ``float`` arithmetic for checking that instructions move
            and combine the expected values; not hardware rounding behavior.
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
        numeric_profile: Rule for storing numbers and computing instruction results.
        trace_level: Amount of typed execution tracing returned to the caller.
    """

    numeric_profile: CentNumericProfile = CentNumericProfile.REFERENCE_MATH
    trace_level: CentTraceLevel = CentTraceLevel.NONE


@dataclass(frozen=True, slots=True, kw_only=True)
class CentExecutionRequest:
    """Combine a reusable executable with named values for one run.

    Attributes:
        executable: Program plus the named memory ranges for its inputs and outputs.
        inputs: Host numbers matched to input bindings by name, regardless of
            the order supplied here. Counts must match the bound memory ranges.
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
    """Return the output values and a record of completed execution.

    Values follow the physical memory layout in the manifest. This result does
    not reshape tensors or remove padding; the logical artifact layer does that.

    Attributes:
        outputs: Named scalar sequences, in the manifest's output order.
        events: Per-instruction read/write summaries when tracing is enabled;
            otherwise an empty tuple.
        executed_instruction_count: Number of instructions successfully applied.
    """

    outputs: tuple[CentNamedScalars, ...]
    events: tuple[CentExecutionEvent, ...]
    executed_instruction_count: int


def execute_functionally(request: CentExecutionRequest) -> CentExecutionResult:
    """Check support, load inputs, run the program, and read its outputs.

    Preflight checks every instruction before loading any input. Once support
    is confirmed, names and counts are checked, inputs are written, and
    instructions execute in order on fresh state.
    Input values are used in the order given within each binding; no tensor
    reshaping or zero padding is added here.

    Args:
        request: Executable, named input values, and simulator configuration.

    Returns:
        Named outputs plus the configured trace and instruction count.

    Raises:
        CentManifestError: If request inputs disagree with the manifest.
        CentSimulationError: If preflight or instruction execution fails.

    Examples:
        If a program copies values and then uses an unsupported instruction,
        preflight rejects the request before loading inputs or running the copy.
    """

    executable = request.executable
    state = CentDeviceState(
        hardware=executable.program.hardware,
        numeric=_numeric_semantics(request.configuration.numeric_profile),
    )

    # Reject unsupported instructions anywhere in the program before loading
    # inputs. A supported prefix must not run before a later rejection.
    preflight_program(executable.program, state)
    try:
        ordered_inputs = executable.manifest.order_input_values(request.inputs)
    except ValueError as error:
        raise CentManifestError(str(error)) from error

    # Values may arrive in any name order. The manifest has now reordered
    # and checked them, so each pair writes exactly its own destination.
    for binding, value in zip(
        executable.manifest.inputs,
        ordered_inputs,
        strict=True,
    ):
        _materialize_input(state, binding, value)

    summary = _execute_preflighted_program(
        executable.program,
        state,
        retain_effects=request.configuration.trace_level is CentTraceLevel.SUMMARY,
    )
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
    """Write one named sequence of host numbers to its bound memory range.

    Args:
        state: Fresh device state receiving the input.
        binding: Named raw physical destination declared by the manifest.
        value: Host numbers whose name and count have already been checked
            against the binding.

    Raises:
        ValueError: The destination is out of bounds or conversion rejects a value.
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
    if isinstance(region, CentAccumulatorRegion):
        raise ValueError("accumulator regions are trace-only; use buffer host bindings")
    assert_never(region)  # pragma: no cover - closed union guard


def _extract_output(
    state: CentDeviceState,
    binding: CentOutputBinding,
) -> CentNamedScalars:
    """Read the exact number of scalars requested by one output binding.

    Args:
        state: Successfully executed device state.
        binding: Named raw physical source declared by the manifest.

    Returns:
        Named host values in increasing column order, or by slot then lane
        for Shared Buffer output.

    Raises:
        ValueError: The requested range is outside device memory.
        CentUninitializedReadError: An output scalar was never written.
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
    elif isinstance(region, CentAccumulatorRegion):
        raise ValueError("accumulator regions are trace-only; use buffer host bindings")
    else:
        assert_never(region)  # pragma: no cover - closed union guard
    return CentNamedScalars(name=binding.name, values=values)


def _execution_events(
    request: CentExecutionRequest,
    state: CentDeviceState,
    summary: CentExecutionSummary,
) -> tuple[CentExecutionEvent, ...]:
    """Attach read and write ranges to each instruction when tracing is enabled.

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
