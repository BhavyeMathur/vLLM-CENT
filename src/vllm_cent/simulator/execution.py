"""Execute preflighted CENT instructions and commit their physical effects."""

from dataclasses import dataclass

from vllm_cent.cent import CentInstruction, CentProgram
from .effects import _InstructionEffects
from .errors import (
    CentExecutionFault,
    CentUninitializedReadError,
)
from .kernels import _KernelArithmeticError
from .semantics import _prepare_instruction_effects, preflight_program
from .state import CentDeviceState

__all__ = ["CentExecutionSummary", "execute_program", "preflight_program"]


@dataclass(frozen=True, slots=True, kw_only=True)
class CentExecutionSummary:
    """Summarize one successfully completed state-level execution.

    Attributes:
        executed_instruction_count: Number of instructions committed in program
            order.
        committed_effects: Complete read and write effects for each committed
            instruction, in program order. The tuple index is the instruction
            index.
    """

    executed_instruction_count: int
    committed_effects: tuple[_InstructionEffects, ...]


def execute_program(
        program: CentProgram,
        state: CentDeviceState,
) -> CentExecutionSummary:
    """Execute a supported program sequentially against one device state.

    Preflight scans the complete program before instruction zero. Each kernel
    resolves and reads all sources into immutable physical effects. The state
    then validates and commits the complete write tuple as one transaction.
    Earlier completed instructions remain committed if a later instruction
    faults.

    Args:
        program: Validated typed instructions in execution order.
        state: Mutable state whose hardware exactly matches ``program``.

    Returns:
        Count and exact physical effects for committed instructions.

    Raises:
        CentExecutionFault: If a dynamic constraint fails during execution.
        CentUninitializedReadError: If an instruction consumes any unwritten
            source lane.
        CentUnsupportedSemanticsError: If preflight finds unsupported meaning.
    """

    preflight_program(program, state)
    return _execute_preflighted_program(program, state)


def _execute_preflighted_program(
        program: CentProgram,
        state: CentDeviceState,
) -> CentExecutionSummary:
    """Execute a program whose complete semantics were already preflighted.

    The public request path calls this helper after preflight and input
    materialization. Keeping that sequencing explicit prevents a second scan
    without allowing callers to accidentally bypass preflight through the
    public state-level entry point.

    Args:
        program: Fully preflighted instructions in execution order.
        state: Mutable state whose hardware matches ``program``.

    Returns:
        Count and exact committed read/write effects in program order.

    Raises:
        CentExecutionFault: If a dynamic constraint fails during execution.
        CentUninitializedReadError: If an instruction consumes unwritten state.
    """

    committed_effects = tuple(
        _execute_instruction(instruction, state, instruction_index)
        for instruction_index, instruction in enumerate(program.instructions)
    )
    return CentExecutionSummary(
        executed_instruction_count=len(program.instructions),
        committed_effects=committed_effects,
    )


def _execute_instruction(
        instruction: CentInstruction,
        state: CentDeviceState,
        instruction_index: int,
) -> _InstructionEffects:
    """Prepare and atomically commit one instruction's complete write set.

    Args:
        instruction: Supported typed instruction to execute.
        state: Device state supplying sources and receiving effects.
        instruction_index: Zero-based program position used in failure context.

    Returns:
        Complete read regions and writes committed for ``instruction``.

    Raises:
        CentExecutionFault: If arithmetic or effect validation fails.
        CentUninitializedReadError: If any complete source is not initialized.
        CentUnsupportedSemanticsError: If no kernel handles ``instruction``.
    """

    try:
        effects = _prepare_instruction_effects(instruction, state)
        state.commit_effects(effects.writes)
        return effects
    except _KernelArithmeticError as error:
        raise CentExecutionFault(
            error.reason,
            device_id=state.device_id,
            instruction_index=instruction_index,
            instruction=instruction,
            location=error.location,
        ) from error
    except CentUninitializedReadError as error:
        raise CentUninitializedReadError(
            error.reason,
            device_id=state.device_id,
            instruction_index=instruction_index,
            instruction=instruction,
            location=error.location,
        ) from error
    except ValueError as error:  # pragma: no cover - validated destination guard
        raise CentExecutionFault(
            str(error),
            device_id=state.device_id,
            instruction_index=instruction_index,
            instruction=instruction,
        ) from error
