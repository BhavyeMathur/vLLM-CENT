"""Execute checked CENT instructions in order, applying each result as a unit."""

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
    """Report instructions completed against an existing device state.

    Attributes:
        executed_instruction_count: Number of instructions committed in program
            order.
        committed_effects: Complete read and write effects for each committed
            instruction, in program order. The tuple index is the instruction
            index. Empty when the internal request executor disables history.
    """

    executed_instruction_count: int
    committed_effects: tuple[_InstructionEffects, ...]


def execute_program(
    program: CentProgram,
    state: CentDeviceState,
) -> CentExecutionSummary:
    """Execute a supported program sequentially against one device state.

    Preflight first checks whether every instruction is supported. Each
    instruction then reads its sources and computes all proposed writes before
    changing memory. Commit applies those writes only after every destination
    passes validation. Earlier completed instructions remain applied if a
    later instruction fails.

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

    Examples:
        If the third instruction fails while reading an unwritten operand, the
        first two remain applied, but no writes from the third become visible.
    """

    preflight_program(program, state)
    return _execute_preflighted_program(program, state)


def _execute_preflighted_program(
    program: CentProgram,
    state: CentDeviceState,
    *,
    retain_effects: bool = True,
) -> CentExecutionSummary:
    """Execute instructions after support checks and input loading are complete.

    The public request API has already checked the whole program before loading
    inputs. It calls this helper to avoid repeating that scan. Callers with an
    existing device state use ``execute_program``, which performs the scan.

    Args:
        program: Fully preflighted instructions in execution order.
        state: Mutable state whose hardware matches ``program``.
        retain_effects: Keep completed writes and read ranges for tracing. When
            false, discard each instruction's temporary values after commit.

    Returns:
        Completed instruction count and, when requested, exact read/write
        effects in program order. Without history, ``committed_effects`` is empty.

    Raises:
        CentExecutionFault: If a dynamic constraint fails during execution.
        CentUninitializedReadError: If an instruction consumes unwritten state.
    """

    # Execution needs only the current instruction's proposed writes. Keeping
    # every overwritten value would make memory grow with the whole program,
    # even when the caller asked for no trace.
    committed_effects: list[_InstructionEffects] = []
    for instruction_index, instruction in enumerate(program.instructions):
        effects = _execute_instruction(instruction, state, instruction_index)
        if retain_effects:
            committed_effects.append(effects)
    return CentExecutionSummary(
        executed_instruction_count=len(program.instructions),
        committed_effects=tuple(committed_effects),
    )


def _execute_instruction(
    instruction: CentInstruction,
    state: CentDeviceState,
    instruction_index: int,
) -> _InstructionEffects:
    """Compute one instruction's writes, then apply all of them together.

    Args:
        instruction: Supported typed instruction to execute.
        state: Memory read by the kernel and updated only after all proposed
            writes pass validation.
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
