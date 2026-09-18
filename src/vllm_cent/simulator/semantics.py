"""Resolve typed CENT instructions to supported functional semantics."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from vllm_cent.cent import (
    Accumulate,
    CentInstruction,
    CentProgram,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    ReadSingleBank,
    WriteGlobalBuffer,
    WriteSingleBank,
)
from .effects import _InstructionEffects
from .errors import (
    CentExecutionFault,
    CentUnsupportedSemanticsError,
)
from .kernels import (
    _prepare_accumulate,
    _prepare_copy_bank_to_global_buffer,
    _prepare_copy_global_buffer_to_bank,
    _prepare_elementwise_multiply,
    _prepare_read_single_bank,
    _prepare_write_global_buffer,
    _prepare_write_single_bank,
)
from .state import CentDeviceState

__all__: list[str] = []


class _ResolvedInstructionBehavior(Protocol):
    """Type-erased interface shared by heterogeneous instruction handlers."""

    def matches(self, instruction: CentInstruction) -> bool:
        """Return whether this behavior handles an instruction.

        Args:
            instruction: Typed instruction being resolved.

        Returns:
            ``True`` when the configured instruction class accepts the value.
        """

        ...

    def preflight(
            self,
            instruction: CentInstruction,
            state: CentDeviceState,
            instruction_index: int,
    ) -> None:
        """Apply this behavior's instruction-specific capability checks.

        Args:
            instruction: Typed instruction previously matched to this behavior.
            state: Target state used for structured error context.
            instruction_index: Zero-based program position of ``instruction``.

        Raises:
            TypeError: If ``instruction`` was not matched to this behavior.
            CentUnsupportedSemanticsError: If simulator semantics are incomplete.
        """

        ...

    def prepare_effects(
            self,
            instruction: CentInstruction,
            state: CentDeviceState,
    ) -> _InstructionEffects:
        """Prepare this behavior's complete physical effects.

        Args:
            instruction: Typed instruction previously matched to this behavior.
            state: Device state supplying the instruction's source values.

        Returns:
            Complete immutable reads and writes for ``instruction``.

        Raises:
            TypeError: If ``instruction`` was not matched to this behavior.
            CentUninitializedReadError: If any source value is unwritten.
            _KernelArithmeticError: If the numeric policy rejects arithmetic.
        """

        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class _InstructionBehavior[InstructionType: CentInstruction]:
    """Bundle all simulator behavior for one supported instruction class.

    The resolver intentionally uses ``isinstance`` through :meth:`matches`.
    Subclasses therefore inherit the first compatible registered instruction's
    semantics, preserving the simulator's existing polymorphic behavior.

    Attributes:
        instruction_type: Instruction class accepted by this behavior.
        preflight_function: Instruction-specific capability validation.
        prepare_function: Kernel that prepares complete immutable effects.
    """

    instruction_type: type[InstructionType]
    preflight_function: Callable[[InstructionType, CentDeviceState, int], None]
    prepare_function: Callable[[InstructionType, CentDeviceState], _InstructionEffects]

    def matches(self, instruction: CentInstruction) -> bool:
        """Return whether this behavior accepts an instruction instance.

        Args:
            instruction: Typed instruction being resolved.

        Returns:
            ``True`` for the configured class and its subclasses.
        """

        return isinstance(instruction, self.instruction_type)

    def preflight(
            self,
            instruction: CentInstruction,
            state: CentDeviceState,
            instruction_index: int,
    ) -> None:
        """Run the capability check after verifying handler resolution.

        Args:
            instruction: Typed instruction previously matched to this behavior.
            state: Target state used for structured error context.
            instruction_index: Zero-based program position of ``instruction``.

        Raises:
            TypeError: If a caller supplies an instruction this behavior does
                not handle.
            CentUnsupportedSemanticsError: If simulator semantics are incomplete.
        """

        if not isinstance(instruction, self.instruction_type):
            raise TypeError("instruction does not match its resolved behavior")
        self.preflight_function(instruction, state, instruction_index)

    def prepare_effects(
            self,
            instruction: CentInstruction,
            state: CentDeviceState,
    ) -> _InstructionEffects:
        """Run the effect kernel after verifying handler resolution.

        Args:
            instruction: Typed instruction previously matched to this behavior.
            state: Device state supplying the instruction's source values.

        Returns:
            Complete immutable reads and writes for ``instruction``.

        Raises:
            TypeError: If a caller supplies an instruction this behavior does
                not handle.
            CentUninitializedReadError: If any source value is unwritten.
            _KernelArithmeticError: If the numeric policy rejects arithmetic.
        """

        if not isinstance(instruction, self.instruction_type):
            raise TypeError("instruction does not match its resolved behavior")
        return self.prepare_function(instruction, state)


def preflight_program(program: CentProgram, state: CentDeviceState) -> None:
    """Reject unsupported program meaning before mutating device state.

    Args:
        program: Validated typed instructions to inspect.
        state: Device state that would receive program effects.

    Raises:
        CentExecutionFault: If the program and state describe different target
            hardware.
        CentUnsupportedSemanticsError: If any instruction lacks implemented
            functional meaning or ``ACC`` ranges overlap partially.
    """

    if program.hardware != state.hardware:
        raise CentExecutionFault(
            "program hardware does not match device state hardware",
            device_id=state.device_id,
        )

    for instruction_index, instruction in enumerate(program.instructions):
        _preflight_instruction(instruction, state, instruction_index)


def _preflight_instruction(
        instruction: CentInstruction,
        state: CentDeviceState,
        instruction_index: int,
) -> None:
    """Check simulator support for one structurally valid instruction.

    Args:
        instruction: Typed instruction to inspect.
        state: Target state used for structured error context.
        instruction_index: Zero-based program position of ``instruction``.

    Raises:
        CentUnsupportedSemanticsError: If the instruction type is unsupported or
            an ``ACC`` uses unresolved partial-overlap semantics.
    """

    behavior = _resolve_instruction_behavior(
        instruction,
        state,
        instruction_index=instruction_index,
    )
    behavior.preflight(instruction, state, instruction_index)


def _prepare_instruction_effects(
        instruction: CentInstruction,
        state: CentDeviceState,
) -> _InstructionEffects:
    """Dispatch one supported instruction to its effect-producing kernel.

    Args:
        instruction: Typed instruction whose effects are required.
        state: Device state from which the kernel reads source values.

    Returns:
        Complete immutable physical reads and writes for ``instruction``.

    Raises:
        CentUninitializedReadError: If a kernel consumes an unwritten source.
        CentUnsupportedSemanticsError: If no kernel handles ``instruction``.
        _KernelArithmeticError: If the numeric policy rejects arithmetic.
    """

    behavior = _resolve_instruction_behavior(instruction, state)
    return behavior.prepare_effects(instruction, state)


def _preflight_defined_instruction(
        instruction: CentInstruction,
        state: CentDeviceState,
        instruction_index: int,
) -> None:
    """Accept an instruction whose complete simulator meaning is defined.

    The arguments intentionally mirror instruction-specific preflight handlers.
    Most supported transfer and arithmetic instructions need no semantic check
    beyond the target validation already performed by :class:`CentProgram`.

    Args:
        instruction: Supported instruction whose semantics are fully defined.
        state: Target state used by handlers that need execution context.
        instruction_index: Zero-based program position used by handlers that
            report a semantic restriction.
    """

    # Every handler has the same callable shape. These instructions require no
    # additional capability check after CentProgram's structural validation.
    del instruction, state, instruction_index


def _preflight_accumulate(
        instruction: Accumulate,
        state: CentDeviceState,
        instruction_index: int,
) -> None:
    """Reject unresolved partial overlap between ``ACC`` source and destination.

    Args:
        instruction: Shared Buffer accumulation to inspect.
        state: Target state used for structured error context.
        instruction_index: Zero-based program position of ``instruction``.

    Raises:
        CentUnsupportedSemanticsError: If the source and destination overlap but
            are not exactly aliased.
    """

    destination_start = instruction.destination.slot
    destination_end = destination_start + instruction.operation_size
    source_start = instruction.source.slot
    source_end = source_start + instruction.operation_size
    ranges_overlap = destination_start < source_end and source_start < destination_end
    exact_alias = destination_start == source_start
    if ranges_overlap and not exact_alias:
        raise CentUnsupportedSemanticsError(
            "ACC partial overlap is undefined; ranges must be disjoint "
            "or exactly aliased",
            device_id=state.device_id,
            instruction_index=instruction_index,
            instruction=instruction,
            location=instruction.destination,
        )


# Each registration is the single source of truth for one simulator-supported
# instruction. A new opcode cannot pass preflight without registering the same
# handler that supplies its execution kernel.
_INSTRUCTION_BEHAVIORS: tuple[_ResolvedInstructionBehavior, ...] = (
    _InstructionBehavior(
        instruction_type=WriteSingleBank,
        preflight_function=_preflight_defined_instruction,
        prepare_function=_prepare_write_single_bank,
    ),
    _InstructionBehavior(
        instruction_type=ReadSingleBank,
        preflight_function=_preflight_defined_instruction,
        prepare_function=_prepare_read_single_bank,
    ),
    _InstructionBehavior(
        instruction_type=WriteGlobalBuffer,
        preflight_function=_preflight_defined_instruction,
        prepare_function=_prepare_write_global_buffer,
    ),
    _InstructionBehavior(
        instruction_type=CopyBankToGlobalBuffer,
        preflight_function=_preflight_defined_instruction,
        prepare_function=_prepare_copy_bank_to_global_buffer,
    ),
    _InstructionBehavior(
        instruction_type=CopyGlobalBufferToBank,
        preflight_function=_preflight_defined_instruction,
        prepare_function=_prepare_copy_global_buffer_to_bank,
    ),
    _InstructionBehavior(
        instruction_type=ElementwiseMultiply,
        preflight_function=_preflight_defined_instruction,
        prepare_function=_prepare_elementwise_multiply,
    ),
    _InstructionBehavior(
        instruction_type=Accumulate,
        preflight_function=_preflight_accumulate,
        prepare_function=_prepare_accumulate,
    ),
)


def _resolve_instruction_behavior(
        instruction: CentInstruction,
        state: CentDeviceState,
        *,
        instruction_index: int | None = None,
) -> _ResolvedInstructionBehavior:
    """Resolve one instruction to its complete external simulator behavior.

    Resolution uses the ordered registrations above and deliberately preserves
    ``isinstance`` semantics. The optional instruction index distinguishes the
    existing preflight failure from the defensive execution-dispatch failure.

    Args:
        instruction: Typed instruction whose simulator behavior is required.
        state: Device state used for structured unsupported-semantics context.
        instruction_index: Program position during preflight, or ``None`` when
            resolving the defensive execution path.

    Returns:
        First registered behavior compatible with ``instruction``.

    Raises:
        CentUnsupportedSemanticsError: If no registered behavior accepts the
            instruction.
    """

    for behavior in _INSTRUCTION_BEHAVIORS:
        if behavior.matches(instruction):
            return behavior

    reason = (
        f"{instruction.opcode.value} has no execution kernel"
        if instruction_index is None
        else f"{instruction.opcode.value} has no implemented functional semantics"
    )
    raise CentUnsupportedSemanticsError(
        reason,
        device_id=state.device_id,
        instruction_index=instruction_index,
        instruction=instruction,
    )
