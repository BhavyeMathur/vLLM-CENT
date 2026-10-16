"""Choose the simulator implementation for each supported CENT instruction.

Preflight means checking the entire program before execution starts. The same
registration supplies both those support checks and the kernel that computes
results, so accepting an instruction always selects a matching implementation.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

from vllm_cent.cent import (
    Accumulate,
    MacAllBanks,
    CentInstruction,
    CentProgram,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    ReadSingleBank,
    WriteGlobalBuffer,
    WriteSingleBank,
)
from vllm_cent.cent.reference import (
    ReferenceInstruction,
    ReferenceRmsScale,
    ReferenceActivation,
    ReferenceSoftmax,
)
from .reference import _prepare_reference, _prepare_mac
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
from .numeric import ReferenceMathSemantics

__all__: list[str] = []


class _ResolvedInstructionBehavior(Protocol):
    """Common interface for handlers of different instruction classes.

    Each concrete handler accepts one instruction class. The executor uses
    this interface to check support and compute writes without knowing which
    instruction class that handler accepts.
    """

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
        """Check restrictions specific to the instruction this handler accepts.

        Args:
            instruction: Typed instruction previously matched to this behavior.
            state: Device state supplying the device identifier for errors.
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
        """Read operands and compute proposed writes without changing device memory.

        Args:
            instruction: Typed instruction previously matched to this behavior.
            state: Device state supplying the instruction's source values.

        Returns:
            Source memory ranges and proposed destination values for the
            instruction; no writes have been applied yet.

        Raises:
            TypeError: If ``instruction`` was not matched to this behavior.
            CentUninitializedReadError: If any source value is unwritten.
            _KernelArithmeticError: If the numeric policy rejects arithmetic.
        """

        ...


@dataclass(frozen=True, slots=True, kw_only=True)
class _InstructionBehavior[InstructionType: CentInstruction]:
    """Bundle all simulator behavior for one supported instruction class.

    ``matches`` accepts the configured class and its subclasses. Registration
    order decides which handler is used if more than one class could match.

    Attributes:
        instruction_type: Instruction class accepted by this behavior.
        preflight_function: Function that rejects unsupported forms of this instruction.
        prepare_function: Function that reads operands and returns proposed writes.
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
            state: Device state supplying the device identifier for errors.
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
        """Check the instruction class, then compute its proposed writes.

        Args:
            instruction: Typed instruction previously matched to this behavior.
            state: Device state supplying the instruction's source values.

        Returns:
            Source memory ranges and proposed destination values for the
            instruction; no writes have been applied yet.

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
    """Check every instruction is supported before changing device memory.

    This scan checks support, not whether every source has been written;
    reads check that during execution.

    Args:
        program: Validated typed instructions to inspect.
        state: Device state that would receive program effects.

    Raises:
        CentExecutionFault: If the program and state describe different target
            hardware.
        CentUnsupportedSemanticsError: If any instruction lacks implemented
            functional meaning or ``ACC`` ranges overlap partially.

    Examples:
        A supported copy followed by an unsupported reduction is rejected
        before the copy runs.
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
        state: Device state supplying the device identifier for errors.
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
    """Find the instruction's implementation and calculate its proposed writes.

    The implementation reads memory and returns the locations and new values to
    write. It does not apply those writes; the execution loop does that only
    after this entire calculation succeeds.

    Args:
        instruction: Supported CENT instruction to compute.
        state: Simulated memory containing the instruction's input values.

    Returns:
        Record of the memory regions read and the complete set of proposed writes.

    Raises:
        CentUninitializedReadError: If any input location has never been written.
        CentUnsupportedSemanticsError: If no implementation handles this instruction.
        _KernelArithmeticError: If the selected arithmetic rules reject a calculation.
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
    beyond the target validation already performed by ``CentProgram``.

    Args:
        instruction: Supported instruction whose semantics are fully defined.
        state: Target state used by handlers that need execution context.
        instruction_index: Zero-based program position used by handlers that
            report a semantic restriction.
    """

    if isinstance(
        instruction, (ReferenceRmsScale, ReferenceActivation, ReferenceSoftmax)
    ) and not isinstance(state.numeric, ReferenceMathSemantics):
        raise CentUnsupportedSemanticsError(
            "reference nonlinear arithmetic requires ReferenceMathSemantics",
            device_id=state.device_id,
            instruction_index=instruction_index,
            instruction=instruction,
        )

    # Every handler has the same callable shape. These instructions require no
    # additional capability check after CentProgram's structural validation.
    del instruction, state, instruction_index


def _preflight_accumulate(
    instruction: Accumulate,
    state: CentDeviceState,
    instruction_index: int,
) -> None:
    """Reject ambiguous overlap for accumulation (``ACC``) buffer addition.

    Equal source and destination ranges are supported: each value is doubled.
    Disjoint ranges are also supported. Partly overlapping ranges are rejected
    because the instruction's read/write order is unresolved.

    Args:
        instruction: Shared Buffer accumulation to inspect.
        state: Device state supplying the device identifier for errors.
        instruction_index: Zero-based program position of ``instruction``.

    Raises:
        CentUnsupportedSemanticsError: If the source and destination overlap but
            are not exactly aliased.

    Examples:
        A source covering slots 0 and 1 and a destination covering slots 1 and 2
        share only slot 1. This partial overlap is rejected.
    """

    # Compare ranges in slots; both use the same operation_size. With equal
    # starts they are identical, otherwise an intersection is only partial.
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
        instruction_type=ReferenceInstruction,
        preflight_function=_preflight_defined_instruction,
        prepare_function=_prepare_reference,
    ),
    _InstructionBehavior(
        instruction_type=MacAllBanks,
        preflight_function=_preflight_defined_instruction,
        prepare_function=_prepare_mac,
    ),
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
    """Find the first registered handler that accepts this instruction.

    Matching uses ``isinstance``, so subclasses use their parent instruction
    handler unless an earlier registration matches them. The optional index
    adds a program position to errors found by the initial support scan.

    Args:
        instruction: Typed instruction whose simulator behavior is required.
        state: Device state used for structured unsupported-semantics context.
        instruction_index: Zero-based program position during the support scan, or
            ``None`` when finding a handler during execution.

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
