"""Stable failures raised by the CENT functional simulator."""

from typing import TypeAlias

from vllm_cent.cent import (
    CentBankRegisterAddress,
    CentGlobalBufferAddress,
    CentInstruction,
    CentMemoryAddress,
    CentSharedBufferAddress,
)

__all__ = [
    "CentExecutionFault",
    "CentManifestError",
    "CentSimulationError",
    "CentSimulationLocation",
    "CentUninitializedReadError",
    "CentUnsupportedSemanticsError",
]

CentSimulationLocation: TypeAlias = (
        CentMemoryAddress
        | CentSharedBufferAddress
        | CentGlobalBufferAddress
        | CentBankRegisterAddress
)


class CentSimulationError(Exception):
    """Base class for deterministic simulator failures.

    Attributes:
        reason: Plain-language explanation of the failure.
        device_id: Zero-based simulated device identifier.
        instruction_index: Zero-based program position, when an instruction was
            active.
        instruction: Typed instruction associated with the failure, when any.
        location: Typed state location associated with the failure, when any.
    """

    reason: str
    device_id: int
    instruction_index: int | None
    instruction: CentInstruction | None
    location: CentSimulationLocation | None

    def __init__(
            self,
            reason: str,
            *,
            device_id: int = 0,
            instruction_index: int | None = None,
            instruction: CentInstruction | None = None,
            location: CentSimulationLocation | None = None,
    ) -> None:
        """Create one structured simulator failure.

        Args:
            reason: Plain-language explanation of the failure.
            device_id: Zero-based simulated device identifier.
            instruction_index: Zero-based program position, when available.
            instruction: Typed instruction associated with the failure.
            location: Typed state location associated with the failure.

        Raises:
            ValueError: If ``device_id`` or ``instruction_index`` is negative.
        """

        if device_id < 0:
            raise ValueError("device_id cannot be negative")
        if instruction_index is not None and instruction_index < 0:
            raise ValueError("instruction_index cannot be negative")

        self.reason = reason
        self.device_id = device_id
        self.instruction_index = instruction_index
        self.instruction = instruction
        self.location = location

        instruction_text = (
            "" if instruction is None else f" instruction={instruction.opcode.value}"
        )
        index_text = "" if instruction_index is None else f" index={instruction_index}"
        location_text = "" if location is None else f" location={location!r}"
        super().__init__(
            f"device={device_id}{index_text}{instruction_text}{location_text}: {reason}"
        )


class CentUnsupportedSemanticsError(CentSimulationError):
    """Report an instruction or configuration without defined semantics."""


class CentManifestError(CentSimulationError):
    """Report disagreement between host values and runtime bindings."""


class CentUninitializedReadError(CentSimulationError):
    """Report a read whose complete source has not been initialized."""


class CentExecutionFault(CentSimulationError):  # noqa: N818
    """Report a dynamic execution constraint that the program violates."""
