"""Model-independent inputs to CENT block compilation."""

from dataclasses import dataclass

from .cent import CentBlockPlacementSpec, CentHardwareSpec
from .models.base import ModelSpec

__all__ = ["CompileRequest", "DecodeStepSpec"]


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodeStepSpec:
    """Token counts needed to compile one decode step.

    Attributes:
        sequence_length: Number of tokens currently in the attention context.
            This includes the token being decoded and cannot exceed
            ``max_sequence_length``.
        max_sequence_length: Number of tokens reserved in the KV cache.
    """

    sequence_length: int
    max_sequence_length: int

    def __post_init__(self) -> None:
        """Check that the current context fits in the KV-cache reservation.

        Raises:
            ValueError: If either length is invalid or the current context is
                larger than the reservation.
        """

        if self.max_sequence_length < 1:
            raise ValueError("max_sequence_length must be at least 1")
        if not 1 <= self.sequence_length <= self.max_sequence_length:
            raise ValueError(
                "sequence_length must be between 1 and max_sequence_length"
            )


@dataclass(frozen=True, slots=True, kw_only=True)
class CompileRequest:
    """Inputs needed to compile one model block.

    Attributes:
        model: Description of the model architecture.
        hardware: Description of the CENT device that will run the program.
        placement: Channels assigned to this model block.
        step: Token counts for the current decode step.
    """

    model: ModelSpec
    hardware: CentHardwareSpec
    placement: CentBlockPlacementSpec
    step: DecodeStepSpec
