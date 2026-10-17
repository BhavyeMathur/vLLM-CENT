"""Model-independent inputs to CENT block compilation."""

from dataclasses import dataclass

from .cent import CentBlockPlacementSpec, CentHardwareSpec
from .models.base import ModelSpec

__all__ = ["CompileRequest", "DecodeStepSpec"]


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodeStepSpec:
    """Describe the new token's position and the cache space reserved for the request.

    Decoding processes one new token while attending to earlier tokens.
    ``max_sequence_length`` controls how much memory is reserved even when
    the current context is shorter.

    Attributes:
        sequence_length: Number of tokens currently in the attention context.
            This includes the token being decoded and cannot exceed
            ``max_sequence_length``.
        max_sequence_length: Maximum token positions reserved for cached keys
            and values (the KV cache); this determines cache row allocation.

    Examples:
        If ``sequence_length`` is 5, this step includes tokens 0 through 4 and
        writes the new key/value pair at position 4.
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
    """Bundle the model, device, and token dimensions used by the block compiler.

    This describes shapes and placement, not actual tensors. Loading weights,
    populating input slots, and providing runtime cache contents are separate
    responsibilities.

    Attributes:
        model: Description of the model architecture.
        hardware: Description of the CENT device that will run the program.
        placement: Number of channels allotted to one block copy; determines
            how model values are split among that copy's banks.
        step: Token counts for the current decode step.
    """

    model: ModelSpec
    hardware: CentHardwareSpec
    placement: CentBlockPlacementSpec
    step: DecodeStepSpec
