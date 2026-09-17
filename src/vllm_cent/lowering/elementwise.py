"""Lower reusable elementwise vector operations to CENT."""

from dataclasses import dataclass

from ..cent import Accumulate, CentProgramBuilder
from .bindings import CentSharedBufferVector

__all__ = ["CentAccumulatePlan", "lower_accumulate"]


@dataclass(frozen=True, slots=True, kw_only=True)
class CentAccumulatePlan:
    """Bind an elementwise addition to two Shared Buffer vectors.

    Attributes:
        destination: Zero-padded first input and destination vector.
        source: Zero-padded second input vector.
    """

    destination: CentSharedBufferVector
    source: CentSharedBufferVector

    def __post_init__(self) -> None:
        """Require both operands to use identical padding positions.

        Raises:
            ValueError: If the vector layouts differ.
        """

        if self.destination.layout != self.source.layout:
            raise ValueError("source and destination vector layouts must match")


def lower_accumulate(
    builder: CentProgramBuilder,
    plan: CentAccumulatePlan,
) -> None:
    """Add a source vector into a destination vector.

    Args:
        builder: Program builder that receives the ACC instruction.
        plan: Vector size and Shared Buffer bindings for the addition.

    Raises:
        ValueError: If the vector's burst width differs from the target.
    """

    layout = plan.destination.layout
    if layout.burst_length != builder.hardware.burst_length:
        raise ValueError("vector burst_length does not match the target")
    operation_size = layout.slot_count

    # Both operands contain zero in exactly the same padding lanes. ACC writes
    # every lane, and zero plus zero keeps those output lanes zero.
    builder.append(
        Accumulate(
            operation_size=operation_size,
            destination=plan.destination.start,
            source=plan.source.start,
        )
    )
