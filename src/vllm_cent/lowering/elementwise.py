"""Lower reusable elementwise vector operations to CENT."""

from ..cent import Accumulate, CentProgramBuilder, ceil_div
from ..cent.utils import require_positive
from .bindings import CentSharedBufferSpan

__all__ = ["lower_accumulate"]


def lower_accumulate(
    builder: CentProgramBuilder,
    *,
    destination: CentSharedBufferSpan,
    source: CentSharedBufferSpan,
    value_count: int,
) -> None:
    """Add a source vector into a destination vector.

    Args:
        builder: Program builder that receives the ACC instruction.
        destination: Slots containing the first input and receiving the result.
        source: Slots containing the second input vector.
        value_count: Number of values in each vector.

    Raises:
        ValueError: If the vector is empty or either span is too small.
    """

    # TODO(ISA): Confirm which vector additions should use ACC.
    #
    # The paper defines ACC but does not explain its intended dataflow. Figure
    # 10 places some vector additions on the RISC-V cores instead.

    require_positive("value_count", value_count)
    operation_size = ceil_div(value_count, builder.hardware.burst_length)
    for name, span in (("destination", destination), ("source", source)):
        if span.slot_count < operation_size:
            raise ValueError(
                f"{name} needs {operation_size} Shared Buffer slots, "
                f"but its span contains {span.slot_count}"
            )

    builder.append(
        Accumulate(
            operation_size=operation_size,
            destination=destination.start,
            source=source.start,
        )
    )
