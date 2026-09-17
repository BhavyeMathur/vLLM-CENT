"""Lower reusable elementwise vector operations to CENT."""

from ..cent import Accumulate, CentProgramBuilder, ceil_div
from ..cent.utils import require_positive
from .bindings import CentSharedBufferSpan
from .utils import _require_shared_buffer_capacity

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

    require_positive("value_count", value_count)
    operation_size = ceil_div(value_count, builder.hardware.burst_length)
    for name, span in (("destination", destination), ("source", source)):
        _require_shared_buffer_capacity(name, span, operation_size)

    builder.append(
        Accumulate(
            operation_size=operation_size,
            destination=destination.start,
            source=source.start,
        )
    )
