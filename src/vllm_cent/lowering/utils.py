"""Internal calculations and capacity checks shared by lowerers."""

from ..cent import ceil_div
from ..cent.utils import require_positive
from .bindings import CentDramRowRange, CentSharedBufferSpan

__all__: list[str] = []


def _row_operation_sizes(
    value_count: int,
    row_width: int,
    burst_length: int,
) -> tuple[int, ...]:
    """Split a value count into row-local ``OPsize`` values.

    Args:
        value_count: Scalar values processed by the complete operation.
        row_width: Scalar values stored in one DRAM row.
        burst_length: Scalar values processed by one micro-operation.

    Returns:
        Operation count for each consecutive DRAM row. The final count includes
        a partly occupied burst when the values do not fill it completely.

    Raises:
        ValueError: If a size is invalid or a row cannot contain whole bursts.
    """

    require_positive("value_count", value_count)
    require_positive("row_width", row_width)
    require_positive("burst_length", burst_length)
    if row_width % burst_length:
        raise ValueError("row_width must be divisible by burst_length")

    remaining_operations = ceil_div(value_count, burst_length)
    operations_per_row = row_width // burst_length
    row_sizes: list[int] = []
    while remaining_operations:
        operation_size = min(remaining_operations, operations_per_row)
        row_sizes.append(operation_size)
        remaining_operations -= operation_size
    return tuple(row_sizes)


def _require_shared_buffer_capacity(
    name: str,
    span: CentSharedBufferSpan,
    required_slots: int,
) -> None:
    """Check that a Shared Buffer span covers every accessed slot.

    Args:
        name: Buffer name used in an error message.
        span: Shared Buffer region assigned to an operation.
        required_slots: Slots the operation will access.

    Raises:
        ValueError: If ``required_slots`` is invalid or the span is too small.
    """

    require_positive("required_slots", required_slots)
    if span.slot_count < required_slots:
        raise ValueError(
            f"{name} needs {required_slots} Shared Buffer slots, "
            f"but its span contains {span.slot_count}"
        )


def _require_dram_row_capacity(
    name: str,
    rows: CentDramRowRange,
    required_rows: int,
) -> None:
    """Check that a DRAM row range covers every accessed row.

    Args:
        name: Row-range name used in an error message.
        rows: DRAM row range assigned to an operation.
        required_rows: Rows the operation will access.

    Raises:
        ValueError: If ``required_rows`` is invalid or the range is too small.
    """

    require_positive("required_rows", required_rows)
    if rows.row_count < required_rows:
        raise ValueError(
            f"{name} needs {required_rows} DRAM rows, "
            f"but its range contains {rows.row_count}"
        )
