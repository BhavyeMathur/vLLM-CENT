"""Lower model-independent matrix-vector multiplication to CENT."""

from ..cent import (
    ApplyActivation,
    CentProgramBuilder,
    MacAllBanks,
    ReadMac,
    WriteBias,
    WriteGlobalBuffer,
    ceil_div,
)
from ..cent.utils import require_positive
from .bindings import CentDramRowRange, CentSharedBufferSpan

__all__ = ["lower_weight_gemv"]


def _require_span_capacity(
    name: str,
    span: CentSharedBufferSpan,
    required_slots: int,
) -> None:
    """Check that a Shared Buffer span can hold an operation operand.

    Args:
        name: Operand name used in an error message.
        span: Shared Buffer region assigned to the operand.
        required_slots: Slots that the operation will access.

    Raises:
        ValueError: If the span has fewer than ``required_slots`` slots.
    """

    if span.slot_count < required_slots:
        raise ValueError(
            f"{name} needs {required_slots} Shared Buffer slots, "
            f"but its span contains {span.slot_count}"
        )


def lower_weight_gemv(
    builder: CentProgramBuilder,
    *,
    weights: CentDramRowRange,
    input_buffer: CentSharedBufferSpan,
    output_buffer: CentSharedBufferSpan,
    vector_size: int,
    output_size: int,
    activated_output_buffer: CentSharedBufferSpan | None = None,
) -> None:
    """Multiply a Shared Buffer vector by a weight matrix in DRAM.

    The matrix is distributed across the banks assigned to the block. Output
    values at the same bank position use consecutive DRAM rows.

    Args:
        builder: Program builder that receives the multiplication instructions.
        weights: DRAM rows containing the distributed weight matrix.
        input_buffer: Shared Buffer slots containing the input vector.
        output_buffer: Shared Buffer slots receiving raw matrix results. These
            slots also carry partial sums when the input crosses DRAM rows.
        vector_size: Number of values in the input vector.
        output_size: Number of values produced by the matrix.
        activated_output_buffer: Separate slots receiving sigmoid results. When
            omitted, the lowerer emits only the matrix results.

    Raises:
        ValueError: If a dimension is invalid or a memory region is too small.
    """

    # TODO(ISA): Define how a dot product continues across input rows.
    #
    # RD_MAC reads a partial sum and WR_BIAS writes it back. WR_BIAS has no
    # register operand, so the paper does not say which register receives it.

    # TODO(ISA): Define how to read an activated result.
    #
    # This code uses RD_MAC after AF. The reference simulator uses RD_AF, but
    # RD_AF does not appear in the paper's instruction table.

    # TODO(dataflow): Define how RD_MAC packs results selected by CHmask.
    #
    # The current mapping assigns one Shared Buffer slot to each accumulator
    # register. We still need the exact ordering of results from several banks
    # and channels.

    # TODO(runtime): Initialize each output slot before its first WR_BIAS.
    #
    # Later input rows reload a real partial sum. The first input row instead
    # needs zero, but this compiler does not yet emit or bind that value.

    require_positive("vector_size", vector_size)
    require_positive("output_size", output_size)

    hardware = builder.hardware
    rows_per_output = ceil_div(vector_size, hardware.dram_columns)
    outputs_per_bank = ceil_div(output_size, builder.total_banks)
    required_weight_rows = outputs_per_bank * rows_per_output
    if weights.row_count < required_weight_rows:
        raise ValueError(
            f"weights needs {required_weight_rows} DRAM rows, "
            f"but its range contains {weights.row_count}"
        )

    input_slots = ceil_div(vector_size, hardware.burst_length)
    _require_span_capacity("input_buffer", input_buffer, input_slots)
    _require_span_capacity(
        "output_buffer", output_buffer, outputs_per_bank
    )
    if activated_output_buffer is not None:
        _require_span_capacity(
            "activated_output_buffer",
            activated_output_buffer,
            outputs_per_bank,
        )
        raw_start = output_buffer.start.slot
        raw_end = raw_start + outputs_per_bank
        activated_start = activated_output_buffer.start.slot
        activated_end = activated_start + outputs_per_bank
        if raw_start < activated_end and activated_start < raw_end:
            raise ValueError("raw and activated output spans cannot overlap")

    utilized_banks = ceil_div(output_size, outputs_per_bank)
    channels = builder.channels_for_matrix(utilized_banks)

    # Activation uses the same physical accumulator after the raw result has
    # been read. Keep the conservative register grouping from the reference
    # lowering until the AF register behavior is confirmed.
    accumulator_limit = hardware.accumulator_slots_per_bank
    if activated_output_buffer is not None:
        if accumulator_limit < 2:
            raise ValueError(
                "activated GEMV requires at least two accumulator slots"
            )
        accumulator_limit //= 2

    accumulator_groups = ceil_div(outputs_per_bank, accumulator_limit)
    group_size = ceil_div(outputs_per_bank, accumulator_groups)
    slots_per_full_input_row = (
        hardware.dram_columns // hardware.burst_length
    )

    for vector_row in range(rows_per_output):
        # Move the next input slice into each selected channel's Global Buffer.
        # A vector wider than one DRAM row advances through its input span.
        remaining_values = vector_size - vector_row * hardware.dram_columns
        row_value_count = min(remaining_values, hardware.dram_columns)
        operation_size = ceil_div(
            row_value_count, hardware.burst_length
        )
        builder.append(
            WriteGlobalBuffer(
                operation_size=operation_size,
                column=0,
                source=input_buffer.address(
                    vector_row * slots_per_full_input_row
                ),
                channels=channels,
            )
        )

        for group in range(accumulator_groups):
            first_group_output = group * group_size
            group_output_count = min(
                group_size,
                outputs_per_bank - first_group_output,
            )

            # Output slots start at a global group offset. Earlier code reset
            # this index to zero and overwrote results from preceding groups.
            for local_output in range(group_output_count):
                output_offset = first_group_output + local_output
                builder.append(
                    WriteBias(
                        source=output_buffer.address(output_offset),
                        channels=channels,
                    )
                )

            for local_output in range(group_output_count):
                output_index = first_group_output + local_output
                weight_row_offset = (
                    output_index * rows_per_output + vector_row
                )
                builder.append(
                    MacAllBanks(
                        operation_size=operation_size,
                        channels=channels,
                        row=weights.row(weight_row_offset),
                        column=0,
                        accumulation_register=local_output,
                    )
                )

            # Always preserve the raw value. On earlier vector rows this value
            # is the partial sum loaded again by the next WR_BIAS pass.
            for local_output in range(group_output_count):
                output_offset = first_group_output + local_output
                builder.append(
                    ReadMac(
                        destination=output_buffer.address(output_offset),
                        accumulation_register=local_output,
                        channels=channels,
                    )
                )

            # Sigmoid is meaningful only after the final input slice completes
            # the dot product. Its result has a different explicit destination.
            if (
                activated_output_buffer is not None
                and vector_row == rows_per_output - 1
            ):
                for local_output in range(group_output_count):
                    builder.append(
                        ApplyActivation(
                            channels=channels,
                            activation_function_id=(
                                hardware.sigmoid_activation_function_id
                            ),
                            accumulation_register=local_output,
                        )
                    )
                    output_offset = first_group_output + local_output
                    builder.append(
                        ReadMac(
                            destination=activated_output_buffer.address(
                                output_offset
                            ),
                            accumulation_register=local_output,
                            channels=channels,
                        )
                    )
