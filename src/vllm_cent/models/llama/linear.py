"""Build CENT instructions for Llama matrix-vector multiplication."""

from ...cent import (
    ApplyActivation,
    CentProgramBuilder,
    CentSharedBufferAddress,
    MacAllBanks,
    ReadMac,
    WriteBias,
    WriteGlobalBuffer,
    ceil_div,
)

__all__: list[str] = []


def _lower_weight_gemv(
    builder: CentProgramBuilder,
    matrix_row: int,
    vector_size: int,
    output_size: int,
    accumulator_slots: int,
    *,
    apply_activation: bool = False,
) -> None:
    """Multiply an input vector by a weight matrix stored in DRAM.

    Args:
        builder: Program builder that receives the multiplication instructions.
        matrix_row: DRAM row where the matrix weights begin.
        vector_size: Number of values in the input vector.
        output_size: Number of values produced by the matrix.
        accumulator_slots: Results each bank can hold at once. At least two are
            needed when ``apply_activation`` is true.
        apply_activation: Whether to apply sigmoid after the dot products finish.

    Raises:
        ValueError: If a size is invalid or sigmoid lacks enough registers.
    """

    # TODO(ISA): Define how a dot product continues across input rows.
    #
    # RD_MAC reads the partial sum and WR_BIAS writes it back. WR_BIAS has no
    # register operand, so it is unclear which accumulator receives that value.

    # TODO(ISA): Define how to read an activated result.
    #
    # This code uses RD_MAC after AF. The reference simulator uses RD_AF, but
    # RD_AF does not appear in the paper's instruction table.

    if matrix_row < 0:
        raise ValueError("matrix_row cannot be negative")
    if vector_size < 1 or output_size < 1 or accumulator_slots < 1:
        raise ValueError("GEMV sizes must be at least 1")
    if apply_activation and accumulator_slots < 2:
        raise ValueError("fused activation requires at least two accumulator slots")

    hardware = builder.hardware

    # Output values are divided among the banks. One output uses several DRAM
    # rows when its input vector is wider than one row.
    outputs_per_bank = ceil_div(output_size, builder.total_banks)
    rows_per_output = ceil_div(vector_size, hardware.dram_columns)
    utilized_banks = ceil_div(output_size, outputs_per_bank)
    channels = builder.channels_for_matrix(utilized_banks)

    # Process outputs in groups that fit the available accumulator registers.
    # The current sigmoid path reserves half of those registers for its results.
    accumulator_limit = (
        accumulator_slots // 2 if apply_activation else accumulator_slots
    )
    accumulator_groups = ceil_div(outputs_per_bank, accumulator_limit)
    group_size = ceil_div(outputs_per_bank, accumulator_groups)

    for vector_row in range(rows_per_output):
        # Copy this input slice from Shared Buffer slot 0 to each selected
        # channel. Only the final slice may use less than a full DRAM row.
        remaining = vector_size - vector_row * hardware.dram_columns
        op_size = ceil_div(
            min(remaining, hardware.dram_columns), hardware.burst_length
        )
        builder.append(
            WriteGlobalBuffer(
                operation_size=op_size,
                column=0,
                source=CentSharedBufferAddress(slot=0),
                channels=channels,
            )
        )
        for group in range(accumulator_groups):
            group_outputs = min(
                group_size, outputs_per_bank - group * group_size
            )

            # Load the starting sum for each output. For later input rows, this
            # is the partial sum returned by the previous RD_MAC.
            for output in range(group_outputs):
                builder.append(
                    WriteBias(
                        source=CentSharedBufferAddress(slot=output),
                        channels=channels,
                    )
                )
            for output in range(group_outputs):
                # Matrix rows for one output are consecutive. Choose the output
                # first, then the row matching the current input slice.
                output_index = group * group_size + output
                row = matrix_row + output_index * rows_per_output + vector_row
                builder.append(
                    MacAllBanks(
                        operation_size=op_size,
                        channels=channels,
                        row=row,
                        column=0,
                        accumulation_register=output,
                    )
                )
            # Apply sigmoid only after the complete dot product is available.
            if apply_activation and vector_row == rows_per_output - 1:
                for output in range(group_outputs):
                    builder.append(
                        ApplyActivation(
                            channels=channels,
                            activation_function_id=(
                                hardware.sigmoid_activation_function_id
                            ),
                            accumulation_register=output,
                        )
                    )
                    # The current design assumes AF changes this accumulator and
                    # RD_MAC can read the changed value. The TODO above tracks
                    # why this still needs confirmation.
            # Return completed or partial results to the Shared Buffer.
            for output in range(group_outputs):
                builder.append(
                    ReadMac(
                        destination=CentSharedBufferAddress(slot=output),
                        accumulation_register=output,
                        channels=channels,
                    )
                )
