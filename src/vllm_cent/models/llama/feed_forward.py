"""Build CENT instructions for the elementwise part of Llama's FFN."""

from ...cent import (
    BANKS_PER_PU,
    CentProgramBuilder,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    ReadSingleBank,
    WriteSingleBank,
    ceil_div,
)
from .planning import _LlamaCompileContext, _LlamaMemoryLayout

__all__: list[str] = []


def _lower_silu_product(
    builder: CentProgramBuilder,
    context: _LlamaCompileContext,
    layout: _LlamaMemoryLayout,
) -> None:
    """Multiply the SiLU gate by the other feed-forward projection.

    Args:
        builder: Program builder that receives the FFN instructions.
        context: Model sizes and hardware limits used by this block.
        layout: Starting DRAM rows of the intermediate FFN values.
    """

    # TODO(FFN): Connect both inputs of the gated multiplication.
    #
    # The multiplication needs sigmoid(W1 output) and W3 output in different
    # banks. This function never uses ``layout.x3``, so W3 is not connected.

    channels = builder.all_channels()
    intermediate_size = context.model.intermediate_size

    # One activation pass has limited capacity. Split a wider FFN across the two
    # work areas reserved by the memory planner.
    if intermediate_size <= context.activation_capacity:
        chunks = ((layout.x1_sigmoid, intermediate_size),)
    else:
        # Context validation ensures two work areas are enough.
        chunks = (
            (layout.x1, context.activation_capacity),
            (layout.x1_sigmoid, intermediate_size - context.activation_capacity),
        )

    chunk_details: list[tuple[int, int, int]] = []
    for row, chunk_size in chunks:
        # Divide this chunk among the four-bank PU groups. The final group may
        # contain fewer real values.
        group_length = ceil_div(
            chunk_size,
            context.total_banks // BANKS_PER_PU,
        )
        utilized_banks = ceil_div(chunk_size, group_length)
        chunk_details.append((row, group_length, utilized_banks))
        # Bank positions 0 and 1 hold W1 output and sigmoid(W1 output).
        for bank_group in (1, 0):
            builder.emit_bank_group_transfer(
                WriteSingleBank,
                context.placement.channels_per_block,
                utilized_banks,
                bank_group,
                row,
                group_length,
            )
        op_size = ceil_div(group_length, context.hardware.burst_length)
        builder.append(
            ElementwiseMultiply(
                operation_size=op_size,
                channels=channels,
                row=row,
                column=0,
            )
        )
        # Move the SiLU result through the Global Buffer so it can become an
        # operand of the next multiplication.
        builder.append(
            CopyBankToGlobalBuffer(
                operation_size=op_size,
                channels=channels,
                row=row,
                column=0,
            )
        )
        builder.append(
            CopyGlobalBufferToBank(
                operation_size=op_size,
                channels=channels,
                row=row,
                column=0,
            )
        )

    # Put the SiLU result in bank position 1, multiply it by W3 output, and read
    # the result from bank position 2 for the W2 projection.
    for row, group_length, utilized_banks in chunk_details:
        builder.emit_bank_group_transfer(
            WriteSingleBank,
            context.placement.channels_per_block,
            utilized_banks,
            1,
            row,
            group_length,
        )
    for row, group_length, _ in chunk_details:
        builder.append(
            ElementwiseMultiply(
                operation_size=ceil_div(
                    group_length, context.hardware.burst_length
                ),
                channels=channels,
                row=row,
                column=0,
            )
        )
    for row, group_length, utilized_banks in chunk_details:
        builder.emit_bank_group_transfer(
            ReadSingleBank,
            context.placement.channels_per_block,
            utilized_banks,
            2,
            row,
            group_length,
        )
