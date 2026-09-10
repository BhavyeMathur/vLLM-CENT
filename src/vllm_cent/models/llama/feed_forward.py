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
from ...lowering import CentSharedBufferSpan
from .planning import _LlamaCompileContext, _LlamaMemoryLayout

__all__: list[str] = []


def _lower_silu_product(
    builder: CentProgramBuilder,
    context: _LlamaCompileContext,
    layout: _LlamaMemoryLayout,
    workspace_buffer: CentSharedBufferSpan,
) -> None:
    """Multiply the SiLU gate by the other feed-forward projection.

    Args:
        builder: Program builder that receives the FFN instructions.
        context: Model sizes and hardware limits used by this block.
        layout: Starting DRAM rows of the intermediate FFN values.
        workspace_buffer: Slots used while staging each FFN vector chunk.
    """

    # TODO(dataflow): Connect W1, sigmoid(W1), and W3 to this workspace.
    #
    # The three projections now have separate compiler bindings. This function
    # still needs instructions that repack each RD_MAC result into the common
    # bank-group layout used below. The paper does not define that conversion.

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
        # Bank positions 0 and 1 hold W1 output and sigmoid(W1 output). The
        # missing repacking step must stage each value here before its write.
        for bank_group in (0, 1):
            builder.emit_bank_group_transfer(
                WriteSingleBank,
                context.placement.channels_per_block,
                utilized_banks,
                bank_group,
                row,
                group_length,
                shared_buffer=workspace_buffer.start,
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

    # Put W3 in bank position 1 beside the SiLU result. The TODO above tracks
    # the missing step that places W3 in this shared workspace first.
    for row, group_length, utilized_banks in chunk_details:
        builder.emit_bank_group_transfer(
            WriteSingleBank,
            context.placement.channels_per_block,
            utilized_banks,
            1,
            row,
            group_length,
            shared_buffer=workspace_buffer.start,
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
            shared_buffer=workspace_buffer.start,
        )
