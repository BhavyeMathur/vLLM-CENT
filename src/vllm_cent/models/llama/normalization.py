"""Build CENT instructions for Llama RMSNorm."""

from ...cent import (
    BANKS_PER_PU,
    CentProgramBuilder,
    CentSharedBufferAddress,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    MacAllBanks,
    ReadMac,
    ReadSingleBank,
    WriteBias,
    WriteSingleBank,
    ceil_div,
)
from .planning import _LlamaCompileContext

__all__: list[str] = []


def _lower_rms_norm(
    builder: CentProgramBuilder,
    context: _LlamaCompileContext,
    input_row: int,
    work_row: int,
    norm_row: int,
) -> None:
    """Add the available instructions for one RMSNorm operation.

    Args:
        builder: Program builder that receives the RMSNorm instructions.
        context: Model sizes and hardware limits used by this block.
        input_row: DRAM row where the input vector begins.
        work_row: DRAM row used while calculating the scale.
        norm_row: DRAM row holding learned weights and the final result.
    """

    # TODO(RMSNorm): Add the missing RMSNorm steps.
    #
    # The code below produces partial sums of squares. It does not combine all
    # banks and channels, divide by hidden size, add epsilon, calculate the
    # reciprocal square root, or broadcast that scale. The model spec also needs
    # an epsilon value.

    channels = builder.all_channels()
    hidden_size = context.model.hidden_size

    # Put two copies of the input in neighboring banks. MAC_ABK multiplies each
    # pair and adds the products into register 0, producing partial sums of x^2.
    neighbor_length = ceil_div(hidden_size, context.total_banks // 2)
    for bank_group in (0, 1):
        builder.emit_neighbor_bank_transfer(
            WriteSingleBank,
            hidden_size,
            bank_group,
            input_row,
            neighbor_length,
        )
    neighbor_op_size = ceil_div(neighbor_length, context.hardware.burst_length)
    # Start register 0 from Shared Buffer slot 0. RD_MAC returns each partial
    # sum to that slot.
    builder.append(
        WriteBias(
            source=CentSharedBufferAddress(slot=0), channels=channels
        )
    )
    builder.append(
        MacAllBanks(
            operation_size=neighbor_op_size,
            channels=channels,
            row=input_row,
            column=0,
            accumulation_register=0,
        )
    )
    builder.append(
        ReadMac(
            destination=CentSharedBufferAddress(slot=0),
            accumulation_register=0,
            channels=channels,
        )
    )

    # The remaining instructions assume the missing RMS scale has already been
    # copied into the work row. The paper assigns reciprocal square root to
    # RISC-V, but does not give the program address needed to call it.
    group_length = ceil_div(
        hidden_size,
        context.total_banks // BANKS_PER_PU,
    )
    utilized_banks = ceil_div(hidden_size, group_length)
    # Bank positions 0 and 1 hold the input and repeated scale. EW_MUL writes
    # their product to position 2.
    for bank_group in (0, 1):
        builder.emit_bank_group_transfer(
            WriteSingleBank,
            context.placement.channels_per_block,
            utilized_banks,
            bank_group,
            work_row,
            group_length,
        )
    group_op_size = ceil_div(group_length, context.hardware.burst_length)
    builder.append(
        ElementwiseMultiply(
            operation_size=group_op_size,
            channels=channels,
            row=work_row,
            column=0,
        )
    )
    # Copy the scaled vector through the Global Buffer to ``norm_row``, where
    # another EW_MUL applies the learned RMSNorm weights.
    builder.append(
        CopyBankToGlobalBuffer(
            operation_size=group_op_size,
            channels=channels,
            row=work_row,
            column=0,
        )
    )
    builder.append(
        CopyGlobalBufferToBank(
            operation_size=group_op_size,
            channels=channels,
            row=norm_row,
            column=0,
        )
    )
    builder.append(
        ElementwiseMultiply(
            operation_size=group_op_size,
            channels=channels,
            row=norm_row,
            column=0,
        )
    )
    builder.emit_bank_group_transfer(
        ReadSingleBank,
        context.placement.channels_per_block,
        utilized_banks,
        2,
        norm_row,
        group_length,
    )
