"""Build CENT instructions for the attention part of a Llama block."""

from ...cent import (
    BANKS_PER_PU,
    CentProgramBuilder,
    CentMemoryAddress,
    CentSharedBufferAddress,
    ElementwiseMultiply,
    MacAllBanks,
    ReadMac,
    ReadSingleBank,
    WriteAllBanks,
    WriteBias,
    WriteGlobalBuffer,
    WriteSingleBank,
    ceil_div,
)
from .planning import _LlamaCompileContext, _LlamaMemoryLayout

__all__: list[str] = []


def _lower_rotary_embedding(
    builder: CentProgramBuilder,
    context: _LlamaCompileContext,
    layout: _LlamaMemoryLayout,
) -> None:
    """Add the current partial implementation of rotary position encoding.

    Args:
        builder: Program builder that receives the new instructions.
        context: Model sizes and hardware limits used by this block.
        layout: Starting DRAM row for each Llama tensor.
    """

    # TODO(RoPE): Finish the rotary position calculation.
    #
    # The code below only multiplies values. A complete version must arrange Q
    # and K into pairs, load sine and cosine for this token, and add the signed
    # products. It is still unknown which steps run on RISC-V cores.

    # One CENT processing unit (PU) works with a group of four DRAM banks.
    # EW_MUL multiplies values in two banks and writes the result to a third.
    channels = builder.all_channels()
    group_length = ceil_div(
        context.model.hidden_size,
        context.total_banks // BANKS_PER_PU,
    )
    utilized_banks = ceil_div(context.model.hidden_size, group_length)
    query_size = group_length * 2
    key_size = ceil_div(group_length, context.repeat_count) * 2

    # ``group_length`` is the part of Q handled by one PU group. K is smaller
    # when several query heads share one key head. These writes place the second
    # multiply operand in bank position 1.
    for row, size in ((layout.xq, query_size), (layout.xk, key_size)):
        builder.emit_bank_group_transfer(
            WriteSingleBank,
            context.placement.channels_per_block,
            utilized_banks,
            1,
            row,
            size,
        )
    builder.append(
        ElementwiseMultiply(
            operation_size=ceil_div(
                group_length, context.hardware.burst_length
            ),
            channels=channels,
            row=layout.xq,
            column=0,
        )
    )
    builder.append(
        ElementwiseMultiply(
            operation_size=ceil_div(
                ceil_div(group_length, context.repeat_count),
                context.hardware.burst_length,
            ),
            channels=channels,
            row=layout.xk,
            column=0,
        )
    )

    # Bank position 2 holds EW_MUL results. Move those results through the
    # Shared Buffer so the following attention step can use them.
    for row, size in ((layout.xq, query_size), (layout.xk, key_size)):
        builder.emit_bank_group_transfer(
            WriteSingleBank,
            context.placement.channels_per_block,
            utilized_banks,
            2,
            row,
            size,
        )


def _lower_kv_cache_update(
    builder: CentProgramBuilder,
    context: _LlamaCompileContext,
    layout: _LlamaMemoryLayout,
) -> None:
    """Store the current token's key and value in their caches.

    Args:
        builder: Program builder that receives the cache-write instructions.
        context: Model sizes, decode position, and hardware limits.
        layout: Starting DRAM rows of the key and value caches.
    """

    # TODO(placement): Define the owner of each repeated channel region.
    #
    # Keys are copied to every region, but values use only the first region. We
    # do not know whether the other regions repeat this request or serve
    # different requests.

    # ``sequence_length`` includes the new token. Subtracting one gives its
    # zero-based position in the cache.
    sequence_index = context.step.sequence_length - 1
    copies = context.hardware.num_channels // context.placement.channels_per_block

    # Consecutive tokens go to consecutive banks. After every assigned bank has
    # one token, storage continues in the next row group. ``copies`` repeats the
    # key in each equal-sized channel region.
    channel, bank = builder.bank_index(sequence_index % context.total_banks)
    key_rows = ceil_div(context.kv_width, context.hardware.dram_columns)
    key_row_group = sequence_index // context.total_banks

    # A key shorter than a DRAM row still needs one row and one write.
    for row_offset in range(key_rows):
        row = layout.cache_k + key_row_group * key_rows + row_offset
        row_value_count = min(
            context.hardware.dram_columns,
            context.kv_width - row_offset * context.hardware.dram_columns,
        )
        for copy in range(copies):
            builder.emit_single_bank_transfer(
                WriteSingleBank,
                channel + copy * context.placement.channels_per_block,
                bank,
                row,
                row_value_count,
            )

    # Values use a different layout from keys. Token positions run across
    # columns, while the values within a head use separate rows. This lets the
    # later multiplication read one head value across many tokens.
    rows_per_dimension = ceil_div(
        context.step.max_sequence_length, context.hardware.dram_columns
    )
    sequence_row = sequence_index // context.hardware.dram_columns
    heads_per_channel = ceil_div(
        context.model.num_kv_heads, context.placement.channels_per_block
    )
    dimension_iterations = ceil_div(context.head_size, context.hardware.num_banks)
    for head_slot in range(heads_per_channel):
        head_base = (
            layout.cache_v
            + rows_per_dimension * dimension_iterations * head_slot
        )
        for dimension in range(dimension_iterations):
            for channel_index in range(context.placement.channels_per_block):
                # Each channel owns a consecutive set of KV heads. The final
                # channel may own fewer because the division was rounded up.
                head = channel_index * heads_per_channel + head_slot
                if head >= context.model.num_kv_heads:
                    break
                # First choose the head, then the part spread over the banks,
                # and finally the row containing this token.
                row = head_base + dimension * rows_per_dimension + sequence_row
                builder.append(
                    WriteAllBanks(
                        source=CentSharedBufferAddress(slot=0),
                        channel=channel_index,
                        row=row,
                        column=(
                            sequence_index % context.hardware.dram_columns
                        ),
                        # Table 3 calls this selector Regid. This lowering uses
                        # register 0 as scratch space.
                        accumulation_register=0,
                    )
                )


def _lower_score_gemv(
    builder: CentProgramBuilder,
    context: _LlamaCompileContext,
    layout: _LlamaMemoryLayout,
) -> None:
    """Multiply each query head by the stored keys to produce scores.

    Args:
        builder: Program builder that receives the score instructions.
        context: Model sizes, decode length, and hardware limits.
        layout: Starting DRAM row of the key cache.
    """

    # TODO(attention): Assign Shared Buffer space to each query and score.
    #
    # Every loop below uses slot 0. The program cannot yet distinguish query
    # heads, shared GQA heads, or different score rows in the buffer.

    hardware = context.hardware

    # Several key heads can share one DRAM row. ``heads_per_row`` says how many
    # fit, and ``mac_size`` says how many bursts make up one head.
    rows_per_key = ceil_div(context.kv_width, hardware.dram_columns)
    write_size = hardware.dram_columns // hardware.burst_length
    mac_size = context.head_size // hardware.burst_length
    heads_per_row = hardware.dram_columns // context.head_size
    sequence_iterations = ceil_div(
        context.step.sequence_length, context.total_banks
    )

    for row_offset in range(rows_per_key):
        for sequence_group in range(sequence_iterations):
            # A sequence group places at most one token in each assigned bank.
            # The last group may contain fewer tokens and use fewer channels.
            remaining = (
                context.step.sequence_length
                - sequence_group * context.total_banks
            )
            left_channels = ceil_div(
                min(remaining, context.total_banks), hardware.num_banks
            )
            channel_count = (hardware.num_channels // left_channels) * left_channels
            channels = builder.channel_set(range(channel_count))
            # In grouped-query attention, several query heads use the same key
            # head. Each pass copies one query slice from the Shared Buffer to
            # the selected channels' Global Buffers.
            for _ in range(context.repeat_count):
                builder.append(
                    WriteGlobalBuffer(
                        operation_size=write_size,
                        column=0,
                        source=CentSharedBufferAddress(slot=0),
                        channels=channels,
                    )
                )
                for head_index in range(heads_per_row):
                    # WR_BIAS initializes register 0. The row stays fixed while
                    # the column moves to the next packed head.
                    builder.append(
                        WriteBias(
                            source=CentSharedBufferAddress(slot=0),
                            channels=channels,
                        )
                    )
                    row = layout.cache_k + sequence_group * rows_per_key + row_offset
                    builder.append(
                        MacAllBanks(
                            operation_size=mac_size,
                            channels=channels,
                            row=row,
                            column=head_index * context.head_size,
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


def _lower_score_transfer(
    builder: CentProgramBuilder,
    context: _LlamaCompileContext,
    layout: _LlamaMemoryLayout,
    instruction_type: type[WriteSingleBank] | type[ReadSingleBank],
    bank_group: int,
) -> None:
    """Move attention scores between the Shared Buffer and DRAM.

    Args:
        builder: Program builder that receives the transfer instructions.
        context: Model sizes, decode length, and hardware limits.
        layout: Starting DRAM row of the score workspace.
        instruction_type: Transfer direction. ``WriteSingleBank`` moves scores
            into DRAM; ``ReadSingleBank`` moves scores out.
        bank_group: Position within each four-bank PU group. Positions 0 and 1
            hold inputs; position 2 holds the multiplication result.

    Raises:
        ValueError: If ``bank_group`` is not 0, 1, or 2.
    """

    if bank_group not in range(3):
        raise ValueError("score bank_group must be between 0 and 2")

    hardware = context.hardware

    # Scores are laid out by head and token position. These loops visit every
    # DRAM burst that contains scores.
    rows_per_score = ceil_div(context.step.sequence_length, hardware.dram_columns)
    heads_per_bank = ceil_div(
        context.model.num_attention_heads,
        context.placement.channels_per_block * BANKS_PER_PU,
    )
    copies = hardware.num_channels // context.placement.channels_per_block
    for row_offset in range(rows_per_score):
        row_size = min(
            hardware.dram_columns,
            context.step.sequence_length - row_offset * hardware.dram_columns,
        )
        for head_slot in range(heads_per_bank):
            for column in range(0, row_size, hardware.burst_length):
                for logical_bank in range(context.total_banks):
                    # The remainder selects the same bank position in every
                    # four-bank PU group.
                    if logical_bank % BANKS_PER_PU != bank_group:
                        continue
                    head = (
                        logical_bank // BANKS_PER_PU
                    ) * heads_per_bank + head_slot
                    if head >= context.model.num_attention_heads:
                        break
                    channel, bank = builder.bank_index(logical_bank)
                    for copy in range(copies):
                        # A CENT memory address contains channel, bank, row, and
                        # column. Adding one block width selects the same bank in
                        # the next repeated channel region.
                        address = CentMemoryAddress(
                            channel=(
                                channel
                                + copy
                                * context.placement.channels_per_block
                            ),
                            bank=bank,
                            row=(
                                layout.scores
                                + head_slot * rows_per_score
                                + row_offset
                            ),
                            column=column,
                        )
                        # One Shared Buffer slot holds one burst. Dividing the
                        # DRAM column by burst length finds its staging slot.
                        shared_buffer = CentSharedBufferAddress(
                            slot=column // hardware.burst_length
                        )
                        # Move one burst so each head and bank keeps an explicit
                        # address.
                        if instruction_type is WriteSingleBank:
                            instruction = WriteSingleBank(
                                address=address,
                                operation_size=1,
                                source=shared_buffer,
                            )
                        else:
                            instruction = ReadSingleBank(
                                address=address,
                                operation_size=1,
                                destination=shared_buffer,
                            )
                        builder.append(instruction)


def _lower_softmax(
    builder: CentProgramBuilder,
    context: _LlamaCompileContext,
    layout: _LlamaMemoryLayout,
) -> None:
    """Add the two multiplication passes from the partial softmax path.

    Args:
        builder: Program builder that receives the softmax instructions.
        context: Model sizes, decode length, and hardware limits.
        layout: Starting DRAM row of the score workspace.
    """

    # TODO(softmax): Add the missing softmax steps.
    #
    # The code below only emits two multiplications. It does not emit EXP, RED,
    # the reciprocal calculation, or the movement between those steps. We must
    # also decide whether to subtract the largest score before EXP.

    hardware = context.hardware
    channels = builder.all_channels()
    rows_per_score = ceil_div(context.step.sequence_length, hardware.dram_columns)
    heads_per_bank = ceil_div(
        context.model.num_attention_heads,
        context.placement.channels_per_block * BANKS_PER_PU,
    )

    # Each pass puts two vectors in PU input banks, multiplies them, and reads
    # the result bank. The second vector should be 1/sqrt(head_size) in the first
    # pass and 1/sum(exp(scores)) in the second.
    #
    # Those values are not generated here. The paper assigns reciprocal and
    # square root work to RISC-V cores, but does not give program addresses.
    for _ in range(2):
        _lower_score_transfer(
            builder, context, layout, WriteSingleBank, 0
        )
        _lower_score_transfer(
            builder, context, layout, WriteSingleBank, 1
        )
        for head_slot in range(heads_per_bank):
            for row_offset in range(rows_per_score):
                # Only the last score row may be partly full. OPsize counts the
                # bursts that contain real scores.
                row_size = min(
                    hardware.dram_columns,
                    context.step.sequence_length
                    - row_offset * hardware.dram_columns,
                )
                builder.append(
                    ElementwiseMultiply(
                        operation_size=ceil_div(
                            row_size, hardware.burst_length
                        ),
                        channels=channels,
                        row=layout.scores
                        + head_slot * rows_per_score
                        + row_offset,
                        column=0,
                    )
                )
        _lower_score_transfer(
            builder, context, layout, ReadSingleBank, 2
        )


def _lower_output_gemv(
    builder: CentProgramBuilder,
    context: _LlamaCompileContext,
    layout: _LlamaMemoryLayout,
) -> None:
    """Multiply softmax scores by cached values to form attention output.

    Args:
        builder: Program builder that receives the output instructions.
        context: Model sizes, decode length, and hardware limits.
        layout: Starting DRAM row of the value cache.
    """

    # TODO(attention): Assign a Shared Buffer range to every score row.
    #
    # Every score row currently reads slot 0. A partial result is also read after
    # every row. We do not know how the next row receives that result or whether
    # RD_MAC should happen only after the last row.

    hardware = context.hardware

    # One value-cache row follows one head value across many token positions.
    # Copying scores to the Global Buffer lets different banks calculate
    # different output values from the same tokens.
    rows_per_sequence = ceil_div(context.step.sequence_length, hardware.dram_columns)
    rows_per_dimension = ceil_div(
        context.step.max_sequence_length, hardware.dram_columns
    )
    heads_per_channel = ceil_div(
        context.model.num_kv_heads, context.placement.channels_per_block
    )
    channels = builder.all_channels()
    dimension_iterations = context.head_size // hardware.num_banks

    for head_slot in range(heads_per_channel):
        head_base = (
            layout.cache_v
            + rows_per_dimension * dimension_iterations * head_slot
        )
        # Repeat a KV head for each query head that shares it.
        for _ in range(context.repeat_count):
            for sequence_row in range(rows_per_sequence):
                row_size = min(
                    hardware.dram_columns,
                    context.step.sequence_length
                    - sequence_row * hardware.dram_columns,
                )
                op_size = ceil_div(row_size, hardware.burst_length)
                builder.append(
                    WriteGlobalBuffer(
                        operation_size=op_size,
                        column=0,
                        source=CentSharedBufferAddress(slot=0),
                        channels=channels,
                    )
                )
                for dimension in range(dimension_iterations):
                    # WR_BIAS initializes register 0 for one output value. The
                    # selected row contains that value across this token range.
                    builder.append(
                        WriteBias(
                            source=CentSharedBufferAddress(slot=0),
                            channels=channels,
                        )
                    )
                    builder.append(
                        MacAllBanks(
                            operation_size=op_size,
                            channels=channels,
                            row=head_base
                            + dimension * rows_per_dimension
                            + sequence_row,
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
