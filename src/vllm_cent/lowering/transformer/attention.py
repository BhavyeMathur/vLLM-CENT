"""Lower transformer decode attention into CENT instructions."""

from dataclasses import dataclass, fields

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
from ...cent.utils import require_positive
from ..bindings import CentDramRowRange, CentSharedBufferSpan
from ..utils import _plan_partitioned_vector

__all__ = [
    "TransformerAttentionBuffers",
    "TransformerAttentionRows",
    "TransformerAttentionSpec",
    "lower_attention_output",
    "lower_kv_cache_update",
    "lower_rotary_embedding",
    "lower_score_gemv",
    "lower_softmax",
]


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformerAttentionSpec:
    """Describe one transformer decode-attention operation.

    Attributes:
        hidden_size: Values in the query projection.
        num_attention_heads: Query heads used by attention.
        num_kv_heads: Distinct key and value heads.
        head_size: Values in one attention head.
        kv_width: Values in all distinct key or value heads.
        repeat_count: Query heads that share one key/value head.
        sequence_length: Tokens visible to the current decode step.
        max_sequence_length: Token positions reserved in the KV cache.
    """

    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    head_size: int
    kv_width: int
    repeat_count: int
    sequence_length: int
    max_sequence_length: int

    def __post_init__(self) -> None:
        """Validate relationships used by generic attention lowering.

        Raises:
            ValueError: If a dimension is invalid or derived values disagree.
        """

        for field in fields(self):
            require_positive(field.name, getattr(self, field.name))
        if self.hidden_size != self.num_attention_heads * self.head_size:
            raise ValueError("hidden_size must equal attention heads * head_size")
        if self.kv_width != self.num_kv_heads * self.head_size:
            raise ValueError("kv_width must equal KV heads * head_size")
        if self.num_attention_heads != self.num_kv_heads * self.repeat_count:
            raise ValueError("attention heads must equal KV heads * repeat_count")
        if self.sequence_length > self.max_sequence_length:
            raise ValueError("sequence_length cannot exceed max_sequence_length")


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformerAttentionRows:
    """DRAM regions used by transformer attention lowering.

    Attributes:
        query: Rows used while rotating the query.
        key: Rows used while rotating the key.
        key_cache: Rows storing keys from all token positions.
        scores: Rows storing attention scores for the current token.
        value_cache: Rows storing values from all token positions.
    """

    query: CentDramRowRange
    key: CentDramRowRange
    key_cache: CentDramRowRange
    scores: CentDramRowRange
    value_cache: CentDramRowRange


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformerAttentionBuffers:
    """Shared Buffer regions passed between attention stages.

    Attributes:
        query: Slots containing Q and then its rotated form.
        key: Slots containing K and then its rotated form.
        value: Slots containing V before it enters the value cache.
        scores: Slots staging one row of attention scores.
        output: Slots receiving the weighted value result.
    """

    query: CentSharedBufferSpan
    key: CentSharedBufferSpan
    value: CentSharedBufferSpan
    scores: CentSharedBufferSpan
    output: CentSharedBufferSpan


def _require_span_capacity(
    name: str,
    span: CentSharedBufferSpan,
    required_slots: int,
) -> None:
    """Check that an attention buffer covers every accessed slot.

    Args:
        name: Buffer name used in an error message.
        span: Shared Buffer region assigned to the value.
        required_slots: Slots accessed by the lowering stage.

    Raises:
        ValueError: If the span is too small.
    """

    if span.slot_count < required_slots:
        raise ValueError(
            f"{name} needs {required_slots} Shared Buffer slots, "
            f"but its span contains {span.slot_count}"
        )


def lower_rotary_embedding(
    builder: CentProgramBuilder,
    spec: TransformerAttentionSpec,
    rows: TransformerAttentionRows,
    buffers: TransformerAttentionBuffers,
) -> None:
    """Add the current partial implementation of rotary position encoding.

    Args:
        builder: Program builder that receives the new instructions.
        spec: Model dimensions and decode lengths used by attention.
        rows: DRAM regions used by the attention stages.
        buffers: Shared Buffer regions carrying attention values.
    """

    # TODO(RoPE): Finish the rotary position calculation.
    #
    # The code below only multiplies values. A complete version must arrange Q
    # and K into pairs, load sine and cosine for this token, and add the signed
    # products. It is still unknown which steps run on RISC-V cores.

    # One CENT processing unit (PU) works with a group of four DRAM banks.
    # EW_MUL multiplies values in two banks and writes the result to a third.
    channels = builder.all_channels()
    group_count = builder.total_banks // BANKS_PER_PU
    query_layout = _plan_partitioned_vector(
        spec.hidden_size,
        group_count,
        builder.hardware.burst_length,
    )
    key_layout = _plan_partitioned_vector(
        spec.kv_width,
        group_count,
        builder.hardware.burst_length,
    )
    _require_span_capacity("query", buffers.query, query_layout.slot_count)
    _require_span_capacity("key", buffers.key, key_layout.slot_count)
    query_row_count = ceil_div(
        query_layout.values_per_partition, builder.hardware.dram_columns
    )
    key_row_count = ceil_div(
        key_layout.values_per_partition, builder.hardware.dram_columns
    )
    if rows.query.row_count < query_row_count:
        raise ValueError(
            f"query rows needs {query_row_count} DRAM rows, "
            f"but its range contains {rows.query.row_count}"
        )
    if rows.key.row_count < key_row_count:
        raise ValueError(
            f"key rows needs {key_row_count} DRAM rows, "
            f"but its range contains {rows.key.row_count}"
        )

    # Q and K are partitioned independently because grouped-query attention
    # usually makes K narrower. These writes place the second multiply operand
    # in bank position 1.
    for row, size, groups, buffer in (
        (
            rows.query.start_row,
            query_layout.values_per_partition,
            query_layout.partition_count,
            buffers.query,
        ),
        (
            rows.key.start_row,
            key_layout.values_per_partition,
            key_layout.partition_count,
            buffers.key,
        ),
    ):
        builder.emit_bank_group_transfer(
            WriteSingleBank,
            builder.placement.channels_per_block,
            groups,
            1,
            row,
            size,
            shared_buffer=buffer.start,
        )
    builder.append(
        ElementwiseMultiply(
            operation_size=ceil_div(
                query_layout.values_per_partition,
                builder.hardware.burst_length,
            ),
            channels=channels,
            row=rows.query.start_row,
            column=0,
        )
    )
    builder.append(
        ElementwiseMultiply(
            operation_size=ceil_div(
                key_layout.values_per_partition,
                builder.hardware.burst_length,
            ),
            channels=channels,
            row=rows.key.start_row,
            column=0,
        )
    )

    # Bank position 2 holds EW_MUL results. Move those results through the
    # Shared Buffer so the following attention step can use them.
    for row, size, groups, buffer in (
        (
            rows.query.start_row,
            query_layout.values_per_partition,
            query_layout.partition_count,
            buffers.query,
        ),
        (
            rows.key.start_row,
            key_layout.values_per_partition,
            key_layout.partition_count,
            buffers.key,
        ),
    ):
        builder.emit_bank_group_transfer(
            ReadSingleBank,
            builder.placement.channels_per_block,
            groups,
            2,
            row,
            size,
            shared_buffer=buffer.start,
        )


def lower_kv_cache_update(
    builder: CentProgramBuilder,
    spec: TransformerAttentionSpec,
    rows: TransformerAttentionRows,
    buffers: TransformerAttentionBuffers,
) -> None:
    """Store the current token's key and value in their caches.

    Args:
        builder: Program builder that receives the cache-write instructions.
        spec: Model dimensions and decode lengths used by attention.
        rows: DRAM regions used by the attention stages.
        buffers: Shared Buffer regions carrying attention values.
    """

    # TODO(placement): Define the owner of each repeated channel region.
    #
    # Keys are copied to every region, but values use only the first region. We
    # do not know whether the other regions repeat this request or serve
    # different requests.

    # ``sequence_length`` includes the new token. Subtracting one gives its
    # zero-based position in the cache.
    hardware = builder.hardware
    sequence_index = spec.sequence_length - 1
    channels_per_block = builder.placement.channels_per_block
    copies = hardware.num_channels // channels_per_block
    key_slots = ceil_div(spec.kv_width, hardware.burst_length)
    _require_span_capacity("key", buffers.key, key_slots)

    # Consecutive tokens go to consecutive banks. After every assigned bank has
    # one token, storage continues in the next row group. ``copies`` repeats the
    # key in each equal-sized channel region.
    channel, bank = builder.bank_index(sequence_index % builder.total_banks)
    key_rows = ceil_div(spec.kv_width, hardware.dram_columns)
    key_row_group = sequence_index // builder.total_banks

    # A key shorter than a DRAM row still needs one row and one write.
    for row_offset in range(key_rows):
        row = rows.key_cache.row(key_row_group * key_rows + row_offset)
        row_value_count = min(
            hardware.dram_columns,
            spec.kv_width - row_offset * hardware.dram_columns,
        )
        for copy in range(copies):
            builder.emit_single_bank_transfer(
                WriteSingleBank,
                channel + copy * channels_per_block,
                bank,
                row,
                row_value_count,
                shared_buffer=buffers.key.address(
                    row_offset
                    * (
                        hardware.dram_columns
                        // hardware.burst_length
                    )
                ),
            )

    # Values use a different layout from keys. Token positions run across
    # columns, while the values within a head use separate rows. This lets the
    # later multiplication read one head value across many tokens.
    rows_per_dimension = ceil_div(
        spec.max_sequence_length, hardware.dram_columns
    )
    sequence_row = sequence_index // hardware.dram_columns
    heads_per_channel = ceil_div(
        spec.num_kv_heads, channels_per_block
    )
    dimension_iterations = ceil_div(spec.head_size, hardware.num_banks)
    value_slots = spec.num_kv_heads * dimension_iterations
    _require_span_capacity("value", buffers.value, value_slots)
    for head_slot in range(heads_per_channel):
        head_row_offset = (
            rows_per_dimension * dimension_iterations * head_slot
        )
        for dimension in range(dimension_iterations):
            for channel_index in range(channels_per_block):
                # Each channel owns a consecutive set of KV heads. The final
                # channel may own fewer because the division was rounded up.
                head = channel_index * heads_per_channel + head_slot
                if head >= spec.num_kv_heads:
                    break
                # First choose the head, then the part spread over the banks,
                # and finally the row containing this token.
                row_offset = (
                    head_row_offset
                    + dimension * rows_per_dimension
                    + sequence_row
                )
                row = rows.value_cache.row(row_offset)
                # One WR_ABK source slot contains the values written across the
                # channel's banks for this head and dimension group.
                value_slot = head * dimension_iterations + dimension
                builder.append(
                    WriteAllBanks(
                        source=buffers.value.address(value_slot),
                        channel=channel_index,
                        row=row,
                        column=(
                            sequence_index % hardware.dram_columns
                        ),
                        # Table 3 calls this selector Regid. This lowering uses
                        # register 0 as scratch space.
                        accumulation_register=0,
                    )
                )


def lower_score_gemv(
    builder: CentProgramBuilder,
    spec: TransformerAttentionSpec,
    rows: TransformerAttentionRows,
    buffers: TransformerAttentionBuffers,
) -> None:
    """Multiply each query head by the stored keys to produce scores.

    Args:
        builder: Program builder that receives the score instructions.
        spec: Model dimensions and decode lengths used by attention.
        rows: DRAM regions used by the attention stages.
        buffers: Shared Buffer regions carrying attention values.
    """

    # TODO(attention): Assign a separate offset to each query and score.
    #
    # The buffers now have explicit base addresses, but every head still uses
    # that same base. We need the exact packing of GQA heads and score rows.

    hardware = builder.hardware

    # Several key heads can share one DRAM row. ``heads_per_row`` says how many
    # fit, and ``mac_size`` says how many bursts make up one head.
    rows_per_key = ceil_div(spec.kv_width, hardware.dram_columns)
    mac_size = spec.head_size // hardware.burst_length
    _require_span_capacity("query", buffers.query, mac_size)
    _require_span_capacity("scores", buffers.scores, 1)
    heads_per_row = hardware.dram_columns // spec.head_size
    sequence_iterations = ceil_div(
        spec.sequence_length, builder.total_banks
    )

    for row_offset in range(rows_per_key):
        for sequence_group in range(sequence_iterations):
            # A sequence group places at most one token in each assigned bank.
            # The last group may contain fewer tokens and use fewer channels.
            remaining = (
                spec.sequence_length
                - sequence_group * builder.total_banks
            )
            left_channels = ceil_div(
                min(remaining, builder.total_banks), hardware.num_banks
            )
            channel_count = (hardware.num_channels // left_channels) * left_channels
            channels = builder.channel_set(range(channel_count))
            # In grouped-query attention, several query heads use the same key
            # head. Each pass copies one query slice from the Shared Buffer to
            # the selected channels' Global Buffers.
            for _ in range(spec.repeat_count):
                builder.append(
                    WriteGlobalBuffer(
                        operation_size=mac_size,
                        column=0,
                        source=buffers.query.start,
                        channels=channels,
                    )
                )
                for head_index in range(heads_per_row):
                    # WR_BIAS initializes register 0. The row stays fixed while
                    # the column moves to the next packed head.
                    builder.append(
                        WriteBias(
                            source=buffers.scores.start,
                            channels=channels,
                        )
                    )
                    row = rows.key_cache.row(
                        sequence_group * rows_per_key + row_offset
                    )
                    builder.append(
                        MacAllBanks(
                            operation_size=mac_size,
                            channels=channels,
                            row=row,
                            column=head_index * spec.head_size,
                            accumulation_register=0,
                        )
                    )
                    builder.append(
                        ReadMac(
                            destination=buffers.scores.start,
                            accumulation_register=0,
                            channels=channels,
                        )
                    )


def _lower_score_transfer(
    builder: CentProgramBuilder,
    spec: TransformerAttentionSpec,
    rows: TransformerAttentionRows,
    instruction_type: type[WriteSingleBank] | type[ReadSingleBank],
    bank_group: int,
    buffers: TransformerAttentionBuffers,
) -> None:
    """Move attention scores between the Shared Buffer and DRAM.

    Args:
        builder: Program builder that receives the transfer instructions.
        spec: Model dimensions and decode lengths used by attention.
        rows: DRAM regions used by the attention stages.
        instruction_type: Transfer direction. ``WriteSingleBank`` moves scores
            into DRAM; ``ReadSingleBank`` moves scores out.
        bank_group: Position within each four-bank PU group. Positions 0 and 1
            hold inputs; position 2 holds the multiplication result.
        buffers: Shared Buffer regions carrying attention values.

    Raises:
        ValueError: If ``bank_group`` is not 0, 1, or 2.
    """

    if bank_group not in range(3):
        raise ValueError("score bank_group must be between 0 and 2")

    hardware = builder.hardware
    score_slots = ceil_div(
        min(spec.sequence_length, hardware.dram_columns),
        hardware.burst_length,
    )
    _require_span_capacity("scores", buffers.scores, score_slots)

    # Scores are laid out by head and token position. These loops visit every
    # DRAM burst that contains scores.
    rows_per_score = ceil_div(spec.sequence_length, hardware.dram_columns)
    heads_per_bank = ceil_div(
        spec.num_attention_heads,
        builder.placement.channels_per_block * BANKS_PER_PU,
    )
    channels_per_block = builder.placement.channels_per_block
    copies = hardware.num_channels // channels_per_block
    for row_offset in range(rows_per_score):
        row_size = min(
            hardware.dram_columns,
            spec.sequence_length - row_offset * hardware.dram_columns,
        )
        for head_slot in range(heads_per_bank):
            for column in range(0, row_size, hardware.burst_length):
                for logical_bank in range(builder.total_banks):
                    # The remainder selects the same bank position in every
                    # four-bank PU group.
                    if logical_bank % BANKS_PER_PU != bank_group:
                        continue
                    head = (
                        logical_bank // BANKS_PER_PU
                    ) * heads_per_bank + head_slot
                    if head >= spec.num_attention_heads:
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
                                * channels_per_block
                            ),
                            bank=bank,
                            row=rows.scores.row(
                                head_slot * rows_per_score + row_offset
                            ),
                            column=column,
                        )
                        # One Shared Buffer slot holds one burst. Dividing the
                        # DRAM column by burst length finds its staging slot.
                        shared_buffer = CentSharedBufferAddress(
                            slot=(
                                buffers.scores.start.slot
                                + column // hardware.burst_length
                            )
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


def lower_softmax(
    builder: CentProgramBuilder,
    spec: TransformerAttentionSpec,
    rows: TransformerAttentionRows,
    buffers: TransformerAttentionBuffers,
) -> None:
    """Add the two multiplication passes from the partial softmax path.

    Args:
        builder: Program builder that receives the softmax instructions.
        spec: Model dimensions and decode lengths used by attention.
        rows: DRAM regions used by the attention stages.
        buffers: Shared Buffer regions carrying attention values.
    """

    # TODO(softmax): Add the missing softmax steps.
    #
    # The code below only emits two multiplications. It does not emit EXP, RED,
    # the reciprocal calculation, or the movement between those steps. We must
    # also decide whether to subtract the largest score before EXP.

    hardware = builder.hardware
    score_slots = ceil_div(
        min(spec.sequence_length, hardware.dram_columns),
        hardware.burst_length,
    )
    _require_span_capacity("scores", buffers.scores, score_slots)
    channels = builder.all_channels()
    rows_per_score = ceil_div(spec.sequence_length, hardware.dram_columns)
    heads_per_bank = ceil_div(
        spec.num_attention_heads,
        builder.placement.channels_per_block * BANKS_PER_PU,
    )

    # Each pass puts two vectors in PU input banks, multiplies them, and reads
    # the result bank. The second vector should be 1/sqrt(head_size) in the first
    # pass and 1/sum(exp(scores)) in the second.
    #
    # Those values are not generated here. The paper assigns reciprocal and
    # square root work to RISC-V cores, but does not give program addresses.
    for _ in range(2):
        _lower_score_transfer(
            builder, spec, rows, WriteSingleBank, 0, buffers
        )
        _lower_score_transfer(
            builder, spec, rows, WriteSingleBank, 1, buffers
        )
        for head_slot in range(heads_per_bank):
            for row_offset in range(rows_per_score):
                # Only the last score row may be partly full. OPsize counts the
                # bursts that contain real scores.
                row_size = min(
                    hardware.dram_columns,
                    spec.sequence_length
                    - row_offset * hardware.dram_columns,
                )
                builder.append(
                    ElementwiseMultiply(
                        operation_size=ceil_div(
                            row_size, hardware.burst_length
                        ),
                        channels=channels,
                        row=rows.scores.row(
                            head_slot * rows_per_score + row_offset
                        ),
                        column=0,
                    )
                )
        _lower_score_transfer(
            builder, spec, rows, ReadSingleBank, 2, buffers
        )


def lower_attention_output(
    builder: CentProgramBuilder,
    spec: TransformerAttentionSpec,
    rows: TransformerAttentionRows,
    buffers: TransformerAttentionBuffers,
) -> None:
    """Multiply softmax scores by cached values to form attention output.

    Args:
        builder: Program builder that receives the output instructions.
        spec: Model dimensions and decode lengths used by attention.
        rows: DRAM regions used by the attention stages.
        buffers: Shared Buffer regions carrying attention values.
    """

    # TODO(attention): Assign a Shared Buffer offset to every score row.
    #
    # Every score row currently reads the start of the score span. A partial
    # result is also read after every row. We do not know how the next row
    # receives it or whether RD_MAC should happen only after the last row.

    hardware = builder.hardware
    score_slots = ceil_div(
        min(spec.sequence_length, hardware.dram_columns),
        hardware.burst_length,
    )
    _require_span_capacity("scores", buffers.scores, score_slots)
    _require_span_capacity("output", buffers.output, 1)

    # One value-cache row follows one head value across many token positions.
    # Copying scores to the Global Buffer lets different banks calculate
    # different output values from the same tokens.
    rows_per_sequence = ceil_div(spec.sequence_length, hardware.dram_columns)
    rows_per_dimension = ceil_div(
        spec.max_sequence_length, hardware.dram_columns
    )
    heads_per_channel = ceil_div(
        spec.num_kv_heads, builder.placement.channels_per_block
    )
    channels = builder.all_channels()
    dimension_iterations = spec.head_size // hardware.num_banks

    for head_slot in range(heads_per_channel):
        head_row_offset = (
            rows_per_dimension * dimension_iterations * head_slot
        )
        # Repeat a KV head for each query head that shares it.
        for _ in range(spec.repeat_count):
            for sequence_row in range(rows_per_sequence):
                row_size = min(
                    hardware.dram_columns,
                    spec.sequence_length
                    - sequence_row * hardware.dram_columns,
                )
                op_size = ceil_div(row_size, hardware.burst_length)
                builder.append(
                    WriteGlobalBuffer(
                        operation_size=op_size,
                        column=0,
                        source=buffers.scores.start,
                        channels=channels,
                    )
                )
                for dimension in range(dimension_iterations):
                    # WR_BIAS initializes register 0 for one output value. The
                    # selected row contains that value across this token range.
                    builder.append(
                        WriteBias(
                            source=buffers.output.start,
                            channels=channels,
                        )
                    )
                    builder.append(
                        MacAllBanks(
                            operation_size=op_size,
                            channels=channels,
                            row=rows.value_cache.row(
                                head_row_offset
                                + dimension * rows_per_dimension
                                + sequence_row
                            ),
                            column=0,
                            accumulation_register=0,
                        )
                    )
                    builder.append(
                        ReadMac(
                            destination=buffers.output.start,
                            accumulation_register=0,
                            channels=channels,
                        )
                    )
