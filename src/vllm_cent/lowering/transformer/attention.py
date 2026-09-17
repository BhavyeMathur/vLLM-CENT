"""Lower transformers decode attention into CENT instructions."""

from dataclasses import dataclass, fields

from ...cent import (
    BANKS_PER_PU,
    CentChannelSet,
    CentProgramBuilder,
    CentMemoryAddress,
    CentSharedBufferAddress,
    ElementwiseMultiply,
    MacAllBanks,
    MacOperandSource,
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
from ..utils import (
    _row_operation_sizes,
    _require_dram_row_capacity,
    _require_shared_buffer_capacity,
)

__all__ = [
    "AttentionOutputPlan",
    "KvCacheUpdatePlan",
    "RotaryEmbeddingPlan",
    "ScoreGemvPlan",
    "ScoreTransferPlan",
    "SoftmaxPassPlan",
    "SoftmaxPlan",
    "TransformerAttentionBuffers",
    "TransformerAttentionPlan",
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


@dataclass(frozen=True, slots=True, kw_only=True)
class RotaryEmbeddingPlan:
    """Describe the physical layout used by rotary embedding lowering.

    Attributes:
        spec: Logical transformer dimensions and decode lengths.
        rows: DRAM regions containing the query and key workspaces.
        buffers: Shared Buffer regions containing query and key values.
        channels: Physical channels that execute both multiply operations.
        query_values_per_partition: Query values assigned to each used PU group.
        query_partition_count: PU groups containing query values.
        key_values_per_partition: Key values assigned to each used PU group.
        key_partition_count: PU groups containing key values.
        transfer_channels_required: Channels occupied by one copied vector layout.
    """

    spec: TransformerAttentionSpec
    rows: TransformerAttentionRows
    buffers: TransformerAttentionBuffers
    channels: CentChannelSet
    query_values_per_partition: int
    query_partition_count: int
    key_values_per_partition: int
    key_partition_count: int
    transfer_channels_required: int


@dataclass(frozen=True, slots=True, kw_only=True)
class KvCacheUpdatePlan:
    """Describe key and value placement for one KV-cache update.

    Attributes:
        spec: Logical transformer dimensions and decode lengths.
        rows: DRAM regions assigned to the key and value caches.
        buffers: Shared Buffer regions containing the new key and value.
        sequence_index: Zero-based cache position of the new token.
        key_logical_bank: Block-local bank receiving the new key.
        key_row_group: Token row group receiving the new key.
        key_rows_per_token: Consecutive rows occupied by one key.
        key_replica_channel_offsets: Channel offsets of every key-cache replica.
        value_channels: Channels receiving distinct value-cache heads.
        value_rows_per_dimension: Rows reserved for one head dimension.
        value_sequence_row: Row offset containing this token position.
        value_heads_per_channel: KV-head slots assigned to each channel.
        value_dimension_iterations: Bank-wide writes needed for one head.
        accumulation_register: Scratch register selected by ``WR_ABK``.
    """

    spec: TransformerAttentionSpec
    rows: TransformerAttentionRows
    buffers: TransformerAttentionBuffers
    sequence_index: int
    key_logical_bank: int
    key_row_group: int
    key_rows_per_token: int
    key_replica_channel_offsets: tuple[int, ...]
    value_channels: tuple[int, ...]
    value_rows_per_dimension: int
    value_sequence_row: int
    value_heads_per_channel: int
    value_dimension_iterations: int
    accumulation_register: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreGemvPlan:
    """Describe query-by-key-cache matrix multiplication.

    Attributes:
        spec: Logical transformer dimensions and decode lengths.
        rows: DRAM regions containing cached keys.
        buffers: Shared Buffer regions carrying queries and scores.
        rows_per_key: DRAM rows occupied by one cached key.
        operation_size: Bursts multiplied for one attention head.
        heads_per_row: Key heads packed into one DRAM row.
        sequence_channels: Channel selection for each token-bank group.
        accumulation_register: Register used for each score dot product.
    """

    spec: TransformerAttentionSpec
    rows: TransformerAttentionRows
    buffers: TransformerAttentionBuffers
    rows_per_key: int
    operation_size: int
    heads_per_row: int
    sequence_channels: tuple[CentChannelSet, ...]
    accumulation_register: int


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreTransferPlan:
    """Describe one direction of attention-score movement.

    Attributes:
        spec: Logical transformer dimensions and decode lengths.
        rows: DRAM region containing attention scores.
        buffers: Shared Buffer region staging one score row.
        instruction_type: Transfer direction between the buffer and DRAM.
        bank_group: Bank position used within every four-bank PU group.
        rows_per_score: DRAM rows occupied by one head's scores.
        heads_per_bank: Head slots assigned to each selected bank position.
        logical_banks: Block-local banks participating in the transfer.
        replica_channel_offsets: Channel offsets of repeated score layouts.
    """

    spec: TransformerAttentionSpec
    rows: TransformerAttentionRows
    buffers: TransformerAttentionBuffers
    instruction_type: type[WriteSingleBank] | type[ReadSingleBank]
    bank_group: int
    rows_per_score: int
    heads_per_bank: int
    logical_banks: tuple[int, ...]
    replica_channel_offsets: tuple[int, ...]


@dataclass(frozen=True, slots=True, kw_only=True)
class SoftmaxPassPlan:
    """Describe transfers surrounding one partial softmax multiply pass.

    Attributes:
        left_input: Transfer placing the first operand in DRAM.
        right_input: Transfer placing the second operand in DRAM.
        output: Transfer returning multiplication results to the Shared Buffer.
    """

    left_input: ScoreTransferPlan
    right_input: ScoreTransferPlan
    output: ScoreTransferPlan


@dataclass(frozen=True, slots=True, kw_only=True)
class SoftmaxPlan:
    """Describe the currently supported multiply passes of softmax.

    Attributes:
        spec: Logical transformer dimensions and decode lengths.
        rows: DRAM region containing attention scores.
        buffers: Shared Buffer region staging attention scores.
        channels: Physical channels executing score multiplication.
        rows_per_score: DRAM rows occupied by one head's scores.
        heads_per_bank: Head slots assigned to each score bank.
        passes: Ordered score multiply passes and their transfers.
    """

    spec: TransformerAttentionSpec
    rows: TransformerAttentionRows
    buffers: TransformerAttentionBuffers
    channels: CentChannelSet
    rows_per_score: int
    heads_per_bank: int
    passes: tuple[SoftmaxPassPlan, ...]

    def __post_init__(self) -> None:
        """Require every softmax pass to use this plan's bindings.

        Raises:
            ValueError: If a child transfer describes different data or
                storage.
        """

        for pass_plan in self.passes:
            for transfer in (
                pass_plan.left_input,
                pass_plan.right_input,
                pass_plan.output,
            ):
                if transfer.spec != self.spec:
                    raise ValueError("softmax transfers must use the same spec")
                if transfer.rows != self.rows:
                    raise ValueError("softmax transfers must use the same rows")
                if transfer.buffers != self.buffers:
                    raise ValueError("softmax transfers must use the same buffers")
                if transfer.rows_per_score != self.rows_per_score:
                    raise ValueError(
                        "softmax transfers must use the same rows_per_score"
                    )
                if transfer.heads_per_bank != self.heads_per_bank:
                    raise ValueError(
                        "softmax transfers must use the same heads_per_bank"
                    )


@dataclass(frozen=True, slots=True, kw_only=True)
class AttentionOutputPlan:
    """Describe score-by-value-cache matrix multiplication.

    Attributes:
        spec: Logical transformer dimensions and decode lengths.
        rows: DRAM region containing cached values.
        buffers: Shared Buffer regions carrying scores and outputs.
        channels: Physical channels executing the multiplication.
        rows_per_sequence: Score rows consumed for the current sequence.
        rows_per_dimension: Reserved value-cache rows per head dimension.
        heads_per_channel: KV-head slots assigned to each channel.
        dimension_iterations: Bank-wide result groups for each attention head.
        accumulation_register: Register used for each output dot product.
    """

    spec: TransformerAttentionSpec
    rows: TransformerAttentionRows
    buffers: TransformerAttentionBuffers
    channels: CentChannelSet
    rows_per_sequence: int
    rows_per_dimension: int
    heads_per_channel: int
    dimension_iterations: int
    accumulation_register: int


@dataclass(frozen=True, slots=True, kw_only=True)
class TransformerAttentionPlan:
    """Compose every planned stage of transformer decode attention.

    Attributes:
        rotary_embedding: Physical plan for query and key rotation.
        kv_cache_update: Physical plan for storing the current key and value.
        score_gemv: Physical plan for producing attention scores.
        softmax: Physical plan for the supported softmax passes.
        output: Physical plan for combining scores with cached values.
    """

    rotary_embedding: RotaryEmbeddingPlan
    kv_cache_update: KvCacheUpdatePlan
    score_gemv: ScoreGemvPlan
    softmax: SoftmaxPlan
    output: AttentionOutputPlan

    def __post_init__(self) -> None:
        """Require all attention stages to share logical bindings.

        Raises:
            ValueError: If a stage describes a different attention operation
                or memory layout.
        """

        stages = (
            self.rotary_embedding,
            self.kv_cache_update,
            self.score_gemv,
            self.softmax,
            self.output,
        )
        first = stages[0]
        for stage in stages[1:]:
            if stage.spec != first.spec:
                raise ValueError("attention stages must use the same spec")
            if stage.rows != first.rows:
                raise ValueError("attention stages must use the same rows")
            if stage.buffers != first.buffers:
                raise ValueError("attention stages must use the same buffers")


def lower_rotary_embedding(
    builder: CentProgramBuilder,
    plan: RotaryEmbeddingPlan,
) -> None:
    """Add the current partial implementation of rotary position encoding.

    Args:
        builder: Program builder that receives the new instructions.
        plan: Checked dimensions, bindings, channels, and PU partitions.

    Raises:
        ValueError: If a planned partition or memory region is invalid.
    """

    # TODO(RoPE): Finish the rotary position calculation.
    #
    # The code below only multiplies values. A complete version must arrange Q
    # and K into pairs, load sine and cosine for this token, and add the signed
    # products. It is still unknown which steps run on RISC-V cores.

    # TODO(ISA): Define how unused PU groups are disabled.
    #
    # The transfer plan may use only part of a channel, while EW_MUL runs every
    # PU group selected by CHmask. Results from unused groups are not read here,
    # but they still consume work and may matter to a future stateful target.

    # One CENT processing unit (PU) works with a group of four DRAM banks.
    # EW_MUL multiplies values in two banks and writes the result to a third.
    spec = plan.spec
    rows = plan.rows
    buffers = plan.buffers
    for name, value in (
        ("query_values_per_partition", plan.query_values_per_partition),
        ("query_partition_count", plan.query_partition_count),
        ("key_values_per_partition", plan.key_values_per_partition),
        ("key_partition_count", plan.key_partition_count),
        ("transfer_channels_required", plan.transfer_channels_required),
    ):
        require_positive(name, value)
    if plan.query_partition_count > builder.total_banks // BANKS_PER_PU:
        raise ValueError("query partitions exceed available PU groups")
    if plan.key_partition_count > builder.total_banks // BANKS_PER_PU:
        raise ValueError("key partitions exceed available PU groups")
    if plan.transfer_channels_required != builder.placement.channels_per_block:
        raise ValueError("transfer_channels_required must match the block placement")
    builder.channel_set(plan.channels.channels)
    if not (
        (plan.query_partition_count - 1) * plan.query_values_per_partition
        < spec.hidden_size
        <= plan.query_partition_count * plan.query_values_per_partition
    ):
        raise ValueError("query partitions must cover hidden_size without gaps")
    if not (
        (plan.key_partition_count - 1) * plan.key_values_per_partition
        < spec.kv_width
        <= plan.key_partition_count * plan.key_values_per_partition
    ):
        raise ValueError("key partitions must cover kv_width without gaps")
    query_slots = plan.query_partition_count * ceil_div(
        plan.query_values_per_partition, builder.hardware.burst_length
    )
    key_slots = plan.key_partition_count * ceil_div(
        plan.key_values_per_partition, builder.hardware.burst_length
    )
    _require_shared_buffer_capacity("query", buffers.query, query_slots)
    _require_shared_buffer_capacity("key", buffers.key, key_slots)
    query_row_sizes = _row_operation_sizes(
        plan.query_values_per_partition,
        builder.hardware.dram_columns,
        builder.hardware.burst_length,
    )
    key_row_sizes = _row_operation_sizes(
        plan.key_values_per_partition,
        builder.hardware.dram_columns,
        builder.hardware.burst_length,
    )
    _require_dram_row_capacity("query rows", rows.query, len(query_row_sizes))
    _require_dram_row_capacity("key rows", rows.key, len(key_row_sizes))

    # Q and K are partitioned independently because grouped-query attention
    # usually makes K narrower. These writes place the second multiply operand
    # in bank position 1.
    for row, size, groups, buffer in (
        (
            rows.query.start_row,
            plan.query_values_per_partition,
            plan.query_partition_count,
            buffers.query,
        ),
        (
            rows.key.start_row,
            plan.key_values_per_partition,
            plan.key_partition_count,
            buffers.key,
        ),
    ):
        builder.emit_bank_group_transfer(
            WriteSingleBank,
            plan.transfer_channels_required,
            groups,
            ElementwiseMultiply.SECOND_OPERAND_BANK,
            row,
            size,
            shared_buffer=buffer.start,
        )
    for row_offset, operation_size in enumerate(query_row_sizes):
        builder.append(
            ElementwiseMultiply(
                operation_size=operation_size,
                channels=plan.channels,
                row=rows.query.row(row_offset),
                column=0,
            )
        )
    for row_offset, operation_size in enumerate(key_row_sizes):
        builder.append(
            ElementwiseMultiply(
                operation_size=operation_size,
                channels=plan.channels,
                row=rows.key.row(row_offset),
                column=0,
            )
        )

    # Bank position 2 holds EW_MUL results. Move those results through the
    # Shared Buffer so the following attention step can use them.
    for row, size, groups, buffer in (
        (
            rows.query.start_row,
            plan.query_values_per_partition,
            plan.query_partition_count,
            buffers.query,
        ),
        (
            rows.key.start_row,
            plan.key_values_per_partition,
            plan.key_partition_count,
            buffers.key,
        ),
    ):
        builder.emit_bank_group_transfer(
            ReadSingleBank,
            plan.transfer_channels_required,
            groups,
            ElementwiseMultiply.RESULT_BANK,
            row,
            size,
            shared_buffer=buffer.start,
        )


def lower_kv_cache_update(
    builder: CentProgramBuilder,
    plan: KvCacheUpdatePlan,
) -> None:
    """Store the current token's key and value in their caches.

    Args:
        builder: Program builder that receives the cache-write instructions.
        plan: Checked cache geometry, memory bindings, and channel placement.

    Raises:
        ValueError: If the planned cache geometry or memory is invalid.
    """

    # TODO(placement): Define the owner of each repeated channel region.
    #
    # Keys are copied to every region, but values use only the first region. We
    # do not know whether the other regions repeat this request or serve
    # different requests.

    spec = plan.spec
    rows = plan.rows
    buffers = plan.buffers
    hardware = builder.hardware
    if plan.sequence_index != spec.sequence_length - 1:
        raise ValueError("sequence_index must identify the current decode token")
    if not 0 <= plan.key_logical_bank < builder.total_banks:
        raise ValueError("key_logical_bank is outside the block allocation")
    for name, value in (
        ("key_rows_per_token", plan.key_rows_per_token),
        ("value_rows_per_dimension", plan.value_rows_per_dimension),
        ("value_heads_per_channel", plan.value_heads_per_channel),
        ("value_dimension_iterations", plan.value_dimension_iterations),
    ):
        require_positive(name, value)
    if plan.key_row_group < 0:
        raise ValueError("key_row_group must be nonnegative")
    if plan.value_sequence_row < 0:
        raise ValueError("value_sequence_row must be nonnegative")
    if not plan.key_replica_channel_offsets:
        raise ValueError("key_replica_channel_offsets cannot be empty")
    if len(set(plan.key_replica_channel_offsets)) != len(
        plan.key_replica_channel_offsets
    ):
        raise ValueError("key_replica_channel_offsets cannot contain duplicates")
    if not plan.value_channels:
        raise ValueError("value_channels cannot be empty")
    if plan.key_rows_per_token * hardware.dram_columns < spec.kv_width:
        raise ValueError("key_rows_per_token does not cover kv_width")
    if plan.value_rows_per_dimension * hardware.dram_columns < spec.max_sequence_length:
        raise ValueError("value_rows_per_dimension does not cover max_sequence_length")
    if len(plan.value_channels) * plan.value_heads_per_channel < spec.num_kv_heads:
        raise ValueError("value channel slots do not cover all KV heads")
    if plan.value_dimension_iterations * hardware.num_banks < spec.head_size:
        raise ValueError("value dimension iterations do not cover head_size")

    expected_key_rows = ceil_div(spec.kv_width, hardware.dram_columns)
    if plan.key_rows_per_token != expected_key_rows:
        raise ValueError("key_rows_per_token must match the packed key width")
    expected_value_rows = ceil_div(spec.max_sequence_length, hardware.dram_columns)
    if plan.value_rows_per_dimension != expected_value_rows:
        raise ValueError(
            "value_rows_per_dimension must match the reserved sequence length"
        )
    expected_dimension_iterations = ceil_div(spec.head_size, hardware.num_banks)
    if plan.value_dimension_iterations != expected_dimension_iterations:
        raise ValueError("value_dimension_iterations must match the packed head width")
    if len(set(plan.value_channels)) != len(plan.value_channels):
        raise ValueError("value_channels cannot contain duplicates")
    builder.channel_set(plan.value_channels)
    if plan.value_heads_per_channel != ceil_div(
        spec.num_kv_heads, len(plan.value_channels)
    ):
        raise ValueError("value_heads_per_channel must match the selected channels")

    # Check every derived address before adding the first cache instruction.
    key_channel, _ = builder.bank_index(plan.key_logical_bank)
    builder.channel_set(
        tuple(key_channel + offset for offset in plan.key_replica_channel_offsets)
    )
    _require_dram_row_capacity(
        "key_cache",
        rows.key_cache,
        (plan.key_row_group + 1) * plan.key_rows_per_token,
    )
    _require_dram_row_capacity(
        "value_cache",
        rows.value_cache,
        plan.value_heads_per_channel
        * plan.value_dimension_iterations
        * plan.value_rows_per_dimension,
    )

    sequence_index = plan.sequence_index
    key_slots = ceil_div(spec.kv_width, hardware.burst_length)
    _require_shared_buffer_capacity("key", buffers.key, key_slots)

    # Consecutive tokens go to consecutive banks. After every assigned bank has
    # one token, storage continues in the next row group. ``copies`` repeats the
    # key in each equal-sized channel region.
    channel, bank = builder.bank_index(plan.key_logical_bank)

    # A key shorter than a DRAM row still needs one row and one write.
    for row_offset in range(plan.key_rows_per_token):
        row = rows.key_cache.row(
            plan.key_row_group * plan.key_rows_per_token + row_offset
        )
        row_value_count = min(
            hardware.dram_columns,
            spec.kv_width - row_offset * hardware.dram_columns,
        )
        for channel_offset in plan.key_replica_channel_offsets:
            builder.emit_single_bank_transfer(
                WriteSingleBank,
                channel + channel_offset,
                bank,
                row,
                row_value_count,
                shared_buffer=buffers.key.address(
                    row_offset * (hardware.dram_columns // hardware.burst_length)
                ),
            )

    # Values use a different layout from keys. Token positions run across
    # columns, while the values within a head use separate rows. This lets the
    # later multiplication read one head value across many tokens.
    value_slots = spec.num_kv_heads * plan.value_dimension_iterations
    _require_shared_buffer_capacity("value", buffers.value, value_slots)
    for head_slot in range(plan.value_heads_per_channel):
        head_row_offset = (
            plan.value_rows_per_dimension * plan.value_dimension_iterations * head_slot
        )
        for dimension in range(plan.value_dimension_iterations):
            for channel_slot, channel_index in enumerate(plan.value_channels):
                # Each channel owns a consecutive set of KV heads. The final
                # channel may own fewer because the division was rounded up.
                head = channel_slot * plan.value_heads_per_channel + head_slot
                if head >= spec.num_kv_heads:
                    break
                # First choose the head, then the part spread over the banks,
                # and finally the row containing this token.
                row_offset = (
                    head_row_offset
                    + dimension * plan.value_rows_per_dimension
                    + plan.value_sequence_row
                )
                row = rows.value_cache.row(row_offset)
                # One WR_ABK source slot contains the values written across the
                # channel's banks for this head and dimension group.
                value_slot = head * plan.value_dimension_iterations + dimension
                builder.append(
                    WriteAllBanks(
                        source=buffers.value.address(value_slot),
                        channel=channel_index,
                        row=row,
                        column=(sequence_index % hardware.dram_columns),
                        # Table 3 calls this selector Regid. This lowering uses
                        # register 0 as scratch space.
                        accumulation_register=plan.accumulation_register,
                    )
                )


def lower_score_gemv(
    builder: CentProgramBuilder,
    plan: ScoreGemvPlan,
) -> None:
    """Multiply each query head by the stored keys to produce scores.

    Args:
        builder: Program builder that receives the score instructions.
        plan: Checked key layout, channel schedule, and buffer bindings.

    Raises:
        ValueError: If the planned geometry or a memory region is invalid.
    """

    # TODO(attention): Assign a separate offset to each query and score.
    #
    # The buffers now have explicit base addresses, but every head still uses
    # that same base. We need the exact packing of GQA heads and score rows.

    # TODO(runtime): Initialize each score accumulator with zero.
    #
    # WR_BIAS reads ``buffers.scores`` before the first RD_MAC writes a score.
    # Reusing that slot after one head would initialize the next head with an old
    # result unless the runtime supplies a separate zero source.

    # TODO(dataflow): Pack attention scores as a zero-padded logical vector.
    #
    # The last sequence group may not occupy every bank, and one head may not
    # fill its final burst. MAC_ABK still executes those positions. RD_MAC must
    # remain a raw result until a repacker copies live scores in order and writes
    # zero to every other lane in each occupied output slot.

    spec = plan.spec
    rows = plan.rows
    buffers = plan.buffers
    hardware = builder.hardware
    for name, value in (
        ("rows_per_key", plan.rows_per_key),
        ("operation_size", plan.operation_size),
        ("heads_per_row", plan.heads_per_row),
    ):
        require_positive(name, value)
    if not plan.sequence_channels:
        raise ValueError("sequence_channels cannot be empty")
    if len(plan.sequence_channels) * builder.total_banks < spec.sequence_length:
        raise ValueError("sequence channel groups do not cover sequence_length")
    if plan.rows_per_key * hardware.dram_columns < spec.kv_width:
        raise ValueError("rows_per_key does not cover kv_width")
    if plan.operation_size * hardware.burst_length < spec.head_size:
        raise ValueError("operation_size does not cover head_size")
    if plan.rows_per_key != ceil_div(spec.kv_width, hardware.dram_columns):
        raise ValueError("rows_per_key must match the packed key width")
    if plan.operation_size != ceil_div(spec.head_size, hardware.burst_length):
        raise ValueError("operation_size must match one attention head")
    expected_heads_per_row = hardware.dram_columns // spec.head_size
    if expected_heads_per_row < 1 or plan.heads_per_row != expected_heads_per_row:
        raise ValueError("heads_per_row must match the DRAM row width")
    expected_sequence_groups = ceil_div(spec.sequence_length, builder.total_banks)
    if len(plan.sequence_channels) != expected_sequence_groups:
        raise ValueError("sequence_channels must match the sequence layout")
    for channels in plan.sequence_channels:
        builder.channel_set(channels.channels)
    _require_dram_row_capacity(
        "key_cache",
        rows.key_cache,
        expected_sequence_groups * plan.rows_per_key,
    )

    # Several key heads can share one DRAM row. ``heads_per_row`` says how many
    # fit, and ``mac_size`` says how many bursts make up one head.
    _require_shared_buffer_capacity("query", buffers.query, plan.operation_size)
    _require_shared_buffer_capacity("scores", buffers.scores, 1)

    for row_offset in range(plan.rows_per_key):
        for sequence_group, channels in enumerate(plan.sequence_channels):
            # In grouped-query attention, several query heads use the same key
            # head. Each pass copies one query slice from the Shared Buffer to
            # the selected channels' Global Buffers.
            for _ in range(spec.repeat_count):
                builder.append(
                    WriteGlobalBuffer(
                        operation_size=plan.operation_size,
                        column=0,
                        source=buffers.query.start,
                        channels=channels,
                    )
                )
                # The final cache row may contain fewer packed KV heads than a
                # full DRAM row. Do not emit MACs for padding columns.
                remaining_heads = spec.num_kv_heads - row_offset * plan.heads_per_row
                heads_in_row = min(plan.heads_per_row, remaining_heads)
                for head_index in range(heads_in_row):
                    # WR_BIAS loads the score's starting value. The ISA does not
                    # yet explain how it selects ``accumulation_register``. The
                    # row stays fixed while the column moves to the next head.
                    builder.append(
                        WriteBias(
                            source=buffers.scores.start,
                            channels=channels,
                        )
                    )
                    row = rows.key_cache.row(
                        sequence_group * plan.rows_per_key + row_offset
                    )
                    builder.append(
                        MacAllBanks(
                            operation_size=plan.operation_size,
                            channels=channels,
                            row=row,
                            column=head_index * spec.head_size,
                            accumulation_register=plan.accumulation_register,
                            operand_source=MacOperandSource.GLOBAL_BUFFER,
                        )
                    )
                    builder.append(
                        ReadMac(
                            destination=buffers.scores.start,
                            accumulation_register=plan.accumulation_register,
                            channels=channels,
                        )
                    )


def _validate_score_transfer(
    builder: CentProgramBuilder,
    plan: ScoreTransferPlan,
) -> None:
    """Check one score transfer without changing the program builder.

    Args:
        builder: Program builder supplying target geometry.
        plan: Score layout, transfer direction, and bank placement to check.

    Raises:
        ValueError: If the planned transfer geometry is invalid.
    """

    spec = plan.spec
    rows = plan.rows
    buffers = plan.buffers
    if plan.instruction_type not in (WriteSingleBank, ReadSingleBank):
        raise ValueError("score transfer instruction_type is unsupported")
    if plan.bank_group not in range(3):
        raise ValueError("score bank_group must be between 0 and 2")
    require_positive("rows_per_score", plan.rows_per_score)
    require_positive("heads_per_bank", plan.heads_per_bank)
    if not plan.replica_channel_offsets:
        raise ValueError("replica_channel_offsets cannot be empty")
    if len(set(plan.replica_channel_offsets)) != len(plan.replica_channel_offsets):
        raise ValueError("replica_channel_offsets cannot contain duplicates")
    if not plan.logical_banks:
        raise ValueError("logical_banks cannot be empty")
    if plan.rows_per_score * builder.hardware.dram_columns < spec.sequence_length:
        raise ValueError("rows_per_score does not cover sequence_length")
    if len(plan.logical_banks) * plan.heads_per_bank < spec.num_attention_heads:
        raise ValueError("logical score banks do not cover all attention heads")
    if tuple(sorted(set(plan.logical_banks))) != plan.logical_banks:
        raise ValueError("logical_banks must be unique and sorted")
    if plan.rows_per_score != ceil_div(
        spec.sequence_length, builder.hardware.dram_columns
    ):
        raise ValueError("rows_per_score must match the sequence length")
    if plan.heads_per_bank != ceil_div(
        spec.num_attention_heads, len(plan.logical_banks)
    ):
        raise ValueError("heads_per_bank must match the selected score banks")
    for logical_bank in plan.logical_banks:
        if not 0 <= logical_bank < builder.total_banks:
            raise ValueError("logical score bank is outside the block allocation")
        if logical_bank % BANKS_PER_PU != plan.bank_group:
            raise ValueError("logical score bank does not match bank_group")

    physical_channels = tuple(
        builder.bank_index(logical_bank)[0] + channel_offset
        for logical_bank in plan.logical_banks
        for channel_offset in plan.replica_channel_offsets
    )
    builder.channel_set(tuple(sorted(set(physical_channels))))
    _require_dram_row_capacity(
        "scores",
        rows.scores,
        plan.heads_per_bank * plan.rows_per_score,
    )

    hardware = builder.hardware
    score_slots = ceil_div(
        min(spec.sequence_length, hardware.dram_columns),
        hardware.burst_length,
    )
    _require_shared_buffer_capacity("scores", buffers.scores, score_slots)


def _lower_score_transfer(
    builder: CentProgramBuilder,
    plan: ScoreTransferPlan,
) -> None:
    """Move attention scores between the Shared Buffer and DRAM.

    Args:
        builder: Program builder that receives the transfer instructions.
        plan: Checked score layout, transfer direction, and bank placement.

    Raises:
        ValueError: If the planned transfer geometry is invalid.
    """

    _validate_score_transfer(builder, plan)
    spec = plan.spec
    rows = plan.rows
    buffers = plan.buffers
    hardware = builder.hardware

    # Scores are laid out by head and token position. These loops visit every
    # DRAM burst that contains scores.
    for row_offset in range(plan.rows_per_score):
        row_size = min(
            hardware.dram_columns,
            spec.sequence_length - row_offset * hardware.dram_columns,
        )
        for head_slot in range(plan.heads_per_bank):
            for column in range(0, row_size, hardware.burst_length):
                for logical_bank in plan.logical_banks:
                    head = (
                        logical_bank // BANKS_PER_PU
                    ) * plan.heads_per_bank + head_slot
                    if head >= spec.num_attention_heads:
                        break
                    channel, bank = builder.bank_index(logical_bank)
                    for channel_offset in plan.replica_channel_offsets:
                        # A CENT memory address contains channel, bank, row, and
                        # column. Adding one block width selects the same bank in
                        # the next repeated channel region.
                        address = CentMemoryAddress(
                            channel=(channel + channel_offset),
                            bank=bank,
                            row=rows.scores.row(
                                head_slot * plan.rows_per_score + row_offset
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
                        instruction: WriteSingleBank | ReadSingleBank
                        if plan.instruction_type is WriteSingleBank:
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
    plan: SoftmaxPlan,
) -> None:
    """Add the two multiplication passes from the partial softmax path.

    Args:
        builder: Program builder that receives the softmax instructions.
        plan: Checked score layout, multiply passes, and channel placement.

    Raises:
        ValueError: If the planned score geometry or buffer is invalid.
    """

    # TODO(softmax): Add the missing softmax steps.
    #
    # The code below only emits two multiplications. It does not emit EXP, RED,
    # the reciprocal calculation, or the movement between those steps. We must
    # also decide whether to subtract the largest score before EXP. EXP changes
    # a zero padding lane to one, so its producer must restore zero padding
    # before the result can be treated as a logical vector or consumed by RED.

    spec = plan.spec
    rows = plan.rows
    buffers = plan.buffers
    hardware = builder.hardware
    require_positive("rows_per_score", plan.rows_per_score)
    require_positive("heads_per_bank", plan.heads_per_bank)
    if not plan.passes:
        raise ValueError("softmax passes cannot be empty")
    if plan.rows_per_score * hardware.dram_columns < spec.sequence_length:
        raise ValueError("rows_per_score does not cover sequence_length")
    if plan.rows_per_score != ceil_div(spec.sequence_length, hardware.dram_columns):
        raise ValueError("rows_per_score must match the sequence length")
    builder.channel_set(plan.channels.channels)
    _require_dram_row_capacity(
        "scores",
        rows.scores,
        plan.heads_per_bank * plan.rows_per_score,
    )
    score_slots = ceil_div(
        min(spec.sequence_length, hardware.dram_columns),
        hardware.burst_length,
    )
    _require_shared_buffer_capacity("scores", buffers.scores, score_slots)

    # Validate every pass before emitting the first one. A bad later pass must
    # not leave an apparently valid prefix in the caller's program.
    for pass_plan in plan.passes:
        _validate_score_transfer(builder, pass_plan.left_input)
        _validate_score_transfer(builder, pass_plan.right_input)
        _validate_score_transfer(builder, pass_plan.output)
    # Each pass puts two vectors in PU input banks, multiplies them, and reads
    # the result bank. The second vector should be 1/sqrt(head_size) in the first
    # pass and 1/sum(exp(scores)) in the second.
    #
    # Those values are not generated here. The paper assigns reciprocal and
    # square root work to RISC-V cores, but does not give program addresses.
    for pass_plan in plan.passes:
        _lower_score_transfer(builder, pass_plan.left_input)
        _lower_score_transfer(builder, pass_plan.right_input)
        for head_slot in range(plan.heads_per_bank):
            for row_offset in range(plan.rows_per_score):
                # Only the last score row may be partly full. OPsize counts the
                # bursts that contain real scores.
                row_size = min(
                    hardware.dram_columns,
                    spec.sequence_length - row_offset * hardware.dram_columns,
                )
                builder.append(
                    ElementwiseMultiply(
                        operation_size=ceil_div(row_size, hardware.burst_length),
                        channels=plan.channels,
                        row=rows.scores.row(
                            head_slot * plan.rows_per_score + row_offset
                        ),
                        column=0,
                    )
                )
        _lower_score_transfer(builder, pass_plan.output)


def lower_attention_output(
    builder: CentProgramBuilder,
    plan: AttentionOutputPlan,
) -> None:
    """Multiply softmax scores by cached values to form attention output.

    Args:
        builder: Program builder that receives the output instructions.
        plan: Checked value-cache geometry, channels, and buffer bindings.

    Raises:
        ValueError: If the planned geometry or a buffer is invalid.
    """

    # TODO(attention): Assign buffer offsets to every score row and output value.
    #
    # Every score row currently reads the start of the score span. A partial
    # result is also written to the same output slot for every head and dimension.
    # We do not know how the next row receives its prior partial or how completed
    # output values occupy separate slots.

    # TODO(runtime): Initialize the first partial for each output value to zero.
    #
    # Later sequence rows may reload a real partial through WR_BIAS. The first
    # sequence row has no previous partial, so its source must be a zero-filled
    # slot rather than an arbitrary output-buffer value.

    # TODO(dataflow): Pack each attention output as a zero-padded logical vector.
    #
    # MAC_ABK operates on complete bursts across every selected bank. When the
    # head width or sequence length does not fill them, stale values can change
    # the dot product. RD_MAC's raw result also needs a repacker that writes each
    # live output once and explicitly zeros every unused lane.

    spec = plan.spec
    rows = plan.rows
    buffers = plan.buffers
    hardware = builder.hardware
    for name, value in (
        ("rows_per_sequence", plan.rows_per_sequence),
        ("rows_per_dimension", plan.rows_per_dimension),
        ("heads_per_channel", plan.heads_per_channel),
        ("dimension_iterations", plan.dimension_iterations),
    ):
        require_positive(name, value)
    if plan.rows_per_sequence * hardware.dram_columns < spec.sequence_length:
        raise ValueError("rows_per_sequence does not cover sequence_length")
    if plan.rows_per_dimension * hardware.dram_columns < spec.max_sequence_length:
        raise ValueError("rows_per_dimension does not cover max_sequence_length")
    if plan.rows_per_sequence != ceil_div(spec.sequence_length, hardware.dram_columns):
        raise ValueError("rows_per_sequence must match the sequence length")
    if plan.rows_per_dimension != ceil_div(
        spec.max_sequence_length, hardware.dram_columns
    ):
        raise ValueError("rows_per_dimension must match the reserved sequence length")
    if plan.heads_per_channel != ceil_div(
        spec.num_kv_heads, builder.placement.channels_per_block
    ):
        raise ValueError("heads_per_channel must match the block placement")
    if plan.dimension_iterations != ceil_div(spec.head_size, hardware.num_banks):
        raise ValueError("dimension_iterations must match the head width")
    builder.channel_set(plan.channels.channels)
    _require_dram_row_capacity(
        "value_cache",
        rows.value_cache,
        plan.heads_per_channel * plan.dimension_iterations * plan.rows_per_dimension,
    )
    score_slots = ceil_div(
        min(spec.sequence_length, hardware.dram_columns),
        hardware.burst_length,
    )
    _require_shared_buffer_capacity("scores", buffers.scores, score_slots)
    _require_shared_buffer_capacity("output", buffers.output, 1)

    # One value-cache row follows one head value across many token positions.
    # Copying scores to the Global Buffer lets different banks calculate
    # different output values from the same tokens.
    for head_slot in range(plan.heads_per_channel):
        head_row_offset = (
            plan.rows_per_dimension * plan.dimension_iterations * head_slot
        )
        # Repeat a KV head for each query head that shares it.
        for _ in range(spec.repeat_count):
            for sequence_row in range(plan.rows_per_sequence):
                row_size = min(
                    hardware.dram_columns,
                    spec.sequence_length - sequence_row * hardware.dram_columns,
                )
                op_size = ceil_div(row_size, hardware.burst_length)
                builder.append(
                    WriteGlobalBuffer(
                        operation_size=op_size,
                        column=0,
                        source=buffers.scores.start,
                        channels=plan.channels,
                    )
                )
                for dimension in range(plan.dimension_iterations):
                    # WR_BIAS loads one output's previous partial. The ISA does
                    # not yet explain how it selects ``accumulation_register``.
                    # The selected row contains values across this token range.
                    builder.append(
                        WriteBias(
                            source=buffers.output.start,
                            channels=plan.channels,
                        )
                    )
                    builder.append(
                        MacAllBanks(
                            operation_size=op_size,
                            channels=plan.channels,
                            row=rows.value_cache.row(
                                head_row_offset
                                + dimension * plan.rows_per_dimension
                                + sequence_row
                            ),
                            column=0,
                            accumulation_register=plan.accumulation_register,
                            operand_source=MacOperandSource.GLOBAL_BUFFER,
                        )
                    )
                    builder.append(
                        ReadMac(
                            destination=buffers.output.start,
                            accumulation_register=plan.accumulation_register,
                            channels=plan.channels,
                        )
                    )
