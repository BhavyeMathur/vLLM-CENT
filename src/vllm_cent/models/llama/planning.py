"""Check Llama dimensions and assign DRAM rows to block tensors."""

from dataclasses import dataclass
from ...cent import (
    BANKS_PER_PU,
    CentBlockPlacementSpec,
    CentChannelSet,
    CentHardwareSpec,
    CentSharedBufferAddress,
    ElementwiseMultiply,
    ReadSingleBank,
    WriteSingleBank,
)
from ...cent.utils import ceil_div, require_positive
from ...lowering import (
    CentAccumulatePlan,
    CentBankGroupVectorTransferPlan,
    CentDramRowRange,
    CentDramVector,
    CentL2NormPlan,
    CentPartitionedVectorLayout,
    CentRmsNormPlan,
    CentSharedBufferSpan,
    CentSharedBufferVector,
    CentSumOfSquaresPlan,
    CentWeightGemvPlan,
    plan_partitioned_vector,
    plan_weight_gemv,
)
from ...lowering.transformer import (
    AttentionOutputPlan,
    KvCacheUpdatePlan,
    RotaryEmbeddingPlan,
    ScoreGemvPlan,
    ScoreTransferPlan,
    SoftmaxPassPlan,
    SoftmaxPlan,
    TransformerAttentionBuffers,
    TransformerAttentionPlan,
    TransformerAttentionRows,
    TransformerAttentionSpec,
)
from ...request import CompileRequest, DecodeStepSpec
from .feed_forward import _SiluProductChunkPlan, _SiluProductPlan
from .spec import LlamaModelSpec

__all__: list[str] = []


@dataclass(frozen=True, slots=True)
class _LlamaCompileContext:
    """Sizes and hardware information shared by all Llama compiler steps.

    Attributes:
        model: Sizes of the Llama block being compiled.
        hardware: Channels, banks, rows, columns, and buffer limits.
        placement: Number of channels assigned to this block.
        step: Current token count and reserved cache length.
        head_size: Values in one attention head. A head must fit in a DRAM row
            and divide evenly across banks and bursts.
        kv_width: Values in all distinct key or value heads. This equals
            ``head_size * num_kv_heads``.
        repeat_count: Query heads that share one KV head. This equals
            ``num_attention_heads // num_kv_heads``.
        total_banks: Banks assigned to this block. This equals
            ``channels_per_block * num_banks``.
        activation_capacity: FFN values handled by one activation pass. The
            current lowering supports no more than two passes.
    """

    model: LlamaModelSpec
    hardware: CentHardwareSpec
    placement: CentBlockPlacementSpec
    step: DecodeStepSpec
    head_size: int
    kv_width: int
    repeat_count: int
    total_banks: int
    activation_capacity: int


@dataclass(frozen=True, slots=True)
class _LlamaRowCounts:
    """Number of DRAM rows needed for each kind of Llama tensor.

    Tensors with the same shape reuse a row-count field, but receive different
    row ranges later.

    Attributes:
        x: Rows for an input or residual vector copied across channels.
        wq: Rows for query-projection weights.
        wk: Rows for key-projection weights.
        wv: Rows for value-projection weights.
        projection: Rows for one projected query or key.
        cache_k: Rows for keys at every reserved token position.
        scores: Rows for every attention head's token scores.
        cache_v: Rows for values at every reserved token position.
        hidden_vector: Rows for a hidden-width vector split across the block.
        wo: Rows for attention-output weights.
        w1: Rows for the FFN gate weights.
        w3: Rows for the FFN up-projection weights.
        intermediate_vector: Rows for one expanded FFN vector.
        ffn_vector: Rows for the gated FFN product.
        w2: Rows for the FFN down-projection weights.
    """

    x: int
    wq: int
    wk: int
    wv: int
    projection: int
    cache_k: int
    scores: int
    cache_v: int
    hidden_vector: int
    wo: int
    w1: int
    w3: int
    intermediate_vector: int
    ffn_vector: int
    w2: int


@dataclass(frozen=True, slots=True)
class _LlamaMemoryLayout:
    """Starting DRAM row assigned to every tensor in one Llama block.

    The row ranges are consecutive and do not overlap. ``end`` is the first row
    after all allocations.

    Attributes:
        x: Input vector and first residual value.
        x_copy: Copy of the input used by RMSNorm.
        sa_norm: First RMSNorm weights and result.
        wq: Query-projection weights.
        wk: Key-projection weights.
        wv: Value-projection weights.
        xq: Query result and rotary workspace.
        xk: Key result and rotary workspace.
        cache_k: Stored keys from all tokens.
        scores: Attention scores for the current token.
        cache_v: Stored values from all tokens.
        output: Result of combining scores with cached values.
        wo: Weights that combine attention heads.
        sa: Attention result after adding the input residual.
        sa_copy: Copy of that residual result used by RMSNorm.
        ffn_norm: Second RMSNorm weights and result.
        w1: FFN gate weights.
        w3: FFN up-projection weights.
        x1: Result of the W1 projection.
        x3: Result of the W3 projection.
        x1_sigmoid: Sigmoid applied to ``x1``.
        ffn_vector: Product of SiLU(``x1``) and ``x3``.
        w2: FFN down-projection weights.
        ffn: FFN output before the final residual addition.
        end: First row not assigned to this block.
    """

    x: int
    x_copy: int
    sa_norm: int
    wq: int
    wk: int
    wv: int
    xq: int
    xk: int
    cache_k: int
    scores: int
    cache_v: int
    output: int
    wo: int
    sa: int
    sa_copy: int
    ffn_norm: int
    w1: int
    w3: int
    x1: int
    x3: int
    x1_sigmoid: int
    ffn_vector: int
    w2: int
    ffn: int
    end: int


@dataclass(frozen=True, slots=True)
class _LlamaBufferLayout:
    """Shared Buffer spans used by one Llama block.

    Q, K, and V use separate spans because attention needs all three. FFN spans
    may reuse earlier attention storage after those values are no longer live.

    Attributes:
        input: Slots containing the block input during the first RMSNorm.
        normalized: Slots containing the normalized projection input.
        query_result: Accumulator results produced by the Q projection.
        key_result: Accumulator results produced by the K projection.
        value_result: Accumulator results produced by the V projection.
        query: Slots containing Q in the packed vector layout used by attention.
        key: Slots containing K in the packed vector layout used by attention.
        value: Slots containing V in the packed vector layout used by attention.
        scores: Slots staging scores while ``input`` remains live for the
            attention residual connection.
        ffn_gate: Slots receiving the raw W1 projection.
        ffn_gate_sigmoid: Slots receiving sigmoid applied to W1.
        ffn_up: Slots receiving the W3 projection.
        ffn_product: Slots staging the gated FFN vector for W2.
        end_slot: First slot not used by any span.
    """

    input: CentSharedBufferSpan
    normalized: CentSharedBufferSpan
    query_result: CentSharedBufferSpan
    key_result: CentSharedBufferSpan
    value_result: CentSharedBufferSpan
    query: CentSharedBufferSpan
    key: CentSharedBufferSpan
    value: CentSharedBufferSpan
    scores: CentSharedBufferSpan
    ffn_gate: CentSharedBufferSpan
    ffn_gate_sigmoid: CentSharedBufferSpan
    ffn_up: CentSharedBufferSpan
    ffn_product: CentSharedBufferSpan
    end_slot: int


@dataclass(frozen=True, slots=True)
class _LlamaSelfAttentionPlan:
    """Collect the generic operation plans used by Llama self-attention.

    Attributes:
        normalization: RMS normalization applied to the block input.
        query_projection: GEMV that produces the query vector.
        key_projection: GEMV that produces the key vector.
        value_projection: GEMV that produces the value vector.
        attention: Transformer-family decode-attention plan.
        output_projection: GEMV that combines attention heads.
        residual: Addition of the original block input.
        store_residual: Transfer preserving the residual result in DRAM.
    """

    normalization: CentRmsNormPlan
    query_projection: CentWeightGemvPlan
    key_projection: CentWeightGemvPlan
    value_projection: CentWeightGemvPlan
    attention: TransformerAttentionPlan
    output_projection: CentWeightGemvPlan
    residual: CentAccumulatePlan
    store_residual: CentBankGroupVectorTransferPlan


@dataclass(frozen=True, slots=True)
class _LlamaFeedForwardPlan:
    """Collect the generic operation plans used by Llama's FFN.

    Attributes:
        normalization: RMS normalization applied to the attention residual.
        gate_projection: GEMV that produces the FFN gate values.
        up_projection: GEMV that produces the second expanded vector.
        silu_product: Elementwise plan that forms the gated FFN product.
        down_projection: GEMV that returns the FFN vector to hidden width.
        load_residual: Transfer restoring the attention residual from DRAM.
        residual: Addition that produces the transformer-block output.
    """

    normalization: CentRmsNormPlan
    gate_projection: CentWeightGemvPlan
    up_projection: CentWeightGemvPlan
    silu_product: _SiluProductPlan
    down_projection: CentWeightGemvPlan
    load_residual: CentBankGroupVectorTransferPlan
    residual: CentAccumulatePlan


@dataclass(frozen=True, slots=True)
class _LlamaCompilePlan:
    """All checked inputs and placements needed to lower one Llama block.

    Attributes:
        context: Derived model, hardware, and decode dimensions.
        row_counts: DRAM space required by each tensor kind.
        memory: DRAM row assigned to every Llama tensor.
        buffers: Shared Buffer spans used as values move through the block.
        self_attention: Generic lowerer plans used by self-attention.
        feed_forward: Generic lowerer plans used by the feed-forward network.
    """

    context: _LlamaCompileContext
    row_counts: _LlamaRowCounts
    memory: _LlamaMemoryLayout
    buffers: _LlamaBufferLayout
    self_attention: _LlamaSelfAttentionPlan
    feed_forward: _LlamaFeedForwardPlan


@dataclass(slots=True)
class _RowAllocator:
    """Track the next free row while building a memory layout.

    Attributes:
        cursor: First unallocated DRAM row, starting at row 0.
    """

    cursor: int = 0

    def reserve(self, row_count: int) -> int:
        """Reserve consecutive rows and return the first one.

        Args:
            row_count: Number of rows needed by the tensor.

        Returns:
            First row assigned to the tensor.

        Raises:
            ValueError: If ``row_count`` is less than one.
        """

        require_positive("row_count", row_count)

        # Give the tensor the current free row, then move the cursor past every
        # row it uses. The next reservation therefore cannot overlap it.
        start = self.cursor
        self.cursor += row_count
        return start


def _create_context(request: CompileRequest) -> _LlamaCompileContext:
    """Calculate the shared values used by all Llama compiler steps.

    Args:
        request: Llama model, CENT hardware, placement, and decode sizes.

    Returns:
        Checked model and hardware values needed by the Llama compiler.

    Raises:
        TypeError: If the request does not describe a Llama model.
        ValueError: If the model does not fit the requested CENT layout.
    """

    if not isinstance(request.model, LlamaModelSpec):
        raise TypeError("the Llama compiler requires a LlamaModelSpec")
    model = request.model
    head_size = model.head_size

    # Flatten the assigned channels and their banks into one count. Later code
    # uses this count to divide tensor values among banks.
    total_banks = request.placement.channels_per_block * request.hardware.num_banks
    context = _LlamaCompileContext(
        model=model,
        hardware=request.hardware,
        placement=request.placement,
        step=request.step,
        head_size=head_size,
        # K and V contain only distinct KV heads. Several query heads may share
        # each one.
        kv_width=head_size * model.num_kv_heads,
        repeat_count=(model.num_attention_heads // model.num_kv_heads),
        total_banks=total_banks,
        # Each group of four banks has one PU. Every PU handles one row of FFN
        # values during an activation pass.
        activation_capacity=(
            request.placement.channels_per_block
            * (request.hardware.num_banks // BANKS_PER_PU)
            * request.hardware.dram_columns
        ),
    )
    _validate_context(context)
    return context


def _validate_context(context: _LlamaCompileContext) -> None:
    """Check that the model can use the chosen hardware and placement.

    Args:
        context: Model and hardware values to check together.

    Raises:
        ValueError: If the dimensions cannot form the current memory layout.
    """

    model = context.model
    hardware = context.hardware
    placement = context.placement

    # The current compiler repeats the block in equal channel regions. A partial
    # region would not have enough channels for the same layout.
    if placement.channels_per_block > hardware.num_channels:
        raise ValueError("channels_per_block cannot exceed num_channels")
    if hardware.num_channels % placement.channels_per_block != 0:
        raise ValueError("channels_per_block must divide num_channels")
    # The sigmoid path needs to keep both raw and activated results.
    if hardware.accumulator_slots_per_bank < 2:
        raise ValueError(
            "Llama fused activation requires at least two accumulator slots per bank"
        )
    # One attention head is split evenly among banks and stored wholly inside a
    # DRAM row. Score multiplication depends on both facts.
    if context.head_size < hardware.num_banks:
        raise ValueError("attention head size must be at least num_banks")
    if context.head_size % hardware.num_banks != 0:
        raise ValueError("attention head size must be divisible by num_banks")
    if context.head_size > hardware.dram_columns:
        raise ValueError("attention head size cannot exceed dram_columns")
    if hardware.dram_columns % context.head_size != 0:
        raise ValueError("dram_columns must be divisible by attention head size")
    if context.head_size % hardware.burst_length != 0:
        raise ValueError("attention head size must be divisible by burst_length")
    # The current SiLU path has only two work areas.
    if model.intermediate_size > context.activation_capacity * 2:
        raise ValueError("intermediate_size exceeds two-pass activation capacity")


def _row_counts(context: _LlamaCompileContext) -> _LlamaRowCounts:
    """Calculate how many DRAM rows each tensor shape needs.

    Args:
        context: Checked model, hardware, and decode sizes.

    Returns:
        Row count for each tensor shape used by the block.
    """

    model = context.model
    hardware = context.hardware
    hidden = model.hidden_size
    intermediate = model.intermediate_size
    columns = hardware.dram_columns
    banks = hardware.num_banks
    max_sequence = context.step.max_sequence_length

    # Divide matrix outputs among all assigned banks. Each output needs one or
    # more rows to hold all of its input weights.
    hidden_outputs_per_bank = ceil_div(hidden, context.total_banks)
    kv_outputs_per_bank = ceil_div(context.kv_width, context.total_banks)
    intermediate_outputs_per_bank = ceil_div(intermediate, context.total_banks)
    hidden_rows_per_output = ceil_div(hidden, columns)
    intermediate_rows_per_output = ceil_div(intermediate, columns)

    # Reserve caches for the maximum token count. Keys cycle tokens through
    # banks; scores and values put token positions across row columns.
    sequence_bank_groups = ceil_div(max_sequence, context.total_banks)
    sequence_rows = ceil_div(max_sequence, columns)
    query_head_slots_per_pu = ceil_div(
        model.num_attention_heads,
        context.total_banks // BANKS_PER_PU,
    )
    kv_heads_per_channel = ceil_div(
        model.num_kv_heads,
        context.placement.channels_per_block,
    )
    head_slices_across_banks = ceil_div(context.head_size, banks)

    # Normalization and rotary lowering place hidden vectors across four-bank
    # PU groups. Their row ranges must fit the largest piece held by one group.
    query_layout = plan_partitioned_vector(
        hidden,
        context.total_banks // BANKS_PER_PU,
        hardware.burst_length,
    )
    projection_rows = ceil_div(
        query_layout.values_per_partition,
        columns,
    )

    # TODO(layout): Define when vectors are copied or split across channels.
    #
    # Input and FFN vectors currently repeat in each channel. Other work vectors
    # are split across all assigned banks. This difference is not yet justified.

    return _LlamaRowCounts(
        # Normalization uses the same PU-group layout as other hidden vectors.
        x=projection_rows,
        wq=hidden_outputs_per_bank * hidden_rows_per_output,
        wk=kv_outputs_per_bank * hidden_rows_per_output,
        wv=kv_outputs_per_bank * hidden_rows_per_output,
        projection=projection_rows,
        cache_k=sequence_bank_groups * ceil_div(context.kv_width, columns),
        scores=sequence_rows * query_head_slots_per_pu,
        cache_v=(sequence_rows * kv_heads_per_channel * head_slices_across_banks),
        hidden_vector=projection_rows,
        wo=hidden_outputs_per_bank * hidden_rows_per_output,
        w1=intermediate_outputs_per_bank * hidden_rows_per_output,
        w3=intermediate_outputs_per_bank * hidden_rows_per_output,
        intermediate_vector=ceil_div(intermediate, context.total_banks * columns),
        # Each channel holds a copy of the gated product across its banks.
        ffn_vector=ceil_div(intermediate, columns * banks),
        w2=hidden_outputs_per_bank * intermediate_rows_per_output,
    )


def _plan_memory(context: _LlamaCompileContext) -> _LlamaMemoryLayout:
    """Give every tensor a separate, consecutive DRAM row range.

    Args:
        context: Checked model, hardware, and decode sizes.

    Returns:
        Starting row of each tensor and the first row left unused.

    Raises:
        ValueError: If a bank does not have enough rows for the block.
    """

    rows = _row_counts(context)
    allocator = _RowAllocator()

    # TODO(dataflow): Resolve work areas that are allocated but never used.
    #
    # ``output``, ``x3``, ``ffn_vector``, and ``ffn`` have no users. Add
    # their missing data movement or remove their allocations.

    # Python evaluates these arguments from top to bottom. Each tensor starts
    # immediately after the preceding tensor.
    layout = _LlamaMemoryLayout(
        # Input and first normalization.
        x=allocator.reserve(rows.x),
        x_copy=allocator.reserve(rows.x),
        sa_norm=allocator.reserve(rows.x),
        # Q, K, and V weights and results.
        wq=allocator.reserve(rows.wq),
        wk=allocator.reserve(rows.wk),
        wv=allocator.reserve(rows.wv),
        xq=allocator.reserve(rows.projection),
        xk=allocator.reserve(rows.projection),
        # KV caches and temporary attention scores.
        cache_k=allocator.reserve(rows.cache_k),
        scores=allocator.reserve(rows.scores),
        cache_v=allocator.reserve(rows.cache_v),
        # Attention output and first residual connection.
        output=allocator.reserve(rows.hidden_vector),
        wo=allocator.reserve(rows.wo),
        sa=allocator.reserve(rows.hidden_vector),
        sa_copy=allocator.reserve(rows.hidden_vector),
        ffn_norm=allocator.reserve(rows.hidden_vector),
        # Feed-forward weights, intermediate values, and output.
        w1=allocator.reserve(rows.w1),
        w3=allocator.reserve(rows.w3),
        x1=allocator.reserve(rows.intermediate_vector),
        x3=allocator.reserve(rows.intermediate_vector),
        x1_sigmoid=allocator.reserve(rows.intermediate_vector),
        ffn_vector=allocator.reserve(rows.ffn_vector),
        w2=allocator.reserve(rows.w2),
        ffn=allocator.reserve(rows.hidden_vector),
        # The cursor now points to the first free row.
        end=allocator.cursor,
    )

    if layout.end > context.hardware.dram_rows:
        raise ValueError(
            f"transformer block requires {layout.end} DRAM rows, "
            f"but hardware provides {context.hardware.dram_rows}"
        )
    return layout


def _plan_shared_buffer(context: _LlamaCompileContext) -> _LlamaBufferLayout:
    """Assign Shared Buffer spans according to Llama value lifetimes.

    Args:
        context: Checked Llama dimensions and target hardware.

    Returns:
        Named spans used by attention and feed-forward lowering.

    Raises:
        ValueError: If the target Shared Buffer cannot hold the live spans.
    """

    burst_length = context.hardware.burst_length
    pu_groups = context.total_banks // BANKS_PER_PU

    hidden_layout = plan_partitioned_vector(
        context.model.hidden_size, pu_groups, burst_length
    )
    hidden_slots = hidden_layout.slot_count
    kv_slots = plan_partitioned_vector(
        context.kv_width, pu_groups, burst_length
    ).slot_count
    intermediate_slots = plan_partitioned_vector(
        context.model.intermediate_size, pu_groups, burst_length
    ).slot_count
    intermediate_result_slots = ceil_div(
        context.model.intermediate_size, context.total_banks
    )
    hidden_result_slots = ceil_div(context.model.hidden_size, context.total_banks)
    kv_result_slots = ceil_div(context.kv_width, context.total_banks)

    # The values are placed in the same order that the block produces them.
    # Keeping the arithmetic here makes every boundary visible during review.
    input_start = 0
    normalized_start = input_start + hidden_slots
    query_result_start = normalized_start + hidden_slots
    key_result_start = query_result_start + hidden_result_slots
    value_result_start = key_result_start + kv_result_slots
    query_start = value_result_start + kv_result_slots
    key_start = query_start + hidden_slots
    value_start = key_start + kv_slots
    score_start = value_start + kv_slots
    score_slots = ceil_div(
        min(context.step.sequence_length, context.hardware.dram_columns),
        burst_length,
    )
    attention_end_slot = score_start + score_slots

    # FFN lowering runs after attention, so it can reuse those slots. Its three
    # accumulator outputs are adjacent at the start of the buffer.
    ffn_gate_start = input_start
    ffn_gate_sigmoid_start = ffn_gate_start + intermediate_result_slots
    ffn_up_start = ffn_gate_sigmoid_start + intermediate_result_slots
    ffn_product_start = input_start
    ffn_projection_end = ffn_up_start + intermediate_result_slots
    end_slot = max(
        attention_end_slot,
        intermediate_slots,
        ffn_projection_end,
    )

    if end_slot > context.hardware.shared_buffer_slots:
        raise ValueError(
            f"transformer block requires {end_slot} Shared Buffer slots, "
            f"but hardware provides {context.hardware.shared_buffer_slots}"
        )

    return _LlamaBufferLayout(
        input=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=input_start),
            slot_count=hidden_slots,
        ),
        normalized=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=normalized_start),
            slot_count=hidden_slots,
        ),
        query_result=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=query_result_start),
            slot_count=hidden_result_slots,
        ),
        key_result=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=key_result_start),
            slot_count=kv_result_slots,
        ),
        value_result=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=value_result_start),
            slot_count=kv_result_slots,
        ),
        query=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=query_start),
            slot_count=hidden_slots,
        ),
        key=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=key_start),
            slot_count=kv_slots,
        ),
        value=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=value_start),
            slot_count=kv_slots,
        ),
        scores=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=score_start),
            slot_count=score_slots,
        ),
        ffn_gate=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=ffn_gate_start),
            slot_count=intermediate_result_slots,
        ),
        ffn_gate_sigmoid=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=ffn_gate_sigmoid_start),
            slot_count=intermediate_result_slots,
        ),
        ffn_up=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=ffn_up_start),
            slot_count=intermediate_result_slots,
        ),
        ffn_product=CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=ffn_product_start),
            slot_count=intermediate_slots,
        ),
        end_slot=end_slot,
    )


def _create_rotary_embedding_plan(
    context: _LlamaCompileContext,
    spec: TransformerAttentionSpec,
    rows: TransformerAttentionRows,
    buffers: TransformerAttentionBuffers,
) -> RotaryEmbeddingPlan:
    """Plan how query and key values occupy CENT PU groups.

    Args:
        context: Derived Llama and hardware dimensions.
        spec: Logical dimensions of the attention operation.
        rows: DRAM regions assigned to attention tensors.
        buffers: Shared Buffer regions assigned to attention tensors.

    Returns:
        Explicit query and key partitions used by rotary lowering.
    """

    hardware = context.hardware
    pu_groups = context.total_banks // BANKS_PER_PU

    # The baseline layout uses every available four-bank PU group. A future
    # optimizer can create a different plan without changing the lowerer.
    query_layout = plan_partitioned_vector(
        spec.hidden_size,
        pu_groups,
        hardware.burst_length,
    )
    key_layout = plan_partitioned_vector(
        spec.kv_width,
        pu_groups,
        hardware.burst_length,
    )
    return RotaryEmbeddingPlan(
        spec=spec,
        rows=rows,
        buffers=buffers,
        channels=CentChannelSet(channels=tuple(range(hardware.num_channels))),
        query_values_per_partition=query_layout.values_per_partition,
        query_partition_count=query_layout.partition_count,
        key_values_per_partition=key_layout.values_per_partition,
        key_partition_count=key_layout.partition_count,
        transfer_channels_required=context.placement.channels_per_block,
    )


def _create_kv_cache_update_plan(
    context: _LlamaCompileContext,
    spec: TransformerAttentionSpec,
    rows: TransformerAttentionRows,
    buffers: TransformerAttentionBuffers,
) -> KvCacheUpdatePlan:
    """Plan where the current token's key and value enter the KV cache.

    Args:
        context: Derived Llama and hardware dimensions.
        spec: Logical dimensions of the attention operation.
        rows: DRAM regions assigned to attention tensors.
        buffers: Shared Buffer regions assigned to attention tensors.

    Returns:
        Physical key and value cache placement for the current token.
    """

    hardware = context.hardware
    channels_per_block = context.placement.channels_per_block
    sequence_index = spec.sequence_length - 1

    # Each equal-sized channel region receives the same key-cache layout.
    replica_channel_offsets = tuple(
        copy * channels_per_block
        for copy in range(hardware.num_channels // channels_per_block)
    )
    return KvCacheUpdatePlan(
        spec=spec,
        rows=rows,
        buffers=buffers,
        sequence_index=sequence_index,
        key_logical_bank=sequence_index % context.total_banks,
        key_row_group=sequence_index // context.total_banks,
        key_rows_per_token=ceil_div(spec.kv_width, hardware.dram_columns),
        key_replica_channel_offsets=replica_channel_offsets,
        value_channels=tuple(range(channels_per_block)),
        value_rows_per_dimension=ceil_div(
            spec.max_sequence_length,
            hardware.dram_columns,
        ),
        value_sequence_row=sequence_index // hardware.dram_columns,
        value_heads_per_channel=ceil_div(
            spec.num_kv_heads,
            channels_per_block,
        ),
        value_dimension_iterations=ceil_div(
            spec.head_size,
            hardware.num_banks,
        ),
        accumulation_register=0,
    )


def _create_score_gemv_plan(
    context: _LlamaCompileContext,
    spec: TransformerAttentionSpec,
    rows: TransformerAttentionRows,
    buffers: TransformerAttentionBuffers,
) -> ScoreGemvPlan:
    """Plan the query-by-key-cache matrix-vector multiplication.

    Args:
        context: Derived Llama and hardware dimensions.
        spec: Logical dimensions of the attention operation.
        rows: DRAM regions assigned to attention tensors.
        buffers: Shared Buffer regions assigned to attention tensors.

    Returns:
        Key-cache row geometry and channel schedule for score lowering.
    """

    hardware = context.hardware
    sequence_iterations = ceil_div(
        spec.sequence_length,
        context.total_banks,
    )
    sequence_channels: list[CentChannelSet] = []
    for sequence_group in range(sequence_iterations):
        remaining_tokens = spec.sequence_length - sequence_group * context.total_banks
        active_channels_per_copy = ceil_div(
            min(remaining_tokens, context.total_banks),
            hardware.num_banks,
        )
        channels_per_block = context.placement.channels_per_block

        # Each physical copy uses the same prefix of its own channel region.
        # For example, two active channels in three-channel copies select
        # (0, 1, 3, 4), not the unrelated contiguous prefix (0, 1, 2, 3).
        channels = tuple(
            copy_start + local_channel
            for copy_start in range(0, hardware.num_channels, channels_per_block)
            for local_channel in range(active_channels_per_copy)
        )
        sequence_channels.append(CentChannelSet(channels=channels))

    return ScoreGemvPlan(
        spec=spec,
        rows=rows,
        buffers=buffers,
        rows_per_key=ceil_div(spec.kv_width, hardware.dram_columns),
        operation_size=spec.head_size // hardware.burst_length,
        heads_per_row=hardware.dram_columns // spec.head_size,
        sequence_channels=tuple(sequence_channels),
        accumulation_register=0,
    )


def _create_score_transfer_plan(
    context: _LlamaCompileContext,
    spec: TransformerAttentionSpec,
    rows: TransformerAttentionRows,
    buffers: TransformerAttentionBuffers,
    instruction_type: type[WriteSingleBank] | type[ReadSingleBank],
    bank_group: int,
) -> ScoreTransferPlan:
    """Plan one direction of score movement for a softmax pass.

    Args:
        context: Derived Llama and hardware dimensions.
        spec: Logical dimensions of the attention operation.
        rows: DRAM regions assigned to attention tensors.
        buffers: Shared Buffer regions assigned to attention tensors.
        instruction_type: Direction of movement between DRAM and the buffer.
        bank_group: Bank position used within every four-bank PU group.

    Returns:
        Explicit score banks and channel replicas for one transfer.
    """

    channels_per_block = context.placement.channels_per_block
    return ScoreTransferPlan(
        spec=spec,
        rows=rows,
        buffers=buffers,
        instruction_type=instruction_type,
        bank_group=bank_group,
        rows_per_score=ceil_div(
            spec.sequence_length,
            context.hardware.dram_columns,
        ),
        heads_per_bank=ceil_div(
            spec.num_attention_heads,
            context.total_banks // BANKS_PER_PU,
        ),
        logical_banks=tuple(range(bank_group, context.total_banks, BANKS_PER_PU)),
        replica_channel_offsets=tuple(
            copy * channels_per_block
            for copy in range(context.hardware.num_channels // channels_per_block)
        ),
    )


def _create_softmax_plan(
    context: _LlamaCompileContext,
    spec: TransformerAttentionSpec,
    rows: TransformerAttentionRows,
    buffers: TransformerAttentionBuffers,
) -> SoftmaxPlan:
    """Plan score placement for the currently supported softmax passes.

    Args:
        context: Derived Llama and hardware dimensions.
        spec: Logical dimensions of the attention operation.
        rows: DRAM regions assigned to attention tensors.
        buffers: Shared Buffer regions assigned to attention tensors.

    Returns:
        Ordered multiply passes and the transfers surrounding each pass.
    """

    left_input = _create_score_transfer_plan(
        context,
        spec,
        rows,
        buffers,
        WriteSingleBank,
        ElementwiseMultiply.FIRST_OPERAND_BANK,
    )
    right_input = _create_score_transfer_plan(
        context,
        spec,
        rows,
        buffers,
        WriteSingleBank,
        ElementwiseMultiply.SECOND_OPERAND_BANK,
    )
    output = _create_score_transfer_plan(
        context,
        spec,
        rows,
        buffers,
        ReadSingleBank,
        ElementwiseMultiply.RESULT_BANK,
    )
    softmax_pass = SoftmaxPassPlan(
        left_input=left_input,
        right_input=right_input,
        output=output,
    )

    # The current partial softmax repeats the same placement twice. The
    # missing exponent and reduction steps are documented in the lowerer.
    return SoftmaxPlan(
        spec=spec,
        rows=rows,
        buffers=buffers,
        channels=CentChannelSet(channels=tuple(range(context.hardware.num_channels))),
        rows_per_score=left_input.rows_per_score,
        heads_per_bank=left_input.heads_per_bank,
        passes=(softmax_pass, softmax_pass),
    )


def _create_attention_output_plan(
    context: _LlamaCompileContext,
    spec: TransformerAttentionSpec,
    rows: TransformerAttentionRows,
    buffers: TransformerAttentionBuffers,
) -> AttentionOutputPlan:
    """Plan the score-by-value-cache matrix-vector multiplication.

    Args:
        context: Derived Llama and hardware dimensions.
        spec: Logical dimensions of the attention operation.
        rows: DRAM regions assigned to attention tensors.
        buffers: Shared Buffer regions assigned to attention tensors.

    Returns:
        Value-cache row geometry and channel placement for output lowering.
    """

    hardware = context.hardware
    return AttentionOutputPlan(
        spec=spec,
        rows=rows,
        buffers=buffers,
        channels=CentChannelSet(channels=tuple(range(hardware.num_channels))),
        rows_per_sequence=ceil_div(
            spec.sequence_length,
            hardware.dram_columns,
        ),
        rows_per_dimension=ceil_div(
            spec.max_sequence_length,
            hardware.dram_columns,
        ),
        heads_per_channel=ceil_div(
            spec.num_kv_heads,
            context.placement.channels_per_block,
        ),
        dimension_iterations=spec.head_size // hardware.num_banks,
        accumulation_register=0,
    )


def _create_attention_plan(
    context: _LlamaCompileContext,
    row_counts: _LlamaRowCounts,
    memory: _LlamaMemoryLayout,
    buffers: _LlamaBufferLayout,
) -> TransformerAttentionPlan:
    """Convert Llama dimensions and placements into attention operands.

    Args:
        context: Derived Llama and hardware dimensions.
        row_counts: DRAM space required by each tensor kind.
        memory: DRAM row assigned to every Llama tensor.
        buffers: Shared Buffer spans used by the Llama block.

    Returns:
        Complete physical plan for every transformer attention stage.
    """

    spec = TransformerAttentionSpec(
        hidden_size=context.model.hidden_size,
        num_attention_heads=context.model.num_attention_heads,
        num_kv_heads=context.model.num_kv_heads,
        head_size=context.head_size,
        kv_width=context.kv_width,
        repeat_count=context.repeat_count,
        sequence_length=context.step.sequence_length,
        max_sequence_length=context.step.max_sequence_length,
    )
    rows = TransformerAttentionRows(
        query=CentDramRowRange(start_row=memory.xq, row_count=row_counts.projection),
        key=CentDramRowRange(start_row=memory.xk, row_count=row_counts.projection),
        key_cache=CentDramRowRange(
            start_row=memory.cache_k, row_count=row_counts.cache_k
        ),
        scores=CentDramRowRange(start_row=memory.scores, row_count=row_counts.scores),
        value_cache=CentDramRowRange(
            start_row=memory.cache_v, row_count=row_counts.cache_v
        ),
    )
    attention_buffers = TransformerAttentionBuffers(
        query=buffers.query,
        key=buffers.key,
        value=buffers.value,
        # The original input remains live through the first residual addition.
        scores=buffers.scores,
        # The normalized projection input is also dead after QKV projection.
        output=buffers.normalized,
    )
    return TransformerAttentionPlan(
        rotary_embedding=_create_rotary_embedding_plan(
            context, spec, rows, attention_buffers
        ),
        kv_cache_update=_create_kv_cache_update_plan(
            context, spec, rows, attention_buffers
        ),
        score_gemv=_create_score_gemv_plan(context, spec, rows, attention_buffers),
        softmax=_create_softmax_plan(context, spec, rows, attention_buffers),
        output=_create_attention_output_plan(context, spec, rows, attention_buffers),
    )


def _create_silu_product_plan(
    context: _LlamaCompileContext,
    memory: _LlamaMemoryLayout,
    buffers: _LlamaBufferLayout,
) -> _SiluProductPlan:
    """Plan the bank partitions used by Llama's gated FFN product.

    Args:
        context: Derived model dimensions and target hardware.
        memory: DRAM rows assigned to Llama intermediate values.
        buffers: Shared Buffer spans assigned to Llama intermediate values.

    Returns:
        Explicit chunks and physical resources consumed by the lowerer.
    """

    intermediate_size = context.model.intermediate_size
    chunk_specs: tuple[tuple[int, int], ...]
    if intermediate_size <= context.activation_capacity:
        chunk_specs = ((memory.x1_sigmoid, intermediate_size),)
    else:
        # Context validation guarantees that the remaining values fit in the
        # second work area.
        chunk_specs = (
            (memory.x1, context.activation_capacity),
            (
                memory.x1_sigmoid,
                intermediate_size - context.activation_capacity,
            ),
        )

    available_partitions = context.total_banks // BANKS_PER_PU
    chunks: list[_SiluProductChunkPlan] = []
    for row, value_count in chunk_specs:
        # The baseline policy uses as many nonempty PU groups as possible. A
        # future optimizer can replace this plan without changing the lowerer.
        values_per_partition = ceil_div(value_count, available_partitions)
        partition_count = ceil_div(value_count, values_per_partition)
        chunks.append(
            _SiluProductChunkPlan(
                row=row,
                value_count=value_count,
                partition_count=partition_count,
                values_per_partition=values_per_partition,
            )
        )

    return _SiluProductPlan(
        chunks=tuple(chunks),
        workspace_buffer=buffers.ffn_product,
        channels=CentChannelSet(channels=tuple(range(context.hardware.num_channels))),
        channels_per_copy=context.placement.channels_per_block,
        result_banks=tuple(
            range(
                ElementwiseMultiply.RESULT_BANK,
                context.hardware.num_banks,
                BANKS_PER_PU,
            )
        ),
    )


def _create_rms_norm_plan(
    context: _LlamaCompileContext,
    *,
    input_rows: CentDramRowRange,
    work_rows: CentDramRowRange,
    weight_rows: CentDramRowRange,
    input_buffer: CentSharedBufferSpan,
    scale_buffer: CentSharedBufferSpan,
    partial_sum_buffer: CentSharedBufferSpan,
    output_buffer: CentSharedBufferSpan,
) -> CentRmsNormPlan:
    """Plan one hidden-width RMS-normalization operation.

    Args:
        context: Derived model dimensions and target hardware.
        input_rows: DRAM rows used by the sum-of-squares pass.
        work_rows: DRAM rows used by the scale multiplication.
        weight_rows: DRAM rows containing learned normalization weights.
        input_buffer: Slots containing the vector to normalize.
        scale_buffer: Slots containing the repeated normalization scale.
        partial_sum_buffer: Slot receiving partial sums of squares.
        output_buffer: Slots receiving the normalized vector.

    Returns:
        Explicit neighboring-pair and four-bank-group normalization layouts.
    """

    value_count = context.model.hidden_size
    burst_length = context.hardware.burst_length
    pu_layout = plan_partitioned_vector(
        value_count,
        context.total_banks // BANKS_PER_PU,
        burst_length,
    )
    # Both stages read the same Shared Buffer span. Use the PU partition count
    # for the neighboring-bank pass so partition padding cannot change where
    # later logical values appear between the two stages.
    pair_layout = CentPartitionedVectorLayout(
        value_count=value_count,
        partition_count=pu_layout.partition_count,
        burst_length=burst_length,
    )
    sum_of_squares = CentSumOfSquaresPlan(
        input_rows=input_rows,
        input_buffer=CentSharedBufferVector(
            span=input_buffer,
            layout=pair_layout,
        ),
        partial_sum_buffer=partial_sum_buffer,
        layout=pair_layout,
    )
    return CentRmsNormPlan(
        l2_norm=CentL2NormPlan(
            sum_of_squares=sum_of_squares,
            work_rows=work_rows,
            scale_buffer=CentSharedBufferVector(
                span=scale_buffer,
                layout=pu_layout,
            ),
            layout=pu_layout,
        ),
        weight_rows=weight_rows,
        output_buffer=CentSharedBufferVector(
            span=output_buffer,
            layout=pu_layout,
        ),
    )


def _create_self_attention_lowering_plan(
    context: _LlamaCompileContext,
    row_counts: _LlamaRowCounts,
    memory: _LlamaMemoryLayout,
    buffers: _LlamaBufferLayout,
    attention: TransformerAttentionPlan,
) -> _LlamaSelfAttentionPlan:
    """Build every generic operation plan used by self-attention.

    Args:
        context: Derived model dimensions and target hardware.
        row_counts: DRAM row counts for each Llama tensor shape.
        memory: Starting DRAM row assigned to every Llama tensor.
        buffers: Shared Buffer spans assigned to live Llama values.
        attention: Transformer-specific plan between QKV and WO.

    Returns:
        Immutable lowering plans in self-attention execution order.
    """

    hardware = context.hardware
    placement = context.placement
    hidden_size = context.model.hidden_size
    hidden_layout = plan_partitioned_vector(
        hidden_size,
        context.total_banks // BANKS_PER_PU,
        hardware.burst_length,
    )
    input_vector = CentSharedBufferVector(
        span=buffers.input,
        layout=hidden_layout,
    )
    normalized_vector = CentSharedBufferVector(
        span=buffers.normalized,
        layout=hidden_layout,
    )
    query_vector = CentSharedBufferVector(
        span=buffers.query,
        layout=hidden_layout,
    )
    normalization = _create_rms_norm_plan(
        context,
        input_rows=CentDramRowRange(
            start_row=memory.x,
            row_count=row_counts.x,
        ),
        work_rows=CentDramRowRange(
            start_row=memory.x_copy,
            row_count=row_counts.x,
        ),
        weight_rows=CentDramRowRange(
            start_row=memory.sa_norm,
            row_count=row_counts.x,
        ),
        input_buffer=buffers.input,
        scale_buffer=buffers.normalized,
        partial_sum_buffer=buffers.value,
        output_buffer=buffers.normalized,
    )
    query_projection = plan_weight_gemv(
        hardware,
        placement,
        weights=CentDramRowRange(
            start_row=memory.wq,
            row_count=row_counts.wq,
        ),
        input_buffer=normalized_vector,
        output_buffer=buffers.query_result,
        vector_size=hidden_size,
        output_size=hidden_size,
    )
    key_projection = plan_weight_gemv(
        hardware,
        placement,
        weights=CentDramRowRange(
            start_row=memory.wk,
            row_count=row_counts.wk,
        ),
        input_buffer=normalized_vector,
        output_buffer=buffers.key_result,
        vector_size=hidden_size,
        output_size=context.kv_width,
    )
    value_projection = plan_weight_gemv(
        hardware,
        placement,
        weights=CentDramRowRange(
            start_row=memory.wv,
            row_count=row_counts.wv,
        ),
        input_buffer=normalized_vector,
        output_buffer=buffers.value_result,
        vector_size=hidden_size,
        output_size=context.kv_width,
    )
    output_projection = plan_weight_gemv(
        hardware,
        placement,
        weights=CentDramRowRange(
            start_row=memory.wo,
            row_count=row_counts.wo,
        ),
        input_buffer=normalized_vector,
        output_buffer=buffers.query_result,
        vector_size=hidden_size,
        output_size=hidden_size,
    )
    return _LlamaSelfAttentionPlan(
        normalization=normalization,
        query_projection=query_projection,
        key_projection=key_projection,
        value_projection=value_projection,
        attention=attention,
        output_projection=output_projection,
        residual=CentAccumulatePlan(
            destination=query_vector,
            source=input_vector,
        ),
        store_residual=CentBankGroupVectorTransferPlan(
            dram=CentDramVector(
                rows=CentDramRowRange(
                    start_row=memory.sa,
                    row_count=row_counts.hidden_vector,
                ),
                layout=hidden_layout,
            ),
            buffer=query_vector,
        ),
    )


def _create_feed_forward_lowering_plan(
    context: _LlamaCompileContext,
    row_counts: _LlamaRowCounts,
    memory: _LlamaMemoryLayout,
    buffers: _LlamaBufferLayout,
    silu_product: _SiluProductPlan,
) -> _LlamaFeedForwardPlan:
    """Build every generic operation plan used by Llama's FFN.

    Args:
        context: Derived model dimensions and target hardware.
        row_counts: DRAM row counts for each Llama tensor shape.
        memory: Starting DRAM row assigned to every Llama tensor.
        buffers: Shared Buffer spans assigned to live Llama values.
        silu_product: Planned chunk layout for the gated product.

    Returns:
        Immutable lowering plans in feed-forward execution order.
    """

    hardware = context.hardware
    placement = context.placement
    hidden_size = context.model.hidden_size
    intermediate_size = context.model.intermediate_size
    hidden_layout = plan_partitioned_vector(
        hidden_size,
        context.total_banks // BANKS_PER_PU,
        hardware.burst_length,
    )
    normalized_vector = CentSharedBufferVector(
        span=buffers.normalized,
        layout=hidden_layout,
    )
    query_vector = CentSharedBufferVector(
        span=buffers.query,
        layout=hidden_layout,
    )
    intermediate_layout = plan_partitioned_vector(
        intermediate_size,
        context.total_banks // BANKS_PER_PU,
        hardware.burst_length,
    )
    ffn_product_vector = CentSharedBufferVector(
        span=buffers.ffn_product,
        layout=intermediate_layout,
    )
    normalization = _create_rms_norm_plan(
        context,
        input_rows=CentDramRowRange(
            start_row=memory.sa_copy,
            row_count=row_counts.hidden_vector,
        ),
        work_rows=CentDramRowRange(
            start_row=memory.sa_copy,
            row_count=row_counts.hidden_vector,
        ),
        weight_rows=CentDramRowRange(
            start_row=memory.ffn_norm,
            row_count=row_counts.hidden_vector,
        ),
        input_buffer=buffers.query,
        scale_buffer=buffers.normalized,
        partial_sum_buffer=buffers.value,
        output_buffer=buffers.normalized,
    )
    gate_projection = plan_weight_gemv(
        hardware,
        placement,
        weights=CentDramRowRange(
            start_row=memory.w1,
            row_count=row_counts.w1,
        ),
        input_buffer=normalized_vector,
        output_buffer=buffers.ffn_gate,
        activated_output_buffer=buffers.ffn_gate_sigmoid,
        vector_size=hidden_size,
        output_size=intermediate_size,
    )
    up_projection = plan_weight_gemv(
        hardware,
        placement,
        weights=CentDramRowRange(
            start_row=memory.w3,
            row_count=row_counts.w3,
        ),
        input_buffer=normalized_vector,
        output_buffer=buffers.ffn_up,
        vector_size=hidden_size,
        output_size=intermediate_size,
    )
    down_projection = plan_weight_gemv(
        hardware,
        placement,
        weights=CentDramRowRange(
            start_row=memory.w2,
            row_count=row_counts.w2,
        ),
        input_buffer=ffn_product_vector,
        output_buffer=buffers.query_result,
        vector_size=intermediate_size,
        output_size=hidden_size,
    )
    return _LlamaFeedForwardPlan(
        normalization=normalization,
        gate_projection=gate_projection,
        up_projection=up_projection,
        silu_product=silu_product,
        down_projection=down_projection,
        load_residual=CentBankGroupVectorTransferPlan(
            dram=CentDramVector(
                rows=CentDramRowRange(
                    start_row=memory.sa,
                    row_count=row_counts.hidden_vector,
                ),
                layout=hidden_layout,
            ),
            buffer=query_vector,
        ),
        residual=CentAccumulatePlan(
            destination=normalized_vector,
            source=query_vector,
        ),
    )


def _create_compile_plan(request: CompileRequest) -> _LlamaCompilePlan:
    """Build every checked size and placement used by Llama lowering.

    Args:
        request: Llama model, hardware, block placement, and decode sizes.

    Returns:
        Complete immutable plan consumed by the Llama block compiler.

    Raises:
        ValueError: If the model does not fit the selected target.
    """

    context = _create_context(request)
    row_counts = _row_counts(context)
    memory = _plan_memory(context)
    buffers = _plan_shared_buffer(context)
    attention = _create_attention_plan(
        context,
        row_counts,
        memory,
        buffers,
    )
    silu_product = _create_silu_product_plan(context, memory, buffers)
    self_attention = _create_self_attention_lowering_plan(
        context,
        row_counts,
        memory,
        buffers,
        attention,
    )
    feed_forward = _create_feed_forward_lowering_plan(
        context,
        row_counts,
        memory,
        buffers,
        silu_product,
    )
    return _LlamaCompilePlan(
        context=context,
        row_counts=row_counts,
        memory=memory,
        buffers=buffers,
        self_attention=self_attention,
        feed_forward=feed_forward,
    )
