"""Check Llama dimensions and assign DRAM rows to block tensors."""

from dataclasses import dataclass
from typing import cast

from ...cent import BANKS_PER_PU, CentBlockPlacementSpec, CentHardwareSpec
from ...cent.utils import ceil_div, require_positive
from ...request import CompileRequest, DecodeStepSpec
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
        ValueError: If the model does not fit the requested CENT layout.
    """

    # The public dispatcher calls this function only for Llama requests. The
    # cast records that fact for the type checker.
    model = cast(LlamaModelSpec, request.model)
    head_size = model.head_size

    # Flatten the assigned channels and their banks into one count. Later code
    # uses this count to divide tensor values among banks.
    total_banks = (
        request.placement.channels_per_block * request.hardware.num_banks
    )
    context = _LlamaCompileContext(
        model=model,
        hardware=request.hardware,
        placement=request.placement,
        step=request.step,
        head_size=head_size,
        # K and V contain only distinct KV heads. Several query heads may share
        # each one.
        kv_width=head_size * model.num_kv_heads,
        repeat_count=(
            model.num_attention_heads // model.num_kv_heads
        ),
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
            "Llama fused activation requires at least two accumulator slots "
            "per bank"
        )
    # The current layout does not record padding inside the last burst.
    if model.hidden_size % hardware.burst_length != 0:
        raise ValueError("hidden_size must be divisible by burst_length")
    # One attention head is split evenly among banks and stored wholly inside a
    # DRAM row. Score multiplication depends on both facts.
    if context.head_size < hardware.num_banks:
        raise ValueError("attention head size must be at least num_banks")
    if context.head_size % hardware.num_banks != 0:
        raise ValueError("attention head size must be divisible by num_banks")
    if context.head_size > hardware.dram_columns:
        raise ValueError("attention head size cannot exceed dram_columns")
    if hardware.dram_columns % context.head_size != 0:
        raise ValueError(
            "dram_columns must be divisible by attention head size"
        )
    if context.head_size % hardware.burst_length != 0:
        raise ValueError(
            "attention head size must be divisible by burst_length"
        )
    # The current SiLU path has only two work areas.
    if model.intermediate_size > context.activation_capacity * 2:
        raise ValueError(
            "intermediate_size exceeds two-pass activation capacity"
        )


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
    intermediate_outputs_per_bank = ceil_div(
        intermediate, context.total_banks
    )
    hidden_rows_per_output = ceil_div(hidden, columns)
    intermediate_rows_per_output = ceil_div(intermediate, columns)

    # Reserve caches for the maximum token count. Keys cycle tokens through
    # banks; scores and values put token positions across row columns.
    sequence_bank_groups = ceil_div(max_sequence, context.total_banks)
    sequence_rows = ceil_div(max_sequence, columns)
    query_head_slots_per_pu = ceil_div(
        model.num_attention_heads,
        context.placement.channels_per_block * BANKS_PER_PU,
    )
    kv_heads_per_channel = ceil_div(
        model.num_kv_heads,
        context.placement.channels_per_block,
    )
    head_slices_across_banks = ceil_div(context.head_size, banks)

    # TODO(layout): Size Q and K from their real bank-group layouts.
    #
    # Tests fit each workspace in one row. A full Q or K slice can still be
    # wider than one row even when a single head fits.

    # TODO(layout): Define when vectors are copied or split across channels.
    #
    # Input and FFN vectors currently repeat in each channel. Other work vectors
    # are split across all assigned banks. This difference is not yet justified.

    return _LlamaRowCounts(
        # Each channel holds a full input copy, split among its banks.
        x=ceil_div(hidden, banks * columns),
        wq=hidden_outputs_per_bank * hidden_rows_per_output,
        wk=kv_outputs_per_bank * hidden_rows_per_output,
        wv=kv_outputs_per_bank * hidden_rows_per_output,
        # This temporary one-row rule is tracked by the TODO above.
        projection=1,
        cache_k=sequence_bank_groups
        * ceil_div(context.kv_width, columns),
        scores=sequence_rows * query_head_slots_per_pu,
        cache_v=(
            sequence_rows * kv_heads_per_channel * head_slices_across_banks
        ),
        # Split block-wide work vectors across every assigned bank.
        hidden_vector=ceil_div(hidden, context.total_banks * columns),
        wo=hidden_outputs_per_bank * hidden_rows_per_output,
        w1=intermediate_outputs_per_bank * hidden_rows_per_output,
        w3=intermediate_outputs_per_bank * hidden_rows_per_output,
        intermediate_vector=ceil_div(
            intermediate, context.total_banks * columns
        ),
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
    # ``output``, ``sa``, ``x3``, ``ffn_vector``, and ``ffn`` have no users. Add
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
