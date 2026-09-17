"""Tests for Llama planning and transformer-operation lowering."""

import unittest
from collections import Counter
from dataclasses import replace

from vllm_cent import (
    CentBlockPlacementSpec,
    CentHardwareSpec,
    CentOpcode,
    CompileRequest,
    DecodeStepSpec,
    LlamaModelSpec,
)
from vllm_cent.cent import (
    CentChannelSet,
    CentMemoryAddress,
    CentProgramBuilder,
    CentSharedBufferAddress,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    MacAllBanks,
    MacOperandSource,
    ReadMac,
    ReadSingleBank,
    WriteAllBanks,
    WriteGlobalBuffer,
    WriteSingleBank,
)
from vllm_cent.lowering import (
    CentDramRowRange,
    CentPartitionedVectorLayout,
    CentSharedBufferSpan,
    lower_rms_norm,
    plan_partitioned_vector,
)
from vllm_cent.lowering.transformer import (
    TransformerAttentionPlan,
    lower_attention_output,
    lower_kv_cache_update,
    lower_rotary_embedding,
    lower_score_gemv,
    lower_softmax,
)
from vllm_cent.lowering.transformer.attention import _lower_score_transfer
from vllm_cent.models.llama import compile_llama_transformer_block
from vllm_cent.models.llama.compiler import (
    _lower_feed_forward,
    _lower_self_attention,
)
from vllm_cent.models.llama.planning import (
    _LlamaBufferLayout,
    _LlamaCompileContext,
    _LlamaMemoryLayout,
    _LlamaRowCounts,
    _RowAllocator,
    _create_attention_output_plan,
    _create_context,
    _create_attention_plan,
    _create_feed_forward_lowering_plan,
    _create_compile_plan,
    _create_kv_cache_update_plan,
    _create_rms_norm_plan,
    _create_rotary_embedding_plan,
    _create_score_gemv_plan,
    _create_score_transfer_plan,
    _create_self_attention_lowering_plan,
    _create_silu_product_plan,
    _create_softmax_plan,
    _plan_memory,
    _plan_shared_buffer,
    _row_counts,
    _validate_context,
)
from vllm_cent.models.llama.feed_forward import (
    _SiluProductChunkPlan,
    _SiluProductPlan,
    _lower_silu_product,
)


def make_request(
    *,
    hidden_size: int = 16,
    num_attention_heads: int = 1,
    num_kv_heads: int = 1,
    intermediate_size: int = 16,
    num_channels: int = 1,
    channels_per_block: int = 1,
    sequence_length: int = 1,
    max_sequence_length: int = 16,
) -> CompileRequest:
    """Create the small request used by lowering tests.

    Args:
        hidden_size: Residual width in BF16 values.
        num_attention_heads: Query-head count.
        num_kv_heads: Unique key/value-head count.
        intermediate_size: Feed-forward width in BF16 values.
        num_channels: Physical channel count.
        channels_per_block: Channels assigned to one block.
        sequence_length: Current context length.
        max_sequence_length: Reserved context length.

    Returns:
        Request using four banks, sixteen columns, and four-value bursts.
    """

    return CompileRequest(
        model=LlamaModelSpec(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            num_kv_heads=num_kv_heads,
            intermediate_size=intermediate_size,
        ),
        hardware=CentHardwareSpec(
            num_channels=num_channels,
            num_banks=4,
            dram_rows=1_000,
            dram_columns=16,
            burst_length=4,
            accumulator_slots_per_bank=4,
            sigmoid_activation_function_id=0,
        ),
        placement=CentBlockPlacementSpec(channels_per_block=channels_per_block),
        step=DecodeStepSpec(
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
        ),
    )


def make_state(
    **changes: int,
) -> tuple[_LlamaCompileContext, _LlamaMemoryLayout, CentProgramBuilder]:
    """Build a context, DRAM layout, and empty builder.

    Args:
        **changes: Integer overrides accepted by :func:`make_request`.

    Returns:
        Context, planned layout, and builder targeting that context.
    """

    context = _create_context(make_request(**changes))
    layout = _plan_memory(context)
    return (
        context,
        layout,
        CentProgramBuilder(context.hardware, context.placement),
    )


def make_attention_plan(
    context: _LlamaCompileContext,
    layout: _LlamaMemoryLayout,
) -> TransformerAttentionPlan:
    """Create every reusable attention stage plan.

    Args:
        context: Derived Llama and hardware dimensions.
        layout: DRAM rows assigned to the Llama block.

    Returns:
        Physical plans consumed by each attention lowerer.
    """

    return _create_attention_plan(
        context,
        _row_counts(context),
        layout,
        _plan_shared_buffer(context),
    )


class ContextAndLayoutTests(unittest.TestCase):
    """Test derived Llama dimensions and DRAM allocation."""

    def test_context_derives_grouped_query_values(self) -> None:
        """Calculate grouped-query and hardware dimensions."""

        request = make_request(
            hidden_size=64,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=64,
            num_channels=2,
            channels_per_block=2,
        )
        context = _create_context(request)

        # hidden_size / query_heads = 64 / 4 = 16 values per head. Two KV
        # heads give a 2 * 16 = 32-value KV width and each is shared by two
        # query heads. Two channels * four banks gives eight block-local banks;
        # two channels * one four-bank PU * 16 columns gives capacity 32.
        self.assertEqual(
            context,
            _LlamaCompileContext(
                model=request.model,
                hardware=request.hardware,
                placement=request.placement,
                step=request.step,
                head_size=16,
                kv_width=32,
                repeat_count=2,
                total_banks=8,
                activation_capacity=32,
            ),
        )

    def test_context_validator_rejects_invalid_derived_capacity(self) -> None:
        """Reject every unsupported relation between model and hardware."""

        valid = _create_context(make_request())
        cases = (
            (replace(valid, head_size=6), "divisible by num_banks"),
            (replace(valid, head_size=20), "cannot exceed"),
            (
                replace(
                    valid,
                    head_size=8,
                    hardware=replace(valid.hardware, dram_columns=20),
                ),
                "dram_columns",
            ),
            (
                replace(
                    valid,
                    head_size=12,
                    hardware=replace(
                        valid.hardware,
                        dram_columns=24,
                        burst_length=8,
                    ),
                ),
                "burst_length",
            ),
            (
                replace(
                    valid,
                    model=LlamaModelSpec(
                        hidden_size=16,
                        num_attention_heads=1,
                        num_kv_heads=1,
                        intermediate_size=33,
                    ),
                ),
                "two-pass",
            ),
        )
        for context, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    _validate_context(context)

    def test_row_counts_and_layout_have_stable_real_values(self) -> None:
        """Calculate tensor rows and place them next to each other."""

        context, layout, _ = make_state()
        rows = _row_counts(context)

        # A 16x16 weight matrix contains 256 values distributed over four
        # banks, or 64 values per bank. At 16 values per row that is four rows
        # for each full-width weight. Hidden-width vectors need one row. The
        # 16-token key and value caches each need four rows in this layout.
        self.assertEqual(
            rows,
            _LlamaRowCounts(
                x=1,
                wq=4,
                wk=4,
                wv=4,
                projection=1,
                cache_k=4,
                scores=1,
                cache_v=4,
                hidden_vector=1,
                wo=4,
                w1=4,
                w3=4,
                intermediate_vector=1,
                ffn_vector=1,
                w2=4,
            ),
        )
        # Adding every contiguous tensor allocation above consumes 51 rows, so
        # row 51 is the first free row. A target with only rows 0..49 cannot
        # contain the plan.
        self.assertEqual(
            layout,
            _LlamaMemoryLayout(
                x=0,
                x_copy=1,
                sa_norm=2,
                wq=3,
                wk=7,
                wv=11,
                xq=15,
                xk=16,
                cache_k=17,
                scores=21,
                cache_v=22,
                output=26,
                wo=27,
                sa=31,
                sa_copy=32,
                ffn_norm=33,
                w1=34,
                w3=38,
                x1=42,
                x3=43,
                x1_sigmoid=44,
                ffn_vector=45,
                w2=46,
                ffn=50,
                end=51,
            ),
        )
        with self.assertRaisesRegex(ValueError, "requires"):
            _plan_memory(
                replace(
                    context,
                    hardware=replace(context.hardware, dram_rows=50),
                )
            )

    def test_row_counts_reserve_multirow_rotary_partitions(self) -> None:
        """Size Q and K workspaces from their largest PU partition."""

        context = _create_context(
            make_request(
                hidden_size=32,
                num_attention_heads=2,
                num_kv_heads=1,
                intermediate_size=32,
            )
        )

        # One four-bank PU receives all 32 query values. A DRAM row holds 16,
        # so both Q and K workspaces reserve two rows instead of assuming one.
        self.assertEqual(_row_counts(context).projection, 2)

    def test_row_allocator_returns_contiguous_starts(self) -> None:
        """Return each allocation's start and advance the cursor."""

        allocator = _RowAllocator()

        # The first three-row tensor starts at zero. The following two-row
        # tensor begins at row three, leaving row five as the next free row.
        self.assertEqual(allocator.reserve(3), 0)
        self.assertEqual(allocator.reserve(2), 3)
        self.assertEqual(allocator.cursor, 5)
        with self.assertRaises(ValueError):
            allocator.reserve(0)

    def test_grouped_buffer_slots_includes_partition_padding(self) -> None:
        """Round each PU partition to its own Shared Buffer slot."""

        # Ten values split among four groups become 3, 3, 3, and 1 values.
        # With four values per slot, every group still occupies one slot.
        self.assertEqual(
            plan_partitioned_vector(10, 4, 4),
            CentPartitionedVectorLayout(
                value_count=10,
                partition_count=4,
                burst_length=4,
            ),
        )
        for arguments in ((0, 4, 4), (10, 0, 4), (10, 4, 0)):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    plan_partitioned_vector(*arguments)

    def test_shared_buffer_plan_separates_q_k_and_v(self) -> None:
        """Give the attention projections nonoverlapping staging spans."""

        context, _, _ = make_state()
        layout = _plan_shared_buffer(context)

        # Each projection first produces four accumulator-result slots. Those
        # occupy 8..19. The separately packed Q, K, and V vectors then occupy
        # slots 20..31 so the two physical layouts are not conflated. Slot 32
        # stages scores without overwriting the input needed by the residual.
        self.assertEqual(
            layout,
            _LlamaBufferLayout(
                input=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=0),
                    slot_count=4,
                ),
                normalized=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=4),
                    slot_count=4,
                ),
                query_result=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=8),
                    slot_count=4,
                ),
                key_result=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=12),
                    slot_count=4,
                ),
                value_result=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=16),
                    slot_count=4,
                ),
                query=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=20),
                    slot_count=4,
                ),
                key=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=24),
                    slot_count=4,
                ),
                value=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=28),
                    slot_count=4,
                ),
                scores=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=32),
                    slot_count=1,
                ),
                ffn_gate=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=0),
                    slot_count=4,
                ),
                ffn_gate_sigmoid=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=4),
                    slot_count=4,
                ),
                ffn_up=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=8),
                    slot_count=4,
                ),
                ffn_product=CentSharedBufferSpan(
                    start=CentSharedBufferAddress(slot=0),
                    slot_count=4,
                ),
                end_slot=33,
            ),
        )

    def test_shared_buffer_plan_rejects_insufficient_capacity(self) -> None:
        """Reject a target that cannot hold live attention values."""

        context, _, _ = make_state()
        with self.assertRaisesRegex(ValueError, "Shared Buffer slots"):
            _plan_shared_buffer(
                replace(
                    context,
                    hardware=replace(context.hardware, shared_buffer_slots=32),
                )
            )


class NormalizationAndFeedForwardTests(unittest.TestCase):
    """Test normalization and Llama feed-forward lowering."""

    def test_rms_norm_and_silu_emit_paper_copy_shapes(self) -> None:
        """Emit the paper's bank-copy operand shapes."""

        context, layout, norm_builder = make_state()
        buffers = _plan_shared_buffer(context)
        compile_plan = _create_compile_plan(make_request())
        lower_rms_norm(
            norm_builder,
            compile_plan.self_attention.normalization,
        )
        norm_counts = Counter(i.opcode for i in norm_builder.instructions)
        # RMSNorm copies its bank result into the Global Buffer once, then
        # copies that scalar scale back to banks once before the final multiply.
        self.assertEqual(norm_counts[CentOpcode.COPY_BANK_TO_GLOBAL_BUFFER], 1)
        self.assertEqual(norm_counts[CentOpcode.COPY_GLOBAL_BUFFER_TO_BANK], 1)

        silu_builder = CentProgramBuilder(context.hardware, context.placement)
        _lower_silu_product(
            silu_builder,
            compile_plan.feed_forward.silu_product,
        )
        silu_counts = Counter(i.opcode for i in silu_builder.instructions)
        # The tiny FFN fits one activation chunk. It performs one multiply to
        # form SiLU, one copy through the Global Buffer, and a second multiply
        # combining the activated gate with the W3 projection.
        self.assertEqual(silu_counts[CentOpcode.COPY_BANK_TO_GLOBAL_BUFFER], 1)
        self.assertEqual(silu_counts[CentOpcode.ELEMENTWISE_MULTIPLY], 2)
        copies = [
            instruction
            for instruction in silu_builder.instructions
            if isinstance(
                instruction,
                (CopyBankToGlobalBuffer, CopyGlobalBufferToBank),
            )
        ]
        # The first multiply leaves SiLU in bank two. The copy moves it to bank
        # one; W3 is then staged in bank zero for the second multiply.
        self.assertEqual(
            copies,
            [
                CopyBankToGlobalBuffer(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=4,
                    bank=2,
                    row=layout.x1_sigmoid,
                    column=0,
                ),
                CopyGlobalBufferToBank(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=4,
                    bank=1,
                    row=layout.x1_sigmoid,
                    column=0,
                ),
            ],
        )
        self.assertEqual(
            silu_builder.instructions[5],
            WriteSingleBank(
                address=CentMemoryAddress(
                    channel=0,
                    bank=0,
                    row=layout.x1_sigmoid,
                    column=0,
                ),
                operation_size=4,
                source=buffers.ffn_product.start,
            ),
        )

    def test_silu_product_rejects_invalid_explicit_plans(self) -> None:
        """Validate chunk coverage, target geometry, and workspace capacity."""

        context, _, _ = make_state()
        valid = _create_compile_plan(make_request()).feed_forward.silu_product
        chunk = valid.chunks[0]

        with self.assertRaisesRegex(ValueError, "cover every value"):
            replace(chunk, value_count=17)
        with self.assertRaisesRegex(ValueError, "chunks"):
            replace(valid, chunks=())
        with self.assertRaisesRegex(ValueError, "result_banks"):
            replace(valid, result_banks=())

        cases = (
            (replace(valid, channels_per_copy=2), "must divide"),
            (
                replace(
                    valid,
                    chunks=(replace(chunk, partition_count=2),),
                ),
                "partition_count",
            ),
            (replace(valid, result_banks=(6,)), "outside"),
            (
                replace(
                    valid,
                    workspace_buffer=replace(
                        valid.workspace_buffer,
                        slot_count=3,
                    ),
                ),
                "workspace_buffer",
            ),
        )
        for invalid_plan, message in cases:
            with self.subTest(message=message):
                builder = CentProgramBuilder(
                    context.hardware,
                    context.placement,
                )
                with self.assertRaisesRegex(ValueError, message):
                    _lower_silu_product(builder, invalid_plan)

        with self.assertRaisesRegex(ValueError, "EW_MUL result"):
            replace(valid, result_banks=(3,))
        with self.assertRaisesRegex(ValueError, "duplicates"):
            replace(valid, result_banks=(2, 2))
        with self.assertRaisesRegex(ValueError, "cannot cross"):
            _lower_silu_product(
                CentProgramBuilder(context.hardware, context.placement),
                replace(
                    valid,
                    chunks=(
                        _SiluProductChunkPlan(
                            row=chunk.row,
                            value_count=17,
                            partition_count=1,
                            values_per_partition=17,
                        ),
                    ),
                ),
            )

        # Private plan types validate their own primitive fields as well.
        with self.assertRaises(ValueError):
            _SiluProductChunkPlan(
                row=-1,
                value_count=1,
                partition_count=1,
                values_per_partition=1,
            )
        with self.assertRaises(ValueError):
            _SiluProductPlan(
                chunks=(chunk,),
                workspace_buffer=valid.workspace_buffer,
                channels=valid.channels,
                channels_per_copy=1,
                result_banks=(-1,),
            )

    def test_silu_product_obeys_a_single_partition_plan(self) -> None:
        """Keep a small FFN chunk on one PU when two PUs are available."""

        request = make_request(
            num_channels=2,
            channels_per_block=2,
        )
        compile_plan = _create_compile_plan(request)
        baseline = compile_plan.feed_forward.silu_product
        chunk = baseline.chunks[0]
        plan = replace(
            baseline,
            chunks=(
                replace(
                    chunk,
                    partition_count=1,
                    values_per_partition=16,
                ),
            ),
            # The alternative layout uses only channel zero. The baseline plan
            # uses both channels because each contains one four-bank PU.
            channels=CentChannelSet(channels=(0,)),
        )
        builder = CentProgramBuilder(request.hardware, request.placement)

        _lower_silu_product(builder, plan)

        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=ElementwiseMultiply.FIRST_OPERAND_BANK,
                        row=chunk.row,
                        column=0,
                    ),
                    operation_size=4,
                    source=baseline.workspace_buffer.start,
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=ElementwiseMultiply.SECOND_OPERAND_BANK,
                        row=chunk.row,
                        column=0,
                    ),
                    operation_size=4,
                    source=baseline.workspace_buffer.start,
                ),
                ElementwiseMultiply(
                    operation_size=4,
                    channels=CentChannelSet(channels=(0,)),
                    row=chunk.row,
                    column=0,
                ),
                CopyBankToGlobalBuffer(
                    operation_size=4,
                    channels=CentChannelSet(channels=(0,)),
                    bank=ElementwiseMultiply.RESULT_BANK,
                    row=chunk.row,
                    column=0,
                ),
                CopyGlobalBufferToBank(
                    operation_size=4,
                    channels=CentChannelSet(channels=(0,)),
                    bank=ElementwiseMultiply.SECOND_OPERAND_BANK,
                    row=chunk.row,
                    column=0,
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=ElementwiseMultiply.FIRST_OPERAND_BANK,
                        row=chunk.row,
                        column=0,
                    ),
                    operation_size=4,
                    source=baseline.workspace_buffer.start,
                ),
                ElementwiseMultiply(
                    operation_size=4,
                    channels=CentChannelSet(channels=(0,)),
                    row=chunk.row,
                    column=0,
                ),
                ReadSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=ElementwiseMultiply.RESULT_BANK,
                        row=chunk.row,
                        column=0,
                    ),
                    operation_size=4,
                    destination=baseline.workspace_buffer.start,
                ),
            ],
        )


class AttentionLoweringTests(unittest.TestCase):
    """Test each attention lowering stage."""

    def test_stage_planners_compose_the_aggregate_attention_plan(self) -> None:
        """Build every attention stage independently from shared bindings."""

        context, layout, _ = make_state(
            num_channels=2,
            channels_per_block=1,
            sequence_length=5,
        )
        plan = make_attention_plan(context, layout)
        spec = plan.rotary_embedding.spec
        rows = plan.rotary_embedding.rows
        buffers = plan.rotary_embedding.buffers

        # Each helper owns one placement decision. These comparisons ensure the
        # aggregate planner only composes those stage plans and does not make a
        # second, hidden choice about the physical layout.
        self.assertEqual(
            _create_rotary_embedding_plan(context, spec, rows, buffers),
            plan.rotary_embedding,
        )
        self.assertEqual(
            _create_kv_cache_update_plan(context, spec, rows, buffers),
            plan.kv_cache_update,
        )
        self.assertEqual(
            _create_score_gemv_plan(context, spec, rows, buffers),
            plan.score_gemv,
        )
        self.assertEqual(
            _create_softmax_plan(context, spec, rows, buffers),
            plan.softmax,
        )
        self.assertEqual(
            _create_attention_output_plan(context, spec, rows, buffers),
            plan.output,
        )

        # The first softmax input occupies bank position zero in every
        # four-bank PU group. Testing the transfer helper directly keeps this
        # bank-role calculation visible.
        self.assertEqual(
            _create_score_transfer_plan(
                context,
                spec,
                rows,
                buffers,
                WriteSingleBank,
                ElementwiseMultiply.FIRST_OPERAND_BANK,
            ),
            plan.softmax.passes[0].left_input,
        )

        with self.assertRaisesRegex(ValueError, "same spec"):
            TransformerAttentionPlan(
                rotary_embedding=plan.rotary_embedding,
                kv_cache_update=plan.kv_cache_update,
                score_gemv=replace(
                    plan.score_gemv,
                    spec=replace(
                        plan.score_gemv.spec,
                        sequence_length=4,
                    ),
                ),
                softmax=plan.softmax,
                output=plan.output,
            )
        with self.assertRaisesRegex(ValueError, "same rows"):
            TransformerAttentionPlan(
                rotary_embedding=plan.rotary_embedding,
                kv_cache_update=plan.kv_cache_update,
                score_gemv=replace(
                    plan.score_gemv,
                    rows=replace(
                        plan.score_gemv.rows,
                        key=CentDramRowRange(start_row=999, row_count=1),
                    ),
                ),
                softmax=plan.softmax,
                output=plan.output,
            )
        with self.assertRaisesRegex(ValueError, "same buffers"):
            TransformerAttentionPlan(
                rotary_embedding=plan.rotary_embedding,
                kv_cache_update=plan.kv_cache_update,
                score_gemv=replace(
                    plan.score_gemv,
                    buffers=replace(
                        plan.score_gemv.buffers,
                        query=CentSharedBufferSpan(
                            start=CentSharedBufferAddress(slot=60),
                            slot_count=1,
                        ),
                    ),
                ),
                softmax=plan.softmax,
                output=plan.output,
            )

        softmax_pass = plan.softmax.passes[0]
        transfer = softmax_pass.left_input
        softmax_mismatches = (
            (
                replace(
                    transfer,
                    rows=replace(
                        transfer.rows,
                        scores=CentDramRowRange(start_row=999, row_count=1),
                    ),
                ),
                "same rows",
            ),
            (
                replace(
                    transfer,
                    buffers=replace(
                        transfer.buffers,
                        scores=CentSharedBufferSpan(
                            start=CentSharedBufferAddress(slot=60),
                            slot_count=1,
                        ),
                    ),
                ),
                "same buffers",
            ),
            (replace(transfer, rows_per_score=2), "same rows_per_score"),
            (replace(transfer, heads_per_bank=2), "same heads_per_bank"),
        )
        for changed_transfer, message in softmax_mismatches:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    replace(
                        plan.softmax,
                        passes=(
                            replace(
                                softmax_pass,
                                left_input=changed_transfer,
                            ),
                        ),
                    )

    def test_rotary_embedding_uses_explicit_single_bank_transfers(self) -> None:
        """Use WR_SBK and EW_MUL for rotary data flow."""

        context, layout, builder = make_state()
        plan = make_attention_plan(context, layout)
        lower_rotary_embedding(builder, plan.rotary_embedding)
        counts = Counter(i.opcode for i in builder.instructions)

        # Query and key are written into operand bank group 1. Their computed
        # bank-group-2 results are then read back to the same named buffers.
        self.assertEqual(counts[CentOpcode.WRITE_SINGLE_BANK], 2)
        self.assertEqual(counts[CentOpcode.READ_SINGLE_BANK], 2)
        self.assertEqual(counts[CentOpcode.ELEMENTWISE_MULTIPLY], 2)

    def test_rotary_embedding_obeys_a_single_partition_plan(self) -> None:
        """Keep query and key on one PU when the block owns two PUs."""

        context, layout, builder = make_state(
            num_channels=2,
            channels_per_block=2,
        )
        baseline = make_attention_plan(context, layout).rotary_embedding
        plan = replace(
            baseline,
            channels=CentChannelSet(channels=(0,)),
            query_values_per_partition=16,
            query_partition_count=1,
            key_values_per_partition=16,
            key_partition_count=1,
        )

        lower_rotary_embedding(builder, plan)

        # The baseline uses one PU in each channel. This exact instruction list
        # proves the supplied plan keeps both vectors on channel zero instead.
        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=ElementwiseMultiply.SECOND_OPERAND_BANK,
                        row=plan.rows.query.start_row,
                        column=0,
                    ),
                    operation_size=4,
                    source=plan.buffers.query.start,
                ),
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=ElementwiseMultiply.SECOND_OPERAND_BANK,
                        row=plan.rows.key.start_row,
                        column=0,
                    ),
                    operation_size=4,
                    source=plan.buffers.key.start,
                ),
                ElementwiseMultiply(
                    operation_size=4,
                    channels=CentChannelSet(channels=(0,)),
                    row=plan.rows.query.start_row,
                    column=0,
                ),
                ElementwiseMultiply(
                    operation_size=4,
                    channels=CentChannelSet(channels=(0,)),
                    row=plan.rows.key.start_row,
                    column=0,
                ),
                ReadSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=ElementwiseMultiply.RESULT_BANK,
                        row=plan.rows.query.start_row,
                        column=0,
                    ),
                    operation_size=4,
                    destination=plan.buffers.query.start,
                ),
                ReadSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=ElementwiseMultiply.RESULT_BANK,
                        row=plan.rows.key.start_row,
                        column=0,
                    ),
                    operation_size=4,
                    destination=plan.buffers.key.start,
                ),
            ],
        )

    def test_rotary_embedding_splits_multirow_query_arithmetic(self) -> None:
        """Emit one row-local multiply for each query and key row."""

        context, layout, builder = make_state(
            hidden_size=32,
            num_attention_heads=2,
            num_kv_heads=1,
            intermediate_size=32,
        )
        plan = make_attention_plan(context, layout)
        lower_rotary_embedding(builder, plan.rotary_embedding)

        self.assertEqual(
            [
                instruction
                for instruction in builder.instructions
                if isinstance(instruction, ElementwiseMultiply)
            ],
            [
                ElementwiseMultiply(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=4,
                    row=38,
                    column=0,
                ),
                ElementwiseMultiply(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=4,
                    row=39,
                    column=0,
                ),
                ElementwiseMultiply(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=4,
                    row=40,
                    column=0,
                ),
            ],
        )

    def test_rotary_embedding_rejects_an_undersized_query_span(self) -> None:
        """Reject a query buffer that cannot hold all PU partitions."""

        context, layout, builder = make_state()
        plan = make_attention_plan(context, layout)
        buffers = plan.rotary_embedding.buffers
        small_query = CentSharedBufferSpan(
            start=buffers.query.start,
            slot_count=buffers.query.slot_count - 1,
        )

        with self.assertRaisesRegex(ValueError, "query needs"):
            lower_rotary_embedding(
                builder,
                replace(
                    plan.rotary_embedding,
                    buffers=replace(buffers, query=small_query),
                ),
            )

    def test_rotary_embedding_rejects_an_undersized_query_row_range(
        self,
    ) -> None:
        """Reject a row range shorter than one query partition."""

        context, layout, builder = make_state(
            hidden_size=32,
            num_attention_heads=2,
            num_kv_heads=1,
            intermediate_size=32,
        )
        plan = make_attention_plan(context, layout)
        short_rows = replace(
            plan.rotary_embedding.rows,
            query=CentDramRowRange(
                start_row=plan.rotary_embedding.rows.query.start_row,
                row_count=1,
            ),
        )

        # This one-channel target has one PU group, so all 32 query values need
        # two sixteen-value DRAM rows. Replacing its planned two-row range with
        # one row must be rejected before any instructions are emitted.
        with self.assertRaisesRegex(ValueError, "query rows needs 2"):
            lower_rotary_embedding(
                builder,
                replace(plan.rotary_embedding, rows=short_rows),
            )
        self.assertEqual(builder.instructions, [])

    def test_kv_cache_update_writes_column_and_channel(self) -> None:
        """Put the new cache value in the expected column and channel."""

        context, layout, builder = make_state(sequence_length=5)
        plan = make_attention_plan(context, layout)
        lower_kv_cache_update(builder, plan.kv_cache_update)

        # Head size 16 spread across four banks gives four dimension writes.
        # sequence_length 5 means token index 4, which is column 4 of row zero;
        # the one-channel placement selects physical channel zero.
        self.assertEqual(
            builder.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=0,
                        row=18,
                        column=0,
                    ),
                    operation_size=4,
                    source=CentSharedBufferAddress(slot=24),
                ),
                WriteAllBanks(
                    channel=0,
                    row=22,
                    column=4,
                    source=CentSharedBufferAddress(slot=28),
                    accumulation_register=0,
                ),
                WriteAllBanks(
                    channel=0,
                    row=23,
                    column=4,
                    source=CentSharedBufferAddress(slot=29),
                    accumulation_register=0,
                ),
                WriteAllBanks(
                    channel=0,
                    row=24,
                    column=4,
                    source=CentSharedBufferAddress(slot=30),
                    accumulation_register=0,
                ),
                WriteAllBanks(
                    channel=0,
                    row=25,
                    column=4,
                    source=CentSharedBufferAddress(slot=31),
                    accumulation_register=0,
                ),
            ],
        )

    def test_attention_stages_reject_undersized_input_spans(self) -> None:
        """Check the buffer capacity required by each attention stage."""

        context, layout, _ = make_state(sequence_length=16)
        plan = make_attention_plan(context, layout)
        one_slot = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=0),
            slot_count=1,
        )
        undersized_score_buffers = replace(
            plan.softmax.buffers,
            scores=one_slot,
        )
        undersized_score_passes = tuple(
            replace(
                pass_plan,
                left_input=replace(
                    pass_plan.left_input,
                    buffers=undersized_score_buffers,
                ),
                right_input=replace(
                    pass_plan.right_input,
                    buffers=undersized_score_buffers,
                ),
                output=replace(
                    pass_plan.output,
                    buffers=undersized_score_buffers,
                ),
            )
            for pass_plan in plan.softmax.passes
        )
        cases = (
            (
                lower_kv_cache_update,
                replace(
                    plan.kv_cache_update,
                    buffers=replace(
                        plan.kv_cache_update.buffers,
                        key=one_slot,
                    ),
                ),
                "key needs",
            ),
            (
                lower_score_gemv,
                replace(
                    plan.score_gemv,
                    buffers=replace(
                        plan.score_gemv.buffers,
                        query=one_slot,
                    ),
                ),
                "query needs",
            ),
            (
                lower_softmax,
                replace(
                    plan.softmax,
                    buffers=undersized_score_buffers,
                    passes=undersized_score_passes,
                ),
                "scores needs",
            ),
            (
                lower_attention_output,
                replace(
                    plan.output,
                    buffers=replace(
                        plan.output.buffers,
                        scores=one_slot,
                    ),
                ),
                "scores needs",
            ),
        )
        for lowerer, stage_plan, message in cases:
            with self.subTest(lowerer=lowerer.__name__):
                builder = CentProgramBuilder(
                    context.hardware,
                    context.placement,
                )
                with self.assertRaisesRegex(ValueError, message):
                    lowerer(builder, stage_plan)

    def test_attention_lowerers_reject_inconsistent_physical_plans(self) -> None:
        """Validate every explicit geometry before emitting instructions."""

        context, layout, _ = make_state()
        plan = make_attention_plan(context, layout)
        rotary_cases = (
            (
                replace(plan.rotary_embedding, query_partition_count=2),
                "query partitions",
            ),
            (
                replace(plan.rotary_embedding, key_partition_count=2),
                "key partitions",
            ),
            (
                replace(plan.rotary_embedding, query_values_per_partition=15),
                "query partitions must cover",
            ),
            (
                replace(plan.rotary_embedding, key_values_per_partition=15),
                "key partitions must cover",
            ),
        )
        for stage_plan, message in rotary_cases:
            with self.subTest(stage="rotary", message=message):
                builder = CentProgramBuilder(
                    context.hardware,
                    context.placement,
                )
                with self.assertRaisesRegex(ValueError, message):
                    lower_rotary_embedding(builder, stage_plan)

        kv_cases = (
            (
                replace(plan.kv_cache_update, sequence_index=1),
                "sequence_index",
            ),
            (
                replace(plan.kv_cache_update, key_logical_bank=4),
                "key_logical_bank",
            ),
            (
                replace(plan.kv_cache_update, key_row_group=-1),
                "key_row_group",
            ),
            (
                replace(plan.kv_cache_update, value_sequence_row=-1),
                "value_sequence_row",
            ),
            (
                replace(
                    plan.kv_cache_update,
                    key_replica_channel_offsets=(),
                ),
                "key_replica_channel_offsets",
            ),
            (
                replace(
                    plan.kv_cache_update,
                    key_replica_channel_offsets=(0, 0),
                ),
                "duplicates",
            ),
            (
                replace(plan.kv_cache_update, value_channels=()),
                "value_channels",
            ),
            (
                replace(
                    plan.kv_cache_update,
                    spec=replace(
                        plan.kv_cache_update.spec,
                        max_sequence_length=32,
                    ),
                ),
                "value_rows_per_dimension",
            ),
            (
                replace(plan.kv_cache_update, value_dimension_iterations=3),
                "value dimension iterations",
            ),
        )
        for stage_plan, message in kv_cases:
            with self.subTest(stage="kv", message=message):
                builder = CentProgramBuilder(
                    context.hardware,
                    context.placement,
                )
                with self.assertRaisesRegex(ValueError, message):
                    lower_kv_cache_update(builder, stage_plan)

        wide_context, wide_layout, _ = make_state(
            hidden_size=32,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=16,
        )
        wide_plan = make_attention_plan(wide_context, wide_layout)
        for stage_plan, message in (
            (
                replace(wide_plan.kv_cache_update, key_rows_per_token=1),
                "key_rows_per_token",
            ),
            (
                replace(wide_plan.kv_cache_update, value_heads_per_channel=1),
                "value channel slots",
            ),
            (
                replace(wide_plan.score_gemv, rows_per_key=1),
                "rows_per_key",
            ),
        ):
            with self.subTest(stage="wide attention", message=message):
                builder = CentProgramBuilder(
                    wide_context.hardware,
                    wide_context.placement,
                )
                lowerer = (
                    lower_score_gemv
                    if "rows_per_key" in message
                    else lower_kv_cache_update
                )
                with self.assertRaisesRegex(ValueError, message):
                    lowerer(builder, stage_plan)

        score_cases = (
            (
                replace(plan.score_gemv, sequence_channels=()),
                "sequence_channels",
            ),
            (
                replace(plan.score_gemv, operation_size=3),
                "operation_size",
            ),
            (
                replace(
                    plan.score_gemv,
                    spec=replace(
                        plan.score_gemv.spec,
                        sequence_length=5,
                    ),
                ),
                "sequence channel groups",
            ),
        )
        for stage_plan, message in score_cases:
            with self.subTest(stage="score", message=message):
                builder = CentProgramBuilder(
                    context.hardware,
                    context.placement,
                )
                with self.assertRaisesRegex(ValueError, message):
                    lower_score_gemv(builder, stage_plan)

        transfer = plan.softmax.passes[0].left_input
        transfer_cases = (
            (
                replace(transfer, instruction_type=WriteAllBanks),
                "instruction_type",
            ),
            (replace(transfer, replica_channel_offsets=()), "replica"),
            (replace(transfer, replica_channel_offsets=(0, 0)), "duplicates"),
            (replace(transfer, logical_banks=()), "logical_banks"),
            (replace(transfer, logical_banks=(4,)), "outside"),
            (replace(transfer, logical_banks=(1,)), "bank_group"),
            (
                replace(
                    transfer,
                    heads_per_bank=1,
                    spec=wide_plan.softmax.spec,
                ),
                "do not cover",
            ),
            (
                replace(
                    transfer,
                    spec=replace(
                        transfer.spec,
                        sequence_length=17,
                        max_sequence_length=17,
                    ),
                ),
                "rows_per_score",
            ),
        )
        for stage_plan, message in transfer_cases:
            with self.subTest(stage="transfer", message=message):
                builder = CentProgramBuilder(
                    context.hardware,
                    context.placement,
                )
                with self.assertRaisesRegex(ValueError, message):
                    _lower_score_transfer(builder, stage_plan)

        with self.assertRaisesRegex(ValueError, "passes"):
            lower_softmax(
                CentProgramBuilder(context.hardware, context.placement),
                replace(plan.softmax, passes=()),
            )
        with self.assertRaisesRegex(ValueError, "same spec"):
            replace(
                plan.softmax,
                spec=replace(
                    plan.softmax.spec,
                    sequence_length=17,
                    max_sequence_length=17,
                ),
            )
        output_cases = (
            (
                replace(plan.output, rows_per_sequence=0),
                "rows_per_sequence",
            ),
            (
                replace(plan.output, rows_per_dimension=0),
                "rows_per_dimension",
            ),
            (
                replace(
                    plan.output,
                    spec=replace(
                        plan.output.spec,
                        sequence_length=17,
                        max_sequence_length=17,
                    ),
                ),
                "rows_per_sequence",
            ),
            (
                replace(
                    plan.output,
                    spec=replace(
                        plan.output.spec,
                        max_sequence_length=32,
                    ),
                ),
                "rows_per_dimension",
            ),
        )
        for stage_plan, message in output_cases:
            with self.subTest(stage="output", message=message):
                builder = CentProgramBuilder(
                    context.hardware,
                    context.placement,
                )
                with self.assertRaisesRegex(ValueError, message):
                    lower_attention_output(builder, stage_plan)

    def test_score_gemv_uses_head_column_offsets(self) -> None:
        """Advance the MAC column between heads packed in one row."""

        context, layout, builder = make_state(
            hidden_size=32,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=16,
        )
        plan = make_attention_plan(context, layout)
        lower_score_gemv(builder, plan.score_gemv)
        macs = [
            instruction
            for instruction in builder.instructions
            if isinstance(instruction, MacAllBanks)
        ]

        # Four heads over hidden width 32 produce eight-value heads. Two heads
        # therefore fit in each 16-column row, starting at columns zero and
        # eight. The 32-value KV width occupies two rows, repeating the pair.
        self.assertEqual(
            macs,
            [
                MacAllBanks(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=2,
                    row=58,
                    column=0,
                    accumulation_register=0,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                ),
                MacAllBanks(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=2,
                    row=58,
                    column=8,
                    accumulation_register=0,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                ),
                MacAllBanks(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=2,
                    row=59,
                    column=0,
                    accumulation_register=0,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                ),
                MacAllBanks(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=2,
                    row=59,
                    column=8,
                    accumulation_register=0,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                ),
            ],
        )

    def test_score_gemv_skips_padded_kv_heads(self) -> None:
        """Do not emit dot products for unused head slots in the final row."""

        context, layout, builder = make_state(
            hidden_size=16,
            num_attention_heads=2,
            num_kv_heads=1,
            intermediate_size=16,
        )
        plan = make_attention_plan(context, layout)
        lower_score_gemv(builder, plan.score_gemv)

        # One real KV head is reused by two query heads. Although two eight-value
        # heads fit in the row, the second packed position is only padding.
        self.assertEqual(
            [
                instruction
                for instruction in builder.instructions
                if isinstance(instruction, MacAllBanks)
            ],
            [
                MacAllBanks(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=2,
                    row=13,
                    column=0,
                    accumulation_register=0,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                ),
                MacAllBanks(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=2,
                    row=13,
                    column=0,
                    accumulation_register=0,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                ),
            ],
        )

    def test_score_gemv_repeats_partial_channel_prefixes_per_block(self) -> None:
        """Select a partial final token group inside every block copy."""

        context, layout, _ = make_state(
            hidden_size=16,
            num_attention_heads=2,
            num_kv_heads=1,
            num_channels=6,
            channels_per_block=3,
            sequence_length=17,
            max_sequence_length=17,
        )
        plan = make_attention_plan(context, layout)

        # The first twelve-token group uses all three channels in both copies.
        # Five remaining tokens need two channels in each three-channel copy.
        self.assertEqual(
            plan.score_gemv.sequence_channels,
            (
                CentChannelSet(channels=(0, 1, 2, 3, 4, 5)),
                CentChannelSet(channels=(0, 1, 3, 4)),
            ),
        )

    def test_score_transfer_distinguishes_rs_and_rd_directions(self) -> None:
        """Use source addresses for writes and destinations for reads."""

        context, layout, writes = make_state()
        plan = make_attention_plan(context, layout)
        _lower_score_transfer(
            writes,
            plan.softmax.passes[0].left_input,
        )
        self.assertEqual(
            writes.instructions,
            [
                WriteSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=0,
                        row=21,
                        column=0,
                    ),
                    operation_size=1,
                    source=CentSharedBufferAddress(slot=32),
                )
            ],
        )

        reads = CentProgramBuilder(context.hardware, context.placement)
        _lower_score_transfer(
            reads,
            plan.softmax.passes[0].output,
        )
        self.assertEqual(
            reads.instructions,
            [
                ReadSingleBank(
                    address=CentMemoryAddress(
                        channel=0,
                        bank=2,
                        row=21,
                        column=0,
                    ),
                    operation_size=1,
                    destination=CentSharedBufferAddress(slot=32),
                )
            ],
        )
        # Score storage defines exactly three four-bank roles, numbered 0, 1,
        # and 2; group 3 therefore has no physical meaning.
        with self.assertRaises(ValueError):
            _lower_score_transfer(
                reads,
                replace(
                    plan.softmax.passes[0].output,
                    bank_group=3,
                    logical_banks=(3,),
                ),
            )

    def test_softmax_and_output_gemv_emit_expected_operation_families(
        self,
    ) -> None:
        """Emit the expected softmax and output-GEMV instructions."""

        context, layout, softmax = make_state()
        plan = make_attention_plan(context, layout)
        lower_softmax(softmax, plan.softmax)
        softmax_counts = Counter(i.opcode for i in softmax.instructions)
        # Softmax has two scaling passes: one for 1/sqrt(head_size), then one
        # for the reciprocal exponent sum. Each pass contributes one EW_MUL.
        self.assertEqual(softmax_counts[CentOpcode.ELEMENTWISE_MULTIPLY], 2)
        self.assertGreater(softmax_counts[CentOpcode.WRITE_SINGLE_BANK], 0)

        output = CentProgramBuilder(context.hardware, context.placement)
        lower_attention_output(output, plan.output)
        output_counts = Counter(i.opcode for i in output.instructions)
        # One score row is loaded into the Global Buffer once. A 16-value head
        # distributed over four banks has four value dimensions, so output
        # GEMV emits four MAC_ABK operations.
        self.assertEqual(output_counts[CentOpcode.WRITE_GLOBAL_BUFFER], 1)
        self.assertEqual(output_counts[CentOpcode.MAC_ALL_BANKS], 4)


class LlamaOrchestrationTests(unittest.TestCase):
    """Test the Llama compiler entry point."""

    def test_frontend_returns_a_nonempty_structural_isa_program(self) -> None:
        """Combine all lowering stages into an instruction program."""

        # TODO(test): Add numerical end-to-end expected values after tensor
        # bindings and executor behavior exist. Known opcodes do not prove that
        # the Llama data flows through the program correctly.

        program = compile_llama_transformer_block(make_request())
        self.assertGreater(len(program.instructions), 0)
        self.assertTrue(
            all(
                instruction.opcode in CentOpcode for instruction in program.instructions
            )
        )

    def test_private_block_stages_emit_their_expected_instruction_counts(
        self,
    ) -> None:
        """Keep attention and FFN orchestration independently testable."""

        plan = _create_compile_plan(make_request())
        attention_builder = CentProgramBuilder(
            plan.context.hardware,
            plan.context.placement,
        )
        _lower_self_attention(attention_builder, plan)
        feed_forward_builder = CentProgramBuilder(
            plan.context.hardware,
            plan.context.placement,
        )
        _lower_feed_forward(feed_forward_builder, plan)

        # The tiny block has 102 attention instructions and 69 FFN
        # instructions. Each stage owns exactly one residual ACC operation.
        self.assertEqual(len(attention_builder.instructions), 102)
        self.assertEqual(len(feed_forward_builder.instructions), 69)
        self.assertEqual(
            sum(
                instruction.opcode is CentOpcode.ACCUMULATION
                for instruction in attention_builder.instructions
            ),
            1,
        )
        self.assertEqual(
            sum(
                instruction.opcode is CentOpcode.ACCUMULATION
                for instruction in feed_forward_builder.instructions
            ),
            1,
        )

    def test_q_k_and_v_share_an_input_but_not_an_output(self) -> None:
        """Connect three projections to distinct Shared Buffer spans."""

        program = compile_llama_transformer_block(make_request())
        global_buffer_writes = [
            instruction
            for instruction in program.instructions
            if isinstance(instruction, WriteGlobalBuffer)
        ]

        # RMSNorm writes normalized x to slots 4..7. The first three WR_GB
        # instructions are Wq, Wk, and Wv, and all must read that same tensor.
        self.assertEqual(
            [instruction.source for instruction in global_buffer_writes[:3]],
            [CentSharedBufferAddress(slot=4)] * 3,
        )

        mac_reads = [
            instruction
            for instruction in program.instructions
            if isinstance(instruction, ReadMac)
        ]
        # RMSNorm first uses slot 28 for its partial sum. The following twelve
        # reads are Q results in 8..11, K results in 12..15, and V results in
        # 16..19. The packed attention vectors begin later at slot 20.
        self.assertEqual(
            [instruction.destination for instruction in mac_reads[:13]],
            [CentSharedBufferAddress(slot=28)]
            + [CentSharedBufferAddress(slot=slot) for slot in range(8, 20)],
        )

    def test_private_planners_reproduce_the_compiler_plan(self) -> None:
        """Keep each nontrivial Llama planner independently testable."""

        plan = _create_compile_plan(make_request())
        silu = _create_silu_product_plan(
            plan.context,
            plan.memory,
            plan.buffers,
        )
        self_attention = _create_self_attention_lowering_plan(
            plan.context,
            plan.row_counts,
            plan.memory,
            plan.buffers,
            plan.self_attention.attention,
        )
        feed_forward = _create_feed_forward_lowering_plan(
            plan.context,
            plan.row_counts,
            plan.memory,
            plan.buffers,
            silu,
        )

        self.assertEqual(silu, plan.feed_forward.silu_product)
        self.assertEqual(self_attention, plan.self_attention)
        self.assertEqual(feed_forward, plan.feed_forward)

        normalization = plan.self_attention.normalization
        self.assertEqual(
            _create_rms_norm_plan(
                plan.context,
                input_rows=normalization.l2_norm.sum_of_squares.input_rows,
                work_rows=normalization.l2_norm.work_rows,
                weight_rows=normalization.weight_rows,
                input_buffer=normalization.l2_norm.sum_of_squares.input_buffer.span,
                scale_buffer=normalization.l2_norm.scale_buffer.span,
                partial_sum_buffer=(
                    normalization.l2_norm.sum_of_squares.partial_sum_buffer
                ),
                output_buffer=normalization.output_buffer.span,
            ),
            normalization,
        )

    def test_attention_scores_do_not_overlap_the_live_residual_input(self) -> None:
        """Preserve the original block input until the first residual add."""

        plan = _create_compile_plan(make_request())
        input_span = plan.buffers.input
        score_span = plan.buffers.scores
        input_end = input_span.start.slot + input_span.slot_count
        score_end = score_span.start.slot + score_span.slot_count

        self.assertTrue(
            input_end <= score_span.start.slot or score_end <= input_span.start.slot
        )
        self.assertEqual(
            plan.self_attention.residual.source.span,
            input_span,
        )


if __name__ == "__main__":
    unittest.main()
