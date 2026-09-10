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
    MacAllBanks,
    ReadMac,
    ReadSingleBank,
    WriteAllBanks,
    WriteGlobalBuffer,
    WriteSingleBank,
)
from vllm_cent.lowering import (
    CentDramRowRange,
    CentSharedBufferSpan,
    lower_rms_norm,
)
from vllm_cent.lowering.transformer import (
    TransformerAttentionBuffers,
    TransformerAttentionRows,
    TransformerAttentionSpec,
    lower_attention_output,
    lower_kv_cache_update,
    lower_rotary_embedding,
    lower_score_gemv,
    lower_softmax,
)
from vllm_cent.lowering.transformer.attention import _lower_score_transfer
from vllm_cent.lowering.utils import (
    _PartitionedVectorLayout,
    _plan_partitioned_vector,
)
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
    _create_context,
    _create_attention_plan,
    _create_compile_plan,
    _plan_memory,
    _plan_shared_buffer,
    _row_counts,
    _validate_context,
)
from vllm_cent.models.llama.feed_forward import _lower_silu_product


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
        placement=CentBlockPlacementSpec(
            channels_per_block=channels_per_block
        ),
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


def make_attention_bindings(
    context: _LlamaCompileContext,
    layout: _LlamaMemoryLayout,
) -> tuple[
    TransformerAttentionSpec,
    TransformerAttentionRows,
    TransformerAttentionBuffers,
]:
    """Convert a Llama plan into reusable attention operands.

    Args:
        context: Derived Llama and hardware dimensions.
        layout: DRAM rows assigned to the Llama block.

    Returns:
        Generic attention sizes, DRAM ranges, and Shared Buffer spans.
    """

    attention = _create_attention_plan(
        context,
        _row_counts(context),
        layout,
        _plan_shared_buffer(context),
    )
    return attention.spec, attention.rows, attention.buffers


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
        """Reject an FFN that exceeds two activation passes."""

        valid = _create_context(make_request())
        with self.assertRaisesRegex(ValueError, "two-pass"):
            _validate_context(
                replace(
                    valid,
                    model=LlamaModelSpec(
                        hidden_size=16,
                        num_attention_heads=1,
                        num_kv_heads=1,
                        intermediate_size=33,
                    ),
                )
            )

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
            _plan_partitioned_vector(10, 4, 4),
            _PartitionedVectorLayout(
                values_per_partition=3,
                partition_count=4,
                slot_count=4,
            ),
        )
        for arguments in ((0, 4, 4), (10, 0, 4), (10, 4, 0)):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    _plan_partitioned_vector(*arguments)

    def test_shared_buffer_plan_separates_q_k_and_v(self) -> None:
        """Give the attention projections nonoverlapping staging spans."""

        context, _, _ = make_state()
        layout = _plan_shared_buffer(context)

        # Each projection first produces four accumulator-result slots. Those
        # occupy 8..19. The separately packed Q, K, and V vectors then occupy
        # slots 20..31 so the two physical layouts are not conflated.
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
                end_slot=32,
            ),
        )

    def test_shared_buffer_plan_rejects_insufficient_capacity(self) -> None:
        """Reject a target that cannot hold live attention values."""

        context, _, _ = make_state()
        with self.assertRaisesRegex(ValueError, "Shared Buffer slots"):
            _plan_shared_buffer(
                replace(
                    context,
                    hardware=replace(context.hardware, shared_buffer_slots=31),
                )
            )


class NormalizationAndFeedForwardTests(unittest.TestCase):
    """Test normalization and Llama feed-forward lowering."""

    def test_rms_norm_and_silu_emit_paper_copy_shapes(self) -> None:
        """Emit the paper's bank-copy operand shapes."""

        context, layout, norm_builder = make_state()
        rows = _row_counts(context)
        buffers = _plan_shared_buffer(context)
        lower_rms_norm(
            norm_builder,
            input_rows=CentDramRowRange(
                start_row=layout.x, row_count=rows.x
            ),
            work_rows=CentDramRowRange(
                start_row=layout.x_copy, row_count=rows.x
            ),
            weight_rows=CentDramRowRange(
                start_row=layout.sa_norm, row_count=rows.x
            ),
            input_buffer=buffers.input,
            scale_buffer=buffers.normalized,
            partial_sum_buffer=buffers.value,
            output_buffer=buffers.normalized,
            value_count=context.model.hidden_size,
        )
        norm_counts = Counter(i.opcode for i in norm_builder.instructions)
        # RMSNorm copies its bank result into the Global Buffer once, then
        # copies that scalar scale back to banks once before the final multiply.
        self.assertEqual(norm_counts[CentOpcode.COPY_BANK_TO_GLOBAL_BUFFER], 1)
        self.assertEqual(norm_counts[CentOpcode.COPY_GLOBAL_BUFFER_TO_BANK], 1)

        silu_builder = CentProgramBuilder(context.hardware, context.placement)
        _lower_silu_product(
            silu_builder,
            context,
            layout,
            buffers.ffn_product,
        )
        silu_counts = Counter(i.opcode for i in silu_builder.instructions)
        # The tiny FFN fits one activation chunk. It performs one multiply to
        # form SiLU, one copy through the Global Buffer, and a second multiply
        # combining the activated gate with the W3 projection.
        self.assertEqual(silu_counts[CentOpcode.COPY_BANK_TO_GLOBAL_BUFFER], 1)
        self.assertEqual(silu_counts[CentOpcode.ELEMENTWISE_MULTIPLY], 2)

class AttentionLoweringTests(unittest.TestCase):
    """Test each attention lowering stage."""

    def test_rotary_embedding_uses_explicit_single_bank_transfers(self) -> None:
        """Use WR_SBK and EW_MUL for rotary data flow."""

        context, layout, builder = make_state()
        spec, rows, buffers = make_attention_bindings(context, layout)
        lower_rotary_embedding(builder, spec, rows, buffers)
        counts = Counter(i.opcode for i in builder.instructions)

        # Query and key are written into operand bank group 1. Their computed
        # bank-group-2 results are then read back to the same named buffers.
        self.assertEqual(counts[CentOpcode.WRITE_SINGLE_BANK], 2)
        self.assertEqual(counts[CentOpcode.READ_SINGLE_BANK], 2)
        self.assertEqual(counts[CentOpcode.ELEMENTWISE_MULTIPLY], 2)

    def test_rotary_embedding_rejects_an_undersized_query_span(self) -> None:
        """Reject a query buffer that cannot hold all PU partitions."""

        context, layout, builder = make_state()
        spec, rows, buffers = make_attention_bindings(context, layout)
        small_query = CentSharedBufferSpan(
            start=buffers.query.start,
            slot_count=buffers.query.slot_count - 1,
        )

        with self.assertRaisesRegex(ValueError, "query needs"):
            lower_rotary_embedding(
                builder,
                spec,
                rows,
                replace(buffers, query=small_query),
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
        spec, rows, buffers = make_attention_bindings(context, layout)

        # This one-channel target has one PU group, so all 32 query values need
        # two sixteen-value DRAM rows. The planned one-row range is too short.
        with self.assertRaisesRegex(ValueError, "query rows needs 2"):
            lower_rotary_embedding(builder, spec, rows, buffers)

    def test_kv_cache_update_writes_column_and_channel(self) -> None:
        """Put the new cache value in the expected column and channel."""

        context, layout, builder = make_state(sequence_length=5)
        spec, rows, buffers = make_attention_bindings(context, layout)
        lower_kv_cache_update(builder, spec, rows, buffers)

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
        spec, rows, buffers = make_attention_bindings(context, layout)
        one_slot = CentSharedBufferSpan(
            start=CentSharedBufferAddress(slot=0),
            slot_count=1,
        )
        cases = (
            (
                lower_kv_cache_update,
                replace(buffers, key=one_slot),
                "key needs",
            ),
            (
                lower_score_gemv,
                replace(buffers, query=one_slot),
                "query needs",
            ),
            (
                lower_softmax,
                replace(buffers, scores=one_slot),
                "scores needs",
            ),
            (
                lower_attention_output,
                replace(buffers, scores=one_slot),
                "scores needs",
            ),
        )
        for lowerer, stage_buffers, message in cases:
            with self.subTest(lowerer=lowerer.__name__):
                builder = CentProgramBuilder(
                    context.hardware,
                    context.placement,
                )
                with self.assertRaisesRegex(ValueError, message):
                    lowerer(builder, spec, rows, stage_buffers)

    def test_score_gemv_uses_head_column_offsets(self) -> None:
        """Advance the MAC column between heads packed in one row."""

        context, layout, builder = make_state(
            hidden_size=32,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=16,
        )
        spec, rows, buffers = make_attention_bindings(context, layout)
        lower_score_gemv(builder, spec, rows, buffers)
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
                    row=53,
                    column=0,
                    accumulation_register=0,
                ),
                MacAllBanks(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=2,
                    row=53,
                    column=8,
                    accumulation_register=0,
                ),
                MacAllBanks(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=2,
                    row=54,
                    column=0,
                    accumulation_register=0,
                ),
                MacAllBanks(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=2,
                    row=54,
                    column=8,
                    accumulation_register=0,
                ),
            ],
        )

    def test_score_transfer_distinguishes_rs_and_rd_directions(self) -> None:
        """Use source addresses for writes and destinations for reads."""

        context, layout, writes = make_state()
        spec, rows, buffers = make_attention_bindings(context, layout)
        _lower_score_transfer(
            writes,
            spec,
            rows,
            WriteSingleBank,
            0,
            buffers,
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
                    source=CentSharedBufferAddress(slot=0),
                )
            ],
        )

        reads = CentProgramBuilder(context.hardware, context.placement)
        _lower_score_transfer(
            reads,
            spec,
            rows,
            ReadSingleBank,
            2,
            buffers,
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
                    destination=CentSharedBufferAddress(slot=0),
                )
            ],
        )
        # Score storage defines exactly three four-bank roles, numbered 0, 1,
        # and 2; group 3 therefore has no physical meaning.
        with self.assertRaises(ValueError):
            _lower_score_transfer(
                reads,
                spec,
                rows,
                ReadSingleBank,
                3,
                buffers,
            )

    def test_softmax_and_output_gemv_emit_expected_operation_families(
        self,
    ) -> None:
        """Emit the expected softmax and output-GEMV instructions."""

        context, layout, softmax = make_state()
        spec, rows, buffers = make_attention_bindings(context, layout)
        lower_softmax(softmax, spec, rows, buffers)
        softmax_counts = Counter(i.opcode for i in softmax.instructions)
        # Softmax has two scaling passes: one for 1/sqrt(head_size), then one
        # for the reciprocal exponent sum. Each pass contributes one EW_MUL.
        self.assertEqual(softmax_counts[CentOpcode.ELEMENTWISE_MULTIPLY], 2)
        self.assertGreater(softmax_counts[CentOpcode.WRITE_SINGLE_BANK], 0)

        output = CentProgramBuilder(context.hardware, context.placement)
        lower_attention_output(output, spec, rows, buffers)
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
                instruction.opcode in CentOpcode
                for instruction in program.instructions
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

        # The tiny block has 104 attention instructions and 71 FFN
        # instructions. Each stage owns exactly one residual ACC operation.
        self.assertEqual(len(attention_builder.instructions), 104)
        self.assertEqual(len(feed_forward_builder.instructions), 71)
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


if __name__ == "__main__":
    unittest.main()
