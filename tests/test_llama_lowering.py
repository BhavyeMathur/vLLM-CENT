"""Tests for the private Llama lowering helpers."""

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
    Accumulate,
    CentChannelSet,
    CentMemoryAddress,
    CentProgramBuilder,
    CentSharedBufferAddress,
    MacAllBanks,
    ReadSingleBank,
    WriteAllBanks,
    WriteSingleBank,
)
from vllm_cent.models.llama.attention import (
    _lower_kv_cache_update,
    _lower_output_gemv,
    _lower_rotary_embedding,
    _lower_score_gemv,
    _lower_score_transfer,
    _lower_softmax,
)
from vllm_cent.models.llama import compile_llama_transformer_block
from vllm_cent.models.llama.compiler import _lower_residual_add
from vllm_cent.models.llama.planning import (
    _LlamaCompileContext,
    _LlamaMemoryLayout,
    _LlamaRowCounts,
    _RowAllocator,
    _create_context,
    _plan_memory,
    _row_counts,
    _validate_context,
)
from vllm_cent.models.llama.feed_forward import _lower_silu_product
from vllm_cent.models.llama.linear import _lower_weight_gemv
from vllm_cent.models.llama.normalization import _lower_rms_norm

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


class LinearAndNormalizationTests(unittest.TestCase):
    """Test linear, normalization, and feed-forward lowering."""

    def test_weight_gemv_preserves_columns_registers_and_activation(self) -> None:
        """Emit the expected GEMV rows, registers, and activations."""

        _, _, builder = make_state()
        _lower_weight_gemv(builder, 7, 16, 8, 4, apply_activation=True)
        counts = Counter(i.opcode for i in builder.instructions)

        # Eight outputs over four banks produce two outputs per bank. Fused
        # activation reserves half of the four registers for activation state,
        # leaving a two-register group. The 16-value input fits in one row, so
        # the group needs one WR_GB and two MAC/AF/read sequences. Consecutive
        # output weights occupy rows 7 and 8 and registers 0 and 1.
        self.assertEqual(counts[CentOpcode.WRITE_GLOBAL_BUFFER], 1)
        self.assertEqual(counts[CentOpcode.MAC_ALL_BANKS], 2)
        self.assertEqual(counts[CentOpcode.ACTIVATION_FUNCTION], 2)
        self.assertEqual(counts[CentOpcode.READ_MAC], 2)
        macs = [
            instruction
            for instruction in builder.instructions
            if isinstance(instruction, MacAllBanks)
        ]
        self.assertEqual(
            macs,
            [
                MacAllBanks(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=4,
                    row=7,
                    column=0,
                    accumulation_register=0,
                ),
                MacAllBanks(
                    channels=CentChannelSet(channels=(0,)),
                    operation_size=4,
                    row=8,
                    column=0,
                    accumulation_register=1,
                ),
            ],
        )
        with self.assertRaises(ValueError):
            _lower_weight_gemv(builder, 0, 16, 1, 1, apply_activation=True)

    def test_rms_norm_and_silu_emit_paper_copy_shapes(self) -> None:
        """Emit the paper's bank-copy operand shapes."""

        context, layout, norm_builder = make_state()
        _lower_rms_norm(norm_builder, context, layout.x, layout.x_copy, layout.sa_norm)
        norm_counts = Counter(i.opcode for i in norm_builder.instructions)
        # RMSNorm copies its bank result into the Global Buffer once, then
        # copies that scalar scale back to banks once before the final multiply.
        self.assertEqual(norm_counts[CentOpcode.COPY_BANK_TO_GLOBAL_BUFFER], 1)
        self.assertEqual(norm_counts[CentOpcode.COPY_GLOBAL_BUFFER_TO_BANK], 1)

        silu_builder = CentProgramBuilder(context.hardware, context.placement)
        _lower_silu_product(silu_builder, context, layout)
        silu_counts = Counter(i.opcode for i in silu_builder.instructions)
        # The tiny FFN fits one activation chunk. It performs one multiply to
        # form SiLU, one copy through the Global Buffer, and a second multiply
        # combining the activated gate with the W3 projection.
        self.assertEqual(silu_counts[CentOpcode.COPY_BANK_TO_GLOBAL_BUFFER], 1)
        self.assertEqual(silu_counts[CentOpcode.ELEMENTWISE_MULTIPLY], 2)

    def test_residual_add_uses_acc_instruction(self) -> None:
        """Lower residual addition to the paper's ACC instruction."""

        _, _, builder = make_state()
        _lower_residual_add(builder, 16)

        # Sixteen values occupy four four-value Shared Buffer slots. ACC reads
        # its destination vector from Rd slot 0 and the nonoverlapping residual
        # from Rs slot 4, and OPsize 4 covers the complete vectors.
        self.assertEqual(
            builder.instructions,
            [
                Accumulate(
                    operation_size=4,
                    destination=CentSharedBufferAddress(slot=0),
                    source=CentSharedBufferAddress(slot=4),
                )
            ],
        )


class AttentionLoweringTests(unittest.TestCase):
    """Test each attention lowering stage."""

    def test_rotary_embedding_uses_explicit_single_bank_transfers(self) -> None:
        """Use WR_SBK and EW_MUL for rotary data flow."""

        context, layout, builder = make_state()
        _lower_rotary_embedding(builder, context, layout)
        counts = Counter(i.opcode for i in builder.instructions)

        # The one-head query and key each transfer into bank group 1 and again
        # into result bank group 2. Each group transfer expands across two live
        # banks, giving 2 tensors * 2 groups * 2 banks = 8 WR_SBK. Query and key
        # each need one elementwise rotation multiply.
        self.assertEqual(counts[CentOpcode.WRITE_SINGLE_BANK], 8)
        self.assertEqual(counts[CentOpcode.ELEMENTWISE_MULTIPLY], 2)

    def test_kv_cache_update_writes_column_and_channel(self) -> None:
        """Put the new cache value in the expected column and channel."""

        context, layout, builder = make_state(sequence_length=5)
        _lower_kv_cache_update(builder, context, layout)

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
                    source=CentSharedBufferAddress(slot=0),
                ),
                WriteAllBanks(
                    channel=0,
                    row=22,
                    column=4,
                    source=CentSharedBufferAddress(slot=0),
                    accumulation_register=0,
                ),
                WriteAllBanks(
                    channel=0,
                    row=23,
                    column=4,
                    source=CentSharedBufferAddress(slot=0),
                    accumulation_register=0,
                ),
                WriteAllBanks(
                    channel=0,
                    row=24,
                    column=4,
                    source=CentSharedBufferAddress(slot=0),
                    accumulation_register=0,
                ),
                WriteAllBanks(
                    channel=0,
                    row=25,
                    column=4,
                    source=CentSharedBufferAddress(slot=0),
                    accumulation_register=0,
                ),
            ],
        )

    def test_score_gemv_uses_head_column_offsets(self) -> None:
        """Advance the MAC column between heads packed in one row."""

        context, layout, builder = make_state(
            hidden_size=32,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=16,
        )
        _lower_score_gemv(builder, context, layout)
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
        _lower_score_transfer(writes, context, layout, WriteSingleBank, 0)
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
        _lower_score_transfer(reads, context, layout, ReadSingleBank, 2)
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
            _lower_score_transfer(reads, context, layout, ReadSingleBank, 3)

    def test_softmax_and_output_gemv_emit_expected_operation_families(
        self,
    ) -> None:
        """Emit the expected softmax and output-GEMV instructions."""

        context, layout, softmax = make_state()
        _lower_softmax(softmax, context, layout)
        softmax_counts = Counter(i.opcode for i in softmax.instructions)
        # Softmax has two scaling passes: one for 1/sqrt(head_size), then one
        # for the reciprocal exponent sum. Each pass contributes one EW_MUL.
        self.assertEqual(softmax_counts[CentOpcode.ELEMENTWISE_MULTIPLY], 2)
        self.assertGreater(softmax_counts[CentOpcode.WRITE_SINGLE_BANK], 0)

        output = CentProgramBuilder(context.hardware, context.placement)
        _lower_output_gemv(output, context, layout)
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


if __name__ == "__main__":
    unittest.main()
