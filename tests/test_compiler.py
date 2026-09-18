"""Tests for the public model-to-CENT compiler."""

import unittest
from collections import Counter

from vllm_cent import (
    CentBlockPlacementSpec,
    CentHardwareSpec,
    CentOpcode,
    CompileRequest,
    DecodeStepSpec,
    LlamaModelSpec,
    compile_transformer_block,
)
from vllm_cent.cent import (
    Accumulate,
    CentMemoryAddress,
    CentSharedBufferAddress,
    ElementwiseMultiply,
    ReadSingleBank,
    WriteSingleBank,
    render_text_program,
)
from vllm_cent.models.base import ModelSpec
from vllm_cent.models.llama import compile_llama_transformer_block
from vllm_cent.models.llama.planning import _create_compile_plan


class UnsupportedModelSpec(ModelSpec):
    """Represent a model with no compiler."""


class LlamaModelVariant(LlamaModelSpec):
    """Represent a Llama specification with inherited Llama behavior."""


def small_request(
    *,
    hidden_size: int = 16,
    num_attention_heads: int = 1,
    num_kv_heads: int = 1,
    intermediate_size: int = 16,
    num_channels: int = 1,
    channels_per_block: int = 1,
    dram_rows: int = 1_000,
    sequence_length: int = 1,
    max_sequence_length: int = 16,
    accumulator_slots_per_bank: int = 2,
) -> CompileRequest:
    """Build the small Llama request used by compiler tests.

    Args:
        hidden_size: Residual width in BF16 values.
        num_attention_heads: Query-head count.
        num_kv_heads: Unique key/value-head count.
        intermediate_size: Feed-forward width in BF16 values.
        num_channels: Physical CENT channel count.
        channels_per_block: Channels assigned to one block.
        dram_rows: Rows available in every bank.
        sequence_length: Tokens in the current context.
        max_sequence_length: Reserved KV-cache length.
        accumulator_slots_per_bank: Accumulator registers in each PU.

    Returns:
        Complete request using four banks and four-value micro-operations.
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
            dram_rows=dram_rows,
            dram_columns=16,
            global_buffer_columns=16,
            burst_length=4,
            accumulator_slots_per_bank=accumulator_slots_per_bank,
            # The paper does not assign numeric AFid values; zero is a fake
            # target-ABI value chosen explicitly for these tests.
            sigmoid_activation_function_id=0,
        ),
        placement=CentBlockPlacementSpec(channels_per_block=channels_per_block),
        step=DecodeStepSpec(
            sequence_length=sequence_length,
            max_sequence_length=max_sequence_length,
        ),
    )


class CompilerInputTests(unittest.TestCase):
    """Test compiler input validation."""

    def test_rejects_unsupported_models_and_invalid_placement(self) -> None:
        """Reject unknown models and invalid block placement."""

        base = small_request()
        unsupported = CompileRequest(
            model=UnsupportedModelSpec(),
            hardware=base.hardware,
            placement=base.placement,
            step=base.step,
        )
        with self.assertRaisesRegex(TypeError, "UnsupportedModelSpec"):
            compile_transformer_block(unsupported)
        with self.assertRaisesRegex(TypeError, "LlamaModelSpec"):
            compile_llama_transformer_block(unsupported)
        with self.assertRaisesRegex(ValueError, "cannot exceed"):
            compile_transformer_block(small_request(channels_per_block=2))
        with self.assertRaisesRegex(ValueError, "must divide"):
            compile_transformer_block(
                small_request(num_channels=3, channels_per_block=2)
            )

    def test_accepts_llama_model_subclasses(self) -> None:
        """Use the Llama compiler for a Llama specification subclass."""

        base = small_request()
        variant = CompileRequest(
            # These are the same dimensions as ``small_request``. Only the
            # model's Python class differs, so both programs should match.
            model=LlamaModelVariant(
                hidden_size=16,
                num_attention_heads=1,
                num_kv_heads=1,
                intermediate_size=16,
            ),
            hardware=base.hardware,
            placement=base.placement,
            step=base.step,
        )

        self.assertEqual(
            compile_transformer_block(variant),
            compile_transformer_block(base),
        )

    def test_rejects_incompatible_llama_and_capacity_dimensions(self) -> None:
        """Reject incompatible model and hardware dimensions."""

        cases = (
            (small_request(accumulator_slots_per_bank=1), "activation"),
            (small_request(hidden_size=4, num_attention_heads=2), "at least"),
            (small_request(intermediate_size=33), "two-pass"),
            (small_request(dram_rows=10), "requires"),
        )
        for request, message in cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    compile_transformer_block(request)

    def test_decode_step_rejects_invalid_lengths_and_accepts_capacity(self) -> None:
        """Validate both decode lengths and their inclusive upper boundary."""

        self.assertEqual(
            DecodeStepSpec(sequence_length=8, max_sequence_length=8),
            DecodeStepSpec(sequence_length=8, max_sequence_length=8),
        )
        for sequence_length, max_sequence_length in ((1, 0), (0, 1), (2, 1)):
            with self.subTest(
                sequence_length=sequence_length,
                max_sequence_length=max_sequence_length,
            ):
                with self.assertRaises(ValueError):
                    DecodeStepSpec(
                        sequence_length=sequence_length,
                        max_sequence_length=max_sequence_length,
                    )


class TransformerBlockCompilerTests(unittest.TestCase):
    """Test complete Llama block compilation."""

    def test_tiny_block_has_stable_paper_operand_instruction_counts(self) -> None:
        """Keep the tiny block's instruction counts stable."""

        program = compile_transformer_block(small_request())
        counts = Counter(instruction.opcode for instruction in program.instructions)

        # These are the expected counts for the smallest complete block.
        # WR_SBK and RD_SBK count row transfers; OPsize stores the burst count.
        #
        # The totals below are sums of the compiler's ordered lowering stages:
        # - Each of two RMSNorms emits 4 WR_SBK, 1 WR_BIAS, 1 MAC_ABK,
        #   1 RD_MAC, 2 EW_MUL, 1 COPY_BKGB, 1 COPY_GBBK, and 1 RD_SBK.
        # - Wq, Wk, Wv, Wo, W3, and W2 each emit 1 WR_GB and four groups of
        #   WR_BIAS + MAC_ABK + RD_MAC for the four outputs assigned per bank.
        # - W1 emits that same GEMV shape, then preserves four raw results and
        #   reads four separately activated results.
        # - Rotary emits 2 WR_SBK, 2 RD_SBK, and 2 EW_MUL; the KV update emits
        #   1 WR_SBK for the key and 4 WR_ABK for the value dimensions.
        # - Score GEMV emits one WR_GB/WR_BIAS/MAC_ABK/RD_MAC group. Two
        #   softmax passes total 4 WR_SBK, 2 EW_MUL, and 2 RD_SBK.
        # - Output GEMV emits 1 WR_GB and four WR_BIAS/MAC_ABK/RD_MAC groups.
        # - The gated SiLU product emits 3 WR_SBK, 2 EW_MUL, one of each copy,
        #   and 1 RD_SBK. Spilling the attention residual adds one WR_SBK and
        #   reloading it adds one RD_SBK. The residuals emit 2 ACC in total.
        self.assertEqual(
            counts,
            {
                CentOpcode.WRITE_SINGLE_BANK: 19,
                CentOpcode.READ_SINGLE_BANK: 8,
                CentOpcode.WRITE_ALL_BANKS: 4,
                CentOpcode.WRITE_BIAS: 35,
                CentOpcode.MAC_ALL_BANKS: 35,
                CentOpcode.READ_MAC: 35,
                CentOpcode.READ_ACTIVATION: 4,
                CentOpcode.ELEMENTWISE_MULTIPLY: 10,
                CentOpcode.WRITE_GLOBAL_BUFFER: 9,
                CentOpcode.COPY_BANK_TO_GLOBAL_BUFFER: 3,
                CentOpcode.COPY_GLOBAL_BUFFER_TO_BANK: 3,
                CentOpcode.ACTIVATION_FUNCTION: 4,
                CentOpcode.ACCUMULATION: 2,
            },
        )
        self.assertEqual(len(program.instructions), 171)

    def test_residual_add_uses_nonoverlapping_shared_buffer_vectors(self) -> None:
        """Use separate Shared Buffer vectors for residual addition."""

        program = compile_transformer_block(small_request())
        residuals = [
            instruction
            for instruction in program.instructions
            if isinstance(instruction, Accumulate)
        ]

        # A sixteen-value vector occupies four four-value slots. The packed
        # attention projection in slots 20..23 first adds the original input
        # in slots 0..3. The final FFN projection in slots 4..7 then adds the
        # attention residual reloaded into slots 20..23.
        self.assertEqual(
            residuals,
            [
                Accumulate(
                    operation_size=4,
                    destination=CentSharedBufferAddress(slot=20),
                    source=CentSharedBufferAddress(slot=0),
                ),
                Accumulate(
                    operation_size=4,
                    destination=CentSharedBufferAddress(slot=4),
                    source=CentSharedBufferAddress(slot=20),
                ),
            ],
        )

    def test_feed_forward_spills_and_reloads_the_attention_residual(self) -> None:
        """Preserve the residual while FFN scratch reuses its buffer slots."""

        program = compile_transformer_block(small_request())
        instructions = list(program.instructions)
        residual_indices = [
            index
            for index, instruction in enumerate(instructions)
            if isinstance(instruction, Accumulate)
        ]
        first_residual, final_residual = residual_indices

        # Row 31 is the planned ``sa`` workspace for this tiny model. The first
        # ACC leaves its packed result in slots 20..23. It is stored immediately
        # after that ACC and loaded immediately before the final ACC.
        self.assertEqual(
            instructions[first_residual + 1],
            WriteSingleBank(
                address=CentMemoryAddress(channel=0, bank=0, row=31, column=0),
                operation_size=4,
                source=CentSharedBufferAddress(slot=20),
            ),
        )
        self.assertEqual(
            instructions[final_residual - 1],
            ReadSingleBank(
                address=CentMemoryAddress(channel=0, bank=0, row=31, column=0),
                operation_size=4,
                destination=CentSharedBufferAddress(slot=20),
            ),
        )

    def test_output_distinguishes_target_extension_from_trace_controls(
        self,
    ) -> None:
        """Render RD_AF without adding AiM trace framing to paper text."""

        assembly = render_text_program(compile_transformer_block(small_request()))

        self.assertIn("MAC_ABK", assembly)
        self.assertIn("WR_SBK", assembly)
        self.assertNotIn("SYNC", assembly)
        self.assertNotIn("EOC", assembly)
        self.assertIn("RD_AF", assembly)

    def test_grouped_query_attention_and_two_pass_activation_compile(self) -> None:
        """Compile grouped-query attention with two activation passes."""

        request = small_request(
            hidden_size=64,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=128,
            num_channels=4,
            channels_per_block=4,
            sequence_length=17,
            max_sequence_length=17,
            accumulator_slots_per_bank=4,
        )
        plan = _create_compile_plan(request)
        program = compile_transformer_block(request)
        # Four query heads share two KV heads. Four channels, one PU per
        # channel, and 16 columns give 4 * 1 * 16 = 64 activation values per
        # pass. The 128-value FFN therefore needs both supported passes.
        self.assertGreater(len(program.instructions), 0)
        self.assertEqual(
            [
                (
                    chunk.row,
                    chunk.value_count,
                    chunk.partition_count,
                    chunk.values_per_partition,
                )
                for chunk in plan.feed_forward.silu_product.chunks
            ],
            [
                (plan.memory.x1, 64, 4, 16),
                (plan.memory.x1_sigmoid, 64, 4, 16),
            ],
        )

    def test_small_and_multirow_normalization_layouts_compile(self) -> None:
        """Compile partition-padding and multirow normalization boundaries."""

        small = compile_transformer_block(
            small_request(
                hidden_size=4,
                num_attention_heads=1,
                intermediate_size=4,
            )
        )
        multirow = compile_transformer_block(
            small_request(
                hidden_size=32,
                num_attention_heads=2,
                intermediate_size=32,
            )
        )

        self.assertGreater(len(small.instructions), 0)
        # The target row holds four bursts. Every multirow elementwise pass is
        # split into separate instructions instead of crossing that boundary.
        self.assertTrue(
            all(
                instruction.operation_size <= 4
                for instruction in multirow.instructions
                if isinstance(instruction, ElementwiseMultiply)
            )
        )

    def test_llama_3_8b_and_70b_shapes_compile(self) -> None:
        """Compile the Llama 3 8B and 70B block dimensions."""

        target = CentHardwareSpec(
            num_channels=32,
            num_banks=16,
            dram_rows=16_384,
            dram_columns=1_024,
            global_buffer_columns=1_024,
            burst_length=16,
            accumulator_slots_per_bank=32,
            sigmoid_activation_function_id=0,
        )
        # These are the published Llama 3 8B and 70B block dimensions. Both
        # have 128-value heads: 4096/32 for 8B and 8192/64 for 70B. Their eight
        # KV heads therefore produce a 1024-value KV width. The channel counts
        # allocate 8B to 8*16=128 banks and 70B to 16*16=256 banks.
        cases = (
            (
                LlamaModelSpec(
                    hidden_size=4_096,
                    num_attention_heads=32,
                    num_kv_heads=8,
                    intermediate_size=14_336,
                ),
                8,
            ),
            (
                LlamaModelSpec(
                    hidden_size=8_192,
                    num_attention_heads=64,
                    num_kv_heads=8,
                    intermediate_size=28_672,
                ),
                16,
            ),
        )
        for model, channels_per_block in cases:
            with self.subTest(hidden_size=model.hidden_size):
                request = CompileRequest(
                    model=model,
                    hardware=target,
                    placement=CentBlockPlacementSpec(
                        channels_per_block=channels_per_block
                    ),
                    step=DecodeStepSpec(
                        # A one-token decode keeps the emitted test trace small
                        # while reserving Llama's 8192-token KV cache.
                        sequence_length=1,
                        max_sequence_length=8_192,
                    ),
                )
                plan = _create_compile_plan(request)
                program = compile_transformer_block(request)
                self.assertGreater(len(program.instructions), 0)
                self.assertEqual(plan.context.kv_width, 1_024)
                self.assertEqual(
                    plan.context.total_banks,
                    channels_per_block * target.num_banks,
                )
                self.assertEqual(
                    plan.self_attention.query_projection.utilized_banks,
                    plan.context.total_banks,
                )
                self.assertLessEqual(plan.memory.end, target.dram_rows)


if __name__ == "__main__":
    unittest.main()
