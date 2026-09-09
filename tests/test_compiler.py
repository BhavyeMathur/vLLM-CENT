import unittest

from vllm_cent import (
    CentHardwareSpec,
    CentInstruction,
    CentMappingSpec,
    CentOpcode,
    CentProgram,
    CompileRequest,
    DecodeStepSpec,
    LlamaModelSpec,
    compile_transformer_block,
    render_text_trace,
)


class CompilerTests(unittest.TestCase):
    def test_tiny_llama_block_has_expected_boundaries(self) -> None:
        # This is a deliberately small, valid Llama block. A hidden width of 16
        # split across four query heads gives four values per head. Two KV heads
        # mean that each KV head is shared by two query heads. The FFN width is
        # kept at 32 so its later mapping arithmetic remains easy to inspect.
        request = CompileRequest(
            model=LlamaModelSpec(
                hidden_size=16,
                num_attention_heads=4,
                num_kv_heads=2,
                intermediate_size=32,
            ),
            # Sixteen banks and a burst length of 16 match the current CENT
            # trace generator. The small row/column counts keep test addresses
            # human-readable; this test does not model storage capacity.
            hardware=CentHardwareSpec(
                num_channels=2,
                num_banks=16,
                dram_rows=128,
                dram_columns=64,
                burst_length=16,
            ),
            # Both channels execute this block, so their bit mask is binary 11,
            # rendered as 0x3. reuse_size=1 disables multi-chunk GB reuse.
            mapping=CentMappingSpec(channels_per_block=2, reuse_size=1),
            # Sequence length 1 is the first autoregressive decode step: the KV
            # cache contains only the token currently being decoded.
            step=DecodeStepSpec(sequence_length=1),
        )

        program = compile_transformer_block(request)

        self.assertEqual(
            program.instructions[:3],
            (
                # RMSNorm begins by zeroing MAC latch 0, accumulating the
                # one-burst input vector from DRAM row 0, then reading the sum.
                CentInstruction(opcode=CentOpcode.WRITE_BIAS, operands=(0, "0x3")),
                CentInstruction(opcode=CentOpcode.MAC_ALL_BANKS, operands=(1, "0x3", 0)),
                CentInstruction(opcode=CentOpcode.READ_MAC, operands=(0, "0x3")),
            ),
        )
        self.assertEqual(
            program.instructions[-1],
            CentInstruction(opcode=CentOpcode.END_OF_COMPUTATION),
        )

    def test_renders_exact_cent_trace_text(self) -> None:
        program = CentProgram(
            instructions=(
                # Write one burst to channel 0, bank 2, DRAM row 17.
                CentInstruction(opcode=CentOpcode.WRITE_MEMORY, operands=(0, 2, 17)),
                # Run four elementwise-multiply bursts on channels 0 and 1
                # (mask 0x3), using operands stored at DRAM row 18.
                CentInstruction(
                    opcode=CentOpcode.ELEMENTWISE_MULTIPLY,
                    operands=(4, "0x3", 18),
                ),
                # Every complete simulator trace ends with this marker.
                CentInstruction(opcode=CentOpcode.END_OF_COMPUTATION),
            )
        )

        self.assertEqual(
            render_text_trace(program),
            "W MEM 0 2 17\nAiM EWMUL 4 0x3 18\nAiM EOC\n",
        )


if __name__ == "__main__":
    unittest.main()
