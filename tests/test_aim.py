"""Tests for the AiM simulator trace adapter."""

import unittest
from dataclasses import replace

from vllm_cent.cent import (
    AIM_SIMULATOR_BANKS,
    AIM_SIMULATOR_BURST_LENGTH,
    AIM_SIMULATOR_CHANNELS,
    Accumulate,
    AimSimulatorCompatibilityError,
    ApplyActivation,
    CentChannelSet,
    CentHardwareSpec,
    CentInstruction,
    CentMemoryAddress,
    CentProgram,
    CentSharedBufferAddress,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    MacAllBanks,
    MacOperandSource,
    ReadActivation,
    ReadMac,
    ReadSingleBank,
    WriteAllBanks,
    WriteBias,
    WriteGlobalBuffer,
    WriteSingleBank,
    render_aim_channel_mask,
    render_aim_instruction,
    render_aim_trace,
    validate_aim_hardware,
)


def aim_hardware() -> CentHardwareSpec:
    """Create the fixed geometry accepted by the AiM trace adapter.

    Returns:
        Thirty-two-channel, sixteen-bank target with 256-bit transfers.
    """

    return CentHardwareSpec(
        num_channels=AIM_SIMULATOR_CHANNELS,
        num_banks=AIM_SIMULATOR_BANKS,
        dram_rows=64,
        dram_columns=64,
        burst_length=AIM_SIMULATOR_BURST_LENGTH,
        accumulator_slots_per_bank=4,
        sigmoid_activation_function_id=3,
        shared_buffer_slots=64,
    )


def program_with(instruction: CentInstruction) -> CentProgram:
    """Wrap one typed instruction in an AiM-geometry program.

    Args:
        instruction: CENT instruction used by the test.

    Returns:
        Program containing only ``instruction``.
    """

    return CentProgram(
        hardware=aim_hardware(),
        instructions=(instruction,),
    )


class AimTraceRenderingTests(unittest.TestCase):
    """Test exact trace records accepted by AiM's parser."""

    def test_channel_mask_uses_simulator_physical_bit_order(self) -> None:
        """Map channel zero to bit 31 and channel 31 to bit zero."""

        # AiM's FindFirstChannelIndex reverses a set-bit index. Selecting the
        # two endpoint channels therefore sets both endpoint bits.
        self.assertEqual(
            render_aim_channel_mask(CentChannelSet(channels=(0, 31))),
            "0x80000001",
        )
        with self.assertRaises(AimSimulatorCompatibilityError):
            render_aim_channel_mask(CentChannelSet(channels=(32,)))

    def test_renders_every_supported_dram_instruction_and_eoc(self) -> None:
        """Serialize data movement, MAC mode, activation, and completion."""

        channels = CentChannelSet(channels=(0, 31))
        channel_zero = CentChannelSet(channels=(0,))
        instructions = (
            WriteSingleBank(
                address=CentMemoryAddress(
                    channel=0, bank=2, row=3, column=0
                ),
                operation_size=2,
                source=CentSharedBufferAddress(slot=5),
            ),
            ReadSingleBank(
                address=CentMemoryAddress(
                    channel=0, bank=2, row=3, column=0
                ),
                operation_size=1,
                destination=CentSharedBufferAddress(slot=7),
            ),
            WriteAllBanks(
                channel=0,
                row=4,
                column=0,
                source=CentSharedBufferAddress(slot=8),
                accumulation_register=0,
            ),
            WriteGlobalBuffer(
                channels=channels,
                operation_size=2,
                column=0,
                source=CentSharedBufferAddress(slot=9),
            ),
            WriteBias(
                channels=channels,
                source=CentSharedBufferAddress(slot=11),
            ),
            MacAllBanks(
                channels=channels,
                operation_size=2,
                row=5,
                column=0,
                accumulation_register=0,
                operand_source=MacOperandSource.GLOBAL_BUFFER,
            ),
            MacAllBanks(
                channels=channel_zero,
                operation_size=1,
                row=6,
                column=0,
                accumulation_register=0,
                operand_source=MacOperandSource.NEXT_BANK,
            ),
            ReadMac(
                channels=channels,
                destination=CentSharedBufferAddress(slot=12),
                accumulation_register=0,
            ),
            ApplyActivation(
                channels=channels,
                activation_function_id=3,
                accumulation_register=0,
            ),
            ReadActivation(
                channels=channels,
                destination=CentSharedBufferAddress(slot=13),
                accumulation_register=0,
            ),
            ElementwiseMultiply(
                channels=channels,
                operation_size=2,
                row=7,
                column=0,
            ),
            CopyBankToGlobalBuffer(
                channels=channels,
                operation_size=2,
                bank=2,
                row=7,
                column=0,
            ),
            CopyGlobalBufferToBank(
                channels=channels,
                operation_size=2,
                bank=1,
                row=8,
                column=0,
            ),
        )
        program = CentProgram(
            hardware=aim_hardware(), instructions=instructions
        )

        # WR_SBK expands into two one-burst records because AiM's grammar has
        # no OPsize field. MAC mode and AFid become explicit CFR writes.
        expected = """AiM WR_SBK 5 0x80000000 2 3
AiM WR_SBK 6 0x80000000 2 3
AiM RD_SBK 7 0x80000000 2 3
AiM WR_ABK 8 0x80000000 4
AiM WR_GB 2 9 0x80000001
AiM WR_BIAS 11 0x80000001
W CFR 0 0
AiM MAC_ABK 2 0x80000001 5
W CFR 0 1
AiM MAC_ABK 1 0x80000000 6
AiM RD_MAC 12 0x80000001
W CFR 2 3
AiM AF 0x80000001
AiM RD_AF 13 0x80000001
AiM EWMUL 2 0x80000001 7
AiM COPY_BKGB 2 0x80000001 2 7
AiM COPY_GBBK 2 0x80000001 1 8
AiM EOC
"""
        self.assertEqual(render_aim_trace(program), expected)

    def test_single_instruction_renderer_returns_all_control_records(self) -> None:
        """Return a CFR write and MAC record as one ordered tuple."""

        instruction = MacAllBanks(
            channels=CentChannelSet(channels=(0,)),
            operation_size=1,
            row=2,
            column=0,
            accumulation_register=0,
            operand_source=MacOperandSource.NEXT_BANK,
        )
        self.assertEqual(
            render_aim_instruction(instruction),
            ("W CFR 0 1", "AiM MAC_ABK 1 0x80000000 2"),
        )


class AimTraceCompatibilityTests(unittest.TestCase):
    """Reject geometry and semantics absent from AiM's trace ABI."""

    def test_requires_fixed_simulator_geometry(self) -> None:
        """Reject independently mismatched channels, banks, and burst width."""

        base = program_with(
            WriteBias(
                channels=CentChannelSet(channels=(0,)),
                source=CentSharedBufferAddress(slot=0),
            )
        )
        validate_aim_hardware(base)
        for hardware in (
            replace(base.hardware, num_channels=16),
            replace(base.hardware, num_banks=8),
            replace(base.hardware, burst_length=8),
        ):
            with self.subTest(hardware=hardware):
                changed = CentProgram(
                    hardware=hardware, instructions=base.instructions
                )
                with self.assertRaises(AimSimulatorCompatibilityError):
                    validate_aim_hardware(changed)

    def test_rejects_unencoded_columns_and_registers(self) -> None:
        """Fail instead of silently dropping target-invisible operands."""

        cases = (
            WriteGlobalBuffer(
                channels=CentChannelSet(channels=(0,)),
                operation_size=1,
                column=16,
                source=CentSharedBufferAddress(slot=0),
            ),
            ReadMac(
                channels=CentChannelSet(channels=(0,)),
                destination=CentSharedBufferAddress(slot=0),
                accumulation_register=1,
            ),
            WriteAllBanks(
                channel=0,
                row=0,
                column=16,
                source=CentSharedBufferAddress(slot=0),
                accumulation_register=0,
            ),
        )
        for instruction in cases:
            with self.subTest(instruction=instruction):
                with self.assertRaises(AimSimulatorCompatibilityError):
                    render_aim_instruction(instruction)

    def test_rejects_pnm_instruction_missing_from_simulator(self) -> None:
        """Keep PNM execution outside the DRAM timing simulator."""

        instruction = Accumulate(
            operation_size=1,
            destination=CentSharedBufferAddress(slot=0),
            source=CentSharedBufferAddress(slot=1),
        )
        with self.assertRaisesRegex(
            AimSimulatorCompatibilityError,
            "does not implement ACC",
        ):
            render_aim_trace(program_with(instruction))


if __name__ == "__main__":
    unittest.main()
