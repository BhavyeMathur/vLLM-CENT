"""Tests for the CENT instruction layer."""

import unittest
from dataclasses import replace

from vllm_cent.cent import (
    Accumulate,
    ApplyActivation,
    BroadcastCxl,
    CentChannelSet,
    CentHardwareSpec,
    CentInstruction,
    CentMemoryAddress,
    CentOpcode,
    CentProgram,
    CentSharedBufferAddress,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    Exponent,
    MacAllBanks,
    MacOperandSource,
    ReadActivation,
    ReadMac,
    ReadSingleBank,
    ReceiveCxl,
    Reduction,
    RunRiscV,
    SendCxl,
    WriteAllBanks,
    WriteBias,
    WriteGlobalBuffer,
    WriteSingleBank,
    render_instruction,
    render_text_program,
)
from vllm_cent.cent.instructions.validation import (
    _validate_accumulation_register,
    _validate_operation_span,
    _validate_row_column,
    _validate_shared_buffer_span,
    validate_address,
    validate_channels,
    validate_instruction,
    validate_shared_buffer_address,
)
from vllm_cent.cent.render import (
    _shared_buffer_operands,
    render_channel_mask,
)
from vllm_cent.cent.utils import require_nonnegative, require_positive


def hardware() -> CentHardwareSpec:
    """Create the small hardware target used by these tests.

    Returns:
        Two channels, four banks, eight rows, sixteen BF16 columns per row,
        four BF16 values per micro-operation, and eight Shared Buffer slots.
    """

    return CentHardwareSpec(
        num_channels=2,
        num_banks=4,
        dram_rows=8,
        dram_columns=16,
        burst_length=4,
        accumulator_slots_per_bank=2,
        sigmoid_activation_function_id=0,
        shared_buffer_slots=8,
    )


class _WriteSingleBankVariant(WriteSingleBank):
    """Test-only variant with the same instruction behavior."""


class PrimitiveValueTests(unittest.TestCase):
    """Test basic values used by CENT instructions."""

    def test_numeric_range_helpers_enforce_their_domains(self) -> None:
        """Reject values below each helper's allowed minimum."""

        require_nonnegative("field", 0)
        require_positive("field", 1)
        for helper, invalid in (
            (require_nonnegative, -1),
            (require_positive, 0),
        ):
            with self.subTest(helper=helper.__name__, invalid=invalid):
                with self.assertRaises(ValueError):
                    helper("field", invalid)

    def test_typed_addresses_preserve_paper_coordinates(self) -> None:
        """Keep DRAM and Shared Buffer address fields unchanged."""

        dram = CentMemoryAddress(channel=1, bank=2, row=3, column=4)
        shared = CentSharedBufferAddress(slot=5)

        self.assertEqual(
            dram,
            CentMemoryAddress(channel=1, bank=2, row=3, column=4),
        )
        self.assertEqual(shared, CentSharedBufferAddress(slot=5))
        for constructor in (
            lambda: CentMemoryAddress(channel=0, bank=0, row=0, column=-1),
            lambda: CentSharedBufferAddress(slot=-1),
        ):
            with self.assertRaises(ValueError):
                constructor()

    def test_channel_helpers_validate_semantic_masks(self) -> None:
        """Keep channel selections typed until rendering."""

        channels = CentChannelSet(channels=(0, 3))

        self.assertEqual(channels, CentChannelSet(channels=(0, 3)))
        self.assertEqual(channels, CentChannelSet(channels=(3, 0)))
        self.assertEqual(hash(channels), hash(CentChannelSet(channels=(3, 0))))
        # CHmask uses one bit per channel: channels 0 and 3 set binary 1001,
        # which is rendered as hexadecimal 0x9.
        self.assertEqual(render_channel_mask(channels), "0x9")
        for invalid in ((), (0, 0), (-1,)):
            with self.assertRaises(ValueError):
                CentChannelSet(channels=invalid)


class PaperInstructionTests(unittest.TestCase):
    """Test the instructions listed in paper Tables 2 and 3."""

    def test_renders_the_complete_paper_isa(self) -> None:
        """Render every instruction in the paper's operand order."""

        channels = CentChannelSet(channels=(0, 1))
        address = CentMemoryAddress(channel=1, bank=2, row=3, column=4)
        source = CentSharedBufferAddress(slot=5)
        destination = CentSharedBufferAddress(slot=6)
        # Every number is intentionally distinct. This makes swapping the
        # paper's CHid/OPsize/BK/RO/CO/Rs/Rd/Regid operands visible.
        cases = (
            (
                MacAllBanks(
                    channels=channels,
                    operation_size=2,
                    row=3,
                    column=4,
                    accumulation_register=1,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                ),
                "MAC_ABK 0x3 2 3 4 1",
            ),
            (
                ElementwiseMultiply(
                    channels=channels,
                    operation_size=2,
                    row=3,
                    column=4,
                ),
                "EW_MUL 0x3 2 3 4",
            ),
            (
                ApplyActivation(
                    channels=channels,
                    activation_function_id=7,
                    accumulation_register=1,
                ),
                "AF 0x3 7 1",
            ),
            (
                Exponent(
                    operation_size=2,
                    destination=destination,
                    source=source,
                ),
                "EXP 2 6 5",
            ),
            (
                Reduction(
                    operation_size=2,
                    destination=destination,
                    source=source,
                ),
                "RED 2 6 5",
            ),
            (
                Accumulate(
                    operation_size=2,
                    destination=destination,
                    source=source,
                ),
                "ACC 2 6 5",
            ),
            (
                RunRiscV(
                    operation_size=2,
                    program_counter=9,
                    destination=destination,
                    source=source,
                ),
                "RISCV 2 9 6 5",
            ),
            (
                SendCxl(
                    destination_device=3,
                    source=source,
                    destination=destination,
                ),
                "SEND_CXL 3 5 6",
            ),
            (ReceiveCxl(), "RECV_CXL"),
            (
                BroadcastCxl(
                    device_count=3,
                    source=source,
                    destination=destination,
                ),
                "BCAST_CXL 3 5 6",
            ),
            (
                WriteSingleBank(
                    address=address,
                    operation_size=2,
                    source=source,
                ),
                "WR_SBK 1 2 2 3 4 5",
            ),
            (
                ReadSingleBank(
                    address=address,
                    operation_size=2,
                    destination=destination,
                ),
                "RD_SBK 1 2 2 3 4 6",
            ),
            (
                WriteAllBanks(
                    channel=1,
                    row=3,
                    column=4,
                    source=source,
                    accumulation_register=1,
                ),
                "WR_ABK 1 3 4 5 1",
            ),
            (
                CopyBankToGlobalBuffer(
                    channels=channels,
                    operation_size=2,
                    bank=2,
                    row=3,
                    column=4,
                ),
                "COPY_BKGB 0x3 2 3 4",
            ),
            (
                CopyGlobalBufferToBank(
                    channels=channels,
                    operation_size=2,
                    bank=1,
                    row=3,
                    column=4,
                ),
                "COPY_GBBK 0x3 2 3 4",
            ),
            (WriteBias(channels=channels, source=source), "WR_BIAS 0x3 5"),
            (
                ReadMac(
                    channels=channels,
                    destination=destination,
                    accumulation_register=1,
                ),
                "RD_MAC 0x3 6 1",
            ),
            (
                ReadActivation(
                    channels=channels,
                    destination=destination,
                    accumulation_register=1,
                ),
                "RD_AF 0x3 6 1",
            ),
            (
                WriteGlobalBuffer(
                    channels=channels,
                    operation_size=2,
                    column=4,
                    source=source,
                ),
                "WR_GB 0x3 2 4 5",
            ),
        )

        # Tables 2 and 3 define 18 instructions. RD_AF is the additional AiM
        # target operation needed to retrieve activation-register results.
        self.assertEqual(len(cases), 19)
        for instruction, expected in cases:
            with self.subTest(instruction=type(instruction).__name__):
                validate_instruction(instruction, hardware())
                self.assertEqual(render_instruction(instruction), expected)
                self.assertEqual(instruction.opcode.value, expected.split()[0])

    def test_instruction_exposes_its_class_opcode(self) -> None:
        """Read an instruction's opcode from its concrete class constant."""

        instruction = ReceiveCxl()

        self.assertIs(ReceiveCxl.OPCODE, CentOpcode.RECEIVE_CXL)
        self.assertIs(instruction.opcode, ReceiveCxl.OPCODE)
        self.assertEqual(MacAllBanks.NEXT_BANK_FIRST_OPERAND_BANK, 0)
        self.assertEqual(MacAllBanks.NEXT_BANK_SECOND_OPERAND_BANK, 1)

    def test_local_validation_rejects_invalid_operand_ranges(self) -> None:
        """Reject invalid instruction operands during construction."""

        channels = CentChannelSet(channels=(0,))
        source = CentSharedBufferAddress(slot=0)
        invalid = (
            lambda: MacAllBanks(
                channels=channels,
                operation_size=0,
                row=0,
                column=0,
                accumulation_register=0,
                operand_source=MacOperandSource.GLOBAL_BUFFER,
            ),
            lambda: ApplyActivation(
                channels=channels,
                activation_function_id=-1,
                accumulation_register=0,
            ),
            lambda: RunRiscV(
                operation_size=1,
                program_counter=-1,
                destination=source,
                source=source,
            ),
            lambda: BroadcastCxl(
                device_count=0,
                source=source,
                destination=source,
            ),
            lambda: WriteAllBanks(
                channel=-1,
                row=0,
                column=0,
                source=source,
                accumulation_register=0,
            ),
        )
        for constructor in invalid:
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()


class HardwareValidationTests(unittest.TestCase):
    """Test validation that depends on hardware dimensions."""

    def test_hardware_spec_validates_every_local_dimension_rule(self) -> None:
        """Reject invalid hardware dimensions and activation IDs."""

        target = hardware()
        # Each case changes one field on the valid target. Channels must be
        # 1..32, banks must be a multiple of four, and sizes must be positive.
        # A row must hold whole bursts, and an AFid cannot be negative.
        invalid_factories = (
            lambda: replace(target, num_channels=0),
            lambda: replace(target, num_channels=33),
            lambda: replace(target, num_banks=5),
            lambda: replace(target, dram_rows=0),
            lambda: replace(target, dram_columns=0),
            lambda: replace(target, burst_length=0),
            lambda: replace(target, dram_columns=15),
            lambda: replace(target, accumulator_slots_per_bank=0),
            lambda: replace(target, sigmoid_activation_function_id=-1),
            lambda: replace(target, shared_buffer_slots=0),
        )
        for factory in invalid_factories:
            with self.subTest(factory=factory):
                with self.assertRaises(ValueError):
                    factory()

    def test_instruction_subclasses_use_parent_rules(self) -> None:
        """Validate and render an instruction subclass like its parent."""

        instruction = _WriteSingleBankVariant(
            address=CentMemoryAddress(channel=0, bank=1, row=2, column=4),
            operation_size=2,
            source=CentSharedBufferAddress(slot=3),
        )

        validate_instruction(instruction, hardware())
        self.assertEqual(
            render_instruction(instruction),
            "WR_SBK 0 2 1 2 4 3",
        )

    def test_validates_each_address_space_and_operation_span(self) -> None:
        """Accept the last valid address in each address space."""

        target = hardware()
        # These are the last valid channel, bank, row, column, Shared Buffer
        # slot, and register in the fake target. A four-value operation at
        # column 12 ends at column 16. Two slots starting at 6 end at slot 8.
        validate_address(
            CentMemoryAddress(channel=1, bank=3, row=7, column=15),
            target,
        )
        validate_channels(CentChannelSet(channels=(0, 1)), target)
        validate_shared_buffer_address(CentSharedBufferAddress(slot=7), target)
        _validate_row_column(7, 15, target)
        _validate_operation_span(1, 12, target)
        _validate_shared_buffer_span(CentSharedBufferAddress(slot=6), 2, target)
        _validate_accumulation_register(1, target)

        # Each invalid case moves one step past a limit. The two-operation
        # spans also overflow the DRAM row or Shared Buffer.
        invalid_calls = (
            lambda: validate_address(
                CentMemoryAddress(channel=2, bank=0, row=0, column=0),
                target,
            ),
            lambda: validate_channels(CentChannelSet(channels=(2,)), target),
            lambda: validate_shared_buffer_address(
                CentSharedBufferAddress(slot=8), target
            ),
            lambda: _validate_row_column(8, 0, target),
            lambda: _validate_operation_span(2, 12, target),
            lambda: _validate_shared_buffer_span(
                CentSharedBufferAddress(slot=7), 2, target
            ),
            lambda: _validate_accumulation_register(2, target),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()

    def test_instruction_validation_checks_role_specific_capacities(self) -> None:
        """Check the hardware limits for each kind of operand."""

        channels = CentChannelSet(channels=(0,))
        slot = CentSharedBufferAddress(slot=0)
        # Each otherwise-valid instruction violates the capacity associated
        # with one operand: WR_SBK and WR_GB cross the 16-column row, WR_ABK
        # selects nonexistent channel 2, RD_MAC selects nonexistent register 2,
        # and EXP crosses the eight-slot Shared Buffer from slot 7.
        invalid = (
            WriteSingleBank(
                address=CentMemoryAddress(
                    channel=0,
                    bank=0,
                    row=0,
                    column=12,
                ),
                operation_size=2,
                source=slot,
            ),
            WriteAllBanks(
                channel=2,
                row=0,
                column=0,
                source=slot,
                accumulation_register=0,
            ),
            ReadMac(
                channels=channels,
                destination=slot,
                accumulation_register=2,
            ),
            Exponent(
                operation_size=2,
                destination=CentSharedBufferAddress(slot=7),
                source=slot,
            ),
            WriteGlobalBuffer(
                channels=channels,
                operation_size=2,
                column=12,
                source=slot,
            ),
        )
        for instruction in invalid:
            with self.subTest(instruction=instruction):
                with self.assertRaises(ValueError):
                    validate_instruction(instruction, hardware())


class ProgramAndRenderingTests(unittest.TestCase):
    """Test CENT programs and assembly rendering."""

    def test_program_is_a_nonempty_sequence_of_paper_instructions(self) -> None:
        """Require at least one instruction in a program."""

        instruction = ReceiveCxl()
        program = CentProgram(hardware=hardware(), instructions=(instruction,))
        self.assertEqual(program.hardware, hardware())
        self.assertEqual(program.instructions, (instruction,))
        with self.assertRaises(ValueError):
            CentProgram(hardware=hardware(), instructions=())

    def test_program_renderer_serializes_every_program_instruction(self) -> None:
        """Render every program instruction in order."""

        instruction = WriteBias(
            channels=CentChannelSet(channels=(0,)),
            source=CentSharedBufferAddress(slot=3),
        )
        program = CentProgram(
            hardware=hardware(),
            instructions=(instruction, ReceiveCxl()),
        )

        # Channel 0 renders as CHmask 0x1 and Rs is Shared Buffer slot 3. The
        # renderer preserves instruction order and terminates each with '\n'.
        self.assertEqual(
            render_text_program(program),
            "WR_BIAS 0x1 3\nRECV_CXL\n",
        )

    def test_private_shared_buffer_renderer_uses_rd_then_rs(self) -> None:
        """Render PNM operands as OPsize, Rd, then Rs."""

        instruction = Accumulate(
            operation_size=2,
            destination=CentSharedBufferAddress(slot=6),
            source=CentSharedBufferAddress(slot=5),
        )
        # Table 2 orders these operands as OPsize, destination Rd, then source
        # Rs, hence two operations followed by slots 6 and 5.
        self.assertEqual(_shared_buffer_operands(instruction), "2 6 5")

    def test_renderer_rejects_an_instruction_without_an_opcode(self) -> None:
        """Report an unsupported base instruction with the documented error."""

        with self.assertRaisesRegex(TypeError, "CentInstruction"):
            render_instruction(CentInstruction())


if __name__ == "__main__":
    unittest.main()
