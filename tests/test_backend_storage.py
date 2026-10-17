"""Regression coverage for explicit Global Buffer capacity."""

import unittest
from dataclasses import replace

from vllm_cent.cent import (
    CentChannelSet,
    CentHardwareSpec,
    CentProgram,
    CentSharedBufferAddress,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    WriteGlobalBuffer,
)


class GlobalBufferCapacityTests(unittest.TestCase):
    """Keep Global Buffer limits independent of DRAM row widths."""

    def setUp(self) -> None:
        """Use a four-value Global Buffer beside an eight-value DRAM row."""
        self.hardware = CentHardwareSpec(
            num_channels=1,
            num_banks=4,
            dram_rows=2,
            dram_columns=8,
            burst_length=2,
            accumulator_slots_per_bank=1,
            sigmoid_activation_function_id=0,
            global_buffer_columns=4,
        )
        self.channels = CentChannelSet(channels=(0,))

    def test_capacity_must_hold_a_positive_whole_number_of_bursts(self) -> None:
        """Reject empty, negative, and partial-burst capacities."""
        for capacity in (0, -2, 3):
            with self.subTest(capacity=capacity), self.assertRaises(ValueError):
                replace(self.hardware, global_buffer_columns=capacity)

    def test_write_uses_global_capacity_independently_of_dram(self) -> None:
        """Accept the exact buffer end and reject the next complete burst."""
        # One two-value burst starting at scalar 2 ends exactly at capacity 4.
        # A second burst would end at 6 and must be rejected.
        instruction = WriteGlobalBuffer(
            channels=self.channels,
            column=2,
            operation_size=1,
            source=CentSharedBufferAddress(slot=0),
        )
        self.assertEqual(
            CentProgram(
                hardware=self.hardware, instructions=(instruction,)
            ).instructions,
            (instruction,),
        )
        with self.assertRaisesRegex(ValueError, "Global Buffer"):
            CentProgram(
                hardware=self.hardware,
                instructions=(replace(instruction, operation_size=2),),
            )
        # A larger Global Buffer can address columns beyond a DRAM row when no
        # DRAM operand participates in this Shared Buffer-to-Global Buffer copy.
        wide = replace(self.hardware, global_buffer_columns=12)
        wide_instruction = replace(instruction, column=10)
        self.assertEqual(
            CentProgram(hardware=wide, instructions=(wide_instruction,)).instructions,
            (wide_instruction,),
        )

    def test_bank_copies_must_fit_both_address_spaces(self) -> None:
        """Reject a transfer that fits DRAM but exceeds the Global Buffer."""
        # [2:4] fits both stores. Extending to [2:6] still fits the eight-value
        # DRAM row, but overflows the four-value Global Buffer in either direction.
        for instruction_type in (CopyBankToGlobalBuffer, CopyGlobalBufferToBank):
            instruction = instruction_type(
                channels=self.channels, bank=0, row=0, column=2, operation_size=1
            )
            with self.subTest(instruction_type=instruction_type):
                self.assertEqual(
                    CentProgram(
                        hardware=self.hardware, instructions=(instruction,)
                    ).instructions,
                    (instruction,),
                )
                with self.assertRaisesRegex(ValueError, "Global Buffer"):
                    CentProgram(
                        hardware=self.hardware,
                        instructions=(replace(instruction, operation_size=2),),
                    )
