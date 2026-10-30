"""Tests for the CENT architecture in the DeviceSpec."""

import unittest
from dataclasses import replace

from vllm_cent.cent import CentHardwareSpec
from vllm_cent.device.cent import CentArchitecture
from vllm_cent.device.spec import MemorySpace, MemorySpaceSpec
from vllm_cent.ir import DType

# Paper-scale CENT geometry in bytes: 32 channels of 16 banks, 16384 rows of
# 2 KiB each, 256-bit (32-byte) bursts, a 2 KiB Global Buffer per channel and
# a 64 KiB Shared Buffer per device.
PAPER = CentArchitecture(
    channels=32,
    banks_per_channel=16,
    banks_per_group=4,
    rows_per_bank=16384,
    row_size=2048,
    burst_size=32,
    global_buffer_size=2048,
    shared_buffer_size=65536,
    accumulators_per_bank=32,
)


class ToCentHardwareTest(unittest.TestCase):
    """Byte sizes become the backend's per-value counts for one dtype."""

    def test_bf16_matches_the_existing_paper_scale_spec(self) -> None:
        """2 bytes per value: 2048-byte rows hold 1024 values, bursts 16.

        The Shared Buffer's 65536 bytes are 2048 slots of 32 bytes, whatever
        the dtype. This is the spec the compiler tests already use.
        """

        self.assertEqual(
            PAPER.to_cent_hardware(DType.BF16, sigmoid_function_id=0),
            CentHardwareSpec(
                num_channels=32,
                num_banks=16,
                dram_rows=16384,
                dram_columns=1024,
                burst_length=16,
                accumulator_slots_per_bank=32,
                sigmoid_activation_function_id=0,
                shared_buffer_slots=2048,
                global_buffer_columns=1024,
            ),
        )

    def test_value_counts_follow_the_dtype_width(self) -> None:
        """fp32 halves every value count and fp4 doubles it; slots do not change."""

        counts = {
            dtype: PAPER.to_cent_hardware(dtype, sigmoid_function_id=3)
            for dtype in (DType.FP32, DType.FP4)
        }
        self.assertEqual(
            [
                (s.dram_columns, s.burst_length, s.global_buffer_columns)
                for s in counts.values()
            ],
            [(512, 8, 512), (4096, 64, 4096)],
        )
        self.assertEqual({s.shared_buffer_slots for s in counts.values()}, {2048})
        self.assertEqual(counts[DType.FP32].sigmoid_activation_function_id, 3)

    def test_rejections(self) -> None:
        """Bank groups other than the backend's 4, or bursts of partial values."""

        cases = {
            "eight-bank groups": (replace(PAPER, banks_per_group=8), "groups of 4"),
            # A 1-byte burst is 8 bits: half a bf16 value.
            "partial value": (replace(PAPER, burst_size=1), "burst_size 1 bytes"),
        }
        for name, (architecture, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                architecture.to_cent_hardware(DType.BF16, sigmoid_function_id=0)


class CentArchitectureTest(unittest.TestCase):
    """The geometry checks itself and derives the device's storage."""

    def test_memory_spaces(self) -> None:
        """Paper scale: 512 banks of 32 MiB, 32 Global Buffers, one Shared Buffer.

        A bank holds 16384 rows of 2048 bytes = 32 MiB; 32 channels of 16
        banks give 512 banks. Every space moves 32-byte bursts.
        """

        self.assertEqual(
            PAPER.memory_spaces(),
            (
                MemorySpaceSpec(
                    space=MemorySpace.PIM_DRAM,
                    instance_size=16384 * 2048,
                    instances=512,
                    access_size=32,
                ),
                MemorySpaceSpec(
                    space=MemorySpace.PIM_GB,
                    instance_size=2048,
                    instances=32,
                    access_size=32,
                ),
                MemorySpaceSpec(
                    space=MemorySpace.PNM_SB,
                    instance_size=65536,
                    instances=1,
                    access_size=32,
                ),
            ),
        )

    def test_rejections(self) -> None:
        """Each geometry rule names the field it checks."""

        cases = {
            "33 channels": ({"channels": 33}, "channels must be between 1 and 32"),
            "row not whole bursts": (
                {"row_size": 2040},
                "row_size must be divisible",
            ),
            "global buffer not whole bursts": (
                {"global_buffer_size": 2049},
                "global_buffer_size must be divisible",
            ),
            "shared buffer not whole slots": (
                {"shared_buffer_size": 65536 + 16},
                "shared_buffer_size must be divisible",
            ),
            "partial bank group": (
                {"banks_per_channel": 6},
                "multiple of banks_per_group",
            ),
            "no accumulators": ({"accumulators_per_bank": 0}, "at least 1"),
        }
        for name, (fields, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                replace(PAPER, **fields)


if __name__ == "__main__":
    unittest.main()
