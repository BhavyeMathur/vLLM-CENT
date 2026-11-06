"""Tests for CENT Tile IR PhysicalLayout: mixed-radix decomposition, locate, and allocate."""

from __future__ import annotations

import unittest
from dataclasses import replace

from vllm_cent.device import CentArchitecture, MemorySpace, load_device_spec
from vllm_cent.ir import DType, Value
from vllm_cent.ir.tile import CentTensorType
from vllm_cent.tile import (
    Bank,
    Channel,
    Column,
    DramElement,
    DramTile,
    Lane,
    Mac,
    Row,
    SbElement,
    SbTile,
    Serial,
    Slot,
    allocate,
    resolve,
    split,
)


def _paper_arch() -> CentArchitecture:
    spec = load_device_spec("examples/cent/cent.yaml")
    assert isinstance(spec.architecture, CentArchitecture)
    return spec.architecture


def _cent(
    space: MemorySpace, shape: tuple[int, ...], dtype: DType = DType.BF16
) -> CentTensorType:
    return CentTensorType(shape=shape, dtype=dtype, memory_space=space)


class PaperGemvLayoutTest(unittest.TestCase):
    """Verify the GEMV layouts of examples/Tile_DSL/gemv.py on the paper spec."""

    def setUp(self) -> None:
        """Resolve the four GEMV layouts on the paper spec."""
        self.arch = _paper_arch()
        self.x_type = _cent(MemorySpace.PNM_SB, (4096,))
        self.w_type = _cent(MemorySpace.PIM_DRAM, (4096, 4096))
        self.y_type = _cent(MemorySpace.PNM_SB, (4096,))
        self.zeros_type = _cent(MemorySpace.PNM_SB, (16,))

        self.x_layout = resolve(
            split(k=(Serial("k_tile", 4), Slot(), Lane())),
            self.x_type,
            self.arch,
        )
        self.w_layout = resolve(
            split(
                n=(Serial("n_wave", 8), Channel(), Bank()),
                k=(Row("k_row", 4), Column()),
            ),
            self.w_type,
            self.arch,
        )
        self.y_layout = resolve(
            split(n=(Serial("n_wave", 8), Slot(of=Channel()), Lane(of=Bank()))),
            self.y_type,
            self.arch,
        )
        self.zeros_layout = resolve(
            split(b=(Slot(1), Lane(of=Bank()))),
            self.zeros_type,
            self.arch,
        )

    def test_resolved_extents_and_strides(self) -> None:
        """Each factor gets its concrete extent and within-dimension stride."""

        self.assertEqual(self.x_layout.tile_shape, (64, 16))
        self.assertEqual(self.x_layout.footprint(), 256)

        self.assertEqual(self.w_layout.tile_shape, (32, 16, 1024))
        self.assertEqual(self.w_layout.footprint(), 32)
        n_strides = [rf.stride for rf in self.w_layout.factors if rf.dim == 0]
        k_strides = [rf.stride for rf in self.w_layout.factors if rf.dim == 1]
        self.assertEqual(n_strides, [512, 16, 1])
        self.assertEqual(k_strides, [1024, 1])

        self.assertEqual(self.y_layout.tile_shape, (32, 16))
        self.assertEqual(self.y_layout.footprint(), 256)

        self.assertEqual(self.zeros_layout.tile_shape, (1, 16))
        self.assertEqual(self.zeros_layout.footprint(), 1)

    def test_locate_tiles(self) -> None:
        """Tile-level address formulas match the Step 3 specification."""

        self.assertEqual(
            self.w_layout.locate({"n_wave": 3, "k_row": 2}),
            DramTile(row=14, opsize=64),
        )
        self.assertEqual(
            self.y_layout.locate({"n_wave": 5}),
            SbTile(first_slot=160, slots=32),
        )
        self.assertEqual(
            self.x_layout.locate({"k_tile": 2}),
            SbTile(first_slot=128, slots=64),
        )
        self.assertEqual(
            self.zeros_layout.locate({}),
            SbTile(first_slot=0, slots=1),
        )

    def test_locate_element_bijection(self) -> None:
        """Consecutive n values fill banks 0..15, then channels 0..31, then waves 0..7."""

        # First 16 columns go to channel 0, banks 0..15.
        for b in range(16):
            self.assertEqual(
                self.w_layout.locate_element((b, 0)),
                DramElement(channel=0, bank=b, row=0, col=0),
            )
        # n = 16 advances to channel 1, bank 0.
        self.assertEqual(
            self.w_layout.locate_element((16, 0)),
            DramElement(channel=1, bank=0, row=0, col=0),
        )
        # n = 512 (after all 32 * 16 banks) advances to n_wave = 1 (row = 4), channel 0, bank 0.
        self.assertEqual(
            self.w_layout.locate_element((512, 0)),
            DramElement(channel=0, bank=0, row=4, col=0),
        )
        # n = 3*512 + 5*16 + 7, k = 2*1024 + 99 -> n_wave=3, ch=5, bank=7, k_row=2 (row=14), col=99.
        n = 3 * 512 + 5 * 16 + 7
        k = 2 * 1024 + 99
        self.assertEqual(
            self.w_layout.locate_element((n, k)),
            DramElement(channel=5, bank=7, row=14, col=99),
        )
        # In Shared Buffer, y[n] lands in slot = 3*32 + 5, lane = 7 (matching channel 5, bank 7).
        self.assertEqual(
            self.y_layout.locate_element((n,)),
            SbElement(slot=3 * 32 + 5, lane=7),
        )

    def test_allocate_paper_gemv(self) -> None:
        """Bump allocation places x at 0..255, y aligned at 256..511, zeros at 512."""

        vx = Value(type=self.x_type, name="x")
        vw = Value(type=self.w_type, name="w")
        vy = Value(type=self.y_type, name="y")
        vz = Value(type=self.zeros_type, name="zeros")

        allocated = allocate(
            {
                vx: self.x_layout,
                vw: self.w_layout,
                vy: self.y_layout,
                vz: self.zeros_layout,
            },
            self.arch,
        )
        self.assertEqual(allocated[vx].base, 0)
        self.assertEqual(allocated[vw].base, 0)
        self.assertEqual(allocated[vy].base, 256)
        self.assertEqual(allocated[vz].base, 512)


class LayoutValidationTest(unittest.TestCase):
    """Illegal layouts and allocations are rejected with descriptive errors."""

    def setUp(self) -> None:
        """Use the paper spec."""
        self.arch = _paper_arch()

    def test_rejects_wrong_rank_and_product(self) -> None:
        """A layout must have one factor list per dimension, multiplying to its size."""

        w = _cent(MemorySpace.PIM_DRAM, (4096, 4096))
        with (
            self.subTest(error="has rank 2"),
            self.assertRaisesRegex(ValueError, "has rank 2"),
        ):
            resolve(split(n=(Channel(), Bank(), Column())), w, self.arch)

        with (
            self.subTest(error="multiply to 2048, expected dimension size 4096"),
            self.assertRaisesRegex(
                ValueError, "multiply to 2048, expected dimension size 4096"
            ),
        ):
            resolve(
                split(
                    n=(Serial("n_wave", 4), Channel(), Bank()),
                    k=(Row("k_row", 4), Column()),
                ),
                w,
                self.arch,
            )

    def test_rejects_two_inferred_extents_in_one_dim(self) -> None:
        """At most one factor per dimension may leave its extent to be inferred."""

        x = _cent(MemorySpace.PNM_SB, (4096,))
        with (
            self.subTest(error="at most one is allowed per dimension"),
            self.assertRaisesRegex(ValueError, "at most one is allowed per dimension"),
        ):
            resolve(split(k=(Slot(), Slot(), Lane())), x, self.arch)

    def test_rejects_wrong_space_factors(self) -> None:
        """DRAM factors stay in DRAM layouts, Shared Buffer factors in SB layouts."""

        x = _cent(MemorySpace.PNM_SB, (4096,))
        w = _cent(MemorySpace.PIM_DRAM, (4096, 4096))

        with (
            self.subTest(error="Channel\\(\\) is only allowed in pim.dram"),
            self.assertRaisesRegex(
                ValueError, "Channel\\(\\) is only allowed in pim.dram"
            ),
        ):
            resolve(split(k=(Channel(), Slot(), Lane())), x, self.arch)

        with (
            self.subTest(error="Row\\('k'\\) is only allowed in pim.dram"),
            self.assertRaisesRegex(
                ValueError, "Row\\('k'\\) is only allowed in pim.dram"
            ),
        ):
            resolve(split(k=(Row("k", 4), Slot(), Lane())), x, self.arch)

        with (
            self.subTest(error="Slot\\(\\) is only allowed in pnm.sb"),
            self.assertRaisesRegex(ValueError, "Slot\\(\\) is only allowed in pnm.sb"),
        ):
            resolve(
                split(
                    n=(Serial("n_wave", 8), Channel(), Bank()),
                    k=(Slot(), Column(1024)),
                ),
                w,
                self.arch,
            )

    def test_rejects_bank_outer_to_channel_or_non_innermost_column(self) -> None:
        """Banks fill before channels; Column is innermost in its dimension."""

        w = _cent(MemorySpace.PIM_DRAM, (4096, 4096))
        with (
            self.subTest(error="Channel\\(\\) must be outer to Bank\\(\\)"),
            self.assertRaisesRegex(
                ValueError, "Channel\\(\\) must be outer to Bank\\(\\)"
            ),
        ):
            resolve(
                split(
                    n=(Serial("n_wave", 8), Bank(), Channel()),
                    k=(Row("k_row", 4), Column()),
                ),
                w,
                self.arch,
            )
        with (
            self.subTest(error="Column\\(\\) must be the innermost factor"),
            self.assertRaisesRegex(
                ValueError, "Column\\(\\) must be the innermost factor"
            ),
        ):
            resolve(
                split(
                    n=(Serial("n_wave", 8), Channel(), Bank()),
                    k=(Column(1024), Row("k_row", 4)),
                ),
                w,
                self.arch,
            )

    def test_rejects_column_exceeding_row_size(self) -> None:
        """A tile occupies at most one DRAM row per bank."""

        w = _cent(MemorySpace.PIM_DRAM, (512, 2048))
        with (
            self.subTest(error="exceeds one DRAM row"),
            self.assertRaisesRegex(ValueError, "exceeds one DRAM row"),
        ):
            resolve(
                split(n=(Channel(), Bank()), k=(Column(),)),
                w,
                self.arch,
            )

    def test_rejects_lane_width_mismatch(self) -> None:
        """Lane(of=Bank()) needs one value per bank in each burst."""

        # FP32 has 8 values per 32-byte burst, which does not match 16 banks.
        y_fp32 = _cent(MemorySpace.PNM_SB, (512,), dtype=DType.FP32)
        with (
            self.subTest(
                error="Lane extent \\(16\\) must equal values per burst \\(8\\)"
            ),
            self.assertRaisesRegex(
                ValueError, "Lane extent \\(16\\) must equal values per burst \\(8\\)"
            ),
        ):
            resolve(
                split(n=(Slot(of=Channel()), Lane(of=Bank()))),
                y_fp32,
                self.arch,
            )

    def test_allocate_alignment_and_overflow(self) -> None:
        """RD_MAC tiles are channel-aligned; the Shared Buffer cannot overflow."""

        z_type = _cent(MemorySpace.PNM_SB, (16,))
        y_type = _cent(MemorySpace.PNM_SB, (512,))
        vz = Value(type=z_type, name="zeros")
        vy = Value(type=y_type, name="y")

        z_layout = resolve(split(b=(Slot(1), Lane(of=Bank()))), z_type, self.arch)
        y_layout = resolve(
            split(n=(Slot(of=Channel()), Lane(of=Bank()))), y_type, self.arch
        )
        # zeros takes slot 0; y needs alignment to 32 channels so its base becomes 32.
        allocated = allocate({vz: z_layout, vy: y_layout}, self.arch)
        self.assertEqual(allocated[vz].base, 0)
        self.assertEqual(allocated[vy].base, 32)

        # Exceed Shared Buffer capacity (2048 slots on paper spec).
        huge_type = _cent(MemorySpace.PNM_SB, (2048 * 16,))
        vh = Value(type=huge_type, name="huge")
        huge_layout = resolve(split(k=(Slot(), Lane())), huge_type, self.arch)
        with (
            self.subTest(error="exceeding capacity"),
            self.assertRaisesRegex(ValueError, "exceeding capacity"),
        ):
            allocate({vz: z_layout, vh: huge_layout}, self.arch)


class MultiChannelLayoutTest(unittest.TestCase):
    """Several Channel factors split the channel number in mixed radix."""

    def setUp(self) -> None:
        """Lay out q_proj's w^T (K, N) as two channel groups over K."""

        self.arch = _paper_arch()
        # Channels 0-15 hold the first half of k, channels 16-31 the second;
        # within a group, n fills banks, then channels, then waves.
        self.wt = resolve(
            split(
                k=(Channel(2), Row("k_row", 4), Column()),
                n=(Serial("n_wave", 16), Channel(16), Bank()),
            ),
            _cent(MemorySpace.PIM_DRAM, (8192, 4096)),
            self.arch,
        )

    def test_first_channel_factor_is_most_significant(self) -> None:
        """The channel number is group * 16 + c: layout order, high digit first."""

        k = 4096 + 5  # group 1, k_row 0, column 5
        n = 3 * 256 + 2 * 16 + 7  # n_wave 3, channel 2 of the group, bank 7
        self.assertEqual(
            self.wt.locate_element((k, n)),
            DramElement(channel=18, bank=7, row=3, col=5),
        )
        self.assertEqual([rf.extent for rf in self.wt.channel_factors], [2, 16])
        self.assertEqual(self.wt.tile_shape, (2, 1024, 16, 16))

    def test_one_channel_extent_may_be_inferred(self) -> None:
        """Channel() next to Channel(2) takes the remaining 16 channels."""

        inferred = resolve(
            split(
                k=(Channel(2), Row("k_row", 4), Column()),
                n=(Serial("n_wave", 16), Channel(), Bank()),
            ),
            _cent(MemorySpace.PIM_DRAM, (8192, 4096)),
            self.arch,
        )
        self.assertEqual(
            [(rf.dim, rf.extent, rf.stride) for rf in inferred.factors],
            [(rf.dim, rf.extent, rf.stride) for rf in self.wt.factors],
        )

    def test_shared_buffer_slots_follow_the_channel_number(self) -> None:
        """Split Slot(of=Channel) factors put channel c in slot Rd + c."""

        partial = resolve(
            split(
                p=(Slot(of=Channel(2)),),
                n=(Serial("n_wave", 16), Slot(of=Channel(16)), Lane(of=Bank())),
            ),
            _cent(MemorySpace.PNM_SB, (2, 4096)),
            self.arch,
        )
        n = 3 * 256 + 2 * 16 + 7
        # Group 1 of wave 3: Rd = 3 * 32, channel 18, bank 7.
        self.assertEqual(
            partial.locate_element((1, n)), SbElement(slot=3 * 32 + 18, lane=7)
        )
        self.assertEqual(partial.locate({"n_wave": 3}), SbTile(first_slot=96, slots=32))
        self.assertTrue(partial.needs_channel_alignment)

    def test_rejects_bad_channel_splits(self) -> None:
        """The channel factors must multiply to the device's channel count."""

        wt = _cent(MemorySpace.PIM_DRAM, (8192, 4096))
        cases = (
            (
                (Channel(2), Row("k_row", 4), Column()),
                (Serial("n_wave", 32), Channel(8), Bank()),
                r"Channel extents \[2, 8\] multiply to 16, expected 32",
            ),
            (
                (Channel(), Row("k_row", 4), Column()),
                (Serial("n_wave", 16), Channel(), Bank()),
                "2 Channel factors omit their extent",
            ),
            (
                (Channel(3), Row("k_row", 4), Column()),
                (Serial("n_wave", 16), Channel(), Bank()),
                "extents 3 do not divide the 32 channels",
            ),
        )
        for k, n, message in cases:
            with (
                self.subTest(error=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                resolve(split(k=k, n=n), wt, self.arch)
        with (
            self.subTest(error="Slot extent differs"),
            self.assertRaisesRegex(
                ValueError, "Slot extent 4 differs from its Channel extent 2"
            ),
        ):
            Slot(4, of=Channel(2))


class MacLayoutTest(unittest.TestCase):
    """A Mac factor lays weights out for a MAC array wider than one output."""

    def setUp(self) -> None:
        """A paper-scale device whose banks have 32-output MAC arrays."""

        self.arch = _paper_arch()
        self.wide = replace(self.arch, mac_array_width=32)

    def test_mac_and_column_share_the_row_in_layout_order(self) -> None:
        """Mac before Column: col = mac * 32 + column; after it: column * 32 + mac."""

        n = 5 * 512 + 7 * 32 + 9  # channel 5, bank 7, array output 9
        k = 32 + 3  # k_row 1, column 3
        w = resolve(
            split(n=(Channel(), Bank(), Mac()), k=(Row("k_row", 2), Column(32))),
            _cent(MemorySpace.PIM_DRAM, (16384, 64)),
            self.wide,
        )
        self.assertEqual(
            w.locate_element((n, k)),
            DramElement(channel=5, bank=7, row=1, col=9 * 32 + 3, mac=9),
        )
        # One row of 32 x 32 values per bank: 64 bursts.
        self.assertEqual(w.locate({"k_row": 1}), DramTile(row=1, opsize=64))
        self.assertEqual(w.mac_width, 32)

        wt = resolve(
            split(k=(Row("k_row", 2), Column(32)), n=(Channel(), Bank(), Mac())),
            _cent(MemorySpace.PIM_DRAM, (64, 16384)),
            self.wide,
        )
        self.assertEqual(
            wt.locate_element((k, n)),
            DramElement(channel=5, bank=7, row=1, col=3 * 32 + 9, mac=9),
        )

    def test_aim_mac_is_one_wide(self) -> None:
        """On AiM, Mac() has extent 1 and changes no address."""

        w = _cent(MemorySpace.PIM_DRAM, (4096, 4096))
        k = (Row("k_row", 4), Column())
        with_mac = resolve(
            split(n=(Serial("n_wave", 8), Channel(), Bank(), Mac()), k=k),
            w,
            self.arch,
        )
        without = resolve(
            split(n=(Serial("n_wave", 8), Channel(), Bank()), k=k), w, self.arch
        )
        self.assertEqual(with_mac.mac_width, 1)
        for coords in ((0, 0), (1000, 3000), (4095, 4095)):
            self.assertEqual(
                with_mac.locate_element(coords), without.locate_element(coords)
            )

    def test_rejects_bad_mac_factors(self) -> None:
        """Mac is DRAM-only, matches the device, appears once and fits a row."""

        two_wide = replace(self.arch, mac_array_width=2)
        cases = (
            (
                split(n=(Serial("n_wave", 8), Channel(), Bank(), Mac(32))),
                _cent(MemorySpace.PIM_DRAM, (4096 * 32,)),
                self.arch,
                r"Mac extent 32 does not match the device's MAC array width \(1\)",
            ),
            (
                split(k=(Slot(), Mac(), Lane())),
                _cent(MemorySpace.PNM_SB, (4096,)),
                self.arch,
                r"Mac\(\) is only allowed in pim.dram",
            ),
            (
                split(n=(Mac(), Channel(), Bank(), Mac()), k=(Column(32),)),
                _cent(MemorySpace.PIM_DRAM, (2048, 32)),
                two_wide,
                "Mac\\(\\) at most once, got 2",
            ),
            (
                split(n=(Channel(), Bank(), Mac()), k=(Column(64),)),
                _cent(MemorySpace.PIM_DRAM, (16384, 64)),
                self.wide,
                "Mac x Column extent 32 x 64 exceeds one DRAM row",
            ),
        )
        for spec, tensor, arch, message in cases:
            with (
                self.subTest(error=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                resolve(spec, tensor, arch)


if __name__ == "__main__":
    unittest.main()
