"""Tests for bank-level CENT tile ops and the layouts they need.

Bank groups (``Bank(every, offset)``), column blocks, partial channels,
DRAM reads and writes, MAC between neighbor banks, EWMUL, bank copies,
activation readout, and their verifier rules.
"""

from __future__ import annotations

import unittest
from pathlib import Path

from vllm_cent.device import CentArchitecture, MemorySpace, load_device_spec
from vllm_cent.ir import DType, Value
from vllm_cent.ir.tile import CentEwmulOp, CentGbConcatOp, CentTensorType, CentWrMemOp
from vllm_cent.tile import (
    Bank,
    Channel,
    Column,
    ColumnBlock,
    DramElement,
    DramTile,
    Lane,
    Row,
    Slot,
    resolve,
    split,
)
from vllm_cent.tile.codegen import iter_aim_records
from vllm_cent.tile.dsl import TileKernel, TileValue


def _paper() -> CentArchitecture:
    arch = load_device_spec(Path("examples/cent/cent_paper.yaml")).architecture
    assert isinstance(arch, CentArchitecture)
    return arch


ARCH = _paper()
ALL = tuple(range(32))


def _dram(shape: tuple[int, ...]) -> CentTensorType:
    return CentTensorType(
        shape=shape, dtype=DType.BF16, memory_space=MemorySpace.PIM_DRAM
    )


class BankLayoutTest(unittest.TestCase):
    """Bank groups, column blocks and partial channels place elements."""

    def test_bank_group_member(self) -> None:
        """Chunk b of a Bank(every=4, offset=2) vector is in bank 4b' + 2."""

        layout = resolve(
            split(n=(Channel(), Bank(every=4, offset=2), Column())),
            _dram((8192,)),
            ARCH,
        )
        # 64 values per bank: element 5 * 64 is chunk 5: channel 1, group 1.
        self.assertEqual(
            layout.locate_element((5 * 64 + 3,)),
            DramElement(channel=1, bank=6, row=0, col=3),
        )
        self.assertEqual(
            sorted(layout.occupied_cells())[:4], [(0, 2), (0, 6), (0, 10), (0, 14)]
        )

    def test_column_blocks_share_a_row(self) -> None:
        """Eight 128-value keys per row: the block index is the column offset."""

        layout = resolve(
            split(
                s=(Row("wave", 1), ColumnBlock("segment", 8), Channel(), Bank()),
                d=(Column(128),),
            ),
            _dram((2304, 128)),
            ARCH,
        )
        # Position 2303: segment 4, flat bank 255 (channel 15, bank 15).
        self.assertEqual(
            layout.locate_element((2303, 7)),
            DramElement(channel=15, bank=15, row=0, col=4 * 128 + 7),
        )
        # Segment 4 holds 256 positions: only channels 0-15.
        tile = layout.locate({"wave": 0, "segment": 4})
        self.assertEqual(tile, DramTile(row=0, opsize=8, column=512))
        opsizes = layout.channel_opsizes({"wave": 0, "segment": 4})
        self.assertEqual({c for c, n in opsizes.items() if n}, set(range(16)))
        self.assertEqual(
            layout.channel_opsizes({"wave": 0, "segment": 5}), dict.fromkeys(ALL, 0)
        )

    def test_partial_last_channels(self) -> None:
        """A context split over channel groups: the last groups hold less, or none."""

        layout = resolve(
            split(s=(Channel(4), Column(1024)), d=(Channel(), Bank())),
            _dram((2304, 128)),
            ARCH,
        )
        opsizes = layout.channel_opsizes({})
        self.assertEqual({opsizes[c] for c in range(16)}, {64})
        self.assertEqual({opsizes[c] for c in range(16, 24)}, {16})
        self.assertEqual({opsizes[c] for c in range(24, 32)}, {0})

    def test_rejects_bad_bank_groups(self) -> None:
        """The group must divide the banks, and the member lie inside it."""

        with self.assertRaisesRegex(ValueError, "0 <= offset < every"):
            Bank(every=4, offset=4)
        with self.assertRaisesRegex(ValueError, "does not divide the 16 banks"):
            resolve(split(n=(Channel(), Bank(every=3), Column())), _dram((8192,)), ARCH)


def _vector(k: TileKernel, name: str, row: int, every: int, g: int) -> TileValue:
    t = k.tensor(name, (8192,), DType.BF16, MemorySpace.PIM_DRAM)
    k.bind_layout(
        t, split(n=(Channel(), Bank(every=every, offset=g), Column())), base=row
    )
    return t


def _sb(k: TileKernel, name: str, n: int = 8192) -> TileValue:
    t = k.tensor(name, (n,), DType.BF16, MemorySpace.PNM_SB)
    k.bind_layout(t, split(n=(Slot(), Lane())))
    return t


class BankOpTest(unittest.TestCase):
    """The ops expand into CENT's records and the verifier checks banks."""

    def test_ewmul_round_trip(self) -> None:
        """W MEM burst by burst, EWMUL, a bank copy, R MEM."""

        with TileKernel("ew", ARCH) as k:
            x, y = _sb(k, "x"), _sb(k, "y")
            a, b, c = (_vector(k, n, 1, 4, g) for g, n in enumerate("abc"))
            d = _vector(k, "d", 2, 4, 1)
            a = k.wr_mem(x, into=a)
            b = k.wr_mem(x, into=b)
            c = k.ewmul(a, b, into=c, channels=ALL)
            gb = k.copy_bkgb(c, bank=2, channels=ALL)
            d = k.copy_gbbk(gb, into=d, bank=1, channels=ALL)
            y = k.rd_mem(c, into=y)
            k.sync()
            k.returns(y, d)
        lines = list(iter_aim_records(k.build(), ARCH))
        # 64 values per bank: 4 bursts x 128 banks per write.
        self.assertEqual(lines[:2], ["W MEM 0 0 1", "W MEM 0 4 1"])
        self.assertEqual(lines[128], "W MEM 0 0 1")  # the second burst
        self.assertEqual(lines[1024], "AiM EWMUL 4 0xffffffff 1")
        self.assertEqual(
            lines[1025:1027],
            ["AiM COPY_BKGB 4 0xffffffff 2 1", "AiM COPY_GBBK 4 0xffffffff 1 2"],
        )
        self.assertEqual(lines[1027], "R MEM 0 2 1")
        self.assertEqual(lines[-1], "AiM SYNC")

    def test_mac_between_neighbors(self) -> None:
        """MAC_ABK between CFR writes that select the next bank and restore."""

        with TileKernel("sq", ARCH) as k:
            x = _sb(k, "x")
            sums = k.tensor("sums", (512,), DType.BF16, MemorySpace.PNM_SB)
            k.bind_layout(sums, split(n=(Slot(of=Channel()), Lane(of=Bank()))))
            p0, p1 = (_vector(k, f"p{g}", 0, 2, g) for g in range(2))
            p0, p1 = k.wr_mem(x, into=p0), k.wr_mem(x, into=p1)
            acc = k.wr_bias_zero(channels=ALL)
            acc = k.mac_bk_bk(p0, p1, acc, channels=ALL)
            k.returns(k.rd_mac(acc, into=sums, channels=ALL))
        kernel = k.build()
        sums_base = kernel.layouts[sums.value].base
        lines = list(iter_aim_records(kernel, ARCH))
        self.assertEqual(
            lines[-5:],
            [
                "AiM WR_BIAS 0 0xffffffff",
                "W CFR 0 0",
                "AiM MAC_ABK 2 0xffffffff 0",
                "W CFR 0 1",
                f"AiM RD_MAC {sums_base} 0xffffffff",
            ],
        )

    def test_verifier_rejects_misplaced_operands(self) -> None:
        """Rows, bank members, copies and shared storage are checked."""

        def ewmul_rows(k: TileKernel) -> None:
            a, b = _vector(k, "a", 1, 4, 0), _vector(k, "b", 2, 4, 1)
            c = _vector(k, "c", 1, 4, 2)
            k.returns(k.ewmul(a, b, into=c, channels=ALL))

        def ewmul_members(k: TileKernel) -> None:
            a, b = _vector(k, "a", 1, 4, 1), _vector(k, "b", 1, 4, 0)
            c = _vector(k, "c", 1, 4, 2)
            k.returns(k.ewmul(a, b, into=c, channels=ALL))

        def copy_bank(k: TileKernel) -> None:
            c = _vector(k, "c", 1, 4, 2)
            gb = k.copy_bkgb(c, bank=1, channels=ALL)
            d = _vector(k, "d", 2, 4, 1)
            k.returns(k.copy_gbbk(gb, into=d, bank=1, channels=ALL))

        def shared_storage(k: TileKernel) -> None:
            a = _vector(k, "a", 1, 2, 0)  # banks 0, 2, 4, ... of row 1
            b = _vector(k, "b", 1, 4, 0)  # banks 0, 4, 8, ... of row 1
            k.returns(a, b)

        cases = (
            (ewmul_rows, "must share one DRAM row"),
            (ewmul_members, r"must be in banks 4b \+ 0"),
            (copy_bank, "bank 1 holds no part of the operand"),
            (shared_storage, "in the same banks"),
        )
        for record, message in cases:
            with (
                self.subTest(error=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                with TileKernel("bad", ARCH) as k:
                    record(k)
                k.build()

    def test_mac_needs_one_operation_size(self) -> None:
        """A MAC over channels holding different bursts must be split."""

        with TileKernel("sv", ARCH) as k:
            cache = k.tensor("v", (2304, 128), DType.BF16, MemorySpace.PIM_DRAM)
            k.bind_layout(
                cache,
                split(s=(Channel(4), Column(1024)), d=(Channel(), Bank())),
                base=0,
            )
            p = _sb(k, "p", 1024)
            used = tuple(range(24))
            acc = k.wr_bias_zero(channels=used)
            gb = k.wr_gb(p, channels=used)
            acc = k.mac_abk(cache, gb, acc, channels=used)
            out = k.tensor("out", (512,), DType.BF16, MemorySpace.PNM_SB)
            k.bind_layout(out, split(n=(Slot(of=Channel()), Lane(of=Bank()))))
            k.returns(k.rd_mac(acc, into=out, channels=used))
        with self.assertRaisesRegex(ValueError, "issue one op per size"):
            list(iter_aim_records(k.build(), ARCH))

    def test_op_type_rules(self) -> None:
        """Bounds of a DRAM write, EWMUL shapes, and joining Global Buffers."""

        sb = Value(
            CentTensorType(
                shape=(128,), dtype=DType.BF16, memory_space=MemorySpace.PNM_SB
            )
        )
        cache = Value(_dram((2304, 128)))
        CentWrMemOp(operands=(sb, cache), start=(2303, 0))
        cases = (
            (
                lambda: CentWrMemOp(operands=(sb, cache), start=(2304, 0)),
                "does not fit",
            ),
            (
                lambda: CentEwmulOp(
                    operands=(cache, Value(_dram((128,))), cache), channels=ALL
                ),
                "must have one shape",
            ),
            (
                lambda: CentGbConcatOp(
                    operands=(
                        Value(
                            CentTensorType(
                                shape=(64,),
                                dtype=DType.BF16,
                                memory_space=MemorySpace.PIM_GB,
                            )
                        ),
                    )
                ),
                "at least two",
            ),
        )
        for build, message in cases:
            with (
                self.subTest(error=message),
                self.assertRaisesRegex(ValueError, message),
            ):
                build()


if __name__ == "__main__":
    unittest.main()
