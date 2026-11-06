"""Expand a verified CENT Tile IR kernel into typed CENT instructions.

This is the one owner of the physical expansion rules (v0.21 plan, section
7): every ``cent.*`` op becomes the CENT backend's typed instructions
(``cent.instructions``), with addresses from the kernel's layouts. Loops
are walked in order and every iteration is expanded, lazily, so a caller can
stream a long kernel without holding all its instructions.

Expansion rules, for the loop index values of one iteration:

- ``cent.wr_gb``: ``WriteGlobalBuffer`` from the source tile's first slot,
  ``OPsize`` = its slots (one burst each), column 0.
- ``cent.mac_abk``: one ``MacAllBanks``: the weight tile's row and
  ``OPsize``, column 0, register 0, operand from the Global Buffer.
- ``cent.wr_bias``: ``WriteBias`` from the source tile's first slot ``Rs``.
- ``cent.wr_bias_zero``: ``WriteBias`` from ``ZERO_GPR``, a GPR number the
  simulator requires but no kernel tensor occupies.
- ``cent.rd_mac``: ``ReadMac`` to the destination tile's first slot ``Rd``.
- ``cent.rd_af``: ``ApplyActivation`` then ``ReadActivation``, as ``RD_MAC``.
- ``cent.wr_mem`` / ``cent.rd_mem``: one ``W MEM`` / ``R MEM`` per DRAM burst
  the elements touch, burst by burst and, within a burst, bank by bank
  (channel-major), as CENT stores vectors.
- ``cent.mac_bk_bk``: ``MAC_ABK`` between neighbor banks, between two writes
  of configuration register 0 (0: next bank; back to 1: Global Buffer).
- ``cent.ewmul``, ``cent.copy_bkgb``, ``cent.copy_gbbk``: ``EWMUL``,
  ``COPY_BKGB``, ``COPY_GBBK`` on their operands' row.
- ``cent.sync``: ``SYNC``; ``cent.ewadd``: ``EWADD`` of two Shared Buffer
  tensors; ``cent.pnm``: a comment with its analytical cycles;
  ``cent.gb_concat``: nothing.

An op's ``OPsize`` is the bursts its operand holds in each selected channel;
where a partial tile holds fewer bursts in some channels, every channel an
op selects must hold the same number (one op per operation size).

Shared Buffer convention (ours, not a documented hardware rule; see
``examples/Tile_DSL/gemv.py``): for ``RD_MAC`` and ``WR_BIAS`` with
``stride = 1``, channel ``c`` uses slot ``Rd + c`` (``Rs + c``), where the
base is aligned to the channel count; ``WR_GB`` and a ``stride = 0``
``WR_BIAS`` give every channel the same slot. ``ChannelIssue`` (the run
spec's ``codegen.channel_issue``) chooses how these transfers between the
Shared Buffer and many channels are issued; ``MAC_ABK`` is always one
instruction:

- ``BROADCAST`` (default): one instruction with the whole channel mask and
  the base slot, as CENT's own traces (``../ALOI/cent_simulation``) do; the
  per-channel slots are implied by the convention.
- ``SERIALIZE``: one instruction per channel, each naming its own slot, so
  the trace alone gives every address. ``RD_MAC`` blocks the DRAM system,
  so this costs the most: on the AiM simulator a 1024x1024 GEMV takes 2899
  cycles instead of 700.
"""

from __future__ import annotations

import itertools
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from enum import StrEnum

from ..cent.hardware import CentHardwareSpec
from ..cent.instructions import (
    ApplyActivation,
    CentChannelSet,
    CentInstruction,
    CentSharedBufferAddress,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    MacAllBanks,
    MacOperandSource,
    ReadActivation,
    ReadMac,
    WriteBias,
    WriteGlobalBuffer,
    validate_instruction,
)
from ..device.cent import CentArchitecture
from ..ir.base import Operation, Value
from ..ir.tile import (
    CentCopyBkGbOp,
    CentCopyGbBkOp,
    CentEwaddOp,
    CentEwmulOp,
    CentGbConcatOp,
    CentMacAbkOp,
    CentMacBkBkOp,
    CentPnmOp,
    CentRdAfOp,
    CentRdMacOp,
    CentRdMemOp,
    CentSyncOp,
    CentTensorType,
    CentTileOp,
    CentWrBiasOp,
    CentWrBiasZeroOp,
    CentWrGbOp,
    CentWrMemOp,
    TileConstOp,
    TileForOp,
)
from .kernel import CentKernel
from .layout import DramElement, DramTile, PhysicalLayout, SbTile
from .verify import _analyze, verify_kernel

__all__ = [
    "ZERO_GPR",
    "ChannelIssue",
    "Comment",
    "ConfigWrite",
    "ElementwiseAdd",
    "MemAccess",
    "Sync",
    "TraceInstruction",
    "backend_hardware",
    "iter_instructions",
]

# AiM has one MAC register per bank, so every MAC uses register 0.
_REGISTER = 0

# The GPR a cent.wr_bias_zero names. The simulator needs a non-negative GPR
# (a negative one means "field missing") but ignores its value; CENT's own
# traces write 0. No kernel tensor occupies it.
ZERO_GPR = 0

# Configuration register 0 ("broadcast"): 1 = MAC with the Global Buffer,
# 0 = MAC with the next bank (AiM simulator README, test/all_isr.trace).
_MAC_SOURCE_REGISTER = 0

# The AFid of sigmoid, the only PIM activation table cent.yaml configures.
_SIGMOID_AFID = 0


@dataclass(frozen=True, slots=True)
class MemAccess:
    """An ordinary DRAM access of one burst: ``W MEM`` or ``R MEM``."""

    write: bool
    channel: int
    bank: int
    row: int


@dataclass(frozen=True, slots=True)
class Sync:
    """``SYNC``: wait for every issued command."""


@dataclass(frozen=True, slots=True)
class ElementwiseAdd:
    """``EWADD``: add two Shared Buffer vectors (a no-op in the AiM simulator)."""

    operation_size: int
    first: int
    second: int


@dataclass(frozen=True, slots=True)
class ConfigWrite:
    """A write of an AiM configuration register: ``W CFR``."""

    register: int
    value: int


@dataclass(frozen=True, slots=True)
class Comment:
    """A trace comment, e.g. the analytical latency of a PNM step."""

    text: str


# The ops that name a tile by indices.
_Indexed = CentWrGbOp | CentMacAbkOp | CentWrBiasOp | CentRdMacOp | CentRdAfOp

TraceInstruction = (
    CentInstruction | MemAccess | Sync | ElementwiseAdd | ConfigWrite | Comment
)


class ChannelIssue(StrEnum):
    """How ``WR_GB``, ``WR_BIAS`` and ``RD_MAC`` reach many channels.

    Members:
        BROADCAST: One instruction for all channels, naming the base slot.
        SERIALIZE: One instruction per channel, naming that channel's slot.
    """

    BROADCAST = "broadcast"
    SERIALIZE = "serialize"


def backend_hardware(kernel: CentKernel, arch: CentArchitecture) -> CentHardwareSpec:
    """The CENT backend's description of ``arch`` for the kernel's dtype.

    The instructions are checked against it with the backend's own rules.
    ``to_cent_hardware`` also needs sigmoid's ``AFid``; no instruction made
    here applies an activation, so it is never read and is set to 0.

    Raises:
        ValueError: If the kernel's tensors do not share one dtype.
    """

    dtypes = {argument.type.dtype for argument in kernel.function.arguments}
    if len(dtypes) != 1:
        raise ValueError(f"kernel tensors must share one dtype, got {sorted(dtypes)}")
    return arch.to_cent_hardware(dtypes.pop(), sigmoid_function_id=0)


def iter_instructions(
    kernel: CentKernel,
    arch: CentArchitecture,
    *,
    issue: ChannelIssue = ChannelIssue.BROADCAST,
) -> Iterator[TraceInstruction]:
    """Yield the kernel's instructions in execution order.

    The kernel is verified first, so every address the layouts give is in
    bounds; each backend instruction is also checked against the backend's
    rules.

    Args:
        kernel: The kernel to expand.
        arch: The device it runs on.
        issue: How ``WR_GB``, ``WR_BIAS``, ``RD_MAC`` and ``RD_AF`` reach many
            channels.

    Yields:
        Typed CENT instructions and the trace's own records.

    Raises:
        ValueError: If the kernel is not legal on ``arch``.
    """

    verify_kernel(kernel, arch)
    hardware = backend_hardware(kernel, arch)
    roots = _analyze(kernel.function, {}).root
    expander = _Expander(kernel, roots, issue, arch)
    for instruction in expander.block(kernel.function.operations, {}):
        if isinstance(instruction, CentInstruction):
            validate_instruction(instruction, hardware)
        yield instruction


class _Expander:
    """Walks a kernel's loops and expands each op for one iteration."""

    def __init__(
        self,
        kernel: CentKernel,
        roots: Mapping[Value, Value],
        issue: ChannelIssue,
        arch: CentArchitecture,
    ) -> None:
        self.kernel = kernel
        self.roots = roots
        self.issue = issue
        self.arch = arch

    def block(
        self, operations: tuple[Operation, ...], index: dict[Value, int]
    ) -> Iterator[TraceInstruction]:
        """Expand ``operations`` with the loop indices fixed in ``index``."""

        index = dict(index)
        for op in operations:
            if isinstance(op, TileForOp):
                for i in range(op.extent):
                    yield from self.block(op.body.operations, {**index, op.index: i})
            elif isinstance(op, TileConstOp):
                index[op.results[0]] = op.value
            else:
                yield from self.op(op, index)

    def op(
        self, op: Operation, index: Mapping[Value, int]
    ) -> Iterator[TraceInstruction]:
        """Expand one ``cent.*`` op; see the module docstring for the rules."""

        if isinstance(op, CentWrGbOp):
            tile = self._sb_tile(op.source, op, index)
            for channels, slot in self._transfer(op.channels, tile.first_slot, 0):
                yield WriteGlobalBuffer(
                    channels=channels,
                    operation_size=tile.slots,
                    column=0,
                    source=_slot(slot),
                )
        elif isinstance(op, CentMacAbkOp):
            weight = self._dram_tile(op.weight, op, index)
            yield MacAllBanks(
                channels=CentChannelSet(channels=op.channels),
                operation_size=self._opsize(op, op.weight, index),
                row=weight.row,
                column=weight.column,
                accumulation_register=_REGISTER,
                operand_source=MacOperandSource.GLOBAL_BUFFER,
            )
        elif isinstance(op, CentWrBiasOp):
            base = self._sb_tile(op.source, op, index).first_slot
            if op.stride == 1:
                base = self._layout(op.source).channel_base(base)
            for channels, slot in self._transfer(op.channels, base, op.stride):
                yield WriteBias(channels=channels, source=_slot(slot))
        elif isinstance(op, CentWrBiasZeroOp):
            for channels, slot in self._transfer(op.channels, ZERO_GPR, 0):
                yield WriteBias(channels=channels, source=_slot(slot))
        elif isinstance(op, CentRdMacOp):
            for channels, slot in self._readout(op, index):
                yield ReadMac(
                    channels=channels,
                    destination=_slot(slot),
                    accumulation_register=_REGISTER,
                )
        elif isinstance(op, CentRdAfOp):
            for channels, slot in self._readout(op, index):
                yield ApplyActivation(
                    channels=channels,
                    activation_function_id=_SIGMOID_AFID,
                    accumulation_register=_REGISTER,
                )
                yield ReadActivation(
                    channels=channels,
                    destination=_slot(slot),
                    accumulation_register=_REGISTER,
                )
        elif isinstance(op, CentWrMemOp):
            yield from self._mem(op.operands[1], op.operands[0], op.start, write=True)
        elif isinstance(op, CentRdMemOp):
            yield from self._mem(op.operands[0], op.operands[1], op.start, write=False)
        elif isinstance(op, CentMacBkBkOp):
            yield ConfigWrite(_MAC_SOURCE_REGISTER, 0)
            yield MacAllBanks(
                channels=CentChannelSet(channels=op.channels),
                operation_size=self._opsize(op, op.operands[0], {}),
                row=self._layout(op.operands[0]).base,
                column=0,
                accumulation_register=_REGISTER,
                operand_source=MacOperandSource.NEXT_BANK,
            )
            yield ConfigWrite(_MAC_SOURCE_REGISTER, 1)
        elif isinstance(op, CentEwmulOp):
            yield ElementwiseMultiply(
                channels=CentChannelSet(channels=op.channels),
                operation_size=self._opsize(op, op.operands[0], {}),
                row=self._layout(op.operands[0]).base,
                column=0,
            )
        elif isinstance(op, CentCopyBkGbOp):
            yield CopyBankToGlobalBuffer(
                channels=CentChannelSet(channels=op.channels),
                operation_size=self._opsize(op, op.operands[0], {}),
                bank=op.bank,
                row=self._layout(op.operands[0]).base,
                column=0,
            )
        elif isinstance(op, CentCopyGbBkOp):
            yield CopyGlobalBufferToBank(
                channels=CentChannelSet(channels=op.channels),
                operation_size=self._opsize(op, op.operands[1], {}),
                bank=op.bank,
                row=self._layout(op.operands[1]).base,
                column=0,
            )
        elif isinstance(op, CentSyncOp):
            yield Sync()
        elif isinstance(op, CentEwaddOp):
            first, second = (self._layout(v) for v in op.operands)
            yield ElementwiseAdd(first.footprint(), first.base, second.base)
        elif isinstance(op, CentPnmOp):
            yield Comment(f"PNM {op.kind}: {op.cycles:g} cycles (analytical, as CENT)")
        elif isinstance(op, CentGbConcatOp):
            return
        else:
            raise ValueError(f"cannot expand op {op.NAME}")

    def _readout(
        self, op: CentRdMacOp | CentRdAfOp, index: Mapping[Value, int]
    ) -> tuple[tuple[CentChannelSet, int], ...]:
        """The (mask, slot) pairs of an ``RD_MAC`` or ``RD_AF``: slot ``Rd + c``."""

        first = self._sb_tile(op.destination, op, index).first_slot
        base = self._layout(op.destination).channel_base(first)
        return self._transfer(op.channels, base, 1)

    def _transfer(
        self, channels: tuple[int, ...], base: int, stride: int
    ) -> tuple[tuple[CentChannelSet, int], ...]:
        """The (mask, slot) pairs of one transfer.

        Channel ``c`` uses slot ``base + stride * c``; a broadcast names
        only ``base``.
        """

        if self.issue is ChannelIssue.BROADCAST:
            return ((CentChannelSet(channels=channels), base),)
        return tuple(
            (CentChannelSet(channels=(channel,)), base + stride * channel)
            for channel in channels
        )

    def _opsize(self, op: CentTileOp, tensor: Value, index: Mapping[Value, int]) -> int:
        """The bursts ``tensor`` holds in each channel ``op`` selects.

        Raises:
            ValueError: If the channels hold different numbers, or none.
        """

        values = self._index_values(op, index) if isinstance(op, CentMacAbkOp) else {}
        opsizes = self._layout(tensor).channel_opsizes(values)
        sizes = {opsizes.get(channel, 0) for channel in op.channels}
        if len(sizes) != 1 or 0 in sizes:
            raise ValueError(
                f"{op.NAME}: channels {list(op.channels)} hold "
                f"{sorted(sizes)} bursts of the operand; issue one op per size"
            )
        return sizes.pop()

    def _index_values(self, op: _Indexed, index: Mapping[Value, int]) -> dict[str, int]:
        """The value of each named index of ``op`` in this iteration."""
        return {
            name: index[value] for name, value in zip(op.at, op.indices, strict=True)
        }

    def _mem(
        self, dram: Value, part: Value, start: tuple[int, ...], *, write: bool
    ) -> Iterator[MemAccess]:
        """One access per DRAM burst ``part`` touches at ``start``.

        Burst by burst, then bank by bank in channel-major order: CENT
        stores a vector's first burst in every bank, then its second.
        """

        layout = self._layout(dram)
        lead = len(start) - len(part.type.shape)
        bursts: dict[tuple[int, int, int, int], None] = {}
        for coords in itertools.product(*(range(n) for n in part.type.shape)):
            position = (
                *start[:lead],
                *(s + c for s, c in zip(start[lead:], coords, strict=True)),
            )
            element = layout.locate_element(position)
            assert isinstance(element, DramElement)
            burst = element.col // layout.burst_values
            bursts[(burst, element.channel, element.bank, element.row)] = None
        for _burst, channel, bank, row in sorted(bursts):
            yield MemAccess(write, channel, bank, row)

    def _dram_tile(
        self, tensor: Value, op: CentMacAbkOp, index: Mapping[Value, int]
    ) -> DramTile:
        tile = self._layout(tensor).locate(self._index_values(op, index))
        assert isinstance(tile, DramTile)
        return tile

    def _layout(self, tensor: Value) -> PhysicalLayout:
        """The layout of the kernel argument whose storage holds ``tensor``."""
        return self.kernel.layouts[self.roots[tensor]]

    def _sb_tile(
        self,
        tensor: Value,
        op: CentWrGbOp | CentWrBiasOp | CentRdMacOp | CentRdAfOp,
        index: Mapping[Value, int],
    ) -> SbTile:
        assert isinstance(tensor.type, CentTensorType)
        tile = self._layout(tensor).locate(self._index_values(op, index))
        assert isinstance(tile, SbTile)
        return tile


def _slot(slot: int) -> CentSharedBufferAddress:
    return CentSharedBufferAddress(slot=slot)
