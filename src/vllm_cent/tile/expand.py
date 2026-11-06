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

from collections.abc import Iterator, Mapping
from enum import StrEnum

from ..cent.hardware import CentHardwareSpec
from ..cent.instructions import (
    CentChannelSet,
    CentInstruction,
    CentSharedBufferAddress,
    MacAllBanks,
    MacOperandSource,
    ReadMac,
    WriteBias,
    WriteGlobalBuffer,
    validate_instruction,
)
from ..device.cent import CentArchitecture
from ..ir.base import Operation, Value
from ..ir.tile import (
    CentMacAbkOp,
    CentRdMacOp,
    CentTensorType,
    CentWrBiasOp,
    CentWrBiasZeroOp,
    CentWrGbOp,
    TileConstOp,
    TileForOp,
)
from .kernel import CentKernel
from .layout import DramTile, PhysicalLayout, SbTile
from .verify import _analyze, verify_kernel

__all__ = ["ZERO_GPR", "ChannelIssue", "backend_hardware", "iter_instructions"]

# AiM has one MAC register per bank, so every MAC uses register 0.
_REGISTER = 0

# The GPR a cent.wr_bias_zero names. The simulator needs a non-negative GPR
# (a negative one means "field missing") but ignores its value; CENT's own
# traces write 0. No kernel tensor occupies it.
ZERO_GPR = 0


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
) -> Iterator[CentInstruction]:
    """Yield the kernel's instructions in execution order.

    The kernel is verified first, so every address the layouts give is in
    bounds; each instruction is also checked against the backend's rules.

    Args:
        kernel: The kernel to expand.
        arch: The device it runs on.
        issue: How ``RD_MAC`` and ``WR_BIAS`` (stride 1) reach many channels.

    Yields:
        Typed CENT instructions.

    Raises:
        ValueError: If the kernel is not legal on ``arch``.
    """

    verify_kernel(kernel, arch)
    hardware = backend_hardware(kernel, arch)
    roots = _analyze(kernel.function, {}).root
    expander = _Expander(kernel, roots, issue)
    for instruction in expander.block(kernel.function.operations, {}):
        validate_instruction(instruction, hardware)
        yield instruction


class _Expander:
    """Walks a kernel's loops and expands each op for one iteration."""

    def __init__(
        self, kernel: CentKernel, roots: Mapping[Value, Value], issue: ChannelIssue
    ) -> None:
        self.kernel = kernel
        self.roots = roots
        self.issue = issue

    def block(
        self, operations: tuple[Operation, ...], index: dict[Value, int]
    ) -> Iterator[CentInstruction]:
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
    ) -> tuple[CentInstruction, ...]:
        """Expand one ``cent.*`` op; see the module docstring for the rules."""

        if isinstance(op, CentWrGbOp):
            tile = self._sb_tile(op.source, op, index)
            return tuple(
                WriteGlobalBuffer(
                    channels=channels,
                    operation_size=tile.slots,
                    column=0,
                    source=_slot(slot),
                )
                for channels, slot in self._transfer(op.channels, tile.first_slot, 0)
            )
        if isinstance(op, CentMacAbkOp):
            weight = self._tile(op.weight, op, index)
            assert isinstance(weight, DramTile)
            return (
                MacAllBanks(
                    channels=CentChannelSet(channels=op.channels),
                    operation_size=weight.opsize,
                    row=weight.row,
                    column=0,
                    accumulation_register=_REGISTER,
                    operand_source=MacOperandSource.GLOBAL_BUFFER,
                ),
            )
        if isinstance(op, CentWrBiasOp):
            base = self._sb_tile(op.source, op, index).first_slot
            if op.stride == 1:
                base = self._layout(op.source).channel_base(base)
            return tuple(
                WriteBias(channels=channels, source=_slot(slot))
                for channels, slot in self._transfer(op.channels, base, op.stride)
            )
        if isinstance(op, CentWrBiasZeroOp):
            return tuple(
                WriteBias(channels=channels, source=_slot(slot))
                for channels, slot in self._transfer(op.channels, ZERO_GPR, 0)
            )
        if isinstance(op, CentRdMacOp):
            first = self._sb_tile(op.destination, op, index).first_slot
            base = self._layout(op.destination).channel_base(first)
            return tuple(
                ReadMac(
                    channels=channels,
                    destination=_slot(slot),
                    accumulation_register=_REGISTER,
                )
                for channels, slot in self._transfer(op.channels, base, 1)
            )
        raise ValueError(f"cannot expand op {op.NAME}")

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

    def _tile(
        self,
        tensor: Value,
        op: CentWrGbOp | CentMacAbkOp | CentWrBiasOp | CentRdMacOp,
        index: Mapping[Value, int],
    ) -> DramTile | SbTile:
        """Locate the tile of ``tensor`` that ``op`` names in this iteration."""

        assert isinstance(tensor.type, CentTensorType)
        layout = self._layout(tensor)
        values = {
            name: index[value] for name, value in zip(op.at, op.indices, strict=True)
        }
        return layout.locate(values)

    def _layout(self, tensor: Value) -> PhysicalLayout:
        """The layout of the kernel argument whose storage holds ``tensor``."""
        return self.kernel.layouts[self.roots[tensor]]

    def _sb_tile(
        self,
        tensor: Value,
        op: CentWrGbOp | CentWrBiasOp | CentRdMacOp,
        index: Mapping[Value, int],
    ) -> SbTile:
        tile = self._tile(tensor, op, index)
        assert isinstance(tile, SbTile)
        return tile


def _slot(slot: int) -> CentSharedBufferAddress:
    return CentSharedBufferAddress(slot=slot)
