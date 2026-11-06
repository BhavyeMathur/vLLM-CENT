"""Render a CENT Tile IR kernel as a trace for the AiM simulator.

The records follow CENT's own trace generator
(``../ALOI/cent_simulation/aim_sim.py``), which the AiM simulator
(``../ALOI/aim_simulator``) reads:

    AiM WR_GB <OPsize> <GPR> <channel_mask>
    AiM WR_BIAS <GPR> <channel_mask>
    AiM MAC_ABK <OPsize> <channel_mask> <row>
    AiM RD_MAC <GPR> <channel_mask>
    AiM EOC

Two differences from CENT's generator, neither of which changes timing:

- The GPR field holds the real Shared Buffer slot from the layouts, where
  CENT writes 0. The simulator does not use it.
- The trace starts with ``W CFR 0 1``: configuration register 0
  (``broadcast``) set to 1 selects the Global Buffer as the second MAC
  operand (AiM simulator README and ``test/all_isr.trace``; its GEMV traces
  start the same way). CENT's generator leaves it at the simulator's
  default. Every ``MAC_ABK`` here reads the Global Buffer, so it is written
  once.

The channel mask puts channel 0 in the most significant of 32 bits, as both
CENT's generator and the backend's ``render_aim_channel_mask`` do.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import TextIO

from ..cent.aim import (
    AIM_SIMULATOR_BANKS,
    AIM_SIMULATOR_BURST_LENGTH,
    AimSimulatorCompatibilityError,
    render_aim_channel_mask,
)
from ..cent.instructions import (
    CentInstruction,
    MacAllBanks,
    ReadMac,
    WriteBias,
    WriteGlobalBuffer,
)
from ..device.cent import CentArchitecture
from .expand import ChannelIssue, backend_hardware, iter_instructions
from .kernel import CentKernel

__all__ = [
    "MAC_FROM_GLOBAL_BUFFER",
    "aim_trace",
    "iter_aim_records",
    "iter_aim_trace",
    "render_aim_record",
    "write_aim_trace",
]

# CFR 0 is "broadcast": 1 = MAC with the Global Buffer, 0 = with the next bank.
MAC_FROM_GLOBAL_BUFFER = "W CFR 0 1"


def render_aim_record(instruction: CentInstruction) -> str:
    """Render one instruction as one AiM trace record.

    Raises:
        AimSimulatorCompatibilityError: For an instruction the Tile IR
            expansion does not make.
    """

    if isinstance(instruction, WriteGlobalBuffer):
        mask = render_aim_channel_mask(instruction.channels)
        return (
            f"AiM WR_GB {instruction.operation_size} {instruction.source.slot} {mask}"
        )
    if isinstance(instruction, WriteBias):
        mask = render_aim_channel_mask(instruction.channels)
        return f"AiM WR_BIAS {instruction.source.slot} {mask}"
    if isinstance(instruction, MacAllBanks):
        mask = render_aim_channel_mask(instruction.channels)
        return f"AiM MAC_ABK {instruction.operation_size} {mask} {instruction.row}"
    if isinstance(instruction, ReadMac):
        mask = render_aim_channel_mask(instruction.channels)
        return f"AiM RD_MAC {instruction.destination.slot} {mask}"
    raise AimSimulatorCompatibilityError(
        f"no AiM trace record for {type(instruction).__name__}"
    )


def iter_aim_records(
    kernel: CentKernel,
    arch: CentArchitecture,
    *,
    issue: ChannelIssue = ChannelIssue.BROADCAST,
) -> Iterator[str]:
    """Yield the records of the kernel's instructions only, without newlines.

    For traces made of several kernels; ``iter_aim_trace`` adds the CFR
    write before and the EOC after.

    Raises:
        AimSimulatorCompatibilityError: If ``arch`` is not a geometry the
            simulator models (16 banks, 16-value bursts, at most 32 channels).
        ValueError: If the kernel is not legal on ``arch``.
    """

    hardware = backend_hardware(kernel, arch)
    if (
        hardware.num_banks != AIM_SIMULATOR_BANKS
        or hardware.burst_length != AIM_SIMULATOR_BURST_LENGTH
        or hardware.num_channels > 32
    ):
        raise AimSimulatorCompatibilityError(
            f"the AiM simulator models {AIM_SIMULATOR_BANKS} banks, "
            f"{AIM_SIMULATOR_BURST_LENGTH}-value bursts and up to 32 channels; "
            f"got {hardware.num_banks} banks, {hardware.burst_length}-value "
            f"bursts, {hardware.num_channels} channels"
        )
    for instruction in iter_instructions(kernel, arch, issue=issue):
        yield render_aim_record(instruction)


def iter_aim_trace(
    kernel: CentKernel,
    arch: CentArchitecture,
    *,
    issue: ChannelIssue = ChannelIssue.BROADCAST,
) -> Iterator[str]:
    """Yield the kernel's whole trace, without newlines, ending with EOC.

    Raises:
        AimSimulatorCompatibilityError: If ``arch`` is not a geometry the
            simulator models.
        ValueError: If the kernel is not legal on ``arch``.
    """

    yield MAC_FROM_GLOBAL_BUFFER
    yield from iter_aim_records(kernel, arch, issue=issue)
    yield "AiM EOC"


def write_aim_trace(
    kernel: CentKernel,
    arch: CentArchitecture,
    out: TextIO,
    *,
    issue: ChannelIssue = ChannelIssue.BROADCAST,
) -> int:
    """Stream the kernel's trace to ``out``, one record per line.

    Returns:
        The number of records written.
    """

    count = 0
    for record in iter_aim_trace(kernel, arch, issue=issue):
        out.write(record + "\n")
        count += 1
    return count


def aim_trace(
    kernel: CentKernel,
    arch: CentArchitecture,
    *,
    issue: ChannelIssue = ChannelIssue.BROADCAST,
) -> str:
    """The kernel's whole trace as text, with a final newline."""

    return "".join(
        record + "\n" for record in iter_aim_trace(kernel, arch, issue=issue)
    )
