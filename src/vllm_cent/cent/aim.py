"""The AiM simulator's fixed geometry and its channel-mask encoding.

``vllm_cent.tile.codegen`` writes AiM traces; this module holds what it
shares with the simulator's trace parser: the geometry the simulator models
and the bit order of its channel masks.
"""

from .instructions import CentChannelSet

__all__ = [
    "AIM_SIMULATOR_BANKS",
    "AIM_SIMULATOR_BURST_LENGTH",
    "AIM_SIMULATOR_CHANNELS",
    "AimSimulatorCompatibilityError",
    "render_aim_channel_mask",
]


AIM_SIMULATOR_CHANNELS = 32
"""Fixed channel count in the AiM simulator."""

AIM_SIMULATOR_BANKS = 16
"""Banks per GDDR6 channel modeled by the AiM simulator."""

AIM_SIMULATOR_BURST_LENGTH = 16
"""FP16/BF16 values in one 256-bit AiM transfer."""


class AimSimulatorCompatibilityError(ValueError):
    """Report an operation or operand the AiM trace format cannot represent."""


def render_aim_channel_mask(channels: CentChannelSet) -> str:
    """Pack channels using the AiM simulator's reversed 32-bit mask order.

    The simulator maps the least-significant mask bit to physical channel 31,
    so channel zero occupies bit 31, as CENT's own trace generator writes it.

    Args:
        channels: Physical AiM channel numbers to select.

    Returns:
        Lowercase hexadecimal channel mask accepted by the trace parser.

    Raises:
        AimSimulatorCompatibilityError: If a channel is outside 0 through 31.

    Examples:
        A selection containing only channel zero renders as ``0x80000000``.
        A selection containing only channel 31 renders as ``0x1``.
    """

    mask = 0
    for channel in channels.channels:
        if channel >= AIM_SIMULATOR_CHANNELS:
            raise AimSimulatorCompatibilityError(
                "AiM simulator channel is outside 0 through 31"
            )
        mask |= 1 << (AIM_SIMULATOR_CHANNELS - 1 - channel)
    return hex(mask)
