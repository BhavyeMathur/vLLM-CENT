"""Turn typed CENT instructions into readable assembly text."""

from .instructions import (
    Accumulate,
    ApplyActivation,
    BroadcastCxl,
    CentChannelSet,
    CentInstruction,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    Exponent,
    MacAllBanks,
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
)
from .program import CentProgram

__all__ = ["render_channel_mask", "render_instruction", "render_text_program"]

# This text follows the paper and is useful for inspection. It is intentionally
# distinct from the executable AiM trace ABI, whose channel mask, configuration
# registers, and operand lists are handled by ``cent.aim``.


def render_channel_mask(channels: CentChannelSet) -> str:
    """Turn selected channel numbers into a hexadecimal ``CHmask``.

    This temporary text format sets bit ``n`` for channel ``n``. The paper does
    not define that bit order.

    Args:
        channels: Physical channels that should receive the instruction.

    Returns:
        Lowercase hexadecimal mask, such as ``"0x9"`` for channels 0 and 3.
    """

    # Instructions keep readable channel numbers. Only the text renderer packs
    # them into bits, with channel 0 currently placed in the lowest bit.
    mask = 0
    for channel in channels.channels:
        mask |= 1 << channel
    return hex(mask)


def _shared_buffer_operands(
    instruction: Exponent | Reduction | Accumulate,
) -> str:
    """Render the ``OPsize Rd Rs`` fields shared by three PNM commands.

    Args:
        instruction: EXP, RED, or ACC instruction to render.

    Returns:
        Space-separated operation count, output slot, and input slot.
    """

    return (
        f"{instruction.operation_size} {instruction.destination.slot} "
        f"{instruction.source.slot}"
    )


def render_instruction(instruction: CentInstruction) -> str:
    """Render one instruction in the operand order used by the paper.

    For example, Python's ``address.row`` becomes ``RO``. A Shared Buffer slot
    becomes ``Rs`` when read and ``Rd`` when written. This inspection format is
    not round-trippable: it omits semantic fields that the paper's assembly
    table does not encode, such as the MAC operand source and copy-bank index.

    Args:
        instruction: Typed CENT command to render.

    Returns:
        One line of assembly without a trailing newline.

    Raises:
        TypeError: If ``instruction`` has an unknown type.
    """

    if isinstance(instruction, (WriteSingleBank, ReadSingleBank)):
        opcode = instruction.opcode.value
        address = instruction.address

        # WR_SBK reads its Shared Buffer source. RD_SBK writes its Shared Buffer
        # destination. Their other fields have the same order.
        buffer = (
            instruction.source
            if isinstance(instruction, WriteSingleBank)
            else instruction.destination
        )
        return (
            f"{opcode} {address.channel} {instruction.operation_size} "
            f"{address.bank} {address.row} {address.column} {buffer.slot}"
        )
    if isinstance(instruction, MacAllBanks):
        opcode = instruction.opcode.value
        return (
            f"{opcode} {render_channel_mask(instruction.channels)} "
            f"{instruction.operation_size} {instruction.row} "
            f"{instruction.column} {instruction.accumulation_register}"
        )
    if isinstance(instruction, ElementwiseMultiply):
        opcode = instruction.opcode.value
        return (
            f"{opcode} {render_channel_mask(instruction.channels)} "
            f"{instruction.operation_size} {instruction.row} "
            f"{instruction.column}"
        )
    if isinstance(instruction, ApplyActivation):
        opcode = instruction.opcode.value
        return (
            f"{opcode} {render_channel_mask(instruction.channels)} "
            f"{instruction.activation_function_id} "
            f"{instruction.accumulation_register}"
        )
    # EXP, RED, and ACC use the same OPsize/Rd/Rs field order.
    if isinstance(instruction, (Exponent, Reduction, Accumulate)):
        opcode = instruction.opcode.value
        return f"{opcode} {_shared_buffer_operands(instruction)}"
    if isinstance(instruction, RunRiscV):
        opcode = instruction.opcode.value
        return (
            f"{opcode} {instruction.operation_size} "
            f"{instruction.program_counter} {instruction.destination.slot} "
            f"{instruction.source.slot}"
        )
    if isinstance(instruction, SendCxl):
        opcode = instruction.opcode.value
        return (
            f"{opcode} {instruction.destination_device} "
            f"{instruction.source.slot} {instruction.destination.slot}"
        )
    if isinstance(instruction, ReceiveCxl):
        return instruction.opcode.value
    if isinstance(instruction, BroadcastCxl):
        opcode = instruction.opcode.value
        return (
            f"{opcode} {instruction.device_count} {instruction.source.slot} "
            f"{instruction.destination.slot}"
        )
    if isinstance(instruction, WriteAllBanks):
        opcode = instruction.opcode.value
        return (
            f"{opcode} {instruction.channel} {instruction.row} "
            f"{instruction.column} {instruction.source.slot} "
            f"{instruction.accumulation_register}"
        )
    # The opcode, rather than an operand, records the direction of these copies.
    if isinstance(instruction, (CopyBankToGlobalBuffer, CopyGlobalBufferToBank)):
        opcode = instruction.opcode.value
        return (
            f"{opcode} {render_channel_mask(instruction.channels)} "
            f"{instruction.operation_size} {instruction.row} "
            f"{instruction.column}"
        )
    if isinstance(instruction, WriteBias):
        opcode = instruction.opcode.value
        return (
            f"{opcode} {render_channel_mask(instruction.channels)} "
            f"{instruction.source.slot}"
        )
    if isinstance(instruction, ReadMac):
        opcode = instruction.opcode.value
        return (
            f"{opcode} {render_channel_mask(instruction.channels)} "
            f"{instruction.destination.slot} "
            f"{instruction.accumulation_register}"
        )
    if isinstance(instruction, ReadActivation):
        opcode = instruction.opcode.value
        return (
            f"{opcode} {render_channel_mask(instruction.channels)} "
            f"{instruction.destination.slot} "
            f"{instruction.accumulation_register}"
        )
    if isinstance(instruction, WriteGlobalBuffer):
        opcode = instruction.opcode.value
        return (
            f"{opcode} {render_channel_mask(instruction.channels)} "
            f"{instruction.operation_size} {instruction.column} "
            f"{instruction.source.slot}"
        )
    raise TypeError(f"unsupported CENT instruction type: {type(instruction).__name__}")


def render_text_program(program: CentProgram) -> str:
    """Render every instruction in a program as assembly text.

    Args:
        program: Ordered CENT program to render.

    Returns:
        Assembly text with one instruction per line and a final newline.
    """

    # The final newline makes the result a normal text file.
    return (
        "\n".join(
            render_instruction(instruction) for instruction in program.instructions
        )
        + "\n"
    )
