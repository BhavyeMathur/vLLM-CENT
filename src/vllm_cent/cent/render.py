"""Turn typed CENT instructions into readable assembly text."""

from functools import singledispatch

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


@singledispatch
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

    raise TypeError(f"unsupported CENT instruction type: {type(instruction).__name__}")


def _render_single_bank_transfer(
        instruction: WriteSingleBank | ReadSingleBank,
        *,
        buffer_slot: int,
) -> str:
    """Render fields shared by the two single-bank transfer directions.

    Args:
        instruction: Single-bank transfer whose common address fields to render.
        buffer_slot: Shared Buffer source or destination selected by the opcode.

    Returns:
        Paper-readable single-bank transfer text.
    """

    address = instruction.address
    return (
        f"{instruction.opcode.value} {address.channel} "
        f"{instruction.operation_size} {address.bank} {address.row} "
        f"{address.column} {buffer_slot}"
    )


@render_instruction.register(WriteSingleBank)
def _render_write_single_bank(instruction: WriteSingleBank) -> str:
    """Render one Shared-Buffer-to-bank transfer.

    Args:
        instruction: Transfer whose Shared Buffer source is rendered as ``Rs``.

    Returns:
        Paper-readable ``WR_SBK`` text.
    """

    return _render_single_bank_transfer(
        instruction,
        buffer_slot=instruction.source.slot,
    )


@render_instruction.register(ReadSingleBank)
def _render_read_single_bank(instruction: ReadSingleBank) -> str:
    """Render one bank-to-Shared-Buffer transfer.

    Args:
        instruction: Transfer whose Shared Buffer destination is rendered as
            ``Rd``.

    Returns:
        Paper-readable ``RD_SBK`` text.
    """

    return _render_single_bank_transfer(
        instruction,
        buffer_slot=instruction.destination.slot,
    )


@render_instruction.register(MacAllBanks)
def _render_mac_all_banks(instruction: MacAllBanks) -> str:
    """Render one all-bank MAC instruction.

    Args:
        instruction: MAC operation to render.

    Returns:
        Paper-readable ``MAC_ABK`` text.
    """

    return (
        f"{instruction.opcode.value} {render_channel_mask(instruction.channels)} "
        f"{instruction.operation_size} {instruction.row} {instruction.column} "
        f"{instruction.accumulation_register}"
    )


@render_instruction.register(ElementwiseMultiply)
def _render_elementwise_multiply(instruction: ElementwiseMultiply) -> str:
    """Render one near-bank elementwise multiplication.

    Args:
        instruction: Elementwise operation to render.

    Returns:
        Paper-readable ``EW_MUL`` text.
    """

    return (
        f"{instruction.opcode.value} {render_channel_mask(instruction.channels)} "
        f"{instruction.operation_size} {instruction.row} {instruction.column}"
    )


@render_instruction.register(ApplyActivation)
def _render_apply_activation(instruction: ApplyActivation) -> str:
    """Render one near-bank activation instruction.

    Args:
        instruction: Activation operation to render.

    Returns:
        Paper-readable ``AF`` text.
    """

    return (
        f"{instruction.opcode.value} {render_channel_mask(instruction.channels)} "
        f"{instruction.activation_function_id} "
        f"{instruction.accumulation_register}"
    )


@render_instruction.register(Exponent)
@render_instruction.register(Reduction)
@render_instruction.register(Accumulate)
def _render_shared_buffer_operation(
        instruction: Exponent | Reduction | Accumulate,
) -> str:
    """Render a PNM command with the common ``OPsize Rd Rs`` layout.

    Args:
        instruction: Exponent, reduction, or accumulation operation to render.

    Returns:
        Paper-readable PNM operation text.
    """

    return f"{instruction.opcode.value} {_shared_buffer_operands(instruction)}"


@render_instruction.register(RunRiscV)
def _render_riscv(instruction: RunRiscV) -> str:
    """Render one PNM RISC-V invocation.

    Args:
        instruction: RISC-V operation to render.

    Returns:
        Paper-readable ``RISCV`` text.
    """

    return (
        f"{instruction.opcode.value} {instruction.operation_size} "
        f"{instruction.program_counter} {instruction.destination.slot} "
        f"{instruction.source.slot}"
    )


@render_instruction.register(SendCxl)
def _render_send_cxl(instruction: SendCxl) -> str:
    """Render one point-to-point CXL send.

    Args:
        instruction: CXL send to render.

    Returns:
        Paper-readable ``SEND_CXL`` text.
    """

    return (
        f"{instruction.opcode.value} {instruction.destination_device} "
        f"{instruction.source.slot} {instruction.destination.slot}"
    )


@render_instruction.register(ReceiveCxl)
def _render_receive_cxl(instruction: ReceiveCxl) -> str:
    """Render one operand-free CXL receive.

    Args:
        instruction: CXL receive to render.

    Returns:
        Paper-readable ``RECV_CXL`` text.
    """

    return instruction.opcode.value


@render_instruction.register(BroadcastCxl)
def _render_broadcast_cxl(instruction: BroadcastCxl) -> str:
    """Render one CXL broadcast.

    Args:
        instruction: CXL broadcast to render.

    Returns:
        Paper-readable ``BCAST_CXL`` text.
    """

    return (
        f"{instruction.opcode.value} {instruction.device_count} "
        f"{instruction.source.slot} {instruction.destination.slot}"
    )


@render_instruction.register(WriteAllBanks)
def _render_write_all_banks(instruction: WriteAllBanks) -> str:
    """Render one all-bank write.

    Args:
        instruction: All-bank transfer to render.

    Returns:
        Paper-readable ``WR_ABK`` text.
    """

    return (
        f"{instruction.opcode.value} {instruction.channel} {instruction.row} "
        f"{instruction.column} {instruction.source.slot} "
        f"{instruction.accumulation_register}"
    )


@render_instruction.register(CopyBankToGlobalBuffer)
@render_instruction.register(CopyGlobalBufferToBank)
def _render_bank_global_buffer_copy(
        instruction: CopyBankToGlobalBuffer | CopyGlobalBufferToBank,
) -> str:
    """Render either direction of a bank and Global Buffer copy.

    The opcode, rather than an operand, records the transfer direction.

    Args:
        instruction: Copy operation to render.

    Returns:
        Paper-readable copy text.
    """

    return (
        f"{instruction.opcode.value} {render_channel_mask(instruction.channels)} "
        f"{instruction.operation_size} {instruction.row} {instruction.column}"
    )


@render_instruction.register(WriteBias)
def _render_write_bias(instruction: WriteBias) -> str:
    """Render one MAC-bias initialization.

    Args:
        instruction: Bias write to render.

    Returns:
        Paper-readable ``WR_BIAS`` text.
    """

    return (
        f"{instruction.opcode.value} {render_channel_mask(instruction.channels)} "
        f"{instruction.source.slot}"
    )


@render_instruction.register(ReadMac)
@render_instruction.register(ReadActivation)
def _render_register_read(instruction: ReadMac | ReadActivation) -> str:
    """Render a MAC or activation-register read.

    Args:
        instruction: Register read to render.

    Returns:
        Paper-readable register-read text.
    """

    return (
        f"{instruction.opcode.value} {render_channel_mask(instruction.channels)} "
        f"{instruction.destination.slot} {instruction.accumulation_register}"
    )


@render_instruction.register(WriteGlobalBuffer)
def _render_write_global_buffer(instruction: WriteGlobalBuffer) -> str:
    """Render one Shared-Buffer-to-Global-Buffer transfer.

    Args:
        instruction: Global Buffer write to render.

    Returns:
        Paper-readable ``WR_GB`` text.
    """

    return (
        f"{instruction.opcode.value} {render_channel_mask(instruction.channels)} "
        f"{instruction.operation_size} {instruction.column} "
        f"{instruction.source.slot}"
    )


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
