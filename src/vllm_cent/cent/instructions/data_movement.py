"""Data-movement instructions in Table 3 order from the CENT paper."""

from dataclasses import dataclass
from typing import ClassVar

from ..utils import require_nonnegative, require_positive
from .address import (
    CentChannelSet,
    CentMemoryAddress,
    CentSharedBufferAddress,
)
from .base import CentInstruction, CentOpcode

__all__ = [
    "BroadcastCxl",
    "CopyBankToGlobalBuffer",
    "CopyGlobalBufferToBank",
    "ReadActivation",
    "ReadMac",
    "ReadSingleBank",
    "ReceiveCxl",
    "SendCxl",
    "WriteAllBanks",
    "WriteBias",
    "WriteGlobalBuffer",
    "WriteSingleBank",
]


# CXL device-to-device transfers -------------------------------------------
#
# Rs is a slot on the sender and Rd is a slot on the receiver. DVid or DVcount
# tells CXL where to send the data.

# TODO(paper): Decide how to represent multicast.
#
# The paper mentions multicast but gives no multicast instruction. We need to
# learn whether it uses BCAST_CXL or another command.


# TODO(paper/runtime ABI): Define how much data SEND_CXL moves.
#
# SEND_CXL has no OPsize or byte count. We need its payload size and message
# boundary rules.


@dataclass(frozen=True, slots=True, kw_only=True)
class SendCxl(CentInstruction):
    """Send data from the local Shared Buffer to one CXL device.

    This is ``SEND_CXL DVid Rs Rd`` in the paper.

    Attributes:
        destination_device: Number of the receiving device, or ``DVid``.
        source: First local Shared Buffer slot to read, or ``Rs``.
        destination: First Shared Buffer slot to write on the receiver, or ``Rd``.
        OPCODE: Fixed ``SEND_CXL`` instruction name.
    """

    destination_device: int
    source: CentSharedBufferAddress
    destination: CentSharedBufferAddress
    OPCODE: ClassVar[CentOpcode] = CentOpcode.SEND_CXL

    def __post_init__(self) -> None:
        """Validate CXL send operands.

        Raises:
            ValueError: If the device ID is negative.
        """

        # A future CXL topology must check the upper device-ID bound.
        require_nonnegative("destination_device", self.destination_device)


# TODO(runtime ABI): Define how RECV_CXL chooses a pending message.
#
# We know receive blocks. We need to learn how it chooses a send and reports
# completion or failure when several messages are waiting.


@dataclass(frozen=True, slots=True)
class ReceiveCxl(CentInstruction):
    """Wait for an incoming CXL transfer.

    This is the operand-free ``RECV_CXL`` instruction from the paper.

    Attributes:
        OPCODE: Fixed ``RECV_CXL`` instruction name.
    """

    OPCODE: ClassVar[CentOpcode] = CentOpcode.RECEIVE_CXL


# TODO(paper/runtime ABI): Define exactly which devices BCAST_CXL targets.
#
# We know DVcount is 8 bits, but the paper also mentions a device-ID mask. We
# need the target order, whether it includes the sender, wrapping rules, and
# payload size. Validation also needs a CXL topology.


@dataclass(frozen=True, slots=True, kw_only=True)
class BroadcastCxl(CentInstruction):
    """Send the same Shared Buffer data to several CXL devices.

    This is ``BCAST_CXL DVcount Rs Rd`` in the paper.

    Attributes:
        device_count: Number of receiving devices, or ``DVcount``.
        source: First local Shared Buffer slot to read, or ``Rs``.
        destination: First Shared Buffer slot to write on each receiver, or
            ``Rd``.
        OPCODE: Fixed ``BCAST_CXL`` instruction name.
    """

    device_count: int
    source: CentSharedBufferAddress
    destination: CentSharedBufferAddress
    OPCODE: ClassVar[CentOpcode] = CentOpcode.BROADCAST_CXL

    def __post_init__(self) -> None:
        """Validate CXL broadcast operands.

        Raises:
            ValueError: If ``device_count`` is less than one.
        """

        # A future CXL topology must check the maximum device count.
        require_positive("device_count", self.device_count)


# Shared Buffer and DRAM-bank transfers ------------------------------------
#
# These commands move between one DRAM bank and the Shared Buffer. OPsize moves
# through later DRAM bursts and Shared Buffer slots.


@dataclass(frozen=True, slots=True, kw_only=True)
class WriteSingleBank(CentInstruction):
    """Copy Shared Buffer slots into one DRAM bank.

    This is ``WR_SBK CHid OPsize BK RO CO Rs`` in the paper. ``address`` names
    the first DRAM burst and ``source`` names the first Shared Buffer slot.

    Attributes:
        address: Channel, bank, row, and column of the first DRAM burst.
        operation_size: Number of bursts to copy, or ``OPsize``.
        source: First Shared Buffer slot to read, or ``Rs``.
        OPCODE: Fixed ``WR_SBK`` instruction name.
    """

    address: CentMemoryAddress
    operation_size: int
    source: CentSharedBufferAddress
    OPCODE: ClassVar[CentOpcode] = CentOpcode.WRITE_SINGLE_BANK

    def __post_init__(self) -> None:
        """Check operand rules that do not need a target device.

        Raises:
            ValueError: If ``operation_size`` is less than one.
        """

        # The target later checks the complete DRAM and Shared Buffer spans.
        require_positive("operation_size", self.operation_size)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadSingleBank(CentInstruction):
    """Copy values from one DRAM bank into Shared Buffer slots.

    This is ``RD_SBK CHid OPsize BK RO CO Rd`` in the paper.

    Attributes:
        address: Channel, bank, row, and column of the first DRAM burst.
        operation_size: Number of bursts to copy, or ``OPsize``.
        destination: First Shared Buffer slot to write, or ``Rd``.
        OPCODE: Fixed ``RD_SBK`` instruction name.
    """

    address: CentMemoryAddress
    operation_size: int
    destination: CentSharedBufferAddress
    OPCODE: ClassVar[CentOpcode] = CentOpcode.READ_SINGLE_BANK

    def __post_init__(self) -> None:
        """Check operand rules that do not need a target device.

        Raises:
            ValueError: If ``operation_size`` is less than one.
        """

        # The target later checks the complete DRAM and Shared Buffer spans.
        require_positive("operation_size", self.operation_size)


# TODO(paper): Explain the Regid field in WR_ABK.
#
# WR_ABK moves Shared Buffer values to DRAM but also contains Regid. We need to
# learn what that register does and whether "all banks" means 16 paper banks or
# every bank on a custom target.


@dataclass(frozen=True, slots=True, kw_only=True)
class WriteAllBanks(CentInstruction):
    """Write values from one Shared Buffer slot across a channel's banks.

    This is ``WR_ABK CHid RO CO Rs Regid`` in the paper. On the paper hardware,
    the 16 BF16 source values are split across 16 banks at the same row/column.

    Attributes:
        channel: Physical channel number, or ``CHid``.
        row: Destination DRAM row, or ``RO``.
        column: Destination position within the row, or ``CO``.
        source: Shared Buffer slot containing the values, or ``Rs``.
        accumulation_register: Register number shown as ``Regid`` in Table 3.
        OPCODE: Fixed ``WR_ABK`` instruction name.
    """

    channel: int
    row: int
    column: int
    source: CentSharedBufferAddress
    accumulation_register: int
    OPCODE: ClassVar[CentOpcode] = CentOpcode.WRITE_ALL_BANKS

    def __post_init__(self) -> None:
        """Check operand rules that do not need a target device.

        Raises:
            ValueError: If an index is negative.
        """

        # The selected target supplies the upper bounds.
        require_nonnegative("channel", self.channel)
        require_nonnegative("row", self.row)
        require_nonnegative("column", self.column)
        require_nonnegative("accumulation_register", self.accumulation_register)


# Global Buffer and DRAM-bank transfers ------------------------------------
#
# Each channel has a Global Buffer used by nearby processing units. CHmask
# chooses the channels. These copy instructions contain no bank number.

# TODO(ISA): Reconcile the paper and AiM copy-instruction operands.
#
# The paper provides a row and column but no bank:
#
#     COPY_BKGB CHmask OPsize RO CO
#
# AiM instead requires a bank and omits the column:
#
#     COPY_BKGB OPsize CHmask BK RO
#
# Confirm how CENT chooses the bank, whether AiM's bank is an extension, and
# whether the real instruction can start anywhere other than column zero.


@dataclass(frozen=True, slots=True, kw_only=True)
class CopyBankToGlobalBuffer(CentInstruction):
    """Copy DRAM values into the selected channels' Global Buffers.

    The paper lists ``COPY_BKGB CHmask OPsize RO CO`` without a bank operand.
    AiM's executable trace ABI additionally requires ``bank`` and uses it as the
    source bank on every selected channel.

    Attributes:
        channels: Physical channels that perform the copy, or ``CHmask``.
        operation_size: Number of bursts to copy, or ``OPsize``.
        bank: Source bank within every selected channel.
        row: First source DRAM row, or ``RO``.
        column: First position in that row, or ``CO``.
        OPCODE: Fixed ``COPY_BKGB`` instruction name.
    """

    channels: CentChannelSet
    operation_size: int
    bank: int
    row: int
    column: int
    OPCODE: ClassVar[CentOpcode] = CentOpcode.COPY_BANK_TO_GLOBAL_BUFFER

    def __post_init__(self) -> None:
        """Validate bank-to-Global-Buffer copy operands.

        Raises:
            ValueError: If a size is not positive or an index is negative.
        """

        require_positive("operation_size", self.operation_size)
        require_nonnegative("bank", self.bank)
        require_nonnegative("row", self.row)
        require_nonnegative("column", self.column)


@dataclass(frozen=True, slots=True, kw_only=True)
class CopyGlobalBufferToBank(CentInstruction):
    """Copy Global Buffer values into DRAM on selected channels.

    The paper lists ``COPY_GBBK CHmask OPsize RO CO`` without a bank operand.
    AiM's executable trace ABI additionally requires ``bank`` and uses it as the
    destination bank on every selected channel.

    Attributes:
        channels: Physical channels that perform the copy, or ``CHmask``.
        operation_size: Number of bursts to copy, or ``OPsize``.
        bank: Destination bank within every selected channel.
        row: First destination DRAM row, or ``RO``.
        column: First position in that row, or ``CO``.
        OPCODE: Fixed ``COPY_GBBK`` instruction name.
    """

    channels: CentChannelSet
    operation_size: int
    bank: int
    row: int
    column: int
    OPCODE: ClassVar[CentOpcode] = CentOpcode.COPY_GLOBAL_BUFFER_TO_BANK

    def __post_init__(self) -> None:
        """Validate Global-Buffer-to-bank copy operands.

        Raises:
            ValueError: If a size is not positive or an index is negative.
        """

        require_positive("operation_size", self.operation_size)
        require_nonnegative("bank", self.bank)
        require_nonnegative("row", self.row)
        require_nonnegative("column", self.column)


# Shared Buffer and near-bank PU state -------------------------------------
#
# These commands move data between the Shared Buffer and MAC result registers.
# Regid is a register number; Rs and Rd are Shared Buffer slots.

# TODO(ISA): Define where RD_MAC stores results from several channels.
#
# One channel produces a 256-bit result containing values from its 16 banks.
# RD_MAC still provides only one Rd when CHmask selects several channels. Confirm
# whether those channel results use consecutive slots, channel-local slots, or
# another layout. Also confirm whether WR_BIAS broadcasts one Rs or reads a
# different slot for each channel.


# TODO(ISA): Define which accumulator WR_BIAS initializes.
#
# MAC_ABK and RD_MAC name a register, but WR_BIAS does not. Confirm whether it
# selects an implicit register, initializes every register, or uses hidden state.


# TODO(runtime): Define how WR_BIAS source values enter the Shared Buffer.
#
# WR_BIAS reads Rs; it does not create a constant. Operations that start an
# accumulation need a slot containing zeros, but neither the paper trace nor the
# public AiM timing simulator defines how the runtime loads that slot.


@dataclass(frozen=True, slots=True, kw_only=True)
class WriteBias(CentInstruction):
    """Load MAC starting values from the Shared Buffer.

    This is ``WR_BIAS CHmask Rs`` in the paper. Later MAC operations add products
    to these starting values.

    Attributes:
        channels: Physical channels that load the bias, or ``CHmask``.
        source: Shared Buffer slot containing bias values, or ``Rs``.
        OPCODE: Fixed ``WR_BIAS`` instruction name.
    """

    channels: CentChannelSet
    source: CentSharedBufferAddress
    OPCODE: ClassVar[CentOpcode] = CentOpcode.WRITE_BIAS


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadMac(CentInstruction):
    """Copy a MAC result register into the Shared Buffer.

    This is ``RD_MAC CHmask Rd Regid`` in the paper.

    Attributes:
        channels: Physical channels whose results are read, or ``CHmask``.
        destination: First Shared Buffer output slot, or ``Rd``.
        accumulation_register: MAC result register to read, or ``Regid``.
        OPCODE: Fixed ``RD_MAC`` instruction name.
    """

    channels: CentChannelSet
    destination: CentSharedBufferAddress
    accumulation_register: int
    OPCODE: ClassVar[CentOpcode] = CentOpcode.READ_MAC

    def __post_init__(self) -> None:
        """Validate MAC-register read operands.

        Raises:
            ValueError: If ``accumulation_register`` is negative.
        """

        require_nonnegative("accumulation_register", self.accumulation_register)


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadActivation(CentInstruction):
    """Copy activation-function results into the Shared Buffer.

    AiM calls this target-specific operation ``RD_AF GPR_0 channel_mask``.
    ``destination`` names the corresponding 256-bit Shared Buffer/GPR slot in
    the compiler IR. The target trace has no register operand, so its adapter
    can represent only accumulator register zero.

    Attributes:
        channels: Physical channels whose activation results are read.
        destination: First Shared Buffer output slot.
        accumulation_register: Register whose activation result is expected.
        OPCODE: Fixed ``RD_AF`` instruction name.
    """

    channels: CentChannelSet
    destination: CentSharedBufferAddress
    accumulation_register: int
    OPCODE: ClassVar[CentOpcode] = CentOpcode.READ_ACTIVATION

    def __post_init__(self) -> None:
        """Validate the selected accumulator register.

        Raises:
            ValueError: If ``accumulation_register`` is negative.
        """

        require_nonnegative("accumulation_register", self.accumulation_register)


# Shared Buffer to Global Buffer -------------------------------------------
#
# WR_GB prepares the Global Buffer input used by operations such as MAC_ABK.


@dataclass(frozen=True, slots=True, kw_only=True)
class WriteGlobalBuffer(CentInstruction):
    """Copy Shared Buffer values into selected Global Buffers.

    This is ``WR_GB CHmask OPsize CO Rs`` in the paper. Each operation reads the
    next Shared Buffer slot and writes the next burst in each Global Buffer.

    Attributes:
        channels: Physical channels whose Global Buffers are written, or
            ``CHmask``.
        operation_size: Number of bursts to copy, or ``OPsize``.
        column: First Global Buffer position, or ``CO``.
        source: First Shared Buffer slot to read, or ``Rs``.
        OPCODE: Fixed ``WR_GB`` instruction name.
    """

    channels: CentChannelSet
    operation_size: int
    column: int
    source: CentSharedBufferAddress
    OPCODE: ClassVar[CentOpcode] = CentOpcode.WRITE_GLOBAL_BUFFER

    def __post_init__(self) -> None:
        """Validate Global Buffer write operands.

        Raises:
            ValueError: If a size is not positive or ``column`` is negative.
        """

        require_positive("operation_size", self.operation_size)
        require_nonnegative("column", self.column)
