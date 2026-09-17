"""Define CENT opcode names and the base instruction type."""

from enum import Enum
from typing import ClassVar

__all__ = ["CentInstruction", "CentOpcode"]


class CentOpcode(str, Enum):
    """CENT instruction names used by the compiler IR.

    The opcode says what CENT should do. The matching instruction dataclass
    stores the values needed to do it. Most names come from Tables 2 and 3 of
    the paper. ``RD_AF`` is an AiM target extension used to retrieve activation
    results.

    Members:
        WRITE_SINGLE_BANK: Copy Shared Buffer slots into one DRAM bank.
        READ_SINGLE_BANK: Copy one DRAM bank into Shared Buffer slots.
        WRITE_ALL_BANKS: Copy one Shared Buffer slot into all banks of a channel.
        WRITE_BIAS: Initialize MAC accumulators from the Shared Buffer.
        MAC_ALL_BANKS: Accumulate bank rows against the global buffer.
        READ_MAC: Copy MAC accumulator results into the Shared Buffer.
        READ_ACTIVATION: Copy activation results into the Shared Buffer.
        ELEMENTWISE_MULTIPLY: Multiply paired values in four-bank groups.
        EXPONENT: Apply elementwise exponentiation in PNM units.
        REDUCTION: Reduce groups of Shared Buffer values in PNM units.
        ACCUMULATION: Add source values into destination Shared Buffer values.
        RISCV: Run a RISC-V operation beginning at a specified program counter.
        SEND_CXL: Send Shared Buffer data to one CXL device.
        RECEIVE_CXL: Receive an incoming CXL transfer.
        BROADCAST_CXL: Broadcast Shared Buffer data to multiple CXL devices.
        WRITE_GLOBAL_BUFFER: Copy Shared Buffer values into global buffers.
        COPY_BANK_TO_GLOBAL_BUFFER: Copy one bank row into global buffers.
        COPY_GLOBAL_BUFFER_TO_BANK: Copy global-buffer values into one bank.
        ACTIVATION_FUNCTION: Apply the configured activation function.
    """

    MAC_ALL_BANKS = "MAC_ABK"
    ELEMENTWISE_MULTIPLY = "EW_MUL"
    ACTIVATION_FUNCTION = "AF"
    EXPONENT = "EXP"
    REDUCTION = "RED"
    ACCUMULATION = "ACC"
    RISCV = "RISCV"
    SEND_CXL = "SEND_CXL"
    RECEIVE_CXL = "RECV_CXL"
    BROADCAST_CXL = "BCAST_CXL"
    WRITE_SINGLE_BANK = "WR_SBK"
    READ_SINGLE_BANK = "RD_SBK"
    WRITE_ALL_BANKS = "WR_ABK"
    COPY_BANK_TO_GLOBAL_BUFFER = "COPY_BKGB"
    COPY_GLOBAL_BUFFER_TO_BANK = "COPY_GBBK"
    WRITE_BIAS = "WR_BIAS"
    READ_MAC = "RD_MAC"
    READ_ACTIVATION = "RD_AF"
    WRITE_GLOBAL_BUFFER = "WR_GB"


class CentInstruction:
    """Base class for every typed CENT instruction.

    Concrete dataclasses store operands. Every instance reads the one ``OPCODE``
    constant defined by its class.
    """

    OPCODE: ClassVar[CentOpcode]

    @property
    def opcode(self) -> CentOpcode:
        """Return this instruction class's fixed opcode.

        Returns:
            Opcode assigned to the concrete instruction class.
        """

        # The opcode is shared by the class rather than copied into each object.
        return type(self).OPCODE
