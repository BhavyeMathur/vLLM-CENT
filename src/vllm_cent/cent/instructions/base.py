"""Define CENT opcode names and the base instruction type."""

from enum import Enum
from typing import ClassVar

__all__ = ["CentInstruction", "CentOpcode"]


class CentOpcode(str, Enum):
    """Instruction names from Tables 2 and 3 of the CENT paper.

    The opcode says what CENT should do. The matching instruction dataclass
    stores the values needed to do it.

    Members:
        WRITE_SINGLE_BANK: Copy Shared Buffer slots into one DRAM bank.
        READ_SINGLE_BANK: Copy one DRAM bank into Shared Buffer slots.
        WRITE_ALL_BANKS: Copy one Shared Buffer slot into all banks of a channel.
        WRITE_BIAS: Initialize MAC accumulators from the Shared Buffer.
        MAC_ALL_BANKS: Accumulate bank rows against the global buffer.
        READ_MAC: Copy MAC accumulator results into the Shared Buffer.
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
    WRITE_GLOBAL_BUFFER = "WR_GB"


class _ImmutableOpcodeMeta(type):
    """Prevent an instruction class from changing its opcode.

    ``OPCODE`` belongs to the class because every instance of that class has the
    same operation. A class may define it once and cannot replace or delete it.
    """

    def __setattr__(cls, name: str, value: object) -> None:
        """Set a class attribute unless ``OPCODE`` is already defined.

        Args:
            name: Name of the class attribute.
            value: Value to assign.

        Raises:
            AttributeError: If an existing ``OPCODE`` would be replaced.
        """

        # Python may set OPCODE once while creating a concrete instruction class.
        if name == "OPCODE" and "OPCODE" in cls.__dict__:
            raise AttributeError("OPCODE cannot be modified")
        super().__setattr__(name, value)

    def __delattr__(cls, name: str) -> None:
        """Delete a class attribute unless it is the opcode.

        Args:
            name: Name of the class attribute.

        Raises:
            AttributeError: If ``OPCODE`` would be deleted.
        """

        # Removing OPCODE would leave the instruction without a stable identity.
        if name == "OPCODE":
            raise AttributeError("OPCODE cannot be deleted")
        super().__delattr__(name)


class CentInstruction(metaclass=_ImmutableOpcodeMeta):
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
