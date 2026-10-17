"""Define CENT opcode names and the base instruction type."""

from enum import Enum
from typing import ClassVar

__all__ = ["CentInstruction", "CentOpcode", "ReferenceOpcode"]


class CentOpcode(str, Enum):
    """Name the operations in the compiler's typed instruction representation.

    The opcode says what CENT should do. The matching instruction dataclass
    stores the values needed to do it. Most names come from Tables 2 and 3 of
    the paper. ``RD_AF`` is an AiM target extension used to retrieve activation
    results.

    Members:
        WRITE_SINGLE_BANK: Copy Shared Buffer slots into one DRAM bank.
        READ_SINGLE_BANK: Copy one DRAM bank into Shared Buffer slots.
        WRITE_ALL_BANKS: Copy one Shared Buffer slot into all banks of a channel.
        WRITE_BIAS: Initialize MAC accumulators from the Shared Buffer.
        MAC_ALL_BANKS: Add bank/input products to stored running sums.
        READ_MAC: Copy MAC accumulator results into the Shared Buffer.
        READ_ACTIVATION: Copy activation results into the Shared Buffer.
        ELEMENTWISE_MULTIPLY: Multiply paired values in four-bank groups.
        EXPONENT: Apply exponentiation in processing-near-memory (PNM) units.
        REDUCTION: Reduce groups of Shared Buffer values in PNM units.
        ACCUMULATION: Add source values into destination Shared Buffer values.
        RISCV: Run a RISC-V operation beginning at a specified program counter.
        SEND_CXL: Send Shared Buffer data over Compute Express Link (CXL).
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


class ReferenceOpcode(str, Enum):
    """Name functional services that have no physical instruction encoding.

    Members:
        INITIALIZE_ACCUMULATOR: Set the selected latch in every selected bank.
        READ_ACCUMULATOR: Pack selected latch values into Shared Buffer.
        COPY: Pack a scalar slice with explicit zero padding.
        STORE_SLICE: Replace a slice while preserving neighboring lanes.
        FILL: Initialize a logical vector to a constant.
        REDUCE_SUM: Sum a logical prefix into one scalar.
        RMS_SCALE: Compute reciprocal square root of a mean plus epsilon.
        BROADCAST: Repeat a scalar across a logical vector.
        ACTIVATION: Apply an explicit scalar activation formula.
        SOFTMAX: Normalize consecutive rows with negative-infinity masking.
        TRANSPOSE: Pack the transpose of a selected matrix slice.
        ATTENTION_SCORES: Scale one query score vector and apply its masks.
    """

    INITIALIZE_ACCUMULATOR = "REF_INITIALIZE_ACCUMULATOR"
    READ_ACCUMULATOR = "REF_READ_ACCUMULATOR"
    COPY = "REF_COPY"
    STORE_SLICE = "REF_STORE_SLICE"
    FILL = "REF_FILL"
    REDUCE_SUM = "REF_REDUCE_SUM"
    RMS_SCALE = "REF_RMS_SCALE"
    BROADCAST = "REF_BROADCAST"
    ACTIVATION = "REF_ACTIVATION"
    SOFTMAX = "REF_SOFTMAX"
    TRANSPOSE = "REF_TRANSPOSE"
    ATTENTION_SCORES = "REF_ATTENTION_SCORES"


class CentInstruction:
    """Base class for every typed CENT instruction.

    Concrete dataclasses store operands. Every instance reads the one ``OPCODE``
    constant defined by its class.
    """

    OPCODE: ClassVar[CentOpcode | ReferenceOpcode]

    @property
    def opcode(self) -> CentOpcode | ReferenceOpcode:
        """Return this instruction class's fixed opcode.

        Returns:
            Opcode assigned to the concrete instruction class.
        """

        # The opcode is shared by the class rather than copied into each object.
        return type(self).OPCODE
