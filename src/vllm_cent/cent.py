"""CENT hardware and instruction types."""

from dataclasses import dataclass, field
from enum import Enum


class CentOpcode(str, Enum):
    """CENT trace opcodes used by the initial compiler contract.

    Attributes:
        WRITE_MEMORY: Write one burst to a DRAM row.
        WRITE_BIAS: Clear or initialize the MAC bias register.
        MAC_ALL_BANKS: Multiply and accumulate across all selected banks.
        READ_MAC: Read the MAC result from selected channels.
        ELEMENTWISE_MULTIPLY: Multiply values stored in paired bank groups.
        END_OF_COMPUTATION: Mark the end of a trace.
    """

    WRITE_MEMORY = "W MEM"
    WRITE_BIAS = "AiM WR_BIAS"
    MAC_ALL_BANKS = "AiM MAC_ABK"
    READ_MAC = "AiM RD_MAC"
    ELEMENTWISE_MULTIPLY = "AiM EWMUL"
    END_OF_COMPUTATION = "AiM EOC"


@dataclass(frozen=True, slots=True, kw_only=True)
class CentHardwareSpec:
    """Memory geometry of one CENT device.

    Attributes:
        num_channels: Number of independently addressable channels. Must be at
            least 1.
        num_banks: Number of banks in each channel. Must be at least 1.
        dram_rows: Number of rows in each bank. Must be at least 1.
        dram_columns: Number of scalar values in each row. Must be at least 1
            and divisible by ``burst_length``.
        burst_length: Number of scalar values transferred by one memory
            command. Must be at least 1 and no greater than ``dram_columns``.
    """

    num_channels: int
    num_banks: int
    dram_rows: int
    dram_columns: int
    burst_length: int


InstructionOperand = int | str


@dataclass(frozen=True, slots=True, kw_only=True)
class CentInstruction:
    """One command accepted by the CENT simulator.

    Attributes:
        opcode: Operation performed by the simulator.
        operands: Positional operands written after the opcode. Integers must
            be non-negative; channel masks use hexadecimal strings such as
            ``"0x3"``. The number and meaning of operands depend on ``opcode``.
    """

    opcode: CentOpcode
    operands: tuple[InstructionOperand, ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True, kw_only=True)
class CentProgram:
    """Commands emitted for one compilation unit.

    Attributes:
        instructions: Commands in execution order. A complete program must be
            non-empty and end with ``CentOpcode.END_OF_COMPUTATION``.
    """

    instructions: tuple[CentInstruction, ...]
