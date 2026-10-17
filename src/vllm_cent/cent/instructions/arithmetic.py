"""Arithmetic instructions in the order used by Table 2 of the CENT paper."""

from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from ..utils import require_nonnegative, require_positive
from .address import CentChannelSet, CentSharedBufferAddress
from .base import CentInstruction, CentOpcode

__all__ = [
    "Accumulate",
    "ApplyActivation",
    "ElementwiseMultiply",
    "Exponent",
    "MacAllBanks",
    "MacOperandSource",
    "Reduction",
    "RunRiscV",
]


# Near-bank processing-unit (PU) instructions -------------------------------------------------
#
# These instructions run beside the DRAM banks. CHmask chooses the channels,
# OPsize chooses the number of repeated operations, RO/CO give the first DRAM
# position, and Regid chooses a register that keeps a running result.


class MacOperandSource(Enum):
    """Choose where a bank reads the second input of a multiply-accumulate.

    A multiply-accumulate multiplies input pairs and adds their products to a
    stored running sum. Here the first input lives in a DRAM bank; the second
    comes from the channel's Global Buffer or a neighboring bank.

    AiM exposes this choice through configuration register zero rather than an
    operand in the textual ``MAC_ABK`` instruction. Keeping the state explicit
    in the compiler's intermediate representation (IR) prevents two different
    calculations from becoming the same apparent instruction.

    Members:
        GLOBAL_BUFFER: Multiply each bank value by the channel Global Buffer.
        NEXT_BANK: Multiply even-numbered bank values by the following bank.
    """

    GLOBAL_BUFFER = 0
    NEXT_BANK = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class MacAllBanks(CentInstruction):
    """Multiply values in every selected bank and add them to registers.

    This is ``MAC_ABK CHmask OPsize RO CO Regid`` in the paper. Each multiply
    uses data starting at ``RO``/``CO`` and adds its result to ``Regid``. The
    register must already contain the desired starting sum. ``channels``
    selects whole channels, not individual banks within them.

    Attributes:
        channels: Physical channels that perform the operation, or ``CHmask``.
        operation_size: Number of burst-sized operations, or ``OPsize``.
        row: First DRAM row, or ``RO``.
        column: First scalar position in that row under the current Python
            address convention; see ``CentMemoryAddress`` for hardware units.
        accumulation_register: Register that receives the running sum, or
            ``Regid``.
        operand_source: Location of the second multiply input. The AiM target
            writes this value to its broadcast configuration register.
        NEXT_BANK_FIRST_OPERAND_BANK: Position of the first input within each
            neighboring-bank pair when ``operand_source`` is ``NEXT_BANK``.
        NEXT_BANK_SECOND_OPERAND_BANK: Position of the second input within each
            neighboring-bank pair when ``operand_source`` is ``NEXT_BANK``.
        OPCODE: Fixed ``MAC_ABK`` instruction name.
    """

    channels: CentChannelSet
    operation_size: int
    row: int
    column: int
    accumulation_register: int
    # This is semantic compiler state, not an operand encoded in the physical
    # MAC_ABK record. A target lowers it to the control-state update required
    # before the MAC, which keeps this IR instruction independent of inherited
    # configuration-register state.
    operand_source: MacOperandSource
    NEXT_BANK_FIRST_OPERAND_BANK: ClassVar[int] = 0
    NEXT_BANK_SECOND_OPERAND_BANK: ClassVar[int] = 1
    OPCODE: ClassVar[CentOpcode] = CentOpcode.MAC_ALL_BANKS

    def __post_init__(self) -> None:
        """Check operand rules that do not need a target device.

        Raises:
            ValueError: If a size is not positive or an index is negative.
        """

        # The target later checks whether the row span and register exist.
        require_positive("operation_size", self.operation_size)
        require_nonnegative("row", self.row)
        require_nonnegative("column", self.column)
        require_nonnegative("accumulation_register", self.accumulation_register)


@dataclass(frozen=True, slots=True, kw_only=True)
class ElementwiseMultiply(CentInstruction):
    """Multiply corresponding values stored in neighboring banks.

    This is ``EW_MUL CHmask OPsize RO CO`` in the paper. Within each four-bank
    group, banks zero and one are inputs and bank two receives the result.
    These roles are defined by CENT's functional reference model; this is a
    software convention, not a universal hardware bank mapping. The typed
    instruction has no explicit bank operands.

    Attributes:
        channels: Physical channels that perform the operation, or ``CHmask``.
        operation_size: Number of burst-sized operations, or ``OPsize``.
        row: First DRAM row, or ``RO``.
        column: First scalar position in that row under the current Python
            address convention; see ``CentMemoryAddress`` for hardware units.
        FIRST_OPERAND_BANK: Position of the first input within each four-bank group.
        SECOND_OPERAND_BANK: Position of the second input within each four-bank group.
        RESULT_BANK: Position receiving the product within each four-bank group.
        OPCODE: Fixed ``EW_MUL`` instruction name.

    Examples:
        Under the reference bank mapping, values ``[2, 3]`` in bank zero and
        ``[4, 5]`` in bank one produce ``[8, 15]`` in bank two.
    """

    channels: CentChannelSet
    operation_size: int
    row: int
    column: int
    FIRST_OPERAND_BANK: ClassVar[int] = 0
    SECOND_OPERAND_BANK: ClassVar[int] = 1
    RESULT_BANK: ClassVar[int] = 2
    OPCODE: ClassVar[CentOpcode] = CentOpcode.ELEMENTWISE_MULTIPLY

    def __post_init__(self) -> None:
        """Check operand rules that do not need a target device.

        Raises:
            ValueError: If a size is not positive or an index is negative.
        """

        # The target later checks whether the complete row span fits.
        require_positive("operation_size", self.operation_size)
        require_nonnegative("row", self.row)
        require_nonnegative("column", self.column)


# TODO(target ABI): Define the activation-function IDs.
#
# AiM places AFid in CFR2 and exposes results through RD_AF, but it does not map
# the eight numeric modes to activation functions. The hardware spec therefore
# keeps the sigmoid ID explicit until that mapping has a stronger source.


@dataclass(frozen=True, slots=True, kw_only=True)
class ApplyActivation(CentInstruction):
    """Apply an activation function to one accumulation register.

    This is ``AF CHmask AFid Regid`` in the paper. ``AFid`` chooses the function
    and ``Regid`` chooses an existing result register.

    Attributes:
        channels: Physical channels that apply the function, or ``CHmask``.
        activation_function_id: Target-defined function number, or ``AFid``.
        accumulation_register: Register containing the input, or ``Regid``.
        OPCODE: Fixed ``AF`` instruction name.
    """

    channels: CentChannelSet
    activation_function_id: int
    accumulation_register: int
    OPCODE: ClassVar[CentOpcode] = CentOpcode.ACTIVATION_FUNCTION

    def __post_init__(self) -> None:
        """Check operand rules that do not need a target device.

        Raises:
            ValueError: If an identifier is negative.
        """

        # The target later checks Regid. AFid remains opaque until its ABI exists.
        require_nonnegative("activation_function_id", self.activation_function_id)
        require_nonnegative("accumulation_register", self.accumulation_register)


# PNM instructions ----------------------------------------------------------
#
# Processing-near-memory (PNM) units operate outside the DRAM banks. These
# instructions name Shared Buffer input Rs and output Rd. Current validation
# treats OPsize as a slot count; the exact footprints still need a target contract.


# TODO(emulator ABI): Define the exact EXP calculation.
#
# We know EXP uses a tenth-order Taylor approximation. We still need its
# coefficients, input range, rounding, overflow, and special-value behavior.


@dataclass(frozen=True, slots=True, kw_only=True)
class Exponent(CentInstruction):
    """Apply the exponential function to values in the Shared Buffer.

    This is ``EXP OPsize Rd Rs`` in the paper. It reads values from ``Rs`` and
    writes their elementwise exponential results to ``Rd``.

    Attributes:
        operation_size: Number of consecutive Shared Buffer slots, or ``OPsize``.
        destination: First output slot, or ``Rd``.
        source: First input slot, or ``Rs``.
        OPCODE: Fixed ``EXP`` instruction name.
    """

    operation_size: int
    destination: CentSharedBufferAddress
    source: CentSharedBufferAddress
    OPCODE: ClassVar[CentOpcode] = CentOpcode.EXPONENT

    def __post_init__(self) -> None:
        """Validate exponent operands.

        Raises:
            ValueError: If ``operation_size`` is less than one.
        """

        require_positive("operation_size", self.operation_size)


# TODO(emulator ABI): Finish the RED output and numerical contract.
#
# CENT Figure 7(b) defines a sum of the 16 lanes in one slot; Section 4.2
# puts that BF16 result in the destination's first lane. The other lanes,
# rounding, and overlap behavior are not specified. OPsize repeats per-slot
# work; a whole-vector sum needs a separate stage combining those results.
# See docs/research/execution-contracts.md before adding numerical semantics.


@dataclass(frozen=True, slots=True, kw_only=True)
class Reduction(CentInstruction):
    """Sum the lanes within each Shared Buffer slot.

    CENT Figure 7(b) and Section 4.2 define ``RED OPsize Rd Rs`` as independent
    per-slot sums, each written to the destination slot's first lane. This is
    not a sum across the entire input vector. Numerical execution remains
    unsupported until the unused output lanes and rounding rules are defined.

    Attributes:
        operation_size: Number of consecutive slots to reduce, or ``OPsize``.
        destination: First output slot, or ``Rd``; each result occupies lane zero.
        source: First input slot, or ``Rs``.
        OPCODE: Fixed ``RED`` instruction name.
    """

    operation_size: int
    destination: CentSharedBufferAddress
    source: CentSharedBufferAddress
    OPCODE: ClassVar[CentOpcode] = CentOpcode.REDUCTION

    def __post_init__(self) -> None:
        """Validate reduction operands.

        Raises:
            ValueError: If ``operation_size`` is less than one.
        """

        require_positive("operation_size", self.operation_size)


# TODO(emulator ABI): Define the exact ACC calculation.
#
# We know ACC adds matching BF16 lanes. We still need its intermediate
# precision, rounding, overflow, and source/destination overlap rules.


@dataclass(frozen=True, slots=True, kw_only=True)
class Accumulate(CentInstruction):
    """Add Shared Buffer source values into destination values in place.

    This is ``ACC OPsize Rd Rs`` in the paper. Each lane performs
    ``Rd[i] = Rd[i] + Rs[i]``, so ``Rd`` must already contain valid values.
    A lane is one scalar position inside a slot.

    Attributes:
        operation_size: Number of slots to add, or ``OPsize``.
        destination: First input-and-output slot, or ``Rd``.
        source: First slot whose values are added, or ``Rs``.
        OPCODE: Fixed ``ACC`` instruction name.

    Examples:
        Under reference-math execution, destination values ``[10, 20]`` plus source
        values ``[1, 2]`` become ``[11, 22]``. This example does not specify
        hardware rounding or overlap behavior.
    """

    operation_size: int
    destination: CentSharedBufferAddress
    source: CentSharedBufferAddress
    OPCODE: ClassVar[CentOpcode] = CentOpcode.ACCUMULATION

    def __post_init__(self) -> None:
        """Validate accumulation operands.

        Raises:
            ValueError: If ``operation_size`` is less than one.
        """

        require_positive("operation_size", self.operation_size)


# TODO(target ABI): Define how RISCV programs are called.
#
# We need entry addresses for reciprocal, square root, and RoPE. We also need PC
# limits and alignment, work assignment across cores, and completion behavior.


@dataclass(frozen=True, slots=True, kw_only=True)
class RunRiscV(CentInstruction):
    """Start a routine on the processing-near-memory RISC-V cores.

    This is ``RISCV OPsize PC Rd Rs`` in the paper. ``PC`` chooses the code,
    ``Rs`` supplies its inputs, and ``Rd`` receives its outputs.

    Attributes:
        operation_size: Number of work items, or ``OPsize``.
        program_counter: Entry address of the RISC-V routine, or ``PC``. Its
            address units and calling convention still need a target definition.
        destination: First output Shared Buffer slot, or ``Rd``.
        source: First input Shared Buffer slot, or ``Rs``.
        OPCODE: Fixed ``RISCV`` instruction name.
    """

    operation_size: int
    program_counter: int
    destination: CentSharedBufferAddress
    source: CentSharedBufferAddress
    OPCODE: ClassVar[CentOpcode] = CentOpcode.RISCV

    def __post_init__(self) -> None:
        """Validate RISC-V instruction operands.

        Raises:
            ValueError: If the size is not positive or ``program_counter``
                is negative.
        """

        require_positive("operation_size", self.operation_size)
        require_nonnegative("program_counter", self.program_counter)
