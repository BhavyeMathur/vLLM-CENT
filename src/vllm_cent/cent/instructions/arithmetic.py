"""Arithmetic instructions in the order used by Table 2 of the CENT paper."""

from dataclasses import dataclass
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
    "Reduction",
    "RunRiscV",
]


# Near-bank PU instructions -------------------------------------------------
#
# These instructions run beside the DRAM banks. CHmask chooses the channels,
# OPsize chooses the number of repeated operations, RO/CO give the first DRAM
# position, and Regid chooses a register that keeps a running result.


# TODO(ISA): Explain where MAC_ABK gets its second input.
#
# One input may come from the Global Buffer or a neighboring bank. MAC_ABK has
# no operand that selects one. We need to learn how hardware makes that choice.

@dataclass(frozen=True, slots=True, kw_only=True)
class MacAllBanks(CentInstruction):
    """Multiply values in every selected bank and add them to registers.

    This is ``MAC_ABK CHmask OPsize RO CO Regid`` in the paper. Each multiply
    uses data starting at ``RO``/``CO`` and adds its result to ``Regid``.

    Attributes:
        channels: Physical channels that perform the operation, or ``CHmask``.
        operation_size: Number of burst-sized operations, or ``OPsize``.
        row: First DRAM row, or ``RO``.
        column: First scalar position in that row, or ``CO``.
        accumulation_register: Register that receives the running sum, or
            ``Regid``.
        OPCODE: Fixed ``MAC_ABK`` instruction name.
    """

    channels: CentChannelSet
    operation_size: int
    row: int
    column: int
    accumulation_register: int
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
        require_nonnegative(
            "accumulation_register", self.accumulation_register
        )


# TODO(paper): Define each bank's role during EW_MUL.
#
# A group has two input banks and one result bank. We need to confirm which
# numbered bank has each role. Current lowering follows the reference simulator.

@dataclass(frozen=True, slots=True, kw_only=True)
class ElementwiseMultiply(CentInstruction):
    """Multiply corresponding values stored in neighboring banks.

    This is ``EW_MUL CHmask OPsize RO CO`` in the paper. Bank roles are implicit,
    so the instruction does not contain input or output bank numbers.

    Attributes:
        channels: Physical channels that perform the operation, or ``CHmask``.
        operation_size: Number of burst-sized operations, or ``OPsize``.
        row: First DRAM row, or ``RO``.
        column: First scalar position in that row, or ``CO``.
        OPCODE: Fixed ``EW_MUL`` instruction name.
    """

    channels: CentChannelSet
    operation_size: int
    row: int
    column: int
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


# TODO(target ABI): Define the activation IDs and how results are read.
#
# The paper does not map AFid numbers to functions or explain how to read the
# result. We need both rules from the target ABI.

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
        require_nonnegative(
            "activation_function_id", self.activation_function_id
        )
        require_nonnegative(
            "accumulation_register", self.accumulation_register
        )


# PNM instructions ----------------------------------------------------------
#
# These instructions run in the PNM units. They read Shared Buffer slots from
# Rs and write slots at Rd. OPsize says how many consecutive slots are handled.


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


# TODO(emulator ABI): Define the exact RED result.
#
# We know RED sums 16 BF16 values in each input slot. We still need the addition
# order, intermediate precision, rounding, overflow, and unused output lanes.

@dataclass(frozen=True, slots=True, kw_only=True)
class Reduction(CentInstruction):
    """Sum the 16 BF16 values in each selected Shared Buffer slot.

    This is ``RED OPsize Rd Rs`` in the paper. Each source slot produces one sum
    in the first BF16 lane of its destination slot. Different slots are not
    combined with each other.

    Attributes:
        operation_size: Number of slots to reduce, or ``OPsize``.
        destination: First slot that receives a sum, or ``Rd``.
        source: First slot containing 16 values to sum, or ``Rs``.
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

    Attributes:
        operation_size: Number of slots to add, or ``OPsize``.
        destination: First input-and-output slot, or ``Rd``.
        source: First slot whose values are added, or ``Rs``.
        OPCODE: Fixed ``ACC`` instruction name.
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
    """Start code on the PNM RISC-V cores.

    This is ``RISCV OPsize PC Rd Rs`` in the paper. ``PC`` chooses the code,
    ``Rs`` supplies its inputs, and ``Rd`` receives its outputs.

    Attributes:
        operation_size: Number of work items, or ``OPsize``.
        program_counter: Address of the first RISC-V instruction, or ``PC``.
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
