"""Typed services for functional reference execution, with no physical encoding.

These operations make packing, selected-register access, and scalar arithmetic
explicit. They are not CENT paper opcodes or AiM trace commands.
"""

from dataclasses import dataclass
from math import isfinite
from typing import ClassVar

from ..activations import ActivationKind

from .hardware import CentHardwareSpec
from .instructions.address import CentChannelSet, CentSharedBufferAddress
from .instructions.base import CentInstruction, ReferenceOpcode

__all__ = [
    "InitializeAccumulator",
    "ReadAccumulator",
    "ReferenceActivation",
    "ReferenceAttentionScores",
    "ReferenceBroadcast",
    "ReferenceCopy",
    "ReferenceFill",
    "ReferenceInstruction",
    "ReferenceReduceSum",
    "ReferenceRmsScale",
    "ReferenceRotaryEmbedding",
    "ReferenceSoftmax",
    "ReferenceStoreSlice",
    "ReferenceTranspose",
]


# TODO(physical packing): The reference services below use scalar offsets and
# explicitly zero unused lanes. CENT describes byte-addressable RISC-V buffer
# access, but does not provide the routines, argument locations, or completion
# rules needed to translate these services. Keep physical serializers rejecting
# them until that target interface is defined; see execution-contracts.md.


class ReferenceInstruction(CentInstruction):
    """Mark a functional service that physical serializers must reject."""


# TODO(physical register selection): Hynix JSSC p296 documents MRS-selected
# result latches. The bias-write selector bits and their scope/timing are still
# missing. Translate this explicit destination only after those are known.


@dataclass(frozen=True, slots=True, kw_only=True)
class InitializeAccumulator(ReferenceInstruction):
    """Initialize every bank in selected channels to the same starting sum.

    Attributes:
        channels: Channels whose bank registers are initialized.
        accumulation_register: Selected register in every bank.
        value: Initial scalar; zero starts an unbiased dot product.
    """

    channels: CentChannelSet
    accumulation_register: int
    value: float = 0.0
    OPCODE: ClassVar[ReferenceOpcode] = ReferenceOpcode.INITIALIZE_ACCUMULATOR


# TODO(physical readback): The original Python reference returns bank-order
# values without clearing the latch. It does not specify CENT Shared Buffer
# bytes. Confirm read selector, precision, channel order, and padding before
# translating this local packing rule into a physical RD_MAC sequence.


@dataclass(frozen=True, slots=True, kw_only=True)
class ReadAccumulator(ReferenceInstruction):
    """Pack selected accumulators into Shared Buffer without clearing them.

    Attributes:
        channels: Channels in output order, followed by increasing bank index.
        accumulation_register: Register read from every bank.
        destination: First output slot; final unused lanes are zero.
    """

    channels: CentChannelSet
    accumulation_register: int
    destination: CentSharedBufferAddress
    OPCODE: ClassVar[ReferenceOpcode] = ReferenceOpcode.READ_ACCUMULATOR


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceCopy(ReferenceInstruction):
    """Pack a scalar slice into a zero-filled Shared Buffer region.

    Attributes:
        source: First source slot.
        destination: First output slot.
        source_offset: Scalar offset from source lane zero.
        value_count: Number of source scalars copied.
        destination_count: Logical output length; remaining values and slot padding become zero.
    """

    source: CentSharedBufferAddress
    destination: CentSharedBufferAddress
    source_offset: int
    value_count: int
    destination_count: int
    OPCODE: ClassVar[ReferenceOpcode] = ReferenceOpcode.COPY


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceStoreSlice(ReferenceInstruction):
    """Replace part of an initialized Shared Buffer region.

    Attributes:
        source: First source slot.
        destination: First destination slot; existing values outside the slice survive.
        source_offset: Scalar offset from source lane zero.
        destination_offset: Scalar offset from destination lane zero.
        value_count: Number of scalars replaced.
        destination_count: Logical destination length used to check the slice bounds.
            Complete slots touched by the slice must already be initialized;
            untouched slots are neither read nor rewritten.
    """

    source: CentSharedBufferAddress
    destination: CentSharedBufferAddress
    source_offset: int
    destination_offset: int
    value_count: int
    destination_count: int
    OPCODE: ClassVar[ReferenceOpcode] = ReferenceOpcode.STORE_SLICE


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceFill(ReferenceInstruction):
    """Initialize a logical vector and zero its final padding lanes.

    Attributes:
        destination: First output slot.
        value_count: Logical vector length in scalars.
        value: Value repeated within the logical vector.
    """

    destination: CentSharedBufferAddress
    value_count: int
    value: float = 0.0
    OPCODE: ClassVar[ReferenceOpcode] = ReferenceOpcode.FILL


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceReduceSum(ReferenceInstruction):
    """Sum a logical scalar prefix into lane zero of one output slot.

    Attributes:
        source: First input slot.
        destination: Output slot; lanes after lane zero become zero.
        value_count: Logical scalar count; input padding is excluded.
    """

    source: CentSharedBufferAddress
    destination: CentSharedBufferAddress
    value_count: int
    OPCODE: ClassVar[ReferenceOpcode] = ReferenceOpcode.REDUCE_SUM


# TODO(physical scalar execution): CENT assigns square root and division to
# RISC-V cores, but supplies no executable routine or calling convention. This
# service computes the formula directly; it must not emit a guessed PC value.


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceRmsScale(ReferenceInstruction):
    """Convert a sum of squares into a reciprocal root mean square.

    The reference executor computes Python division and square root, then
    calls ``numeric.store`` once for the scalar result. It requires
    ``ReferenceMathSemantics``; subclasses may customize storage conversion.
    This is an explicit software service, not a RISC-V executable or physical
    scalar instruction. Unknown physical root/packing encodings remain outside
    this contract.

    Attributes:
        source: Slot whose first lane contains the sum of squares.
        destination: Output slot; unused lanes become zero.
        value_count: Divisor used to form the mean.
        epsilon: Positive finite constant added to the mean before reciprocal square root.
    """

    source: CentSharedBufferAddress
    destination: CentSharedBufferAddress
    value_count: int
    epsilon: float
    OPCODE: ClassVar[ReferenceOpcode] = ReferenceOpcode.RMS_SCALE


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceBroadcast(ReferenceInstruction):
    """Repeat one scalar over a logical vector and zero its padding.

    Attributes:
        source: Slot containing the scalar in lane zero.
        destination: First output slot.
        value_count: Number of logical values receiving the scalar.
    """

    source: CentSharedBufferAddress
    destination: CentSharedBufferAddress
    value_count: int
    OPCODE: ClassVar[ReferenceOpcode] = ReferenceOpcode.BROADCAST


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceActivation(ReferenceInstruction):
    """Apply a selected activation to logical values and zero output padding.

    Inputs are read before any output is written, including when both spans
    alias. Python mathematical functions define the reference results; this
    service requires ``ReferenceMathSemantics`` and converts each result with
    ``numeric.store``. It does not model an activation lookup table or hardware
    rounding. NaN inputs produce NaN. Infinite inputs use the limiting values:
    sigmoid tends to zero/one, tanh to minus/plus one, and the other functions
    tend to zero at negative infinity and positive infinity at positive infinity.

    Attributes:
        source: First input slot; only the logical prefix is read.
        destination: First output slot; unused final lanes become zero.
        value_count: Number of logical scalars to transform.
        function: Exact mathematical formula selected by ``ActivationKind``.
    """

    source: CentSharedBufferAddress
    destination: CentSharedBufferAddress
    value_count: int
    function: ActivationKind
    OPCODE: ClassVar[ReferenceOpcode] = ReferenceOpcode.ACTIVATION


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceSoftmax(ReferenceInstruction):
    """Normalize each contiguous row with a stable reference softmax.

    Subtracting the maximum before exponentiation avoids overflow for finite
    inputs. Negative infinity masks a position and produces zero probability.
    NaN, positive infinity, and rows with no unmasked positions are rejected
    before any output is committed. Rows are consecutive logical scalars; only
    the final output slot is padded. Exact input/output aliasing is safe.
    Python exponentiation, summation, and division define this software service,
    which requires ``ReferenceMathSemantics`` and converts final results with
    ``numeric.store``. No physical EXP or reduction encoding is implied.

    Attributes:
        source: First slot containing the flattened rows.
        destination: First output slot; unused final lanes become zero.
        row_count: Positive number of independent rows.
        row_width: Positive number of scalar positions in each row.
    """

    source: CentSharedBufferAddress
    destination: CentSharedBufferAddress
    row_count: int
    row_width: int
    OPCODE: ClassVar[ReferenceOpcode] = ReferenceOpcode.SOFTMAX


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceTranspose(ReferenceInstruction):
    """Pack the transpose of a row-major matrix slice into Shared Buffer.

    The source matrix begins at a scalar offset, so a caller can select one
    head within a larger tensor without copying that head first. The complete
    source is read before writing, making overlapping source/output safe.
    Only the final output slot is padded with zeros; rows have no padding.
    This describes reference movement, not a physical transpose instruction.

    Attributes:
        source: First slot of the containing source tensor.
        destination: First slot of the packed transposed matrix.
        source_offset: Nonnegative scalar offset to the selected matrix.
        row_count: Positive number of input rows and output columns.
        column_count: Positive number of input columns and output rows.
    """

    source: CentSharedBufferAddress
    destination: CentSharedBufferAddress
    source_offset: int
    row_count: int
    column_count: int
    OPCODE: ClassVar[ReferenceOpcode] = ReferenceOpcode.TRANSPOSE


@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceAttentionScores(ReferenceInstruction):
    """Scale one query's scores, then apply additive and causal masks.

    An unmasked key receives ``score * scale + mask``; absent additive masks
    contribute zero. Keys beyond ``causal_limit`` and additive negative-infinity
    entries become negative infinity without evaluating score arithmetic. Thus
    a masked infinite score cannot generate NaN through ``inf + -inf``.
    Every supplied mask entry is checked, including causally hidden positions:
    only finite values and negative infinity are allowed. Unmasked nonfinite
    scores follow the numeric policy; subsequent softmax rejects NaN/positive
    infinity and fully masked rows. Both buffers are snapshotted before writes.

    Multiplication and addition use the numeric policy directly; the immediate
    ``scale`` is converted once through ``numeric.store`` before multiplication.
    No nonlinear profile is required. This service has no physical instruction
    encoding.

    Attributes:
        source: First slot holding this query's logical score vector.
        destination: First output slot; unused final lanes become zero.
        value_count: Positive number of keys represented by the score vector.
        scale: Finite multiplier applied before the additive mask, including
            zero or a negative value when explicitly requested.
        mask: First slot of the additive-mask tensor, or ``None`` for no mask.
        mask_offset: Scalar offset to this query's mask slice; must be zero
            when ``mask`` is absent.
        causal_limit: Greatest visible key index, inclusive, or ``None`` to
            disable causal masking. A limit beyond the final key permits all keys.
    """

    source: CentSharedBufferAddress
    destination: CentSharedBufferAddress
    value_count: int
    scale: float
    mask: CentSharedBufferAddress | None = None
    mask_offset: int = 0
    causal_limit: int | None = None
    OPCODE: ClassVar[ReferenceOpcode] = ReferenceOpcode.ATTENTION_SCORES


# TODO(RoPE on hardware): The paper and the reference CENT code do not say
# which steps of rotary embedding run on PIM banks, PNM units, or RISC-V cores.
# The older structural lowering (lowering/transformer/attention.py) emits only
# the element products. A physical form likely gathers the table rows of each
# position, multiplies with EW_MUL, and adds the signed products with ACC.
#
# This service computes the complete rotation in software instead, like
# ReferenceSoftmax. Replace it once the gather and pairing steps are defined.
@dataclass(frozen=True, slots=True, kw_only=True)
class ReferenceRotaryEmbedding(ReferenceInstruction):
    """Rotate every head vector by the angles stored for its token's position.

    The source holds ``token_count * head_count`` head vectors of
    ``head_dim`` scalars, token-major. The table holds one row per position:
    ``rotary_dim // 2`` cosines followed by as many sines. For token ``t``
    with position ``p``, each pair ``(a, b)`` of its first ``rotary_dim``
    elements becomes ``(a * cos - b * sin, b * cos + a * sin)`` with the
    angles of pair index ``i`` in row ``p``. The remaining elements are
    copied unchanged. Inputs are read before any output is written, so the
    destination may equal the source. Python arithmetic defines the result;
    it requires ``ReferenceMathSemantics``. No physical table lookup or
    pairing encoding is implied.

    Attributes:
        source: First slot of the head vectors.
        positions: First slot of ``token_count`` positions. Each must be an
            integer table row, stored as a scalar value.
        table: First slot of ``position_count`` rows of ``rotary_dim`` values.
        destination: First output slot; unused final lanes become zero.
        token_count: Number of tokens; each has one position.
        head_count: Head vectors per token, all rotated by that token's angles.
        head_dim: Scalars in each head vector.
        rotary_dim: Leading scalars of each head vector that are rotated.
        position_count: Rows in the table.
        is_neox_style: True pairs element ``i`` with ``i + rotary_dim // 2``
            (Hugging Face and vLLM ``rotate_half``); False pairs ``2i`` with
            ``2i + 1`` (GPT-J and Meta's original Llama code).

    Examples:
        With ``head_dim=4``, ``rotary_dim=4`` and NeoX pairing, a head
        ``(a0, a1, b0, b1)`` uses pairs ``(a0, b0)`` and ``(a1, b1)``. A table
        row ``(1, 0, 0, 1)`` holds ``cos=(1, 0)`` and ``sin=(0, 1)``, so the
        result is ``(a0, -b1, b0, a1)``: pair 0 is kept and pair 1 turns by
        ninety degrees.
    """

    source: CentSharedBufferAddress
    positions: CentSharedBufferAddress
    table: CentSharedBufferAddress
    destination: CentSharedBufferAddress
    token_count: int
    head_count: int
    head_dim: int
    rotary_dim: int
    position_count: int
    is_neox_style: bool
    OPCODE: ClassVar[ReferenceOpcode] = ReferenceOpcode.ROTARY_EMBEDDING


def validate_reference(
    instruction: ReferenceInstruction, hardware: CentHardwareSpec
) -> None:
    """Check all scalar spans and selected registers before execution.

    Args:
        instruction: Reference service whose operands are checked.
        hardware: Storage capacities and lane count defining valid spans.

    Raises:
        ValueError: If a count, offset, channel, register or span is invalid.
        TypeError: If an unknown reference service has no validation rules.
    """

    def span(address: CentSharedBufferAddress, count: int) -> None:
        """Check a scalar prefix from a slot against the buffer capacity.

        Args:
            address: First slot occupied by the prefix.
            count: Scalar length, rounded upward to complete slots.

        Raises:
            ValueError: If the count is empty or the final slot exceeds capacity.
        """
        if (
            count < 1
            or address.slot
            + (count + hardware.burst_length - 1) // hardware.burst_length
            > hardware.shared_buffer_slots
        ):
            raise ValueError("reference Shared Buffer span is invalid")

    if isinstance(instruction, ReferenceTranspose):
        if (
            instruction.source_offset < 0
            or instruction.row_count < 1
            or instruction.column_count < 1
        ):
            raise ValueError(
                "transpose requires a nonnegative offset and positive matrix dimensions"
            )
        count = instruction.row_count * instruction.column_count
        span(instruction.source, instruction.source_offset + count)
        span(instruction.destination, count)
        return
    if isinstance(instruction, ReferenceAttentionScores):
        span(instruction.source, instruction.value_count)
        span(instruction.destination, instruction.value_count)
        if not isfinite(instruction.scale):
            raise ValueError("attention scale must be finite")
        if instruction.mask_offset < 0:
            raise ValueError("mask_offset cannot be negative")
        if instruction.mask is None and instruction.mask_offset != 0:
            raise ValueError("mask_offset requires an additive mask")
        if instruction.mask is not None:
            span(instruction.mask, instruction.mask_offset + instruction.value_count)
        if instruction.causal_limit is not None and instruction.causal_limit < 0:
            raise ValueError("causal_limit cannot be negative")
        return
    if isinstance(instruction, ReferenceRotaryEmbedding):
        if (
            min(
                instruction.token_count,
                instruction.head_count,
                instruction.head_dim,
                instruction.rotary_dim,
                instruction.position_count,
            )
            < 1
        ):
            raise ValueError("rotary embedding counts must be positive")
        # Every rotated element needs a partner, so the rotated prefix is
        # an even number of elements that fits inside one head vector.
        if instruction.rotary_dim % 2 or instruction.rotary_dim > instruction.head_dim:
            raise ValueError("rotary_dim must be even and at most head_dim")
        count = instruction.token_count * instruction.head_count * instruction.head_dim
        span(instruction.source, count)
        span(instruction.destination, count)
        span(instruction.positions, instruction.token_count)
        span(instruction.table, instruction.position_count * instruction.rotary_dim)
        return
    if isinstance(instruction, ReferenceSoftmax):
        if instruction.row_count < 1 or instruction.row_width < 1:
            raise ValueError("softmax row_count and row_width must be positive")
        count = instruction.row_count * instruction.row_width
        span(instruction.source, count)
        span(instruction.destination, count)
        return
    if isinstance(instruction, ReferenceActivation):
        span(instruction.source, instruction.value_count)
        span(instruction.destination, instruction.value_count)
        return
    if isinstance(instruction, (InitializeAccumulator, ReadAccumulator)):
        if any(
            channel >= hardware.num_channels
            for channel in instruction.channels.channels
        ):
            raise ValueError("reference channel is outside target")
        if (
            not 0
            <= instruction.accumulation_register
            < hardware.accumulator_slots_per_bank
        ):
            raise ValueError("reference accumulator register is outside target")
        if isinstance(instruction, ReadAccumulator):
            span(
                instruction.destination,
                len(instruction.channels.channels) * hardware.num_banks,
            )
        return
    if not isinstance(
        instruction,
        (
            ReferenceCopy,
            ReferenceStoreSlice,
            ReferenceFill,
            ReferenceReduceSum,
            ReferenceRmsScale,
            ReferenceBroadcast,
        ),
    ):
        raise TypeError("unsupported reference instruction")
    if instruction.value_count < 1:
        raise ValueError("value_count must be positive")
    if isinstance(instruction, (ReferenceCopy, ReferenceStoreSlice)):
        if instruction.source_offset < 0:
            raise ValueError("source_offset cannot be negative")
        span(instruction.source, instruction.source_offset + instruction.value_count)
        span(instruction.destination, instruction.destination_count)
        offset = (
            instruction.destination_offset
            if isinstance(instruction, ReferenceStoreSlice)
            else 0
        )
        if (
            offset < 0
            or offset + instruction.value_count > instruction.destination_count
        ):
            raise ValueError("copied slice exceeds destination_count")
    elif isinstance(instruction, ReferenceFill):
        span(instruction.destination, instruction.value_count)
    elif isinstance(instruction, ReferenceReduceSum):
        span(instruction.source, instruction.value_count)
        span(instruction.destination, 1)
    elif isinstance(instruction, (ReferenceRmsScale, ReferenceBroadcast)):
        span(instruction.source, 1)
        span(
            instruction.destination,
            instruction.value_count
            if isinstance(instruction, ReferenceBroadcast)
            else 1,
        )
        if isinstance(instruction, ReferenceRmsScale) and (
            not isfinite(instruction.epsilon) or instruction.epsilon <= 0
        ):
            raise ValueError("epsilon must be positive and finite")
