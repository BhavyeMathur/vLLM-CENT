"""Execute explicit reference services and bank/global-buffer dot products.

Each service snapshots inputs and proposes writes. The executor commits all
writes together. Packing and scalar root operations are software conventions,
not claims about a physical CENT encoding or rounding rule.
"""

from dataclasses import dataclass, field
from math import erfc, exp, fsum, inf, isnan, pi, sqrt, tanh

from vllm_cent.activations import ActivationKind

from vllm_cent.cent import (
    CentBankRegisterAddress,
    CentGlobalBufferAddress,
    CentMemoryAddress,
    CentSharedBufferAddress,
    MacAllBanks,
    MacOperandSource,
)
from vllm_cent.cent.reference import (
    InitializeAccumulator,
    ReferenceActivation,
    ReferenceTranspose,
    ReferenceAttentionScores,
    ReferenceSoftmax,
    ReadAccumulator,
    ReferenceBroadcast,
    ReferenceCopy,
    ReferenceFill,
    ReferenceInstruction,
    ReferenceReduceSum,
    ReferenceRmsScale,
    ReferenceRotaryEmbedding,
    ReferenceStoreSlice,
)
from vllm_cent.runtime import (
    CentAccumulatorRegion,
    CentDramRegion,
    CentGlobalBufferRegion,
    CentPhysicalRegion,
    CentSharedBufferRegion,
)
from .effects import (
    AccumulatorWriteEffect,
    SharedBufferWriteEffect,
    _InstructionEffects,
)
from .kernels import _KernelArithmeticError
from .state import CentDeviceState

__all__: list[str] = []

_GELU_TANH_CUBIC_COEFFICIENT = 0.044715


def _activate_scalar(value: float, function: ActivationKind) -> float:
    """Evaluate one activation without overflow from large finite inputs.

    Args:
        value: Stored scalar; NaN propagates and infinities use limiting values.
        function: Exact activation formula to evaluate.

    Returns:
        Python floating-point result before the numeric storage conversion.

    Raises:
        ValueError: If the activation choice has no defined formula.
    """
    if isnan(value):
        return value
    if function is ActivationKind.RELU:
        return max(0.0, value)
    if function is ActivationKind.TANH:
        return tanh(value)
    if function is ActivationKind.SIGMOID:
        # Both exponent arguments are nonpositive, so neither branch overflows.
        if value >= 0.0:
            return 1.0 / (1.0 + exp(-value))
        exponential = exp(value)
        return exponential / (1.0 + exponential)
    if function is ActivationKind.SILU:
        if value == -inf:
            return 0.0
        if value >= 0.0:
            return value / (1.0 + exp(-value))
        exponential = exp(value)
        return value * exponential / (1.0 + exponential)
    if function in (ActivationKind.GELU_ERF, ActivationKind.GELU_TANH):
        if value == -inf:
            return 0.0
        if value == inf:
            return inf
        if function is ActivationKind.GELU_ERF:
            # erfc avoids subtracting nearly equal numbers in the negative tail.
            return (0.5 * value) * erfc(-value / sqrt(2.0))
        # Multiplication may saturate to infinity; unlike x**3 it does not
        # raise OverflowError. Tanh then supplies its correct limiting sign.
        argument = sqrt(2.0 / pi) * (
            value + _GELU_TANH_CUBIC_COEFFICIENT * value * value * value
        )
        return (0.5 * value) * (1.0 + tanh(argument))
    raise ValueError("unsupported activation formula")


def _softmax_row(values: tuple[float, ...]) -> tuple[float, ...]:
    """Normalize one finite-or-masked row using stable mathematical stages.

    Args:
        values: Nonempty row of scores; negative infinity masks a position.

    Returns:
        Probabilities in input order, with masked positions exactly zero.

    Raises:
        ValueError: If the row is empty, contains NaN or positive infinity,
            or every position is masked.
    """
    if not values or any(isnan(value) or value == inf for value in values):
        raise ValueError(
            "softmax requires a nonempty row without NaN or positive infinity"
        )
    maximum = max(values)
    if maximum == -inf:
        raise ValueError("softmax requires at least one unmasked position per row")
    # Subtract the largest score so exponents are in [0,1]. At least one is
    # exactly one, preventing division by zero even when other scores underflow.
    exponentials = tuple(exp(value - maximum) for value in values)
    total = fsum(exponentials)
    return tuple(value / total for value in exponentials)


def _rotate_head(
    head: tuple[float, ...],
    cos: tuple[float, ...],
    sin: tuple[float, ...],
    is_neox_style: bool,
) -> tuple[float, ...]:
    """Rotate the leading pairs of one head vector by the given angles.

    Args:
        head: One head vector; its first ``2 * len(cos)`` scalars are rotated.
        cos: Cosine of each pair's angle, in pair order.
        sin: Sine of each pair's angle, in pair order.
        is_neox_style: True pairs element ``i`` with ``i + len(cos)``; False
            pairs ``2i`` with ``2i + 1``.

    Returns:
        The rotated head vector; elements after the rotated prefix are kept.

    Examples:
        GPT-J pairing of ``(a0, b0, a1, b1)`` with ``cos=(0, 1)`` and
        ``sin=(1, 0)`` gives ``(-b0, a0, a1, b1)``.
    """
    half = len(cos)
    rotated = list(head)
    for pair in range(half):
        # Index of each pair's first and second element in the head vector.
        first, second = (
            (pair, pair + half) if is_neox_style else (2 * pair, 2 * pair + 1)
        )
        a, b = head[first], head[second]
        rotated[first] = a * cos[pair] - b * sin[pair]
        rotated[second] = b * cos[pair] + a * sin[pair]
    return tuple(rotated)


@dataclass(slots=True)
class _ReferenceContext:
    """Collect one instruction's snapshots before constructing any writes.

    Attributes:
        state: Live memory read by the instruction; this context never mutates it.
        reads: Physical regions consumed, in read order, for execution tracing.
    """

    state: CentDeviceState
    reads: list[CentPhysicalRegion] = field(default_factory=list)

    def read(self, address: CentSharedBufferAddress, count: int) -> tuple[float, ...]:
        """Snapshot a scalar prefix and record its physical span.

        Args:
            address: First Shared Buffer slot.
            count: Number of initialized scalars to read from lane zero.

        Returns:
            Stored values without another numerical conversion.
        """
        self.reads.append(CentSharedBufferRegion(address=address, scalar_count=count))
        return self.state.read_shared_buffer_values(address, value_count=count)

    def slice(
        self, address: CentSharedBufferAddress, offset: int, count: int
    ) -> tuple[float, ...]:
        """Read a selected tensor slice without traversing all preceding slots.

        Args:
            address: First slot of the containing tensor.
            offset: Scalar offset into that tensor.
            count: Number of selected scalar elements.

        Returns:
            The selected values. Leading lanes in the first touched slot are
            read and traced, then excluded from this tuple.
        """
        slot, lane = divmod(offset, self.state.hardware.burst_length)
        first = CentSharedBufferAddress(slot=address.slot + slot)
        return self.read(first, lane + count)[lane:]

    def zeros(self, count: int) -> tuple[float, ...]:
        """Convert zero only when the result actually contains new zero lanes.

        Args:
            count: Number of new zero-valued lanes, possibly zero.

        Returns:
            Repeated stored zero, or an empty tuple without calling conversion.
        """
        # A policy may reject zero. Complete-slot copies must not fail because
        # a padding value that would never be written was converted anyway.
        return (self.state.numeric.store(0.0),) * count if count else ()

    def finish(
        self, destination: CentSharedBufferAddress, values: tuple[float, ...]
    ) -> _InstructionEffects:
        """Propose one padded Shared Buffer write without committing it.

        Args:
            destination: First output slot, possibly within a larger tensor.
            values: Computed logical scalars already in stored representation.

        Returns:
            Recorded reads and one write with zero-valued final padding.
        """
        padded = values + self.zeros(-len(values) % self.state.hardware.burst_length)
        return _InstructionEffects(
            reads=tuple(self.reads),
            writes=(SharedBufferWriteEffect(address=destination, values=padded),),
        )


def _prepare_accumulator_access(
    instruction: InitializeAccumulator | ReadAccumulator, context: _ReferenceContext
) -> _InstructionEffects:
    """Initialize selected bank registers or pack their current values.

    Args:
        instruction: Register selection and initialization value or output address.
        context: Live state and the read log for this instruction.

    Returns:
        All register writes or one channel-major, bank-major readback write.
    """
    state = context.state
    if isinstance(instruction, InitializeAccumulator):
        value = state.numeric.store(instruction.value)
        return _InstructionEffects(
            reads=(),
            writes=tuple(
                AccumulatorWriteEffect(
                    address=CentBankRegisterAddress(
                        channel=channel,
                        bank=bank,
                        register=instruction.accumulation_register,
                    ),
                    value=value,
                )
                for channel in instruction.channels.channels
                for bank in range(state.hardware.num_banks)
            ),
        )
    values: list[float] = []
    for channel in instruction.channels.channels:
        for bank in range(state.hardware.num_banks):
            address = CentBankRegisterAddress(
                channel=channel, bank=bank, register=instruction.accumulation_register
            )
            context.reads.append(CentAccumulatorRegion(address=address))
            values.append(state.read_accumulator(address))
    return context.finish(instruction.destination, tuple(values))


def _prepare_movement(
    instruction: ReferenceCopy
    | ReferenceStoreSlice
    | ReferenceTranspose
    | ReferenceFill
    | ReferenceBroadcast,
    context: _ReferenceContext,
) -> _InstructionEffects:
    """Pack, transpose, fill, or gather values without reconverting copied data.

    Args:
        instruction: Selected source slice and destination layout.
        context: Snapshot and padding helpers for this instruction.

    Returns:
        One proposed output write; overlapping inputs are read before it exists.
    """
    if isinstance(instruction, (ReferenceCopy, ReferenceStoreSlice)):
        source = context.slice(
            instruction.source, instruction.source_offset, instruction.value_count
        )
        if isinstance(instruction, ReferenceCopy):
            values = source + context.zeros(
                instruction.destination_count - instruction.value_count
            )
        else:
            # Preserve only touched slots. Reading and rewriting the entire
            # output on every gather would make row-by-row GEMV quadratic.
            lanes = context.state.hardware.burst_length
            slot, lane = divmod(instruction.destination_offset, lanes)
            address = CentSharedBufferAddress(slot=instruction.destination.slot + slot)
            count = ((lane + instruction.value_count + lanes - 1) // lanes) * lanes
            preserved = list(context.read(address, count))
            preserved[lane : lane + instruction.value_count] = source
            return context.finish(address, tuple(preserved))
    elif isinstance(instruction, ReferenceTranspose):
        source = context.slice(
            instruction.source,
            instruction.source_offset,
            instruction.row_count * instruction.column_count,
        )
        # Output row c contains input column c, whose values are column_count
        # scalars apart in the original row-major matrix.
        values = tuple(
            source[row * instruction.column_count + column]
            for column in range(instruction.column_count)
            for row in range(instruction.row_count)
        )
    elif isinstance(instruction, ReferenceFill):
        values = (
            context.state.numeric.store(instruction.value),
        ) * instruction.value_count
    else:
        values = context.read(instruction.source, 1) * instruction.value_count
    return context.finish(instruction.destination, values)


def _prepare_attention_scores(
    instruction: ReferenceAttentionScores, context: _ReferenceContext
) -> _InstructionEffects:
    """Scale scores and apply masks before any output replaces live values.

    Args:
        instruction: Score scale, optional additive slice, and causal boundary.
        context: Input snapshots and the numeric policy used for arithmetic.

    Returns:
        A padded score-vector write with excluded positions set to negative infinity.

    Raises:
        ValueError: If any mask entry is invalid or the numeric policy rejects arithmetic.
    """
    source = context.read(instruction.source, instruction.value_count)
    mask = (
        context.slice(
            instruction.mask, instruction.mask_offset, instruction.value_count
        )
        if instruction.mask is not None
        else None
    )
    if mask is not None and any(isnan(value) or value == inf for value in mask):
        raise ValueError("attention mask values must be finite or negative infinity")
    numeric = context.state.numeric
    # The immediate is a host scalar; convert it once before stored arithmetic.
    scale = numeric.store(instruction.scale)
    values: list[float] = []
    for key, score in enumerate(source):
        additive = mask[key] if mask is not None else 0.0
        if (
            instruction.causal_limit is not None and key > instruction.causal_limit
        ) or additive == -inf:
            # Selecting the mask first avoids evaluating inf + -inf for an
            # explicitly excluded position, whose probability must be zero.
            values.append(numeric.store(-inf))
        else:
            scaled = numeric.multiply(score, scale)
            values.append(numeric.add(scaled, additive) if mask is not None else scaled)
    return context.finish(instruction.destination, tuple(values))


def _prepare_nonlinear(
    instruction: ReferenceActivation | ReferenceSoftmax, context: _ReferenceContext
) -> _InstructionEffects:
    """Apply scalar activation formulas or independent row-wise softmax.

    Args:
        instruction: Logical tensor prefix and nonlinear calculation.
        context: Input snapshots and conversion policy for final results.

    Returns:
        One complete output write, created only after every row succeeds.

    Raises:
        ValueError: If the formula, a softmax row, or result conversion is invalid.
    """
    count = (
        instruction.value_count
        if isinstance(instruction, ReferenceActivation)
        else instruction.row_count * instruction.row_width
    )
    source = context.read(instruction.source, count)
    if isinstance(instruction, ReferenceActivation):
        result = tuple(
            _activate_scalar(value, instruction.function) for value in source
        )
    else:
        # A bad final row must not commit earlier rows or destroy aliased input.
        result = tuple(
            value
            for start in range(0, count, instruction.row_width)
            for value in _softmax_row(source[start : start + instruction.row_width])
        )
    return context.finish(
        instruction.destination,
        tuple(context.state.numeric.store(value) for value in result),
    )


def _prepare_rotary_embedding(
    instruction: ReferenceRotaryEmbedding, context: _ReferenceContext
) -> _InstructionEffects:
    """Rotate every head vector by the table angles of its token's position.

    Args:
        instruction: Head vectors, positions, angle table, and pairing style.
        context: Input snapshots and conversion policy for final results.

    Returns:
        One complete output write, created only after every input is read.

    Raises:
        ValueError: If a position is not an integer row of the table, or
            result conversion is invalid.
    """
    width = instruction.head_dim
    vectors_per_token = instruction.head_count * width
    source = context.read(
        instruction.source, instruction.token_count * vectors_per_token
    )
    positions = context.read(instruction.positions, instruction.token_count)
    table = context.read(
        instruction.table, instruction.position_count * instruction.rotary_dim
    )
    half = instruction.rotary_dim // 2
    result: list[float] = []
    for token, position in enumerate(positions):
        # Positions arrive as stored scalars. Only an exact integer names a row.
        if not position.is_integer() or not 0 <= position < instruction.position_count:
            raise ValueError(
                "rotary position must be an integer row of the angle table"
            )
        row = int(position) * instruction.rotary_dim
        cos, sin = table[row : row + half], table[row + half : row + 2 * half]
        # Every head of this token rotates by the same angles.
        for head in range(instruction.head_count):
            start = token * vectors_per_token + head * width
            result.extend(
                _rotate_head(
                    source[start : start + width], cos, sin, instruction.is_neox_style
                )
            )
    return context.finish(
        instruction.destination,
        tuple(context.state.numeric.store(value) for value in result),
    )


def _prepare_reduction(
    instruction: ReferenceReduceSum | ReferenceRmsScale, context: _ReferenceContext
) -> _InstructionEffects:
    """Reduce logical values or turn their square sum into an RMS scale.

    Args:
        instruction: Input prefix or scalar, with normalization parameters.
        context: Initialized inputs and selected arithmetic/conversion policy.

    Returns:
        One scalar in output lane zero, followed by zero padding.

    Raises:
        ValueError: If the root argument is nonpositive or the numeric policy rejects arithmetic.
    """
    numeric = context.state.numeric
    if isinstance(instruction, ReferenceReduceSum):
        source = context.read(instruction.source, instruction.value_count)
        value = numeric.store(0.0)
        for item in source:
            value = numeric.add(value, item)
    else:
        scalar = context.read(instruction.source, 1)[0]
        # Python division/root is an explicit software service, not a RISC-V ABI.
        radicand = scalar / instruction.value_count + instruction.epsilon
        if radicand <= 0:
            raise ValueError("reciprocal square root requires a positive argument")
        value = numeric.store(1.0 / sqrt(radicand))
    return context.finish(instruction.destination, (value,))


def _prepare_reference(
    instruction: ReferenceInstruction, state: CentDeviceState
) -> _InstructionEffects:
    """Select a reference service and attach its destination to arithmetic failures.

    Args:
        instruction: Structurally validated reference command.
        state: Initialized memory and numeric policy; never changed during preparation.

    Returns:
        Exact read regions and proposed writes, committed later as one instruction.

    Raises:
        CentUninitializedReadError: If a required input or preserved edge slot is unwritten.
        _KernelArithmeticError: If arithmetic or storage conversion is rejected.
        TypeError: If the reference command has no implementation.
    """
    context = _ReferenceContext(state)
    location: CentBankRegisterAddress | CentSharedBufferAddress
    try:
        if isinstance(instruction, (InitializeAccumulator, ReadAccumulator)):
            location = (
                CentBankRegisterAddress(
                    channel=instruction.channels.channels[0],
                    bank=0,
                    register=instruction.accumulation_register,
                )
                if isinstance(instruction, InitializeAccumulator)
                else instruction.destination
            )
            return _prepare_accumulator_access(instruction, context)
        if isinstance(
            instruction,
            (
                ReferenceCopy,
                ReferenceStoreSlice,
                ReferenceTranspose,
                ReferenceFill,
                ReferenceBroadcast,
            ),
        ):
            location = instruction.destination
            return _prepare_movement(instruction, context)
        if isinstance(instruction, ReferenceAttentionScores):
            location = instruction.destination
            return _prepare_attention_scores(instruction, context)
        if isinstance(instruction, (ReferenceActivation, ReferenceSoftmax)):
            location = instruction.destination
            return _prepare_nonlinear(instruction, context)
        if isinstance(instruction, (ReferenceReduceSum, ReferenceRmsScale)):
            location = instruction.destination
            return _prepare_reduction(instruction, context)
        if isinstance(instruction, ReferenceRotaryEmbedding):
            location = instruction.destination
            return _prepare_rotary_embedding(instruction, context)
        raise TypeError("unsupported reference instruction")
    except ValueError as error:
        raise _KernelArithmeticError(str(error), location=location) from error


def _prepare_mac(
    instruction: MacAllBanks, state: CentDeviceState
) -> _InstructionEffects:
    """Accumulate bank products into the explicitly selected register.

    Args:
        instruction: Bank row span and source mode; next-bank mode updates even banks.
        state: Initialized DRAM, Global Buffer, and accumulators.

    Returns:
        All operand/register reads and one write per participating bank.

    Raises:
        CentUninitializedReadError: If any operand or starting accumulator is unwritten.
        _KernelArithmeticError: If numeric multiplication or addition is rejected.
    """
    reads: list[CentPhysicalRegion] = []
    writes: list[AccumulatorWriteEffect] = []
    count = instruction.operation_size * state.hardware.burst_length
    for channel in instruction.channels.channels:
        step = 1 if instruction.operand_source is MacOperandSource.GLOBAL_BUFFER else 2
        for bank in range(0, state.hardware.num_banks, step):
            address = CentMemoryAddress(
                channel=channel,
                bank=bank,
                row=instruction.row,
                column=instruction.column,
            )
            reads.append(CentDramRegion(address=address, scalar_count=count))
            left = state.read_dram(address, value_count=count)
            if instruction.operand_source is MacOperandSource.GLOBAL_BUFFER:
                global_address = CentGlobalBufferAddress(
                    channel=channel, column=instruction.column
                )
                reads.append(
                    CentGlobalBufferRegion(address=global_address, scalar_count=count)
                )
                right = state.read_global_buffer(global_address, value_count=count)
            else:
                neighbor = CentMemoryAddress(
                    channel=channel,
                    bank=bank + 1,
                    row=instruction.row,
                    column=instruction.column,
                )
                reads.append(CentDramRegion(address=neighbor, scalar_count=count))
                right = state.read_dram(neighbor, value_count=count)
            register = CentBankRegisterAddress(
                channel=channel, bank=bank, register=instruction.accumulation_register
            )
            reads.append(CentAccumulatorRegion(address=register))
            total = state.read_accumulator(register)
            try:
                for first, second in zip(left, right, strict=True):
                    total = state.numeric.add(
                        total, state.numeric.multiply(first, second)
                    )
            except ValueError as error:
                raise _KernelArithmeticError(str(error), location=register) from error
            writes.append(AccumulatorWriteEffect(address=register, value=total))
    return _InstructionEffects(reads=tuple(reads), writes=tuple(writes))
