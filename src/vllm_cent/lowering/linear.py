"""Plan and lower model-independent matrix-vector multiplication to CENT."""

from dataclasses import dataclass

from ..cent import (
    ApplyActivation,
    CentBlockPlacementSpec,
    CentChannelSet,
    CentHardwareSpec,
    CentProgramBuilder,
    MacAllBanks,
    MacOperandSource,
    ReadActivation,
    ReadMac,
    WriteBias,
    WriteGlobalBuffer,
    ceil_div,
)
from ..cent.utils import require_positive
from .bindings import (
    CentDramRowRange,
    CentSharedBufferSpan,
    CentSharedBufferVector,
)
from .utils import (
    _require_dram_row_capacity,
    _require_shared_buffer_capacity,
)

__all__ = ["CentWeightGemvPlan", "lower_weight_gemv", "plan_weight_gemv"]


@dataclass(frozen=True, slots=True, kw_only=True)
class CentWeightGemvPlan:
    """Describe one matrix-vector multiplication and its physical layout.

    Attributes:
        weights: DRAM rows containing the distributed weight matrix.
        input_buffer: Zero-padded Shared Buffer input vector.
        output_buffer: Slots receiving raw matrix results and partial sums.
        vector_size: Number of values in the input vector.
        output_size: Number of values produced by the matrix.
        outputs_per_bank: Output rows assigned to each used DRAM bank.
        accumulator_group_size: Outputs calculated before accumulator registers
            are reused for the next group.
        channels: Physical channels that execute the matrix operation.
        activated_output_buffer: Separate slots receiving sigmoid results. A
            value of ``None`` disables activation lowering.
    """

    weights: CentDramRowRange
    input_buffer: CentSharedBufferVector
    output_buffer: CentSharedBufferSpan
    vector_size: int
    output_size: int
    outputs_per_bank: int
    accumulator_group_size: int
    channels: CentChannelSet
    activated_output_buffer: CentSharedBufferSpan | None = None

    def __post_init__(self) -> None:
        """Validate dimensions that do not depend on a target device.

        Raises:
            ValueError: If a dimension is invalid, the input layout disagrees,
                or output spans overlap.
        """

        for name, value in (
            ("vector_size", self.vector_size),
            ("output_size", self.output_size),
            ("outputs_per_bank", self.outputs_per_bank),
            ("accumulator_group_size", self.accumulator_group_size),
        ):
            require_positive(name, value)
        if self.accumulator_group_size > self.outputs_per_bank:
            raise ValueError("accumulator_group_size cannot exceed outputs_per_bank")
        if self.input_buffer.layout.value_count != self.vector_size:
            raise ValueError("input_buffer layout must match vector_size")
        if not self.input_buffer.layout.is_contiguously_packed:
            raise ValueError("GEMV input_buffer must be contiguously packed")

        if self.activated_output_buffer is not None:
            raw_start = self.output_buffer.start.slot
            raw_end = raw_start + self.outputs_per_bank
            activated_start = self.activated_output_buffer.start.slot
            activated_end = activated_start + self.outputs_per_bank
            if raw_start < activated_end and activated_start < raw_end:
                raise ValueError("raw and activated output spans cannot overlap")

    @property
    def utilized_banks(self) -> int:
        """Return the number of banks that contain output rows.

        Returns:
            Banks needed for all output values in this layout.
        """

        return ceil_div(self.output_size, self.outputs_per_bank)


def plan_weight_gemv(
    hardware: CentHardwareSpec,
    placement: CentBlockPlacementSpec,
    *,
    weights: CentDramRowRange,
    input_buffer: CentSharedBufferVector,
    output_buffer: CentSharedBufferSpan,
    vector_size: int,
    output_size: int,
    activated_output_buffer: CentSharedBufferSpan | None = None,
) -> CentWeightGemvPlan:
    """Create the baseline maximum-parallelism GEMV plan.

    The policy spreads outputs across all block-local banks, then divides each
    bank's outputs into groups that fit its accumulator registers.

    Args:
        hardware: CENT device that will execute the plan.
        placement: Number of device channels assigned to the model block.
        weights: DRAM rows containing the distributed weight matrix.
        input_buffer: Shared Buffer slots containing the input vector.
        output_buffer: Slots receiving raw matrix results and partial sums.
        vector_size: Number of values in the input vector.
        output_size: Number of values produced by the matrix.
        activated_output_buffer: Separate slots receiving sigmoid results. A
            value of ``None`` disables activation lowering.

    Returns:
        Immutable GEMV plan that preserves the baseline lowering policy.

    Raises:
        ValueError: If a dimension or target capacity is invalid.
    """

    require_positive("vector_size", vector_size)
    require_positive("output_size", output_size)
    if placement.channels_per_block > hardware.num_channels:
        raise ValueError("channels_per_block cannot exceed num_channels")
    if hardware.num_channels % placement.channels_per_block:
        raise ValueError("channels_per_block must divide num_channels")

    total_banks = placement.channels_per_block * hardware.num_banks
    outputs_per_bank = ceil_div(output_size, total_banks)
    utilized_banks = ceil_div(output_size, outputs_per_bank)

    # TODO(placement): Distinguish matrix partitions from replicated block copies.
    #
    # This dense channel prefix can cross from one block copy into another when
    # a matrix uses fewer channels than ``channels_per_block``. The planner must
    # eventually represent each copy explicitly instead of merging them into one
    # channel mask.

    # A copied matrix occupies complete channels. Select every complete copy
    # that fits on the device, matching the current builder mapping.
    channels_per_matrix = ceil_div(utilized_banks, hardware.num_banks)
    channel_count = (hardware.num_channels // channels_per_matrix) * channels_per_matrix
    channels = CentChannelSet(channels=tuple(range(channel_count)))

    # Activation reuses accumulator hardware after the raw result is read. The
    # current implementation reserves half the registers for each role.
    accumulator_limit = hardware.accumulator_slots_per_bank
    if activated_output_buffer is not None:
        if accumulator_limit < 2:
            raise ValueError("activated GEMV requires at least two accumulator slots")
        accumulator_limit //= 2
    group_count = ceil_div(outputs_per_bank, accumulator_limit)
    accumulator_group_size = ceil_div(outputs_per_bank, group_count)

    return CentWeightGemvPlan(
        weights=weights,
        input_buffer=input_buffer,
        output_buffer=output_buffer,
        vector_size=vector_size,
        output_size=output_size,
        outputs_per_bank=outputs_per_bank,
        accumulator_group_size=accumulator_group_size,
        channels=channels,
        activated_output_buffer=activated_output_buffer,
    )


def lower_weight_gemv(
    builder: CentProgramBuilder,
    plan: CentWeightGemvPlan,
) -> None:
    """Multiply a Shared Buffer vector by a weight matrix in DRAM.

    The matrix is distributed according to ``plan``. Output values at the same
    bank position use consecutive DRAM rows.

    Args:
        builder: Program builder that receives the multiplication instructions.
        plan: Matrix dimensions, physical layout, and memory bindings.

    Raises:
        ValueError: If the plan is incompatible with the target or a memory
            region is too small.
    """

    # TODO(ISA): Define how a dot product continues across input rows.
    #
    # RD_MAC reads a partial sum and WR_BIAS writes it back. WR_BIAS has no
    # register operand, so the paper does not say which register receives it.

    # TODO(dataflow): Define how RD_MAC packs results selected by CHmask.
    #
    # The current mapping assigns one Shared Buffer slot to each accumulator
    # register. We still need the exact ordering of results from several banks
    # and channels.

    # TODO(runtime): Initialize each output slot before its first WR_BIAS.
    #
    # Later input rows reload a real partial sum. The first input row instead
    # needs zero, but this compiler does not yet emit or bind that value.

    # TODO(dataflow): Define unused-bank behavior and pack GEMV results.
    #
    # ``input_buffer`` guarantees that the last input burst is zero-padded.
    # MAC_ABK still runs every bank in a selected channel, and RD_MAC's lane
    # order is unknown. Keep its output as a raw span until a repacker omits
    # unused banks and writes zero into every padding lane of the packed vector.

    hardware = builder.hardware
    if plan.utilized_banks > builder.total_banks:
        raise ValueError("plan uses more banks than the block owns")
    builder.channel_set(plan.channels.channels)
    if len(plan.channels.channels) * hardware.num_banks < plan.utilized_banks:
        raise ValueError("selected channels cannot cover every utilized bank")

    accumulator_limit = hardware.accumulator_slots_per_bank
    if plan.activated_output_buffer is not None:
        if accumulator_limit < 2:
            raise ValueError("activated GEMV requires at least two accumulator slots")
        accumulator_limit //= 2
    if plan.accumulator_group_size > accumulator_limit:
        raise ValueError("accumulator_group_size exceeds the target capacity")

    rows_per_output = ceil_div(plan.vector_size, hardware.dram_columns)
    required_weight_rows = plan.outputs_per_bank * rows_per_output
    _require_dram_row_capacity("weights", plan.weights, required_weight_rows)

    _require_shared_buffer_capacity(
        "output_buffer", plan.output_buffer, plan.outputs_per_bank
    )
    if plan.activated_output_buffer is not None:
        _require_shared_buffer_capacity(
            "activated_output_buffer",
            plan.activated_output_buffer,
            plan.outputs_per_bank,
        )

    group_count = ceil_div(plan.outputs_per_bank, plan.accumulator_group_size)
    slots_per_full_input_row = hardware.dram_columns // hardware.burst_length

    for vector_row in range(rows_per_output):
        # Move this input slice into each selected channel's Global Buffer. A
        # vector wider than one DRAM row advances through its input span.
        remaining_values = plan.vector_size - vector_row * hardware.dram_columns
        row_value_count = min(remaining_values, hardware.dram_columns)
        operation_size = ceil_div(row_value_count, hardware.burst_length)
        builder.append(
            WriteGlobalBuffer(
                operation_size=operation_size,
                column=0,
                source=plan.input_buffer.address(vector_row * slots_per_full_input_row),
                channels=plan.channels,
            )
        )

        for group in range(group_count):
            first_output = group * plan.accumulator_group_size
            output_count = min(
                plan.accumulator_group_size,
                plan.outputs_per_bank - first_output,
            )

            # Each output has one partial-sum slot. A later input row reloads
            # that slot before continuing the dot product.
            for local_output in range(output_count):
                output_offset = first_output + local_output
                builder.append(
                    WriteBias(
                        source=plan.output_buffer.address(output_offset),
                        channels=plan.channels,
                    )
                )

            for local_output in range(output_count):
                output_index = first_output + local_output
                weight_row_offset = output_index * rows_per_output + vector_row
                builder.append(
                    MacAllBanks(
                        operation_size=operation_size,
                        channels=plan.channels,
                        row=plan.weights.row(weight_row_offset),
                        column=0,
                        accumulation_register=local_output,
                        operand_source=MacOperandSource.GLOBAL_BUFFER,
                    )
                )

            # Always preserve the raw value. On earlier vector rows this value
            # is the partial sum loaded again by the next WR_BIAS pass.
            for local_output in range(output_count):
                output_offset = first_output + local_output
                builder.append(
                    ReadMac(
                        destination=plan.output_buffer.address(output_offset),
                        accumulation_register=local_output,
                        channels=plan.channels,
                    )
                )

            # Sigmoid is meaningful only after the final input slice completes
            # the dot product. Its result has a different destination.
            if (
                plan.activated_output_buffer is not None
                and vector_row == rows_per_output - 1
            ):
                for local_output in range(output_count):
                    builder.append(
                        ApplyActivation(
                            channels=plan.channels,
                            activation_function_id=(
                                hardware.sigmoid_activation_function_id
                            ),
                            accumulation_register=local_output,
                        )
                    )
                    output_offset = first_output + local_output
                    builder.append(
                        ReadActivation(
                            destination=(
                                plan.activated_output_buffer.address(output_offset)
                            ),
                            accumulation_register=local_output,
                            channels=plan.channels,
                        )
                    )
