"""Pair instructions with the memory locations used for inputs and outputs.

The manifest is the list of named input and output bindings. Its checks prevent
missing inputs, writes outside device memory, and two inputs overwriting each
other. It does not choose a tensor layout or rearrange host values.
"""

from dataclasses import dataclass
from enum import Enum, auto
from typing import Literal, assert_never

from .bindings import (
    CentAccumulatorRegion,
    CentDramRegion,
    CentGlobalBufferRegion,
    CentInputBinding,
    CentNamedScalars,
    CentOutputBinding,
    CentPhysicalRegion,
    CentSharedBufferRegion,
)
from ..cent import CentHardwareSpec, CentProgram
from ..cent.instructions.validation import (
    validate_address,
    validate_shared_buffer_address,
)

__all__ = ["CentExecutable", "CentExecutionManifest"]


class _CentStorageSpace(Enum):
    """Distinguish physical address spaces during overlap validation.

    Members:
        DRAM: One row within one channel and bank.
        SHARED_BUFFER: Device-wide Shared Buffer scalar lanes.
        GLOBAL_BUFFER: Scalar columns in one channel's Global Buffer.
    """

    DRAM = auto()
    SHARED_BUFFER = auto()
    GLOBAL_BUFFER = auto()


@dataclass(frozen=True, slots=True, kw_only=True)
class _CentPhysicalSpan:
    """Describe a memory range with a start index and an exclusive end.

    Two spans can overlap only when their storage kind and coordinates agree.

    Attributes:
        space: Storage kind containing the interval.
        coordinates: Indices selecting one independent storage region. DRAM
            uses channel, bank, and row; a Global Buffer uses its channel; the
            device-wide Shared Buffer uses an empty tuple.
        start: First scalar lane in the selected storage region.
        end: Index immediately after the last scalar included in this span.

    Examples:
        A span with ``start=2`` and ``end=5`` includes scalars 2, 3, and 4.
    """

    space: _CentStorageSpace
    coordinates: tuple[int, ...]
    start: int
    end: int

    def overlaps(self, other: _CentPhysicalSpan) -> bool:
        """Return whether this span and another span share a scalar lane.

        Args:
            other: Normalized physical span to compare.

        Returns:
            ``True`` when both spans select the same storage region and their
            half-open scalar intervals intersect.
        """

        same_region = (
            self.space is other.space and self.coordinates == other.coordinates
        )
        return same_region and self.start < other.end and other.start < self.end


@dataclass(frozen=True, slots=True, kw_only=True)
class CentExecutionManifest:
    """List where program inputs are written and outputs are read.

    A name may occur once among inputs and once among outputs, so an updated
    value can keep its name. Two inputs cannot write the same scalar. Outputs
    may read overlapping ranges because reading does not change the values.
    These bindings describe memory locations only: they do not specify tensor
    shapes, reorder values, add padding, or select a numeric format.

    Attributes:
        inputs: Named physical regions populated before instruction zero.
        outputs: Named physical regions extracted after the final instruction.
    """

    inputs: tuple[CentInputBinding, ...] = ()
    outputs: tuple[CentOutputBinding, ...] = ()

    def __post_init__(self) -> None:
        """Reject repeated names and input overlaps detectable without slot sizes.

        Raises:
            ValueError: If names repeat within one direction or two inputs
                overlap even before the hardware slot size is known.
        """

        _validate_unique_names(self.inputs, category="input")
        _validate_unique_names(self.outputs, category="output")
        _validate_no_input_overlaps(self.inputs)

    def validate_hardware(self, hardware: CentHardwareSpec) -> None:
        """Check that all inputs and outputs fit, and no two inputs share memory.

        The burst length gives the number of scalars in a Shared Buffer slot.

        Args:
            hardware: Hardware geometry that must contain every region.

        Raises:
            ValueError: If any region exceeds its address space or two input
                regions overlap under this target's burst length.

        Examples:
            With ``hardware.burst_length=4``, a six-scalar input at slot 0 overlaps
            an input starting at slot 1. Slot addresses alone do not reveal this;
            the check needs the number of scalars per slot.
        """

        regions: tuple[CentPhysicalRegion, ...] = tuple(
            binding.region for binding in self.inputs
        ) + tuple(binding.region for binding in self.outputs)
        for region in regions:
            _validate_region_hardware(region, hardware)
        _validate_no_input_overlaps(self.inputs, hardware=hardware)

    def order_input_values(
        self,
        values: tuple[CentNamedScalars, ...],
    ) -> tuple[CentNamedScalars, ...]:
        """Validate request values and return them in input-binding order.

        Args:
            values: Named raw host scalars supplied for this execution.

        Returns:
            The same values ordered to match ``inputs``.

        Raises:
            ValueError: If a name is duplicated, missing, unexpected, or has a
                scalar count different from its input region.
        """

        values_by_name: dict[str, CentNamedScalars] = {}
        for value in values:
            if value.name in values_by_name:
                raise ValueError(f"duplicate input value name: {value.name}")
            values_by_name[value.name] = value

        # Validate the entire name set before ordering values. A missing input
        # cannot be repaired by silently reusing another input's position.
        expected_names = {binding.name for binding in self.inputs}
        actual_names = set(values_by_name)
        missing_names = expected_names - actual_names
        if missing_names:
            missing = ", ".join(sorted(missing_names))
            raise ValueError(f"missing input values: {missing}")
        unexpected_names = actual_names - expected_names
        if unexpected_names:
            unexpected = ", ".join(sorted(unexpected_names))
            raise ValueError(f"unexpected input values: {unexpected}")

        ordered_values: list[CentNamedScalars] = []
        for binding in self.inputs:
            value = values_by_name[binding.name]
            expected_count = binding.region.scalar_count
            actual_count = len(value.values)
            if actual_count != expected_count:
                raise ValueError(
                    f"input {binding.name} expects {expected_count} scalars, "
                    f"but received {actual_count}"
                )
            ordered_values.append(value)
        return tuple(ordered_values)


@dataclass(frozen=True, slots=True, kw_only=True)
class CentExecutable:
    """Keep a program and its input/output memory bindings together.

    Attributes:
        program: Ordered typed instructions and their hardware target.
        manifest: Raw runtime inputs and outputs for that program.
    """

    program: CentProgram
    manifest: CentExecutionManifest

    def __post_init__(self) -> None:
        """Require every runtime region to fit the program's hardware.

        Raises:
            ValueError: If a manifest region is incompatible with the target.
        """

        self.manifest.validate_hardware(self.program.hardware)


def _validate_unique_names(
    bindings: tuple[CentInputBinding, ...] | tuple[CentOutputBinding, ...],
    *,
    category: Literal["input", "output"],
) -> None:
    """Require unique names within one manifest direction.

    Args:
        bindings: Input or output bindings to inspect.
        category: Human-readable direction used in validation errors.

    Raises:
        ValueError: If one name occurs more than once in ``bindings``.
    """

    seen_names: set[str] = set()
    for binding in bindings:
        if binding.name in seen_names:
            raise ValueError(f"duplicate {category} binding name: {binding.name}")
        seen_names.add(binding.name)


def _validate_no_input_overlaps(
    bindings: tuple[CentInputBinding, ...],
    *,
    hardware: CentHardwareSpec | None = None,
) -> None:
    """Reject two input bindings that initialize the same scalar lane.

    Args:
        bindings: Input bindings to inspect.
        hardware: Optional target used to convert Shared Buffer slots to scalar
            lanes. Without it, only inputs starting in the same Shared Buffer
            slot can be proven to overlap.

    Raises:
        ValueError: If two input regions share one physical scalar lane.
    """

    # Convert addresses to comparable ranges once, then compare each pair.
    # Different banks or channels remain separate even if their columns match.
    spans = tuple(
        (binding, _physical_span(binding.region, hardware)) for binding in bindings
    )
    for index, (left_binding, left_span) in enumerate(spans):
        for right_binding, right_span in spans[index + 1 :]:
            if left_span.overlaps(right_span):
                raise ValueError(
                    "input regions "
                    f"{left_binding.name} and {right_binding.name} overlap"
                )


def _physical_span(
    region: CentPhysicalRegion,
    hardware: CentHardwareSpec | None,
) -> _CentPhysicalSpan:
    """Convert a memory region to indices that can be checked for overlap.

    Args:
        region: Physical region to normalize.
        hardware: Optional target supplying the Shared Buffer burst length.

    Returns:
        Storage kind, coordinates, start index, and exclusive end index.
        Without ``hardware``, Shared Buffer ranges use slot indices and detect
        only inputs that start in the same slot.
    """

    if isinstance(region, CentDramRegion):
        address = region.address
        return _CentPhysicalSpan(
            space=_CentStorageSpace.DRAM,
            coordinates=(address.channel, address.bank, address.row),
            start=address.column,
            end=address.column + region.scalar_count,
        )
    if isinstance(region, CentSharedBufferRegion):
        # Target-independent validation can still detect two regions that begin
        # at lane zero of the same slot. Exact cross-slot overlap requires the
        # target's number of scalar lanes per slot.
        if hardware is None:
            return _CentPhysicalSpan(
                space=_CentStorageSpace.SHARED_BUFFER,
                coordinates=(),
                start=region.address.slot,
                end=region.address.slot + 1,
            )
        # Convert slots to scalar indices: at four lanes per slot, slot 2
        # begins at scalar 8. Counts in the binding are already in scalars.
        first_lane = region.address.slot * hardware.burst_length
        return _CentPhysicalSpan(
            space=_CentStorageSpace.SHARED_BUFFER,
            coordinates=(),
            start=first_lane,
            end=first_lane + region.scalar_count,
        )
    if isinstance(region, CentGlobalBufferRegion):
        return _CentPhysicalSpan(
            space=_CentStorageSpace.GLOBAL_BUFFER,
            coordinates=(region.address.channel,),
            start=region.address.column,
            end=region.address.column + region.scalar_count,
        )
    if isinstance(region, CentAccumulatorRegion):
        raise ValueError("accumulator regions are trace-only; use buffer host bindings")
    assert_never(region)


def _validate_region_hardware(
    region: CentPhysicalRegion,
    hardware: CentHardwareSpec,
) -> None:
    """Check that an input or output fits inside its selected memory area.

    Args:
        region: Starting address and number of scalar values to access.
        hardware: Device capacities and slot width used to check the entire range.

    Raises:
        ValueError: If an address is invalid, a DRAM range crosses a row, or a
            buffer range extends past its capacity.
    """

    if isinstance(region, CentDramRegion):
        validate_address(region.address, hardware)
        if region.address.column + region.scalar_count > hardware.dram_columns:
            raise ValueError("DRAM region crosses a row boundary")
        return

    if isinstance(region, CentSharedBufferRegion):
        validate_shared_buffer_address(region.address, hardware)
        # Convert slots to scalar indices: at four lanes per slot, slot 2
        # begins at scalar 8. Counts in the binding are already in scalars.
        first_lane = region.address.slot * hardware.burst_length
        buffer_lane_count = hardware.shared_buffer_slots * hardware.burst_length
        if first_lane + region.scalar_count > buffer_lane_count:
            raise ValueError("Shared Buffer region exceeds the target capacity")
        return

    if isinstance(region, CentGlobalBufferRegion):
        if region.address.channel >= hardware.num_channels:
            raise ValueError("Global Buffer region selects an invalid channel")
        if region.address.column + region.scalar_count > hardware.global_buffer_columns:
            raise ValueError("Global Buffer region exceeds the target capacity")
        return

    if isinstance(region, CentAccumulatorRegion):
        raise ValueError("accumulator regions are trace-only; use buffer host bindings")
    assert_never(region)
