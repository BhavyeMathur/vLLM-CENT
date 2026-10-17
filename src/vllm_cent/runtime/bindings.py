"""Describe host values and the device memory locations that hold them.

A scalar is one number. A lane is one scalar position in a buffer; a Shared
Buffer slot contains ``hardware.burst_length`` lanes. DRAM (dynamic random
access memory) addresses select a channel, bank, row, and scalar column.
"""

from dataclasses import dataclass
from typing import TypeAlias

from ..cent import (
    CentBankRegisterAddress,
    CentGlobalBufferAddress,
    CentMemoryAddress,
    CentSharedBufferAddress,
)

__all__ = [
    "CentAccumulatorRegion",
    "CentDramRegion",
    "CentGlobalBufferRegion",
    "CentInputBinding",
    "CentNamedScalars",
    "CentOutputBinding",
    "CentPhysicalRegion",
    "CentSharedBufferRegion",
]


def _validate_name(name: str) -> None:
    """Require a nonempty runtime value name.

    Args:
        name: Name to validate.

    Raises:
        ValueError: If ``name`` is empty.
    """

    if not name:
        raise ValueError("name cannot be empty")


def _validate_scalar_count(scalar_count: int) -> None:
    """Require the memory region to contain at least one value.

    Args:
        scalar_count: Number of consecutive scalar lanes in the region.

    Raises:
        ValueError: If ``scalar_count`` is less than one.
    """

    if scalar_count < 1:
        raise ValueError("scalar_count must be at least 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class CentNamedScalars:
    """Provide host numbers associated with one input or output name.

    The binding with the same name tells the runtime where these numbers go.
    This type does not reshape values, reorder them, or add padding. The
    execution target determines how to represent each number.

    Attributes:
        name: Input or output binding name.
        values: Scalars in the physical lane order declared by the binding.

    Examples:
        A three-number input bound to DRAM column 5 fills columns 5, 6, and 7.
    """

    name: str
    values: tuple[float, ...]

    def __post_init__(self) -> None:
        """Validate the name and require at least one supplied scalar.

        Raises:
            ValueError: If the name is empty or no values are supplied.
        """

        _validate_name(self.name)
        if not self.values:
            raise ValueError("named scalars must contain at least one scalar")


@dataclass(frozen=True, slots=True, kw_only=True)
class CentDramRegion:
    """Identify consecutive scalar columns within one DRAM row.

    Attributes:
        address: Address of the first physical scalar.
        scalar_count: Number of consecutive columns in the row.
    """

    address: CentMemoryAddress
    scalar_count: int

    def __post_init__(self) -> None:
        """Check that the region includes at least one scalar.

        Raises:
            ValueError: If the region contains no scalars.
        """

        _validate_scalar_count(self.scalar_count)


@dataclass(frozen=True, slots=True, kw_only=True)
class CentSharedBufferRegion:
    """Identify consecutive Shared Buffer scalar lanes.

    A Shared Buffer address selects the first complete burst-sized slot. The
    region begins at lane zero of that slot and may end inside a later slot.

    Attributes:
        address: First Shared Buffer slot occupied by the region.
        scalar_count: Number of scalars, filling each slot before moving to the next.

    Examples:
        With four lanes per slot, six values at slot 2 occupy all of slot 2 and
        the first two lanes of slot 3. The other two lanes are outside the region.
    """

    address: CentSharedBufferAddress
    scalar_count: int

    def __post_init__(self) -> None:
        """Check that the region includes at least one scalar.

        Raises:
            ValueError: If the region contains no scalars.
        """

        _validate_scalar_count(self.scalar_count)


@dataclass(frozen=True, slots=True, kw_only=True)
class CentGlobalBufferRegion:
    """Identify consecutive scalars in one channel's Global Buffer.

    Attributes:
        address: Channel and first scalar column occupied by the region.
        scalar_count: Number of consecutive Global Buffer columns.
    """

    address: CentGlobalBufferAddress
    scalar_count: int

    def __post_init__(self) -> None:
        """Check that the region includes at least one scalar.

        Raises:
            ValueError: If the region contains no scalars.
        """

        _validate_scalar_count(self.scalar_count)


@dataclass(frozen=True, slots=True, kw_only=True)
class CentAccumulatorRegion:
    """Identify one register in traces; host bindings must use buffers.

    Attributes:
        address: Channel, bank, and selected accumulation register.
    """

    address: CentBankRegisterAddress

    @property
    def scalar_count(self) -> int:
        """Return one scalar stored in this register."""
        return 1


CentPhysicalRegion: TypeAlias = (
    CentDramRegion
    | CentSharedBufferRegion
    | CentGlobalBufferRegion
    | CentAccumulatorRegion
)
"""A contiguous range of scalar positions in one device memory space."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CentInputBinding:
    """Bind named host scalars to a physical region before execution.

    Attributes:
        name: Name used to match one ``CentNamedScalars`` request value.
        region: Memory range filled with the named input before execution starts.
    """

    name: str
    region: CentPhysicalRegion

    def __post_init__(self) -> None:
        """Require a stable nonempty input name.

        Raises:
            ValueError: If ``name`` is empty.
        """

        _validate_name(self.name)


@dataclass(frozen=True, slots=True, kw_only=True)
class CentOutputBinding:
    """Bind a result name to a physical region read after execution.

    Attributes:
        name: Name assigned to the extracted ``CentNamedScalars`` value.
        region: Memory range read to obtain the named output after execution.
    """

    name: str
    region: CentPhysicalRegion

    def __post_init__(self) -> None:
        """Require a stable nonempty output name.

        Raises:
            ValueError: If ``name`` is empty.
        """

        _validate_name(self.name)
