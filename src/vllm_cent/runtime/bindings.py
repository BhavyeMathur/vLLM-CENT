"""Describe named host values and reusable raw physical regions."""

from dataclasses import dataclass
from typing import TypeAlias

from ..cent import (
    CentGlobalBufferAddress,
    CentMemoryAddress,
    CentSharedBufferAddress,
)

__all__ = [
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
    """Require a raw region to contain at least one scalar.

    Args:
        scalar_count: Number of consecutive scalar lanes in the region.

    Raises:
        ValueError: If ``scalar_count`` is less than one.
    """

    if scalar_count < 1:
        raise ValueError("scalar_count must be at least 1")


@dataclass(frozen=True, slots=True, kw_only=True)
class CentNamedScalars:
    """Provide one named sequence of raw host scalar values.

    A runtime binding determines the values' physical destination or source.
    This type intentionally carries no logical shape, packing, padding, or
    scalar-format claim.

    Attributes:
        name: Input or output binding name.
        values: Scalars in the physical lane order declared by the binding.
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
        """Validate the target-independent scalar count.

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
        scalar_count: Number of consecutive scalar lanes in slot-major order.
    """

    address: CentSharedBufferAddress
    scalar_count: int

    def __post_init__(self) -> None:
        """Validate the target-independent scalar count.

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
        """Validate the target-independent scalar count.

        Raises:
            ValueError: If the region contains no scalars.
        """

        _validate_scalar_count(self.scalar_count)


CentPhysicalRegion: TypeAlias = (
        CentDramRegion | CentSharedBufferRegion | CentGlobalBufferRegion
)
"""A raw physical region understood by runtime targets."""


@dataclass(frozen=True, slots=True, kw_only=True)
class CentInputBinding:
    """Bind named host scalars to a physical region before execution.

    Attributes:
        name: Name used to match one :class:`CentNamedScalars` request value.
        region: Raw physical region initialized before instruction zero.
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
        name: Name assigned to the extracted :class:`CentNamedScalars` value.
        region: Raw physical region read after the final instruction.
    """

    name: str
    region: CentPhysicalRegion

    def __post_init__(self) -> None:
        """Require a stable nonempty output name.

        Raises:
            ValueError: If ``name`` is empty.
        """

        _validate_name(self.name)
