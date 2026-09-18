"""Describe immutable writes prepared by functional instruction kernels."""

from dataclasses import dataclass
from typing import TypeAlias

from vllm_cent.cent import (
    CentGlobalBufferAddress,
    CentMemoryAddress,
    CentSharedBufferAddress,
)
from vllm_cent.runtime import (
    CentDramRegion,
    CentGlobalBufferRegion,
    CentPhysicalRegion,
    CentSharedBufferRegion,
)

__all__: list[str] = []


@dataclass(frozen=True, slots=True, kw_only=True)
class DramWriteEffect:
    """Write stored scalar values to one consecutive DRAM span.

    Attributes:
        address: Physical location of the first destination scalar.
        values: Values already represented in the selected numeric profile.
    """

    address: CentMemoryAddress
    values: tuple[float, ...]

    @property
    def region(self) -> CentDramRegion:
        """Return the typed physical region changed by this effect.

        Returns:
            DRAM region beginning at :attr:`address` and covering every value.
        """

        return CentDramRegion(address=self.address, scalar_count=len(self.values))


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedBufferWriteEffect:
    """Write stored scalar values to consecutive Shared Buffer lanes.

    Attributes:
        address: First destination slot. Values begin at lane zero of this slot.
        values: Values already represented in the selected numeric profile.
    """

    address: CentSharedBufferAddress
    values: tuple[float, ...]

    @property
    def region(self) -> CentSharedBufferRegion:
        """Return the typed physical region changed by this effect.

        Returns:
            Shared Buffer region covering every stored lane in this effect.
        """

        return CentSharedBufferRegion(
            address=self.address,
            scalar_count=len(self.values),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class GlobalBufferWriteEffect:
    """Write stored scalar values to one channel's Global Buffer.

    Attributes:
        address: Channel and first destination scalar column.
        values: Values already represented in the selected numeric profile.
    """

    address: CentGlobalBufferAddress
    values: tuple[float, ...]

    @property
    def region(self) -> CentGlobalBufferRegion:
        """Return the typed physical region changed by this effect.

        Returns:
            Global Buffer region covering every scalar in this effect.
        """

        return CentGlobalBufferRegion(
            address=self.address,
            scalar_count=len(self.values),
        )


CentWriteEffect: TypeAlias = (
        DramWriteEffect | SharedBufferWriteEffect | GlobalBufferWriteEffect
)
"""One immutable state write prepared by an instruction kernel."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _InstructionEffects:
    """Capture the complete physical effect of one prepared instruction.

    Read regions are recorded by the same concrete kernel that consumes them,
    preventing trace logic from becoming a second instruction-semantic
    dispatcher. Writes include stored values for the atomic state commit; their
    public trace regions are available through each effect's ``region``.

    Attributes:
        reads: Physical source regions consumed in deterministic kernel order.
        writes: Complete stored-value writes committed as one transaction.
    """

    reads: tuple[CentPhysicalRegion, ...]
    writes: tuple[CentWriteEffect, ...]
