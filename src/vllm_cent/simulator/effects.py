"""Record proposed writes without changing simulated memory.

An instruction kernel computes values and returns these records. Device state
checks and applies the whole set later, so a failed instruction cannot leave
only some of its destination memory updated.
"""

from dataclasses import dataclass
from typing import TypeAlias

from vllm_cent.cent import (
    CentBankRegisterAddress,
    CentGlobalBufferAddress,
    CentMemoryAddress,
    CentSharedBufferAddress,
)
from vllm_cent.runtime import (
    CentAccumulatorRegion,
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
        values: Numbers already converted by the numeric policy, in write order.
            Committing this record must not convert or round them again.
    """

    address: CentMemoryAddress
    values: tuple[float, ...]

    @property
    def region(self) -> CentDramRegion:
        """Return the typed physical region changed by this effect.

        Returns:
            DRAM region beginning at ``address`` and covering every value.
        """

        return CentDramRegion(address=self.address, scalar_count=len(self.values))


@dataclass(frozen=True, slots=True, kw_only=True)
class SharedBufferWriteEffect:
    """Write stored scalar values to consecutive Shared Buffer lanes.

    Attributes:
        address: First destination slot. Values begin at lane zero of this slot.
        values: Numbers already converted by the numeric policy, in write order.
            Committing this record must not convert or round them again.
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
        values: Numbers already converted by the numeric policy, in write order.
            Committing this record must not convert or round them again.
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


@dataclass(frozen=True, slots=True, kw_only=True)
class AccumulatorWriteEffect:
    """Propose one converted register write without changing device state.

    Attributes:
        address: Channel, bank, and selected register.
        value: Stored scalar; commit does not convert it again.
    """

    address: CentBankRegisterAddress
    value: float

    @property
    def region(self) -> CentAccumulatorRegion:
        """Return the single register changed by this effect."""
        return CentAccumulatorRegion(address=self.address)


CentWriteEffect: TypeAlias = (
    DramWriteEffect
    | SharedBufferWriteEffect
    | GlobalBufferWriteEffect
    | AccumulatorWriteEffect
)
"""One immutable state write prepared by an instruction kernel."""


@dataclass(frozen=True, slots=True, kw_only=True)
class _InstructionEffects:
    """Record every source range read and every write an instruction proposes.

    The kernel records its own reads so tracing follows the operations that
    actually computed the result. Writes carry the computed values. Device
    state applies all writes together after checking them; traces expose only
    their memory ranges through each write record's ``region`` property.

    Attributes:
        reads: Source memory ranges, in the order the kernel read them.
        writes: All proposed writes, applied in order with no partial update
            if a destination check fails.
    """

    reads: tuple[CentPhysicalRegion, ...]
    writes: tuple[CentWriteEffect, ...]
