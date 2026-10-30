"""The CENT architecture: geometry in bytes, and the storage it provides."""

from __future__ import annotations

from dataclasses import dataclass

from ..cent import BANKS_PER_GROUP, CentHardwareSpec
from ..ir.types import DType
from .spec import Architecture, MemorySpace, MemorySpaceSpec

__all__ = ["CentArchitecture"]


def _values(name: str, size: int, dtype: DType) -> int:
    """Count the ``dtype`` values that fit exactly in ``size`` bytes.

    Args:
        name: Field name, for the error message.
        size: A size, in bytes.
        dtype: Element type; its width in bits may be below 8 (fp4).

    Returns:
        ``size * 8 // dtype.bits``, e.g. 32 bytes hold 16 bf16 values.

    Raises:
        ValueError: If the bytes do not hold a whole number of values.
    """

    bits = size * 8
    if bits % dtype.bits != 0:
        raise ValueError(f"{name} {size} bytes is not a whole number of {dtype} values")
    return bits // dtype.bits


@dataclass(frozen=True, slots=True, kw_only=True)
class CentArchitecture(Architecture):
    """Geometry of one CENT device; the single source of its sizes.

    A device has ``channels`` DRAM channels of ``banks_per_channel`` banks;
    each bank has ``rows_per_bank`` rows of ``row_size`` bytes. Every channel
    has a Global Buffer that feeds its banks' MAC units; the device has one
    Shared Buffer near memory. Every ``*_size`` field is in bytes, not
    values, so how many values fit depends on the element type (see
    ``to_cent_hardware``).

    Attributes:
        channels: DRAM channels; the paper's ``CHmask`` covers up to 32.
        banks_per_channel: Banks in each channel.
        banks_per_group: Banks served by one processing unit (four on CENT).
        rows_per_bank: DRAM rows in each bank.
        row_size: Size of one DRAM row, in bytes.
        burst_size: Size of one burst, in bytes: the data one access moves,
            and also the size of one Shared Buffer slot.
        global_buffer_size: Size of each channel's Global Buffer, in bytes.
        shared_buffer_size: Size of the device's Shared Buffer, in bytes.
        accumulators_per_bank: MAC result registers per bank (``Regid``).
    """

    channels: int
    banks_per_channel: int
    banks_per_group: int
    rows_per_bank: int
    row_size: int
    burst_size: int
    global_buffer_size: int
    shared_buffer_size: int
    accumulators_per_bank: int

    def __post_init__(self) -> None:
        """Validate the hardware geometry.

        Raises:
            ValueError: If a dimension is outside its documented range or the
                DRAM row or Global Buffer cannot hold a whole number of bursts.
        """

        # CHmask has one bit for each of the paper's 32 possible channels.
        if not 1 <= self.channels <= 32:
            raise ValueError("channels must be between 1 and 32")

        # One burst cannot cross a row boundary, so a row must contain a whole
        # number of bursts.
        if self.rows_per_bank < 1:
            raise ValueError("rows_per_bank must be at least 1")
        if self.row_size < 1:
            raise ValueError("row_size must be at least 1")
        if self.burst_size < 1 or self.burst_size > self.row_size:
            raise ValueError("burst_size must be between 1 and row_size")
        if self.row_size % self.burst_size != 0:
            raise ValueError("row_size must be divisible by burst_size")
        # Regid and Shared Buffer addresses need at least one valid destination.
        if self.accumulators_per_bank < 1:
            raise ValueError("accumulators_per_bank must be at least 1")
        if self.global_buffer_size < 1:
            raise ValueError("global_buffer_size must be at least 1")
        if self.global_buffer_size % self.burst_size != 0:
            raise ValueError("global_buffer_size must be divisible by burst_size")
        if self.shared_buffer_size < 1:
            raise ValueError("shared_buffer_size must be at least 1")
        # One Shared Buffer slot holds one burst.
        if self.shared_buffer_size % self.burst_size != 0:
            raise ValueError("shared_buffer_size must be divisible by burst_size")
        # Each processing unit serves a whole group of banks.
        if self.banks_per_group < 1 or self.banks_per_channel % self.banks_per_group:
            raise ValueError("banks_per_channel must be a multiple of banks_per_group")

    def memory_spaces(self) -> tuple[MemorySpaceSpec, ...]:
        """Return the DRAM banks, Global Buffers and Shared Buffer of the device.

        Returns:
            ``pim.dram`` (one per bank), ``pim.gb`` (one per channel) and
            ``pnm.sb`` (one per device), each read and written in bursts.
        """

        # TODO(accumulator width): accumulators_per_bank counts the MAC result
        # registers of a bank, but not how many bytes each holds, so pim.acc
        # has no size yet. Find the register width in the CENT paper or the
        # AiM documentation before adding it.
        return (
            MemorySpaceSpec(
                space=MemorySpace.PIM_DRAM,
                instance_size=self.rows_per_bank * self.row_size,
                instances=self.channels * self.banks_per_channel,
                access_size=self.burst_size,
            ),
            MemorySpaceSpec(
                space=MemorySpace.PIM_GB,
                instance_size=self.global_buffer_size,
                instances=self.channels,
                access_size=self.burst_size,
            ),
            MemorySpaceSpec(
                space=MemorySpace.PNM_SB,
                instance_size=self.shared_buffer_size,
                instances=1,
                access_size=self.burst_size,
            ),
        )

    def to_cent_hardware(
        self, dtype: DType, sigmoid_function_id: int
    ) -> CentHardwareSpec:
        """Describe this geometry as the existing CENT backend's hardware spec.

        The architecture measures storage in bytes; ``CentHardwareSpec``
        counts scalar values, so every size is converted for one element
        type. This only converts: the architecture stays the single source
        of the geometry.

        Args:
            dtype: Element type the backend computes in. A 2048-byte row holds
                1024 bf16 values, but 512 fp32 values.
            sigmoid_function_id: The ``AFid`` the target uses for sigmoid. The
                paper does not publish it; take it from
                ``DeviceCapability.function_id(ActivationKind.SIGMOID)``.

        Returns:
            The equivalent hardware spec for ``dtype``.

        Raises:
            ValueError: If the bank groups differ from the backend's fixed
                ``BANKS_PER_GROUP``, or a size is not a whole number of
                ``dtype`` values.
        """

        # The backend's placement rules hard-code four-bank groups.
        if self.banks_per_group != BANKS_PER_GROUP:
            raise ValueError(
                f"banks_per_group is {self.banks_per_group}, but the CENT backend "
                f"uses groups of {BANKS_PER_GROUP} banks"
            )
        return CentHardwareSpec(
            num_channels=self.channels,
            num_banks=self.banks_per_channel,
            dram_rows=self.rows_per_bank,
            dram_columns=_values("row_size", self.row_size, dtype),
            burst_length=_values("burst_size", self.burst_size, dtype),
            accumulator_slots_per_bank=self.accumulators_per_bank,
            sigmoid_activation_function_id=sigmoid_function_id,
            shared_buffer_slots=self.shared_buffer_size // self.burst_size,
            global_buffer_columns=_values(
                "global_buffer_size", self.global_buffer_size, dtype
            ),
        )
