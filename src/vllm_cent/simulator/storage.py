"""Store simulated memory and distinguish unwritten scalars from zero.

DRAM (dynamic random-access memory) rows are allocated on first write. Small
buffers use fixed-size arrays. Both keep an explicit marker for unwritten
positions.

Examples:
    Writing DRAM column 2 does not initialize columns 0 and 1.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import cast

from vllm_cent.cent import (
    CentBankRegisterAddress,
    CentHardwareSpec,
    CentMemoryAddress,
)
from vllm_cent.cent.instructions import validate_address
from .errors import CentUninitializedReadError

__all__: list[str] = []


class _Uninitialized(Enum):
    """Mark a scalar location that has never been written.

    Members:
        VALUE: Marker placed where no scalar has been written yet.
    """

    VALUE = 0


def _require_stored_floats(
    values: list[float | _Uninitialized],
    *,
    region: str,
) -> tuple[float, ...]:
    """Return all selected values after checking that each is a Python ``float``.

    The caller has already rejected unwritten locations. A different value type
    here means storage is corrupted or number conversion returned the wrong type.
    Dropping such a value would shift the remaining positions, so reject the
    whole read instead.

    Args:
        values: Initialized values selected by one memory read, in memory order.
        region: Human-readable name of the memory area, used in the error message.

    Returns:
        Every selected ``float`` in the original order, without dropping positions.

    Raises:
        RuntimeError: If any selected value is not a Python ``float``.
    """

    if any(not isinstance(value, float) for value in values):
        raise RuntimeError(f"{region} storage contains a non-float value")
    return cast("tuple[float, ...]", tuple(values))


class _UninitializedScalarError(Exception):
    """Identify an uninitialized scalar selected from a linear buffer.

    Attributes:
        index: Absolute zero-based scalar index in the linear buffer.
    """

    index: int

    def __init__(self, index: int) -> None:
        """Create an internal missing-scalar result.

        Args:
            index: Absolute scalar index that has not been written.

        Raises:
            ValueError: If ``index`` is negative.
        """

        if index < 0:
            raise ValueError("uninitialized scalar index cannot be negative")
        self.index = index
        super().__init__(f"scalar {index} is uninitialized")


@dataclass(frozen=True, slots=True, kw_only=True)
class _DramRowAddress:
    """Identify a complete DRAM row, without selecting a column within it.

    Attributes:
        channel: Zero-based channel number.
        bank: Zero-based bank number within that channel.
        row: Zero-based row number within that bank.
    """

    channel: int
    bank: int
    row: int


@dataclass(slots=True, kw_only=True)
class _SparseDram:
    """Allocate a DRAM row only when something writes to it.

    Stored rows are immutable tuples. Clones share those tuples until a write
    creates a replacement row, so an abandoned clone cannot change the original.

    Attributes:
        hardware: Geometry used to validate DRAM addresses and spans.
        device_id: Device identifier attached to uninitialized-read failures.
        _rows: Immutable allocated rows keyed by channel, bank, and row.

    Examples:
        With ``hardware.dram_columns=8``, writing two numbers at column 3
        initializes columns 3 and 4. The other six columns remain unwritten.
    """

    hardware: CentHardwareSpec
    device_id: int
    _rows: dict[_DramRowAddress, tuple[float | _Uninitialized, ...]] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate context used by structured storage failures.

        Raises:
            ValueError: If ``device_id`` is negative.
        """

        if self.device_id < 0:
            raise ValueError("device_id cannot be negative")

    def clone(self) -> _SparseDram:
        """Copy the row lookup table while sharing the immutable row values.

        Returns:
            Storage with an independent row mapping and the same immutable row
            values.
        """

        cloned = _SparseDram(hardware=self.hardware, device_id=self.device_id)
        cloned._rows = self._rows.copy()
        return cloned

    def write(self, address: CentMemoryAddress, values: tuple[float, ...]) -> None:
        """Write converted numbers to consecutive columns without converting again.

        Args:
            address: Physical location of the first destination scalar.
            values: Nonempty stored values written at consecutive columns.

        Raises:
            ValueError: If the address, value count, or complete span is
                outside the target.
        """

        validate_address(address, self.hardware)
        if not values:
            raise ValueError("a DRAM write cannot be empty")
        span_end = address.column + len(values)
        if span_end > self.hardware.dram_columns:
            raise ValueError("DRAM write crosses a row boundary")

        row_address = _DramRowAddress(
            channel=address.channel,
            bank=address.bank,
            row=address.row,
        )
        # Build a replacement row so clones sharing the old tuple keep their
        # original values. Untouched columns remain unwritten or unchanged.
        stored_row = self._rows.get(row_address)
        prepared_row: list[float | _Uninitialized] = (
            [_Uninitialized.VALUE] * self.hardware.dram_columns
            if stored_row is None
            else list(stored_row)
        )
        prepared_row[address.column : span_end] = values
        self._rows[row_address] = tuple(prepared_row)

    def read(
        self,
        address: CentMemoryAddress,
        *,
        value_count: int,
    ) -> tuple[float, ...]:
        """Read consecutive initialized values from one DRAM row.

        Args:
            address: Physical location of the first source scalar.
            value_count: Number of consecutive scalar values to read.

        Returns:
            Stored values in increasing column order.

        Raises:
            ValueError: If the address, count, or span is invalid.
            CentUninitializedReadError: If any selected scalar has not been
                written.
        """

        validate_address(address, self.hardware)
        if value_count < 1:
            raise ValueError("value_count must be at least 1")
        span_end = address.column + value_count
        if span_end > self.hardware.dram_columns:
            raise ValueError("DRAM read crosses a row boundary")

        row_address = _DramRowAddress(
            channel=address.channel,
            bank=address.bank,
            row=address.row,
        )
        row = self._rows.get(row_address)
        for column in range(address.column, span_end):
            value = _Uninitialized.VALUE if row is None else row[column]
            if value is _Uninitialized.VALUE:
                raise CentUninitializedReadError(
                    "DRAM source value is uninitialized",
                    device_id=self.device_id,
                    location=CentMemoryAddress(
                        channel=address.channel,
                        bank=address.bank,
                        row=address.row,
                        column=column,
                    ),
                )

        values = list(row[address.column : span_end]) if row is not None else []
        return _require_stored_floats(values, region="DRAM")


@dataclass(slots=True, kw_only=True)
class _LinearBuffer:
    """Store a fixed number of scalars and track which positions were written.

    This class works with scalar indices only. Device state translates buffer
    addresses into those indices and converts a missing index back into the
    appropriate address for an error message.

    Attributes:
        capacity: Number of scalar positions in the buffer.
        _values: Stored numbers, with a special marker in positions never written.
    """

    capacity: int
    _values: list[float | _Uninitialized] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Allocate an entirely uninitialized scalar array.

        Raises:
            ValueError: If ``capacity`` is less than one.
        """

        if self.capacity < 1:
            raise ValueError("linear buffer capacity must be at least 1")
        self._values = [_Uninitialized.VALUE] * self.capacity

    def clone(self) -> _LinearBuffer:
        """Copy the complete initialized-value array.

        Returns:
            Independent buffer containing the same stored values.
        """

        cloned = _LinearBuffer(capacity=self.capacity)
        cloned._values = self._values.copy()
        return cloned

    def write(self, start: int, values: tuple[float, ...]) -> None:
        """Write converted numbers at consecutive indices without converting again.

        Args:
            start: Zero-based destination scalar index.
            values: Nonempty stored values to write.

        Raises:
            ValueError: If the start, value count, or span is invalid.
        """

        if start < 0 or start >= self.capacity:
            raise ValueError("linear buffer start is outside storage")
        if not values:
            raise ValueError("a linear buffer write cannot be empty")
        span_end = start + len(values)
        if span_end > self.capacity:
            raise ValueError("linear buffer write exceeds storage")
        self._values[start:span_end] = values

    def read(self, start: int, *, value_count: int) -> tuple[float, ...]:
        """Read consecutive initialized scalar positions.

        Args:
            start: Zero-based source scalar index.
            value_count: Number of consecutive scalar values to read.

        Returns:
            Stored values in increasing index order.

        Raises:
            ValueError: If the start, count, or span is invalid.
            _UninitializedScalarError: If a selected position has not been
                written.
        """

        if start < 0 or start >= self.capacity:
            raise ValueError("linear buffer start is outside storage")
        if value_count < 1:
            raise ValueError("value_count must be at least 1")
        span_end = start + value_count
        if span_end > self.capacity:
            raise ValueError("linear buffer read exceeds storage")

        values = self._values[start:span_end]
        # The slice starts at zero locally; errors need the original buffer
        # index. Reading indices 5..7 and missing offset 1 must report index 6.
        for offset, value in enumerate(values):
            if value is _Uninitialized.VALUE:
                raise _UninitializedScalarError(start + offset)
        return _require_stored_floats(values, region="linear buffer")


@dataclass(slots=True, kw_only=True)
class _BankRegisterFile:
    """Store written values for one kind of register next to each memory bank.

    A dictionary entry exists only after a register is written. Missing entries
    therefore mean unwritten registers, not registers containing zero.

    Attributes:
        hardware: Geometry that defines channel, bank, and register bounds.
        device_id: Device identifier attached to uninitialized-read failures.
        region_name: Short address-space name used in bound errors.
        uninitialized_reason: Failure reason used when a register is unset.
        _values: Initialized values keyed by complete register address.
    """

    hardware: CentHardwareSpec
    device_id: int
    region_name: str
    uninitialized_reason: str
    _values: dict[CentBankRegisterAddress, float] = field(
        init=False,
        default_factory=dict,
        repr=False,
    )

    def __post_init__(self) -> None:
        """Validate context used by register errors.

        Raises:
            ValueError: If ``device_id`` is negative or an error description is
                empty.
        """

        if self.device_id < 0:
            raise ValueError("device_id cannot be negative")
        if not self.region_name:
            raise ValueError("region_name cannot be empty")
        if not self.uninitialized_reason:
            raise ValueError("uninitialized_reason cannot be empty")

    def clone(self) -> _BankRegisterFile:
        """Copy initialized registers for an atomic instruction commit.

        Returns:
            Independent register storage with the same addresses and values.
        """
        clone = _BankRegisterFile(
            hardware=self.hardware,
            device_id=self.device_id,
            region_name=self.region_name,
            uninitialized_reason=self.uninitialized_reason,
        )
        clone._values = dict(self._values)
        return clone

    def write(self, address: CentBankRegisterAddress, value: float) -> None:
        """Write one converted register value without converting it again.

        Args:
            address: Channel, bank, and register to write.
            value: Stored scalar value.

        Raises:
            ValueError: If any address coordinate is outside the target.
        """

        self._validate_address(address)
        self._values[address] = value

    def read(self, address: CentBankRegisterAddress) -> float:
        """Read one initialized register value.

        Args:
            address: Channel, bank, and register to read.

        Returns:
            Stored scalar value.

        Raises:
            ValueError: If any address coordinate is outside the target.
            CentUninitializedReadError: If the register has not been written.
        """

        self._validate_address(address)
        try:
            return self._values[address]
        except KeyError as error:
            raise CentUninitializedReadError(
                self.uninitialized_reason,
                device_id=self.device_id,
                location=address,
            ) from error

    def _validate_address(self, address: CentBankRegisterAddress) -> None:
        """Check every coordinate against the configured target.

        Args:
            address: Register address to validate.

        Raises:
            ValueError: If the channel, bank, or register does not exist.
        """

        if address.channel >= self.hardware.num_channels:
            raise ValueError(f"{self.region_name} channel is outside the target")
        if address.bank >= self.hardware.num_banks:
            raise ValueError(f"{self.region_name} bank is outside the target")
        if address.register >= self.hardware.accumulator_slots_per_bank:
            raise ValueError(f"{self.region_name} register is outside the target")
