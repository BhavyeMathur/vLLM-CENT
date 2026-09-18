"""Public state facade for one simulated CENT device."""

from dataclasses import dataclass, field
from typing import assert_never

from vllm_cent.cent import (
    CentBankRegisterAddress,
    CentGlobalBufferAddress,
    CentHardwareSpec,
    CentMemoryAddress,
    CentSharedBufferAddress,
)
from vllm_cent.cent.instructions import validate_address
from .effects import (
    CentWriteEffect,
    DramWriteEffect,
    GlobalBufferWriteEffect,
    SharedBufferWriteEffect,
)
from .errors import CentUninitializedReadError
from .numeric import CentNumericSemantics, ReferenceMathSemantics
from .storage import (
    _BankRegisterFile,
    _LinearBuffer,
    _SparseDram,
    _UninitializedScalarError,
)

__all__ = ["CentDeviceState"]


@dataclass(slots=True, kw_only=True)
class CentDeviceState:
    """Coordinate the independent storage regions of one functional device.

    Focused internal components own sparse DRAM, dense buffer, and register-file
    mechanics. This facade preserves typed CENT addresses, numeric conversion,
    structured errors, and atomic commits that span more than one region.

    Attributes:
        hardware: Geometry of the device represented by this state.
        numeric: Scalar conversion and arithmetic policy used by writes and
            instruction kernels.
        device_id: Zero-based identifier included in structured failures.
        _dram: Sparse, initialization-aware DRAM storage.
        _shared_buffer: Shared Buffer lanes in slot-major scalar order.
        _global_buffers: One scalar buffer for each physical channel.
        _accumulators: Initialized near-bank accumulation registers.
        _activation_results: Initialized activation-result registers.
    """

    hardware: CentHardwareSpec
    numeric: CentNumericSemantics = field(default_factory=ReferenceMathSemantics)
    device_id: int = 0
    _dram: _SparseDram = field(init=False, repr=False)
    _shared_buffer: _LinearBuffer = field(init=False, repr=False)
    _global_buffers: tuple[_LinearBuffer, ...] = field(init=False, repr=False)
    _accumulators: _BankRegisterFile = field(init=False, repr=False)
    _activation_results: _BankRegisterFile = field(init=False, repr=False)

    def __post_init__(self) -> None:
        """Construct each storage region and validate the device identifier.

        Raises:
            ValueError: If ``device_id`` is negative.
        """

        if self.device_id < 0:
            raise ValueError("device_id cannot be negative")

        self._dram = _SparseDram(hardware=self.hardware, device_id=self.device_id)
        shared_lane_count = (
                self.hardware.shared_buffer_slots * self.hardware.burst_length
        )
        self._shared_buffer = _LinearBuffer(capacity=shared_lane_count)
        self._global_buffers = tuple(
            _LinearBuffer(capacity=self.hardware.global_buffer_columns)
            for _ in range(self.hardware.num_channels)
        )
        self._accumulators = _BankRegisterFile(
            hardware=self.hardware,
            device_id=self.device_id,
            region_name="accumulator",
            uninitialized_reason="accumulator register is uninitialized",
        )
        self._activation_results = _BankRegisterFile(
            hardware=self.hardware,
            device_id=self.device_id,
            region_name="activation",
            uninitialized_reason="activation-result register is uninitialized",
        )

    def write_dram(
            self,
            address: CentMemoryAddress,
            values: tuple[float, ...],
    ) -> None:
        """Write consecutive host values within one sparse DRAM row.

        Args:
            address: Physical location of the first value.
            values: Nonempty host values written at consecutive columns.

        Raises:
            ValueError: If the address or complete span is outside the target,
                ``values`` is empty, or numeric conversion rejects a value.
        """

        # Validate before invoking a numeric policy that may be stateful.
        validate_address(address, self.hardware)
        if not values:
            raise ValueError("a DRAM write cannot be empty")
        if address.column + len(values) > self.hardware.dram_columns:
            raise ValueError("DRAM write crosses a row boundary")

        stored_values = self._store_host_values(values)
        self.commit_effects((DramWriteEffect(address=address, values=stored_values),))

    def read_dram(
            self,
            address: CentMemoryAddress,
            *,
            value_count: int,
    ) -> tuple[float, ...]:
        """Read consecutive initialized values from one DRAM row.

        Args:
            address: Physical location of the first value.
            value_count: Number of consecutive scalar values to read.

        Returns:
            Values in increasing column order.

        Raises:
            ValueError: If the address, count, or span is invalid.
            CentUninitializedReadError: If any requested value is unwritten.
        """

        return self._dram.read(address, value_count=value_count)

    def write_shared_buffer(
            self,
            address: CentSharedBufferAddress,
            values: tuple[float, ...],
    ) -> None:
        """Write consecutive host lane values at a Shared Buffer slot.

        Args:
            address: First Shared Buffer slot to write.
            values: Nonempty lane values in slot-major order. The final value
                may end inside a slot.

        Raises:
            ValueError: If the slot, count, span, or conversion is invalid.
        """

        if address.slot >= self.hardware.shared_buffer_slots:
            raise ValueError("Shared Buffer address is outside the target")
        if not values:
            raise ValueError("a Shared Buffer write cannot be empty")
        start_lane = address.slot * self.hardware.burst_length
        if start_lane + len(values) > self._shared_buffer.capacity:
            raise ValueError("Shared Buffer write exceeds the target")

        stored_values = self._store_host_values(values)
        self.commit_effects(
            (SharedBufferWriteEffect(address=address, values=stored_values),)
        )

    def read_shared_buffer(
            self,
            address: CentSharedBufferAddress,
            *,
            slot_count: int,
    ) -> tuple[float, ...]:
        """Read complete consecutive initialized Shared Buffer slots.

        Args:
            address: First Shared Buffer slot to read.
            slot_count: Number of complete slots to read.

        Returns:
            Lane values in slot-major order.

        Raises:
            ValueError: If the slot, count, or complete span is invalid.
            CentUninitializedReadError: If any requested lane is unwritten.
        """

        if address.slot >= self.hardware.shared_buffer_slots:
            raise ValueError("Shared Buffer address is outside the target")
        if slot_count < 1:
            raise ValueError("slot_count must be at least 1")
        if address.slot + slot_count > self.hardware.shared_buffer_slots:
            raise ValueError("Shared Buffer read exceeds the target")

        value_count = slot_count * self.hardware.burst_length
        return self.read_shared_buffer_values(address, value_count=value_count)

    def read_shared_buffer_values(
            self,
            address: CentSharedBufferAddress,
            *,
            value_count: int,
    ) -> tuple[float, ...]:
        """Read an exact scalar count starting at a Shared Buffer slot.

        Args:
            address: First Shared Buffer slot to read.
            value_count: Number of consecutive lane values to read.

        Returns:
            Exactly ``value_count`` initialized lanes in slot-major order.

        Raises:
            ValueError: If the slot, count, or complete span is invalid.
            CentUninitializedReadError: If any requested lane is unwritten.
        """

        if address.slot >= self.hardware.shared_buffer_slots:
            raise ValueError("Shared Buffer address is outside the target")
        if value_count < 1:
            raise ValueError("value_count must be at least 1")

        start_lane = address.slot * self.hardware.burst_length
        if start_lane + value_count > self._shared_buffer.capacity:
            raise ValueError("Shared Buffer read exceeds the target")
        try:
            return self._shared_buffer.read(start_lane, value_count=value_count)
        except _UninitializedScalarError as error:
            missing_slot = error.index // self.hardware.burst_length
            raise CentUninitializedReadError(
                "Shared Buffer source lane is uninitialized",
                device_id=self.device_id,
                location=CentSharedBufferAddress(slot=missing_slot),
            ) from error

    def write_global_buffer(
            self,
            address: CentGlobalBufferAddress,
            values: tuple[float, ...],
    ) -> None:
        """Write consecutive host values to one channel's Global Buffer.

        Args:
            address: Channel and first scalar column to write.
            values: Nonempty host values written at consecutive columns.

        Raises:
            ValueError: If the address, count, span, or conversion is invalid.
        """

        if address.channel >= self.hardware.num_channels:
            raise ValueError("Global Buffer channel is outside the target")
        if address.column >= self.hardware.global_buffer_columns:
            raise ValueError("Global Buffer column is outside the target")
        if not values:
            raise ValueError("a Global Buffer write cannot be empty")
        if address.column + len(values) > self.hardware.global_buffer_columns:
            raise ValueError("Global Buffer write exceeds its capacity")

        stored_values = self._store_host_values(values)
        self.commit_effects(
            (GlobalBufferWriteEffect(address=address, values=stored_values),)
        )

    def read_global_buffer(
            self,
            address: CentGlobalBufferAddress,
            *,
            value_count: int,
    ) -> tuple[float, ...]:
        """Read consecutive initialized values from one Global Buffer.

        Args:
            address: Channel and first scalar column to read.
            value_count: Number of consecutive values to read.

        Returns:
            Values in increasing column order.

        Raises:
            ValueError: If the address, count, or span is invalid.
            CentUninitializedReadError: If any requested value is unwritten.
        """

        if address.channel >= self.hardware.num_channels:
            raise ValueError("Global Buffer channel is outside the target")
        if address.column >= self.hardware.global_buffer_columns:
            raise ValueError("Global Buffer column is outside the target")
        if value_count < 1:
            raise ValueError("value_count must be at least 1")
        if address.column + value_count > self.hardware.global_buffer_columns:
            raise ValueError("Global Buffer read exceeds its capacity")

        try:
            return self._global_buffers[address.channel].read(
                address.column,
                value_count=value_count,
            )
        except _UninitializedScalarError as error:
            raise CentUninitializedReadError(
                "Global Buffer source value is uninitialized",
                device_id=self.device_id,
                location=CentGlobalBufferAddress(
                    channel=address.channel,
                    column=error.index,
                ),
            ) from error

    def write_accumulator(
            self,
            address: CentBankRegisterAddress,
            value: float,
    ) -> None:
        """Write one near-bank accumulation register.

        Args:
            address: Channel, bank, and register to write.
            value: Host scalar accumulator value.

        Raises:
            ValueError: If the address or numeric conversion is invalid.
        """

        # Bounds precede conversion to match buffer host-write behavior.
        if address.channel >= self.hardware.num_channels:
            raise ValueError("accumulator channel is outside the target")
        if address.bank >= self.hardware.num_banks:
            raise ValueError("accumulator bank is outside the target")
        if address.register >= self.hardware.accumulator_slots_per_bank:
            raise ValueError("accumulator register is outside the target")
        self._accumulators.write(address, self.numeric.store(value))

    def read_accumulator(self, address: CentBankRegisterAddress) -> float:
        """Read one initialized near-bank accumulation register.

        Args:
            address: Channel, bank, and register to read.

        Returns:
            Stored accumulator scalar.

        Raises:
            ValueError: If any address coordinate is outside the target.
            CentUninitializedReadError: If the register is unwritten.
        """

        return self._accumulators.read(address)

    def write_activation_result(
            self,
            address: CentBankRegisterAddress,
            value: float,
    ) -> None:
        """Write one near-bank activation-result register.

        Args:
            address: Channel, bank, and register to write.
            value: Host scalar activation result.

        Raises:
            ValueError: If the address or numeric conversion is invalid.
        """

        # Bounds precede conversion to match buffer host-write behavior.
        if address.channel >= self.hardware.num_channels:
            raise ValueError("activation channel is outside the target")
        if address.bank >= self.hardware.num_banks:
            raise ValueError("activation bank is outside the target")
        if address.register >= self.hardware.accumulator_slots_per_bank:
            raise ValueError("activation register is outside the target")
        self._activation_results.write(address, self.numeric.store(value))

    def read_activation_result(self, address: CentBankRegisterAddress) -> float:
        """Read one initialized near-bank activation-result register.

        Args:
            address: Channel, bank, and register to read.

        Returns:
            Stored activation-result scalar.

        Raises:
            ValueError: If any address coordinate is outside the target.
            CentUninitializedReadError: If the register is unwritten.
        """

        return self._activation_results.read(address)

    def commit_effects(self, effects: tuple[CentWriteEffect, ...]) -> None:
        """Atomically commit already-represented instruction results.

        Each touched component is cloned at most once. Global Buffers are
        cloned independently by channel, and sparse DRAM copies only rows that
        receive writes. Prepared components replace live state only after every
        effect succeeds.

        Args:
            effects: Complete ordered write set for one logical instruction.

        Raises:
            ValueError: If the tuple is empty, an effect contains no values, or
                a destination span is outside the target.
        """

        if not effects:
            raise ValueError("an instruction commit must contain at least one effect")

        prepared_dram = self._dram
        prepared_shared_buffer = self._shared_buffer
        prepared_global_buffers: list[_LinearBuffer] | None = None
        copied_global_channels: set[int] = set()

        for effect in effects:
            if not effect.values:
                raise ValueError("a write effect cannot be empty")

            if isinstance(effect, DramWriteEffect):
                if prepared_dram is self._dram:
                    prepared_dram = self._dram.clone()
                prepared_dram.write(effect.address, effect.values)
                continue

            if isinstance(effect, SharedBufferWriteEffect):
                if effect.address.slot >= self.hardware.shared_buffer_slots:
                    raise ValueError("Shared Buffer address is outside the target")
                start_lane = effect.address.slot * self.hardware.burst_length
                if start_lane + len(effect.values) > self._shared_buffer.capacity:
                    raise ValueError("Shared Buffer write exceeds the target")
                if prepared_shared_buffer is self._shared_buffer:
                    prepared_shared_buffer = self._shared_buffer.clone()
                prepared_shared_buffer.write(start_lane, effect.values)
                continue

            if isinstance(effect, GlobalBufferWriteEffect):
                channel = effect.address.channel
                if channel >= self.hardware.num_channels:
                    raise ValueError("Global Buffer channel is outside the target")
                if effect.address.column >= self.hardware.global_buffer_columns:
                    raise ValueError("Global Buffer column is outside the target")
                span_end = effect.address.column + len(effect.values)
                if span_end > self.hardware.global_buffer_columns:
                    raise ValueError("Global Buffer write exceeds its capacity")

                if prepared_global_buffers is None:
                    prepared_global_buffers = list(self._global_buffers)
                if channel not in copied_global_channels:
                    prepared_global_buffers[channel] = self._global_buffers[
                        channel
                    ].clone()
                    copied_global_channels.add(channel)
                prepared_global_buffers[channel].write(
                    effect.address.column,
                    effect.values,
                )
                continue

            assert_never(effect)  # pragma: no cover - closed union guard

        # Replacing component references is the only live-state mutation.
        self._dram = prepared_dram
        self._shared_buffer = prepared_shared_buffer
        if prepared_global_buffers is not None:
            self._global_buffers = tuple(prepared_global_buffers)

    def _store_host_values(self, values: tuple[float, ...]) -> tuple[float, ...]:
        """Convert every host scalar before starting an atomic commit.

        Args:
            values: Host scalar values entering simulated device state.

        Returns:
            Complete converted values ready for an atomic state commit.

        Raises:
            ValueError: If the configured numeric policy rejects a value.
        """

        return tuple(self.numeric.store(value) for value in values)
