"""Tests for the deterministic CENT functional simulator core."""

import math
import unittest

from vllm_cent.cent import (
    Accumulate,
    CentBankRegisterAddress,
    CentChannelSet,
    CentGlobalBufferAddress,
    CentHardwareSpec,
    CentInstruction,
    CentMemoryAddress,
    CentProgram,
    CentSharedBufferAddress,
    CopyBankToGlobalBuffer,
    CopyGlobalBufferToBank,
    ElementwiseMultiply,
    Exponent,
    ReadSingleBank,
    WriteGlobalBuffer,
    WriteSingleBank,
)
from vllm_cent.runtime import (
    CentDramRegion,
    CentGlobalBufferRegion,
    CentSharedBufferRegion,
)
from vllm_cent.simulator import (
    CentExecutionFault,
    CentSimulationError,
    CentUninitializedReadError,
    CentUnsupportedSemanticsError,
)
from vllm_cent.simulator.effects import (
    DramWriteEffect,
    GlobalBufferWriteEffect,
    SharedBufferWriteEffect,
    _InstructionEffects,
)
from vllm_cent.simulator.execution import (
    _execute_instruction,
    _execute_preflighted_program,
    execute_program,
)
from vllm_cent.simulator.kernels import (
    _prepare_accumulate,
    _prepare_copy_bank_to_global_buffer,
    _prepare_copy_global_buffer_to_bank,
    _prepare_elementwise_multiply,
    _prepare_read_single_bank,
    _prepare_write_global_buffer,
    _prepare_write_single_bank,
)
from vllm_cent.simulator.numeric import ReferenceMathSemantics
from vllm_cent.simulator.semantics import (
    _INSTRUCTION_BEHAVIORS,
    _InstructionBehavior,
    _preflight_defined_instruction,
    _preflight_instruction,
    _prepare_instruction_effects,
    _resolve_instruction_behavior,
    preflight_program,
)
from vllm_cent.simulator.state import CentDeviceState
from vllm_cent.simulator.storage import (
    _BankRegisterFile,
    _LinearBuffer,
    _SparseDram,
    _UninitializedScalarError,
    _require_stored_floats,
)


def hardware(*, shared_buffer_slots: int = 4) -> CentHardwareSpec:
    """Create the small target used by simulator tests.

    Args:
        shared_buffer_slots: Number of two-lane Shared Buffer slots.

    Returns:
        Two channels with two four-bank PU groups per channel.
    """

    return CentHardwareSpec(
        num_channels=2,
        num_banks=8,
        dram_rows=2,
        dram_columns=8,
        global_buffer_columns=8,
        burst_length=2,
        accumulator_slots_per_bank=2,
        sigmoid_activation_function_id=0,
        shared_buffer_slots=shared_buffer_slots,
    )


def memory_address(
        *, channel: int = 0, bank: int = 0, row: int = 0, column: int = 0
) -> CentMemoryAddress:
    """Create a DRAM address while making changed coordinates visible.

    Args:
        channel: Physical channel number.
        bank: Bank number inside the channel.
        row: Row number inside the bank.
        column: Scalar column number inside the row.

    Returns:
        Address containing the supplied coordinates.
    """

    return CentMemoryAddress(
        channel=channel,
        bank=bank,
        row=row,
        column=column,
    )


def program(*instructions: CentInstruction) -> CentProgram:
    """Create a program for the standard test target.

    Args:
        *instructions: Typed CENT instructions in execution order.

    Returns:
        Validated test program containing the instructions.
    """

    return CentProgram(hardware=hardware(), instructions=instructions)


class CountingStoreSemantics:
    """Quantize host stores visibly while leaving arithmetic deterministic.

    Attributes:
        store_call_count: Number of host values converted by :meth:`store`.
    """

    def __init__(self) -> None:
        """Create a policy with no completed host conversions."""

        self.store_call_count = 0

    def store(self, value: float) -> float:
        """Convert a host value and count the conversion.

        Args:
            value: Host scalar entering device storage.

        Returns:
            Input offset by one quarter so a second conversion is observable.
        """

        self.store_call_count += 1
        return float(value) + 0.25

    def add(self, left: float, right: float) -> float:
        """Add two stored values without another host conversion.

        Args:
            left: Left stored operand.
            right: Right stored operand.

        Returns:
            Sum represented by this test policy.
        """

        return left + right

    def multiply(self, left: float, right: float) -> float:
        """Multiply two stored values without another host conversion.

        Args:
            left: Left stored operand.
            right: Right stored operand.

        Returns:
            Product represented by this test policy.
        """

        return left * right


class FailingArithmeticSemantics:
    """Raise from a selected arithmetic operation after earlier results exist.

    Attributes:
        fail_multiply_at: One-based multiply call that raises ``ValueError``.
        fail_add_at: One-based add call that raises ``ValueError``.
        multiply_call_count: Number of attempted multiplications.
        add_call_count: Number of attempted additions.
    """

    def __init__(
            self,
            *,
            fail_multiply_at: int | None = None,
            fail_add_at: int | None = None,
    ) -> None:
        """Configure the arithmetic calls that fail.

        Args:
            fail_multiply_at: One-based multiply call to reject, if any.
            fail_add_at: One-based add call to reject, if any.
        """

        self.fail_multiply_at = fail_multiply_at
        self.fail_add_at = fail_add_at
        self.multiply_call_count = 0
        self.add_call_count = 0

    def store(self, value: float) -> float:
        """Convert host data without changing it.

        Args:
            value: Host scalar entering device storage.

        Returns:
            Python float representation of ``value``.
        """

        return float(value)

    def add(self, left: float, right: float) -> float:
        """Add stored operands unless this is the configured failing call.

        Args:
            left: Left stored operand.
            right: Right stored operand.

        Returns:
            Sum of ``left`` and ``right``.

        Raises:
            ValueError: On the configured add call.
        """

        self.add_call_count += 1
        if self.add_call_count == self.fail_add_at:
            raise ValueError("configured add failure")
        return left + right

    def multiply(self, left: float, right: float) -> float:
        """Multiply operands unless this is the configured failing call.

        Args:
            left: Left stored operand.
            right: Right stored operand.

        Returns:
            Product of ``left`` and ``right``.

        Raises:
            ValueError: On the configured multiply call.
        """

        self.multiply_call_count += 1
        if self.multiply_call_count == self.fail_multiply_at:
            raise ValueError("configured multiply failure")
        return left * right


class SimulatorValueTests(unittest.TestCase):
    """Test validation and structure of simulator-specific values."""

    def test_generic_state_addresses_reject_negative_coordinates(self) -> None:
        """Reject negative generic Global Buffer and register coordinates."""

        constructors = (
            lambda: CentGlobalBufferAddress(channel=-1, column=0),
            lambda: CentGlobalBufferAddress(channel=0, column=-1),
            lambda: CentBankRegisterAddress(channel=-1, bank=0, register=0),
            lambda: CentBankRegisterAddress(channel=0, bank=-1, register=0),
            lambda: CentBankRegisterAddress(channel=0, bank=0, register=-1),
        )

        for constructor in constructors:
            with self.subTest(constructor=constructor):
                with self.assertRaises(ValueError):
                    constructor()

    def test_simulation_error_rejects_negative_context_indices(self) -> None:
        """Require nonnegative device and instruction identifiers."""

        for arguments in (
                {"device_id": -1},
                {"instruction_index": -1},
        ):
            with self.subTest(arguments=arguments):
                with self.assertRaises(ValueError):
                    CentSimulationError("invalid context", **arguments)


class ReferenceMathSemanticsTests(unittest.TestCase):
    """Test the explicitly non-hardware-fidelity numeric profile."""

    def test_preserves_python_float_values_and_operations(self) -> None:
        """Preserve signed zero and special values while using host arithmetic."""

        semantics = ReferenceMathSemantics()

        self.assertEqual(math.copysign(1.0, semantics.store(-0.0)), -1.0)
        self.assertTrue(math.isnan(semantics.store(math.nan)))
        self.assertTrue(math.isinf(semantics.store(math.inf)))
        self.assertEqual(semantics.add(1.25, 2.5), 3.75)
        self.assertEqual(semantics.multiply(-2.0, 4.5), -9.0)


class StorageComponentTests(unittest.TestCase):
    """Directly test the internal storage components behind device state."""

    def test_sparse_dram_clones_rows_with_copy_on_write(self) -> None:
        """Keep the source row unchanged when a clone overwrites part of it."""

        storage = _SparseDram(hardware=hardware(), device_id=3)
        address = memory_address(column=1)
        storage.write(address, (1.0, 2.0, 3.0))

        cloned = storage.clone()
        cloned.write(memory_address(column=2), (8.0, 9.0))

        self.assertEqual(storage.read(address, value_count=3), (1.0, 2.0, 3.0))
        self.assertEqual(cloned.read(address, value_count=3), (1.0, 8.0, 9.0))

        # The source side also copies a row before its first post-clone write.
        storage.write(memory_address(column=1), (7.0,))
        self.assertEqual(storage.read(address, value_count=3), (7.0, 2.0, 3.0))
        self.assertEqual(cloned.read(address, value_count=3), (1.0, 8.0, 9.0))

    def test_sparse_dram_validates_spans_and_reports_missing_address(self) -> None:
        """Reject invalid sparse spans and identify the first missing scalar."""

        storage = _SparseDram(hardware=hardware(), device_id=4)
        start = memory_address(channel=1, bank=2, row=1, column=6)
        storage.write(start, (1.0,))

        with self.assertRaises(CentUninitializedReadError) as raised:
            storage.read(start, value_count=2)
        self.assertEqual(raised.exception.device_id, 4)
        self.assertEqual(
            raised.exception.location,
            memory_address(channel=1, bank=2, row=1, column=7),
        )

        for operation in (
                lambda: storage.write(memory_address(column=7), (1.0, 2.0)),
                lambda: storage.write(memory_address(), ()),
                lambda: storage.read(memory_address(), value_count=0),
                lambda: storage.read(memory_address(column=7), value_count=2),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    operation()

    def test_linear_buffer_tracks_initialization_and_clone_isolation(self) -> None:
        """Read initialized scalars and keep cloned writes independent."""

        storage = _LinearBuffer(capacity=5)
        storage.write(1, (2.0, 3.0))

        self.assertEqual(storage.read(1, value_count=2), (2.0, 3.0))
        with self.assertRaises(_UninitializedScalarError) as raised:
            storage.read(0, value_count=2)
        self.assertEqual(raised.exception.index, 0)

        cloned = storage.clone()
        cloned.write(2, (9.0, 10.0))
        self.assertEqual(storage.read(1, value_count=2), (2.0, 3.0))
        self.assertEqual(cloned.read(1, value_count=3), (2.0, 9.0, 10.0))

    def test_linear_buffer_validates_capacity_counts_and_spans(self) -> None:
        """Reject unusable capacity and every invalid linear access shape."""

        with self.assertRaises(ValueError):
            _LinearBuffer(capacity=0)
        with self.assertRaises(ValueError):
            _UninitializedScalarError(-1)

        storage = _LinearBuffer(capacity=2)
        for operation in (
                lambda: storage.write(-1, (1.0,)),
                lambda: storage.write(2, (1.0,)),
                lambda: storage.write(0, ()),
                lambda: storage.write(1, (1.0, 2.0)),
                lambda: storage.read(-1, value_count=1),
                lambda: storage.read(2, value_count=1),
                lambda: storage.read(0, value_count=0),
                lambda: storage.read(1, value_count=2),
        ):
            with self.subTest(operation=operation):
                with self.assertRaises(ValueError):
                    operation()

    def test_register_file_keeps_regions_independent_and_checks_bounds(self) -> None:
        """Store one register and reject missing or out-of-range coordinates."""

        registers = _BankRegisterFile(
            hardware=hardware(),
            device_id=5,
            region_name="accumulator",
            uninitialized_reason="accumulator register is uninitialized",
        )
        address = CentBankRegisterAddress(channel=1, bank=7, register=1)
        registers.write(address, 4.0)
        registers._validate_address(address)
        self.assertEqual(registers.read(address), 4.0)

        missing = CentBankRegisterAddress(channel=0, bank=0, register=0)
        with self.assertRaises(CentUninitializedReadError) as raised:
            registers.read(missing)
        self.assertEqual(raised.exception.device_id, 5)
        self.assertEqual(raised.exception.location, missing)

        for invalid in (
                CentBankRegisterAddress(channel=2, bank=0, register=0),
                CentBankRegisterAddress(channel=0, bank=8, register=0),
                CentBankRegisterAddress(channel=0, bank=0, register=2),
        ):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    registers._validate_address(invalid)
                with self.assertRaises(ValueError):
                    registers.write(invalid, 1.0)
                with self.assertRaises(ValueError):
                    registers.read(invalid)

    def test_storage_components_reject_negative_device_identifiers(self) -> None:
        """Require structured storage failures to use nonnegative device IDs."""

        with self.assertRaises(ValueError):
            _SparseDram(hardware=hardware(), device_id=-1)
        with self.assertRaises(ValueError):
            _BankRegisterFile(
                hardware=hardware(),
                device_id=-1,
                region_name="accumulator",
                uninitialized_reason="accumulator register is uninitialized",
            )
        for region_name, reason in (
                ("", "accumulator register is uninitialized"),
                ("accumulator", ""),
        ):
            with self.subTest(region_name=region_name, reason=reason):
                with self.assertRaises(ValueError):
                    _BankRegisterFile(
                        hardware=hardware(),
                        device_id=0,
                        region_name=region_name,
                        uninitialized_reason=reason,
                    )


class DeviceStateTests(unittest.TestCase):
    """Test typed sparse storage and strict initialization tracking."""

    def test_all_state_regions_begin_uninitialized(self) -> None:
        """Reject reads from every region until an initializer writes it."""

        state = CentDeviceState(hardware=hardware())
        register = CentBankRegisterAddress(channel=0, bank=0, register=0)
        cases = (
            lambda: state.read_dram(memory_address(), value_count=1),
            lambda: state.read_shared_buffer(
                CentSharedBufferAddress(slot=0), slot_count=1
            ),
            lambda: state.read_global_buffer(
                CentGlobalBufferAddress(channel=0, column=0), value_count=1
            ),
            lambda: state.read_accumulator(register),
            lambda: state.read_activation_result(register),
        )

        for read in cases:
            with self.subTest(read=read):
                with self.assertRaises(CentUninitializedReadError):
                    read()

    def test_dram_is_sparse_and_rows_are_isolated(self) -> None:
        """Allocate only written rows and keep equal columns in other rows unset."""

        state = CentDeviceState(hardware=hardware())
        written = memory_address(channel=1, bank=3, row=1, column=2)
        other_row = memory_address(channel=1, bank=3, row=0, column=2)

        state.write_dram(written, (1.0, 2.0, 3.0))

        self.assertEqual(state.read_dram(written, value_count=3), (1.0, 2.0, 3.0))
        with self.assertRaises(CentUninitializedReadError):
            state.read_dram(other_row, value_count=1)

    def test_storage_regions_enforce_physical_bounds(self) -> None:
        """Reject spans beyond DRAM, Shared Buffer, and Global Buffer capacities."""

        state = CentDeviceState(hardware=hardware())
        cases = (
            lambda: state.write_dram(
                memory_address(column=7),
                (1.0, 2.0),
            ),
            lambda: state.read_shared_buffer(
                CentSharedBufferAddress(slot=3),
                slot_count=2,
            ),
            lambda: state.write_global_buffer(
                CentGlobalBufferAddress(channel=2, column=0),
                (1.0,),
            ),
            lambda: state.write_global_buffer(
                CentGlobalBufferAddress(channel=0, column=7),
                (1.0, 2.0),
            ),
        )

        for access in cases:
            with self.subTest(access=access):
                with self.assertRaises(ValueError):
                    access()

    def test_storage_regions_reject_invalid_counts_and_starts(self) -> None:
        """Reject empty, zero-length, out-of-range, and crossing raw accesses."""

        state = CentDeviceState(hardware=hardware())
        shared_outside = CentSharedBufferAddress(slot=4)
        global_column_outside = CentGlobalBufferAddress(channel=0, column=8)
        cases = (
            lambda: CentDeviceState(hardware=hardware(), device_id=-1),
            lambda: state.write_dram(memory_address(), ()),
            lambda: state.read_dram(memory_address(), value_count=0),
            lambda: state.read_dram(memory_address(column=7), value_count=2),
            lambda: state.write_shared_buffer(shared_outside, (1.0,)),
            lambda: state.write_shared_buffer(CentSharedBufferAddress(slot=0), ()),
            lambda: state.write_shared_buffer(
                CentSharedBufferAddress(slot=3), (1.0, 2.0, 3.0)
            ),
            lambda: state.read_shared_buffer(shared_outside, slot_count=1),
            lambda: state.read_shared_buffer(
                CentSharedBufferAddress(slot=0), slot_count=0
            ),
            lambda: state.read_shared_buffer_values(shared_outside, value_count=1),
            lambda: state.read_shared_buffer_values(
                CentSharedBufferAddress(slot=0), value_count=0
            ),
            lambda: state.read_shared_buffer_values(
                CentSharedBufferAddress(slot=3), value_count=3
            ),
            lambda: state.write_global_buffer(global_column_outside, (1.0,)),
            lambda: state.write_global_buffer(
                CentGlobalBufferAddress(channel=0, column=0), ()
            ),
            lambda: state.read_global_buffer(
                CentGlobalBufferAddress(channel=2, column=0), value_count=1
            ),
            lambda: state.read_global_buffer(global_column_outside, value_count=1),
            lambda: state.read_global_buffer(
                CentGlobalBufferAddress(channel=0, column=0), value_count=0
            ),
            lambda: state.read_global_buffer(
                CentGlobalBufferAddress(channel=0, column=7), value_count=2
            ),
        )

        for access in cases:
            with self.subTest(access=access):
                with self.assertRaises(ValueError):
                    access()

    def test_register_regions_reject_each_out_of_range_coordinate(self) -> None:
        """Validate channel, bank, and register limits for both register files."""

        state = CentDeviceState(hardware=hardware())
        invalid_addresses = (
            CentBankRegisterAddress(channel=2, bank=0, register=0),
            CentBankRegisterAddress(channel=0, bank=8, register=0),
            CentBankRegisterAddress(channel=0, bank=0, register=2),
        )

        for address in invalid_addresses:
            for access in (
                    lambda address=address: state.write_accumulator(address, 1.0),
                    lambda address=address: state.read_accumulator(address),
                    lambda address=address: state.write_activation_result(address, 1.0),
                    lambda address=address: state.read_activation_result(address),
            ):
                with self.subTest(address=address, access=access):
                    with self.assertRaises(ValueError):
                        access()

    def test_accumulator_and_activation_results_are_independent(self) -> None:
        """Store separate values at the same bank-register coordinates."""

        state = CentDeviceState(hardware=hardware())
        register = CentBankRegisterAddress(channel=1, bank=7, register=1)

        state.write_accumulator(register, 4.0)
        state.write_activation_result(register, -3.0)

        self.assertEqual(state.read_accumulator(register), 4.0)
        self.assertEqual(state.read_activation_result(register), -3.0)

    def test_numeric_zero_is_initialized_data(self) -> None:
        """Return written zero normally instead of confusing it with unset state."""

        state = CentDeviceState(hardware=hardware())
        address = memory_address()

        state.write_dram(address, (0.0,))

        self.assertEqual(state.read_dram(address, value_count=1), (0.0,))

    def test_shared_buffer_exact_count_does_not_read_trailing_lanes(self) -> None:
        """Extract a raw binding that ends inside its final physical slot."""

        state = CentDeviceState(hardware=hardware())
        address = CentSharedBufferAddress(slot=1)
        state.write_shared_buffer(address, (1.0, 2.0, 3.0))

        self.assertEqual(
            state.read_shared_buffer_values(address, value_count=3),
            (1.0, 2.0, 3.0),
        )
        # Reading two complete two-lane slots also consumes the untouched
        # fourth lane and must therefore expose the incomplete initialization.
        with self.assertRaises(CentUninitializedReadError):
            state.read_shared_buffer(address, slot_count=2)

    def test_batch_commit_validates_every_effect_before_mutating_state(self) -> None:
        """Leave all regions unchanged when a later effect is out of bounds."""

        state = CentDeviceState(hardware=hardware())
        shared_address = CentSharedBufferAddress(slot=0)
        state.write_shared_buffer(shared_address, (9.0, 10.0))
        effects = (
            SharedBufferWriteEffect(address=shared_address, values=(1.0, 2.0)),
            GlobalBufferWriteEffect(
                address=CentGlobalBufferAddress(channel=0, column=7),
                # Only one scalar fits at column seven in the eight-column GB.
                values=(3.0, 4.0),
            ),
        )

        with self.assertRaisesRegex(ValueError, "exceeds"):
            state.commit_effects(effects)

        self.assertEqual(
            state.read_shared_buffer(shared_address, slot_count=1),
            (9.0, 10.0),
        )

    def test_batch_commit_requires_nonempty_effects_and_values(self) -> None:
        """Reject an empty transaction or destination span before state mutation."""

        state = CentDeviceState(hardware=hardware())

        with self.assertRaisesRegex(ValueError, "at least one effect"):
            state.commit_effects(())
        with self.assertRaisesRegex(ValueError, "effect cannot be empty"):
            state.commit_effects(
                (
                    DramWriteEffect(
                        address=memory_address(),
                        values=(),
                    ),
                )
            )

    def test_batch_commit_checks_each_effect_address_space(self) -> None:
        """Validate complete DRAM, Shared Buffer, and Global Buffer destinations."""

        state = CentDeviceState(hardware=hardware())
        invalid_effects = (
            DramWriteEffect(
                address=memory_address(column=7),
                values=(1.0, 2.0),
            ),
            SharedBufferWriteEffect(
                address=CentSharedBufferAddress(slot=4),
                values=(1.0,),
            ),
            SharedBufferWriteEffect(
                address=CentSharedBufferAddress(slot=3),
                values=(1.0, 2.0, 3.0),
            ),
            GlobalBufferWriteEffect(
                address=CentGlobalBufferAddress(channel=2, column=0),
                values=(1.0,),
            ),
            GlobalBufferWriteEffect(
                address=CentGlobalBufferAddress(channel=0, column=8),
                values=(1.0,),
            ),
            GlobalBufferWriteEffect(
                address=CentGlobalBufferAddress(channel=0, column=7),
                values=(1.0, 2.0),
            ),
        )

        for effect in invalid_effects:
            with self.subTest(effect=effect):
                with self.assertRaises(ValueError):
                    state.commit_effects((effect,))

    def test_batch_commit_combines_effects_for_one_sparse_dram_row(self) -> None:
        """Apply several effects to one prepared row without losing earlier writes."""

        state = CentDeviceState(hardware=hardware())
        first = memory_address(column=0)
        second = memory_address(column=2)

        state.commit_effects(
            (
                DramWriteEffect(address=first, values=(1.0, 2.0)),
                DramWriteEffect(address=second, values=(3.0, 4.0)),
            )
        )

        self.assertEqual(
            state.read_dram(first, value_count=4),
            (1.0, 2.0, 3.0, 4.0),
        )

    def test_batch_commit_reuses_prepared_dense_components(self) -> None:
        """Apply several writes through one clone of each touched buffer."""

        state = CentDeviceState(hardware=hardware())
        state.commit_effects(
            (
                SharedBufferWriteEffect(
                    address=CentSharedBufferAddress(slot=0),
                    values=(1.0, 2.0),
                ),
                SharedBufferWriteEffect(
                    address=CentSharedBufferAddress(slot=1),
                    values=(3.0, 4.0),
                ),
                GlobalBufferWriteEffect(
                    address=CentGlobalBufferAddress(channel=0, column=0),
                    values=(5.0, 6.0),
                ),
                GlobalBufferWriteEffect(
                    address=CentGlobalBufferAddress(channel=0, column=2),
                    values=(7.0, 8.0),
                ),
            )
        )

        self.assertEqual(
            state.read_shared_buffer(
                CentSharedBufferAddress(slot=0),
                slot_count=2,
            ),
            (1.0, 2.0, 3.0, 4.0),
        )
        self.assertEqual(
            state.read_global_buffer(
                CentGlobalBufferAddress(channel=0, column=0),
                value_count=4,
            ),
            (5.0, 6.0, 7.0, 8.0),
        )

    def test_batch_commit_clones_only_touched_storage_components(self) -> None:
        """Preserve component identity for regions absent from a transaction."""

        state = CentDeviceState(hardware=hardware())
        original_dram = state._dram
        original_shared = state._shared_buffer
        original_globals = state._global_buffers

        state.commit_effects(
            (DramWriteEffect(address=memory_address(), values=(1.0,)),)
        )

        self.assertIsNot(state._dram, original_dram)
        self.assertIs(state._shared_buffer, original_shared)
        for current, original in zip(
                state._global_buffers,
                original_globals,
                strict=True,
        ):
            self.assertIs(current, original)

        dram_after_write = state._dram
        shared_after_write = state._shared_buffer
        globals_after_write = state._global_buffers
        state.commit_effects(
            (
                GlobalBufferWriteEffect(
                    address=CentGlobalBufferAddress(channel=1, column=0),
                    values=(2.0,),
                ),
            )
        )

        self.assertIs(state._dram, dram_after_write)
        self.assertIs(state._shared_buffer, shared_after_write)
        self.assertIs(state._global_buffers[0], globals_after_write[0])
        self.assertIsNot(state._global_buffers[1], globals_after_write[1])

    def test_host_conversion_is_separate_from_stored_value_transport(self) -> None:
        """Quantize host values once and copy their stored representation unchanged."""

        numeric = CountingStoreSemantics()
        state = CentDeviceState(hardware=hardware(), numeric=numeric)
        source = CentSharedBufferAddress(slot=0)
        state.write_shared_buffer(source, (1.0, 2.0))
        self.assertEqual(numeric.store_call_count, 2)

        execute_program(
            program(
                WriteGlobalBuffer(
                    channels=CentChannelSet(channels=(0, 1)),
                    operation_size=1,
                    column=0,
                    source=source,
                )
            ),
            state,
        )

        # A transport instruction carries the already-offset values. Calling
        # store again would both increment this count and produce 1.5/2.5.
        self.assertEqual(numeric.store_call_count, 2)
        for channel in (0, 1):
            self.assertEqual(
                state.read_global_buffer(
                    CentGlobalBufferAddress(channel=channel, column=0),
                    value_count=2,
                ),
                (1.25, 2.25),
            )

    def test_private_host_conversion_prepares_every_value_before_commit(self) -> None:
        """Directly test the helper that converts host values into stored values."""

        numeric = CountingStoreSemantics()
        state = CentDeviceState(hardware=hardware(), numeric=numeric)

        self.assertEqual(state._store_host_values((3.0, 4.0)), (3.25, 4.25))
        self.assertEqual(numeric.store_call_count, 2)

    def test_stored_value_type_check_fails_loudly(self) -> None:
        """Never shorten a read by filtering out a corrupt non-float lane."""

        self.assertEqual(
            _require_stored_floats([1.0, 2.0], region="test"),
            (1.0, 2.0),
        )
        with self.assertRaisesRegex(RuntimeError, "non-float"):
            _require_stored_floats([1.0, 2], region="test")


class InstructionEffectPreparationTests(unittest.TestCase):
    """Directly test dispatch and every private instruction kernel."""

    def test_prepare_write_single_bank_returns_one_dram_effect(self) -> None:
        """Map complete Shared Buffer source slots to the instruction address."""

        state = CentDeviceState(hardware=hardware())
        source = CentSharedBufferAddress(slot=1)
        destination = memory_address(channel=1, bank=3, row=1, column=2)
        state.write_shared_buffer(source, (1.0, 2.0, 3.0, 4.0))
        instruction = WriteSingleBank(
            address=destination,
            operation_size=2,
            source=source,
        )

        self.assertEqual(
            _prepare_write_single_bank(instruction, state),
            _InstructionEffects(
                reads=(CentSharedBufferRegion(address=source, scalar_count=4),),
                writes=(
                    DramWriteEffect(
                        address=destination,
                        values=(1.0, 2.0, 3.0, 4.0),
                    ),
                ),
            ),
        )

    def test_prepare_read_single_bank_returns_one_shared_effect(self) -> None:
        """Map the exact DRAM burst span to its Shared Buffer destination."""

        state = CentDeviceState(hardware=hardware())
        source = memory_address(channel=1, bank=3, row=1, column=2)
        destination = CentSharedBufferAddress(slot=1)
        state.write_dram(source, (1.0, 2.0, 3.0, 4.0))
        instruction = ReadSingleBank(
            address=source,
            operation_size=2,
            destination=destination,
        )

        self.assertEqual(
            _prepare_read_single_bank(instruction, state),
            _InstructionEffects(
                reads=(CentDramRegion(address=source, scalar_count=4),),
                writes=(
                    SharedBufferWriteEffect(
                        address=destination,
                        values=(1.0, 2.0, 3.0, 4.0),
                    ),
                ),
            ),
        )

    def test_prepare_write_global_buffer_reuses_one_source_snapshot(self) -> None:
        """Prepare identical stored values for every selected channel."""

        state = CentDeviceState(hardware=hardware())
        source = CentSharedBufferAddress(slot=0)
        state.write_shared_buffer(source, (2.0, 3.0))
        instruction = WriteGlobalBuffer(
            channels=CentChannelSet(channels=(1, 0)),
            operation_size=1,
            column=2,
            source=source,
        )

        self.assertEqual(
            _prepare_write_global_buffer(instruction, state),
            _InstructionEffects(
                reads=(CentSharedBufferRegion(address=source, scalar_count=2),),
                writes=(
                    GlobalBufferWriteEffect(
                        address=CentGlobalBufferAddress(channel=0, column=2),
                        values=(2.0, 3.0),
                    ),
                    GlobalBufferWriteEffect(
                        address=CentGlobalBufferAddress(channel=1, column=2),
                        values=(2.0, 3.0),
                    ),
                ),
            ),
        )

    def test_prepare_copy_bank_to_global_buffer_keeps_channels_independent(
            self,
    ) -> None:
        """Read each selected channel's named bank into its own Global Buffer."""

        state = CentDeviceState(hardware=hardware())
        for channel, values in ((0, (1.0, 2.0)), (1, (3.0, 4.0))):
            state.write_dram(
                memory_address(channel=channel, bank=3, column=2),
                values,
            )
        instruction = CopyBankToGlobalBuffer(
            channels=CentChannelSet(channels=(0, 1)),
            operation_size=1,
            bank=3,
            row=0,
            column=2,
        )

        self.assertEqual(
            _prepare_copy_bank_to_global_buffer(instruction, state),
            _InstructionEffects(
                reads=(
                    CentDramRegion(
                        address=memory_address(channel=0, bank=3, column=2),
                        scalar_count=2,
                    ),
                    CentDramRegion(
                        address=memory_address(channel=1, bank=3, column=2),
                        scalar_count=2,
                    ),
                ),
                writes=(
                    GlobalBufferWriteEffect(
                        address=CentGlobalBufferAddress(channel=0, column=2),
                        values=(1.0, 2.0),
                    ),
                    GlobalBufferWriteEffect(
                        address=CentGlobalBufferAddress(channel=1, column=2),
                        values=(3.0, 4.0),
                    ),
                ),
            ),
        )

    def test_prepare_copy_global_buffer_to_bank_keeps_channels_independent(
            self,
    ) -> None:
        """Read each selected Global Buffer into the same bank coordinates."""

        state = CentDeviceState(hardware=hardware())
        for channel, values in ((0, (1.0, 2.0)), (1, (3.0, 4.0))):
            state.write_global_buffer(
                CentGlobalBufferAddress(channel=channel, column=2),
                values,
            )
        instruction = CopyGlobalBufferToBank(
            channels=CentChannelSet(channels=(0, 1)),
            operation_size=1,
            bank=3,
            row=1,
            column=2,
        )

        self.assertEqual(
            _prepare_copy_global_buffer_to_bank(instruction, state),
            _InstructionEffects(
                reads=(
                    CentGlobalBufferRegion(
                        address=CentGlobalBufferAddress(channel=0, column=2),
                        scalar_count=2,
                    ),
                    CentGlobalBufferRegion(
                        address=CentGlobalBufferAddress(channel=1, column=2),
                        scalar_count=2,
                    ),
                ),
                writes=(
                    DramWriteEffect(
                        address=memory_address(
                            channel=0,
                            bank=3,
                            row=1,
                            column=2,
                        ),
                        values=(1.0, 2.0),
                    ),
                    DramWriteEffect(
                        address=memory_address(
                            channel=1,
                            bank=3,
                            row=1,
                            column=2,
                        ),
                        values=(3.0, 4.0),
                    ),
                ),
            ),
        )

    def test_prepare_elementwise_multiply_returns_every_group_result(self) -> None:
        """Produce one product effect for each four-bank processing-unit group."""

        state = CentDeviceState(hardware=hardware())
        for bank, values in (
                (0, (1.0, 2.0)),
                (1, (3.0, 4.0)),
                (4, (5.0, 6.0)),
                (5, (7.0, 8.0)),
        ):
            state.write_dram(memory_address(bank=bank), values)
        instruction = ElementwiseMultiply(
            channels=CentChannelSet(channels=(0,)),
            operation_size=1,
            row=0,
            column=0,
        )

        self.assertEqual(
            _prepare_elementwise_multiply(instruction, state),
            _InstructionEffects(
                reads=(
                    CentDramRegion(
                        address=memory_address(bank=0),
                        scalar_count=2,
                    ),
                    CentDramRegion(
                        address=memory_address(bank=1),
                        scalar_count=2,
                    ),
                    CentDramRegion(
                        address=memory_address(bank=4),
                        scalar_count=2,
                    ),
                    CentDramRegion(
                        address=memory_address(bank=5),
                        scalar_count=2,
                    ),
                ),
                writes=(
                    DramWriteEffect(
                        address=memory_address(bank=2),
                        values=(3.0, 8.0),
                    ),
                    DramWriteEffect(
                        address=memory_address(bank=6),
                        values=(35.0, 48.0),
                    ),
                ),
            ),
        )

    def test_prepare_accumulate_snapshots_both_input_ranges(self) -> None:
        """Produce one destination effect containing every lane-wise sum."""

        state = CentDeviceState(hardware=hardware())
        destination = CentSharedBufferAddress(slot=0)
        source = CentSharedBufferAddress(slot=2)
        state.write_shared_buffer(destination, (1.0, 2.0))
        state.write_shared_buffer(source, (3.0, 4.0))
        instruction = Accumulate(
            operation_size=1,
            destination=destination,
            source=source,
        )

        self.assertEqual(
            _prepare_accumulate(instruction, state),
            _InstructionEffects(
                reads=(
                    CentSharedBufferRegion(
                        address=destination,
                        scalar_count=2,
                    ),
                    CentSharedBufferRegion(
                        address=source,
                        scalar_count=2,
                    ),
                ),
                writes=(
                    SharedBufferWriteEffect(
                        address=destination,
                        values=(4.0, 6.0),
                    ),
                ),
            ),
        )

    def test_dispatcher_and_executor_return_the_committed_effects(self) -> None:
        """Dispatch the typed kernel once and return the same effects after commit."""

        state = CentDeviceState(hardware=hardware())
        source = CentSharedBufferAddress(slot=0)
        destination = memory_address()
        state.write_shared_buffer(source, (1.0, 2.0))
        instruction = WriteSingleBank(
            address=destination,
            operation_size=1,
            source=source,
        )
        expected = _InstructionEffects(
            reads=(CentSharedBufferRegion(address=source, scalar_count=2),),
            writes=(DramWriteEffect(address=destination, values=(1.0, 2.0)),),
        )

        self.assertEqual(_prepare_instruction_effects(instruction, state), expected)
        self.assertEqual(_execute_instruction(instruction, state, 0), expected)
        self.assertEqual(state.read_dram(destination, value_count=2), (1.0, 2.0))

        second_state = CentDeviceState(hardware=hardware())
        second_state.write_shared_buffer(source, (1.0, 2.0))
        summary = _execute_preflighted_program(program(instruction), second_state)
        self.assertEqual(summary.committed_effects, (expected,))

    def test_write_effects_expose_typed_physical_regions(self) -> None:
        """Derive exact trace regions without reconstructing effect fields."""

        dram_address = memory_address(column=2)
        shared_address = CentSharedBufferAddress(slot=1)
        global_address = CentGlobalBufferAddress(channel=1, column=2)

        self.assertEqual(
            DramWriteEffect(address=dram_address, values=(1.0, 2.0)).region,
            CentDramRegion(address=dram_address, scalar_count=2),
        )
        self.assertEqual(
            SharedBufferWriteEffect(
                address=shared_address,
                values=(1.0, 2.0),
            ).region,
            CentSharedBufferRegion(address=shared_address, scalar_count=2),
        )
        self.assertEqual(
            GlobalBufferWriteEffect(
                address=global_address,
                values=(1.0, 2.0),
            ).region,
            CentGlobalBufferRegion(address=global_address, scalar_count=2),
        )

    def test_private_preflight_checks_supported_and_unsupported_meaning(self) -> None:
        """Directly validate a supported instruction and reject undefined ones."""

        state = CentDeviceState(hardware=hardware())
        supported = Accumulate(
            operation_size=1,
            destination=CentSharedBufferAddress(slot=0),
            source=CentSharedBufferAddress(slot=0),
        )
        _preflight_instruction(supported, state, 2)

        unsupported = Exponent(
            operation_size=1,
            destination=CentSharedBufferAddress(slot=1),
            source=CentSharedBufferAddress(slot=0),
        )
        with self.assertRaises(CentUnsupportedSemanticsError):
            _preflight_instruction(unsupported, state, 2)
        with self.assertRaises(CentUnsupportedSemanticsError):
            _prepare_instruction_effects(unsupported, state)

    def test_semantics_registry_declares_the_exact_supported_subset(self) -> None:
        """Keep simulator support centralized, unique, and deliberately narrow."""

        registered_types = tuple(
            behavior.instruction_type for behavior in _INSTRUCTION_BEHAVIORS
        )
        expected_types = {
            Accumulate,
            CopyBankToGlobalBuffer,
            CopyGlobalBufferToBank,
            ElementwiseMultiply,
            ReadSingleBank,
            WriteGlobalBuffer,
            WriteSingleBank,
        }

        self.assertEqual(len(registered_types), len(set(registered_types)))
        self.assertSetEqual(set(registered_types), expected_types)

    def test_semantics_registry_preserves_instruction_subclasses(self) -> None:
        """Resolve an instruction subtype through its registered parent behavior."""

        class WriteSingleBankVariant(WriteSingleBank):
            """Test-only subtype with ordinary single-bank-write semantics."""

        state = CentDeviceState(hardware=hardware())
        instruction = WriteSingleBankVariant(
            address=memory_address(),
            operation_size=1,
            source=CentSharedBufferAddress(slot=0),
        )

        behavior = _resolve_instruction_behavior(instruction, state)

        self.assertIs(behavior.instruction_type, WriteSingleBank)

    def test_instruction_behavior_rejects_a_mismatched_direct_call(self) -> None:
        """Defend the type-erased handler boundary against caller mistakes."""

        state = CentDeviceState(hardware=hardware())
        behavior = _InstructionBehavior(
            instruction_type=WriteSingleBank,
            preflight_function=_preflight_defined_instruction,
            prepare_function=_prepare_write_single_bank,
        )
        wrong_instruction = Accumulate(
            operation_size=1,
            destination=CentSharedBufferAddress(slot=0),
            source=CentSharedBufferAddress(slot=0),
        )

        with self.assertRaisesRegex(TypeError, "resolved behavior"):
            behavior.preflight(wrong_instruction, state, 0)
        with self.assertRaisesRegex(TypeError, "resolved behavior"):
            behavior.prepare_effects(wrong_instruction, state)


class TransferExecutionTests(unittest.TestCase):
    """Test the five supported data-movement instructions."""

    def test_single_bank_round_trip_preserves_multiple_bursts(self) -> None:
        """Move two two-lane slots through a nonzero DRAM column and back."""

        state = CentDeviceState(hardware=hardware())
        source = CentSharedBufferAddress(slot=0)
        destination = CentSharedBufferAddress(slot=2)
        address = memory_address(channel=1, bank=5, row=1, column=2)
        state.write_shared_buffer(source, (1.0, 2.0, 3.0, 4.0))

        result = execute_program(
            program(
                WriteSingleBank(
                    address=address,
                    operation_size=2,
                    source=source,
                ),
                ReadSingleBank(
                    address=address,
                    operation_size=2,
                    destination=destination,
                ),
            ),
            state,
        )

        self.assertEqual(result.executed_instruction_count, 2)
        self.assertEqual(
            result.committed_effects,
            (
                _InstructionEffects(
                    reads=(
                        CentSharedBufferRegion(
                            address=source,
                            scalar_count=4,
                        ),
                    ),
                    writes=(
                        DramWriteEffect(
                            address=address,
                            values=(1.0, 2.0, 3.0, 4.0),
                        ),
                    ),
                ),
                _InstructionEffects(
                    reads=(CentDramRegion(address=address, scalar_count=4),),
                    writes=(
                        SharedBufferWriteEffect(
                            address=destination,
                            values=(1.0, 2.0, 3.0, 4.0),
                        ),
                    ),
                ),
            ),
        )
        self.assertEqual(
            state.read_shared_buffer(destination, slot_count=2),
            (1.0, 2.0, 3.0, 4.0),
        )

    def test_write_global_buffer_honors_channel_mask(self) -> None:
        """Replicate source slots only to Global Buffers selected by CHmask."""

        state = CentDeviceState(hardware=hardware())
        state.write_shared_buffer(
            CentSharedBufferAddress(slot=1),
            (2.0, 3.0, 4.0, 5.0),
        )
        instruction = WriteGlobalBuffer(
            channels=CentChannelSet(channels=(1,)),
            operation_size=2,
            column=2,
            source=CentSharedBufferAddress(slot=1),
        )

        execute_program(program(instruction), state)

        self.assertEqual(
            state.read_global_buffer(
                CentGlobalBufferAddress(channel=1, column=2),
                value_count=4,
            ),
            (2.0, 3.0, 4.0, 5.0),
        )
        with self.assertRaises(CentUninitializedReadError):
            state.read_global_buffer(
                CentGlobalBufferAddress(channel=0, column=2),
                value_count=1,
            )

    def test_bank_global_buffer_copies_keep_channels_independent(self) -> None:
        """Copy each selected channel's own bank values out and back."""

        state = CentDeviceState(hardware=hardware())
        state.write_dram(
            memory_address(channel=0, bank=3, row=1, column=2),
            (1.0, 2.0, 3.0, 4.0),
        )
        state.write_dram(
            memory_address(channel=1, bank=3, row=1, column=2),
            (5.0, 6.0, 7.0, 8.0),
        )
        channels = CentChannelSet(channels=(0, 1))

        execute_program(
            program(
                CopyBankToGlobalBuffer(
                    channels=channels,
                    operation_size=2,
                    bank=3,
                    row=1,
                    column=2,
                ),
                CopyGlobalBufferToBank(
                    channels=channels,
                    operation_size=2,
                    bank=6,
                    row=0,
                    column=2,
                ),
            ),
            state,
        )

        self.assertEqual(
            state.read_dram(
                memory_address(channel=0, bank=6, row=0, column=2),
                value_count=4,
            ),
            (1.0, 2.0, 3.0, 4.0),
        )
        self.assertEqual(
            state.read_dram(
                memory_address(channel=1, bank=6, row=0, column=2),
                value_count=4,
            ),
            (5.0, 6.0, 7.0, 8.0),
        )


class ArithmeticExecutionTests(unittest.TestCase):
    """Test lane-wise arithmetic with hand-calculated expected values."""

    def test_elementwise_multiply_runs_in_every_four_bank_group(self) -> None:
        """Multiply bank positions zero and one into position two per PU group."""

        state = CentDeviceState(hardware=hardware())
        # Bank groups are 0-3 and 4-7. The products therefore land in banks 2
        # and 6, respectively, while bank 3 remains untouched.
        for bank, values in (
                (0, (1.0, 2.0, 3.0, 4.0)),
                (1, (5.0, 6.0, 7.0, 8.0)),
                (4, (-1.0, -2.0, -3.0, -4.0)),
                (5, (2.0, 3.0, 4.0, 5.0)),
        ):
            state.write_dram(memory_address(bank=bank, column=2), values)
        instruction = ElementwiseMultiply(
            channels=CentChannelSet(channels=(0,)),
            operation_size=2,
            row=0,
            column=2,
        )

        execute_program(program(instruction), state)

        self.assertEqual(
            state.read_dram(memory_address(bank=2, column=2), value_count=4),
            (5.0, 12.0, 21.0, 32.0),
        )
        self.assertEqual(
            state.read_dram(memory_address(bank=6, column=2), value_count=4),
            (-2.0, -6.0, -12.0, -20.0),
        )
        with self.assertRaises(CentUninitializedReadError):
            state.read_dram(memory_address(bank=3, column=2), value_count=1)

    def test_accumulate_supports_disjoint_and_exact_alias_ranges(self) -> None:
        """Add matching lanes and make exact aliasing double each value."""

        state = CentDeviceState(hardware=hardware())
        state.write_shared_buffer(
            CentSharedBufferAddress(slot=0),
            (1.0, 2.0, 3.0, 4.0),
        )
        state.write_shared_buffer(
            CentSharedBufferAddress(slot=2),
            (10.0, 20.0, 30.0, 40.0),
        )

        execute_program(
            program(
                Accumulate(
                    operation_size=2,
                    destination=CentSharedBufferAddress(slot=0),
                    source=CentSharedBufferAddress(slot=2),
                ),
                Accumulate(
                    operation_size=2,
                    destination=CentSharedBufferAddress(slot=0),
                    source=CentSharedBufferAddress(slot=0),
                ),
            ),
            state,
        )

        # First ACC produces 11, 22, 33, 44. Exact aliasing reads the complete
        # source before committing, so the second ACC doubles those four lanes.
        self.assertEqual(
            state.read_shared_buffer(CentSharedBufferAddress(slot=0), slot_count=2),
            (22.0, 44.0, 66.0, 88.0),
        )


class FailureSemanticsTests(unittest.TestCase):
    """Test preflight and per-instruction transactional behavior."""

    def test_unsupported_preflight_prevents_any_program_mutation(self) -> None:
        """Reject a later unsupported opcode before an earlier valid write runs."""

        state = CentDeviceState(hardware=hardware())
        state.write_shared_buffer(CentSharedBufferAddress(slot=0), (1.0, 2.0))
        destination = memory_address()
        executable = program(
            WriteSingleBank(
                address=destination,
                operation_size=1,
                source=CentSharedBufferAddress(slot=0),
            ),
            Exponent(
                operation_size=1,
                destination=CentSharedBufferAddress(slot=1),
                source=CentSharedBufferAddress(slot=0),
            ),
        )

        with self.assertRaises(CentUnsupportedSemanticsError) as raised:
            execute_program(executable, state)

        self.assertEqual(raised.exception.device_id, 0)
        self.assertEqual(raised.exception.instruction_index, 1)
        self.assertIs(raised.exception.instruction, executable.instructions[1])
        self.assertIn("EXP", raised.exception.reason)
        with self.assertRaises(CentUninitializedReadError):
            state.read_dram(destination, value_count=1)

    def test_preflight_rejects_partial_accumulate_overlap(self) -> None:
        """Reject overlap other than exact alias before changing Shared Buffer."""

        overlap_hardware = hardware(shared_buffer_slots=5)
        state = CentDeviceState(hardware=overlap_hardware)
        state.write_shared_buffer(
            CentSharedBufferAddress(slot=0),
            tuple(float(value) for value in range(10)),
        )
        instruction = Accumulate(
            operation_size=3,
            destination=CentSharedBufferAddress(slot=0),
            source=CentSharedBufferAddress(slot=1),
        )
        executable = CentProgram(
            hardware=overlap_hardware,
            instructions=(instruction,),
        )
        before = state.read_shared_buffer(CentSharedBufferAddress(slot=0), slot_count=5)

        with self.assertRaises(CentUnsupportedSemanticsError):
            preflight_program(executable, state)

        self.assertEqual(
            state.read_shared_buffer(CentSharedBufferAddress(slot=0), slot_count=5),
            before,
        )

    def test_uninitialized_multichannel_source_causes_no_partial_write(self) -> None:
        """Read every source before committing any selected-channel result."""

        state = CentDeviceState(hardware=hardware())
        state.write_dram(memory_address(channel=0, bank=1), (7.0, 8.0))
        instruction = CopyBankToGlobalBuffer(
            channels=CentChannelSet(channels=(0, 1)),
            operation_size=1,
            bank=1,
            row=0,
            column=0,
        )

        with self.assertRaises(CentUninitializedReadError) as raised:
            execute_program(program(instruction), state)

        self.assertEqual(raised.exception.instruction_index, 0)
        self.assertIs(raised.exception.instruction, instruction)
        self.assertEqual(
            raised.exception.location,
            memory_address(channel=1, bank=1),
        )
        with self.assertRaises(CentUninitializedReadError):
            state.read_global_buffer(
                CentGlobalBufferAddress(channel=0, column=0), value_count=1
            )

    def test_write_single_bank_with_late_unset_lane_writes_nothing(self) -> None:
        """Leave the whole DRAM burst span unset when a later source lane fails."""

        state = CentDeviceState(hardware=hardware())
        source = CentSharedBufferAddress(slot=0)
        destination = memory_address(bank=2)
        # WR_SBK requests two slots (four lanes), while only its first three
        # lanes are initialized.
        state.write_shared_buffer(source, (1.0, 2.0, 3.0))
        instruction = WriteSingleBank(
            address=destination,
            operation_size=2,
            source=source,
        )

        with self.assertRaises(CentUninitializedReadError):
            execute_program(program(instruction), state)

        with self.assertRaises(CentUninitializedReadError):
            state.read_dram(destination, value_count=1)

    def test_read_single_bank_with_late_unset_lane_preserves_destination(self) -> None:
        """Leave every destination lane unchanged when the DRAM source is incomplete."""

        state = CentDeviceState(hardware=hardware())
        source = memory_address(bank=2)
        destination = CentSharedBufferAddress(slot=0)
        state.write_dram(source, (1.0, 2.0, 3.0))
        state.write_shared_buffer(destination, (90.0, 91.0, 92.0, 93.0))
        instruction = ReadSingleBank(
            address=source,
            operation_size=2,
            destination=destination,
        )

        with self.assertRaises(CentUninitializedReadError):
            execute_program(program(instruction), state)

        self.assertEqual(
            state.read_shared_buffer(destination, slot_count=2),
            (90.0, 91.0, 92.0, 93.0),
        )

    def test_write_global_buffer_with_late_unset_lane_changes_no_channel(self) -> None:
        """Preserve every selected Global Buffer when the SB snapshot is incomplete."""

        state = CentDeviceState(hardware=hardware())
        source = CentSharedBufferAddress(slot=0)
        for channel, values in (
                (0, (10.0, 11.0, 12.0, 13.0)),
                (1, (20.0, 21.0, 22.0, 23.0)),
        ):
            state.write_global_buffer(
                CentGlobalBufferAddress(channel=channel, column=2),
                values,
            )
        state.write_shared_buffer(source, (1.0, 2.0, 3.0))
        instruction = WriteGlobalBuffer(
            channels=CentChannelSet(channels=(0, 1)),
            operation_size=2,
            column=2,
            source=source,
        )

        with self.assertRaises(CentUninitializedReadError):
            execute_program(program(instruction), state)

        self.assertEqual(
            state.read_global_buffer(
                CentGlobalBufferAddress(channel=0, column=2), value_count=4
            ),
            (10.0, 11.0, 12.0, 13.0),
        )
        self.assertEqual(
            state.read_global_buffer(
                CentGlobalBufferAddress(channel=1, column=2), value_count=4
            ),
            (20.0, 21.0, 22.0, 23.0),
        )

    def test_copy_global_buffer_with_late_missing_channel_writes_no_bank(self) -> None:
        """Preserve every DRAM destination when a later channel source is unset."""

        state = CentDeviceState(hardware=hardware())
        for channel, values in (
                (0, (90.0, 91.0)),
                (1, (92.0, 93.0)),
        ):
            state.write_dram(
                memory_address(channel=channel, bank=3),
                values,
            )
        state.write_global_buffer(
            CentGlobalBufferAddress(channel=0, column=0),
            (1.0, 2.0),
        )
        instruction = CopyGlobalBufferToBank(
            channels=CentChannelSet(channels=(0, 1)),
            operation_size=1,
            bank=3,
            row=0,
            column=0,
        )

        with self.assertRaises(CentUninitializedReadError):
            execute_program(program(instruction), state)

        self.assertEqual(
            state.read_dram(memory_address(channel=0, bank=3), value_count=2),
            (90.0, 91.0),
        )
        self.assertEqual(
            state.read_dram(memory_address(channel=1, bank=3), value_count=2),
            (92.0, 93.0),
        )

    def test_elementwise_multiply_with_late_missing_group_writes_nothing(self) -> None:
        """Preserve all PU results when a later group has an unset operand."""

        state = CentDeviceState(hardware=hardware())
        for bank, values in (
                (0, (1.0, 2.0)),
                (1, (3.0, 4.0)),
                (2, (90.0, 91.0)),
                (4, (5.0, 6.0)),
                # Bank 5 is intentionally uninitialized in the second PU group.
                (6, (92.0, 93.0)),
        ):
            state.write_dram(memory_address(bank=bank), values)
        instruction = ElementwiseMultiply(
            channels=CentChannelSet(channels=(0,)),
            operation_size=1,
            row=0,
            column=0,
        )

        with self.assertRaises(CentUninitializedReadError):
            execute_program(program(instruction), state)

        self.assertEqual(
            state.read_dram(memory_address(bank=2), value_count=2),
            (90.0, 91.0),
        )
        self.assertEqual(
            state.read_dram(memory_address(bank=6), value_count=2),
            (92.0, 93.0),
        )

    def test_accumulate_with_late_unset_source_lane_preserves_destination(self) -> None:
        """Keep the full ACC destination when its last source lane is unset."""

        state = CentDeviceState(hardware=hardware())
        destination = CentSharedBufferAddress(slot=0)
        source = CentSharedBufferAddress(slot=2)
        state.write_shared_buffer(destination, (10.0, 20.0, 30.0, 40.0))
        state.write_shared_buffer(source, (1.0, 2.0, 3.0))
        instruction = Accumulate(
            operation_size=2,
            destination=destination,
            source=source,
        )

        with self.assertRaises(CentUninitializedReadError):
            execute_program(program(instruction), state)

        self.assertEqual(
            state.read_shared_buffer(destination, slot_count=2),
            (10.0, 20.0, 30.0, 40.0),
        )

    def test_late_numeric_failure_changes_no_elementwise_result_bank(self) -> None:
        """Compute all EW_MUL effects before committing any processing-unit group."""

        numeric = FailingArithmeticSemantics(fail_multiply_at=3)
        state = CentDeviceState(hardware=hardware(), numeric=numeric)
        for bank, values in (
                (0, (1.0, 2.0)),
                (1, (3.0, 4.0)),
                (2, (90.0, 91.0)),
                (4, (5.0, 6.0)),
                (5, (7.0, 8.0)),
                (6, (92.0, 93.0)),
        ):
            state.write_dram(memory_address(bank=bank), values)
        instruction = ElementwiseMultiply(
            channels=CentChannelSet(channels=(0,)),
            operation_size=1,
            row=0,
            column=0,
        )

        with self.assertRaisesRegex(
                CentExecutionFault,
                "configured multiply",
        ) as raised:
            execute_program(program(instruction), state)

        self.assertEqual(raised.exception.location, memory_address(bank=6))
        self.assertEqual(
            state.read_dram(memory_address(bank=2), value_count=2),
            (90.0, 91.0),
        )
        self.assertEqual(
            state.read_dram(memory_address(bank=6), value_count=2),
            (92.0, 93.0),
        )

    def test_late_numeric_failure_preserves_accumulate_destination(self) -> None:
        """Keep all ACC lanes and report its destination when addition fails."""

        numeric = FailingArithmeticSemantics(fail_add_at=2)
        state = CentDeviceState(hardware=hardware(), numeric=numeric)
        destination = CentSharedBufferAddress(slot=0)
        source = CentSharedBufferAddress(slot=1)
        state.write_shared_buffer(destination, (1.0, 2.0))
        state.write_shared_buffer(source, (3.0, 4.0))
        instruction = Accumulate(
            operation_size=1,
            destination=destination,
            source=source,
        )

        with self.assertRaisesRegex(
                CentExecutionFault,
                "configured add",
        ) as raised:
            execute_program(program(instruction), state)

        self.assertEqual(raised.exception.location, destination)
        self.assertEqual(
            state.read_shared_buffer(destination, slot_count=1),
            (1.0, 2.0),
        )

    def test_program_keeps_completed_prefix_when_later_instruction_faults(self) -> None:
        """Commit completed instructions but keep a failing instruction atomic."""

        state = CentDeviceState(hardware=hardware())
        state.write_shared_buffer(CentSharedBufferAddress(slot=0), (3.0, 4.0))
        first_destination = memory_address(bank=0)
        instruction = CopyBankToGlobalBuffer(
            channels=CentChannelSet(channels=(0, 1)),
            operation_size=1,
            bank=1,
            row=0,
            column=0,
        )
        executable = program(
            WriteSingleBank(
                address=first_destination,
                operation_size=1,
                source=CentSharedBufferAddress(slot=0),
            ),
            instruction,
        )

        with self.assertRaises(CentUninitializedReadError):
            execute_program(executable, state)

        self.assertEqual(
            state.read_dram(first_destination, value_count=2),
            (3.0, 4.0),
        )

    def test_program_and_state_hardware_must_match(self) -> None:
        """Reject execution when the state belongs to another target geometry."""

        executable = program(
            Accumulate(
                operation_size=1,
                destination=CentSharedBufferAddress(slot=0),
                source=CentSharedBufferAddress(slot=0),
            )
        )
        other_hardware = hardware(shared_buffer_slots=5)

        with self.assertRaises(CentExecutionFault):
            preflight_program(
                executable,
                CentDeviceState(hardware=other_hardware),
            )


if __name__ == "__main__":
    unittest.main()
