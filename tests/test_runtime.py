"""Tests for the reusable CENT runtime-data contract."""

import unittest
from dataclasses import FrozenInstanceError

from vllm_cent.cent import (
    CentGlobalBufferAddress,
    CentHardwareSpec,
    CentMemoryAddress,
    CentProgram,
    CentSharedBufferAddress,
    ReadSingleBank,
)
from vllm_cent.runtime import (
    CentDramRegion,
    CentExecutable,
    CentExecutionManifest,
    CentGlobalBufferRegion,
    CentInputBinding,
    CentNamedScalars,
    CentOutputBinding,
    CentSharedBufferRegion,
)
from vllm_cent.runtime.executable import (
    _CentPhysicalSpan,
    _CentStorageSpace,
    _physical_span,
    _validate_no_input_overlaps,
    _validate_region_hardware,
    _validate_unique_names,
)


def make_hardware() -> CentHardwareSpec:
    """Create the small hardware geometry used by runtime tests.

    Returns:
        Target with two channels, four banks, and four lanes per burst.
    """

    return CentHardwareSpec(
        num_channels=2,
        num_banks=4,
        dram_rows=3,
        dram_columns=8,
        burst_length=4,
        accumulator_slots_per_bank=2,
        sigmoid_activation_function_id=3,
        shared_buffer_slots=4,
        global_buffer_columns=12,
    )


def make_program(hardware: CentHardwareSpec) -> CentProgram:
    """Create one valid instruction so a runtime executable can be built.

    Args:
        hardware: Device targeted by the program.

    Returns:
        Program that reads one burst from DRAM into the Shared Buffer.
    """

    return CentProgram(
        hardware=hardware,
        instructions=(
            ReadSingleBank(
                address=CentMemoryAddress(
                    channel=0,
                    bank=0,
                    row=0,
                    column=0,
                ),
                operation_size=1,
                destination=CentSharedBufferAddress(slot=0),
            ),
        ),
    )


def dram_region(
        *,
        channel: int = 0,
        bank: int = 0,
        row: int = 0,
        column: int = 0,
        scalar_count: int = 1,
) -> CentDramRegion:
    """Create a DRAM region with concise coordinate overrides.

    Args:
        channel: Physical channel containing the region.
        bank: Bank within the selected channel.
        row: Row within the selected bank.
        column: First scalar column in the row.
        scalar_count: Number of consecutive scalar columns.

    Returns:
        Raw DRAM region with the requested coordinates.
    """

    return CentDramRegion(
        address=CentMemoryAddress(
            channel=channel,
            bank=bank,
            row=row,
            column=column,
        ),
        scalar_count=scalar_count,
    )


class CentRuntimeValueTests(unittest.TestCase):
    """Check named host values and reusable physical regions."""

    def test_named_scalars_are_immutable_and_require_values(self) -> None:
        """A request value has a stable name and at least one raw scalar."""

        value = CentNamedScalars(name="input", values=(1.0, 2.0))

        self.assertEqual(value.values, (1.0, 2.0))
        self.assertFalse(hasattr(value, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            value.name = "changed"  # type: ignore[misc]

        with self.assertRaisesRegex(ValueError, "name cannot be empty"):
            CentNamedScalars(name="", values=(1.0,))
        with self.assertRaisesRegex(ValueError, "at least one scalar"):
            CentNamedScalars(name="input", values=())

    def test_regions_and_bindings_validate_target_independent_fields(self) -> None:
        """Reject empty regions and bindings without stable names."""

        region = dram_region()
        binding = CentInputBinding(name="input", region=region)
        self.assertFalse(hasattr(region, "__dict__"))
        self.assertFalse(hasattr(binding, "__dict__"))
        with self.assertRaises(FrozenInstanceError):
            region.scalar_count = 2  # type: ignore[misc]

        invalid_regions = (
            lambda: dram_region(scalar_count=0),
            lambda: CentSharedBufferRegion(
                address=CentSharedBufferAddress(slot=0),
                scalar_count=0,
            ),
            lambda: CentGlobalBufferRegion(
                address=CentGlobalBufferAddress(channel=0, column=0),
                scalar_count=0,
            ),
        )
        for constructor in invalid_regions:
            with self.subTest(constructor=constructor):
                with self.assertRaisesRegex(
                        ValueError, "scalar_count must be at least 1"
                ):
                    constructor()

        with self.assertRaisesRegex(ValueError, "name cannot be empty"):
            CentInputBinding(name="", region=dram_region())
        with self.assertRaisesRegex(ValueError, "name cannot be empty"):
            CentOutputBinding(name="", region=dram_region())


class CentExecutionManifestTests(unittest.TestCase):
    """Check manifest identity, input, bounds, and aliasing rules."""

    def test_executable_accepts_all_raw_address_spaces(self) -> None:
        """DRAM, Shared Buffer, and Global Buffer regions can coexist."""

        hardware = make_hardware()
        manifest = CentExecutionManifest(
            inputs=(
                CentInputBinding(
                    name="dram_input",
                    region=dram_region(bank=1, row=2, column=4, scalar_count=4),
                ),
                CentInputBinding(
                    name="shared_input",
                    region=CentSharedBufferRegion(
                        address=CentSharedBufferAddress(slot=1),
                        scalar_count=5,
                    ),
                ),
                CentInputBinding(
                    name="global_input",
                    region=CentGlobalBufferRegion(
                        address=CentGlobalBufferAddress(channel=1, column=6),
                        scalar_count=6,
                    ),
                ),
            ),
            outputs=(
                CentOutputBinding(
                    name="dram_output",
                    region=dram_region(
                        channel=1,
                        bank=3,
                        row=2,
                        scalar_count=4,
                    ),
                ),
                CentOutputBinding(
                    name="shared_output",
                    region=CentSharedBufferRegion(
                        address=CentSharedBufferAddress(slot=3),
                        scalar_count=4,
                    ),
                ),
                CentOutputBinding(
                    name="global_output",
                    region=CentGlobalBufferRegion(
                        address=CentGlobalBufferAddress(channel=0, column=8),
                        scalar_count=4,
                    ),
                ),
            ),
        )

        executable = CentExecutable(program=make_program(hardware), manifest=manifest)

        self.assertIs(executable.manifest, manifest)
        self.assertFalse(hasattr(executable, "__dict__"))

    def test_manifest_validates_and_orders_request_values(self) -> None:
        """Inputs are returned in binding order after exact validation."""

        manifest = CentExecutionManifest(
            inputs=(
                CentInputBinding(
                    name="first",
                    region=CentSharedBufferRegion(
                        address=CentSharedBufferAddress(slot=0),
                        scalar_count=2,
                    ),
                ),
                CentInputBinding(
                    name="second",
                    region=CentSharedBufferRegion(
                        address=CentSharedBufferAddress(slot=1),
                        scalar_count=1,
                    ),
                ),
            )
        )

        ordered = manifest.order_input_values(
            (
                CentNamedScalars(name="second", values=(3.0,)),
                CentNamedScalars(name="first", values=(1.0, 2.0)),
            )
        )

        self.assertEqual(
            ordered,
            (
                CentNamedScalars(name="first", values=(1.0, 2.0)),
                CentNamedScalars(name="second", values=(3.0,)),
            ),
        )

    def test_manifest_rejects_invalid_request_values(self) -> None:
        """Request names and scalar counts must exactly match input bindings."""

        manifest = CentExecutionManifest(
            inputs=(
                CentInputBinding(
                    name="input",
                    region=CentSharedBufferRegion(
                        address=CentSharedBufferAddress(slot=0),
                        scalar_count=2,
                    ),
                ),
            )
        )

        with self.assertRaisesRegex(ValueError, "missing.*input"):
            manifest.order_input_values(())
        with self.assertRaisesRegex(ValueError, "unexpected.*other"):
            manifest.order_input_values(
                (
                    CentNamedScalars(name="input", values=(1.0, 2.0)),
                    CentNamedScalars(name="other", values=(3.0,)),
                )
            )
        with self.assertRaisesRegex(ValueError, "duplicate.*input"):
            manifest.order_input_values(
                (
                    CentNamedScalars(name="input", values=(1.0, 2.0)),
                    CentNamedScalars(name="input", values=(3.0, 4.0)),
                )
            )
        with self.assertRaisesRegex(ValueError, "expects 2 scalars"):
            manifest.order_input_values(
                (CentNamedScalars(name="input", values=(1.0,)),)
            )

    def test_names_are_unique_per_direction_but_may_cross_directions(self) -> None:
        """Mutable state may preserve one name from input through output."""

        shared_region = CentSharedBufferRegion(
            address=CentSharedBufferAddress(slot=0),
            scalar_count=2,
        )
        manifest = CentExecutionManifest(
            inputs=(CentInputBinding(name="cache", region=shared_region),),
            outputs=(CentOutputBinding(name="cache", region=shared_region),),
        )

        self.assertEqual(manifest.inputs[0].name, manifest.outputs[0].name)
        with self.assertRaisesRegex(ValueError, "duplicate input binding name"):
            CentExecutionManifest(
                inputs=(
                    CentInputBinding(name="duplicate", region=dram_region()),
                    CentInputBinding(
                        name="duplicate",
                        region=dram_region(bank=1),
                    ),
                )
            )
        with self.assertRaisesRegex(ValueError, "duplicate output binding name"):
            CentExecutionManifest(
                outputs=(
                    CentOutputBinding(name="duplicate", region=dram_region()),
                    CentOutputBinding(
                        name="duplicate",
                        region=dram_region(bank=1),
                    ),
                )
            )

    def test_manifest_rejects_overlapping_input_regions(self) -> None:
        """Two host inputs cannot initialize the same physical scalar lane."""

        overlapping_pairs = (
            (
                CentInputBinding(
                    name="left",
                    region=dram_region(column=1, scalar_count=3),
                ),
                CentInputBinding(
                    name="right",
                    region=dram_region(column=3, scalar_count=2),
                ),
            ),
            (
                CentInputBinding(
                    name="left",
                    region=CentSharedBufferRegion(
                        address=CentSharedBufferAddress(slot=0),
                        scalar_count=5,
                    ),
                ),
                CentInputBinding(
                    name="right",
                    region=CentSharedBufferRegion(
                        address=CentSharedBufferAddress(slot=1),
                        scalar_count=1,
                    ),
                ),
            ),
            (
                CentInputBinding(
                    name="left",
                    region=CentGlobalBufferRegion(
                        address=CentGlobalBufferAddress(channel=0, column=0),
                        scalar_count=4,
                    ),
                ),
                CentInputBinding(
                    name="right",
                    region=CentGlobalBufferRegion(
                        address=CentGlobalBufferAddress(channel=0, column=3),
                        scalar_count=2,
                    ),
                ),
            ),
        )

        for left, right in overlapping_pairs:
            with self.subTest(left=left, right=right):
                with self.assertRaisesRegex(ValueError, "input regions.*overlap"):
                    manifest = CentExecutionManifest(inputs=(left, right))
                    CentExecutable(
                        program=make_program(make_hardware()),
                        manifest=manifest,
                    )

    def test_overlapping_output_regions_are_valid_read_only_views(self) -> None:
        """Several names may read intersecting spans without write ambiguity."""

        region = CentSharedBufferRegion(
            address=CentSharedBufferAddress(slot=1),
            scalar_count=4,
        )
        outputs = (
            CentOutputBinding(name="complete", region=region),
            CentOutputBinding(
                name="prefix",
                region=CentSharedBufferRegion(
                    address=CentSharedBufferAddress(slot=1),
                    scalar_count=2,
                ),
            ),
        )
        manifest = CentExecutionManifest(outputs=outputs)

        executable = CentExecutable(
            program=make_program(make_hardware()),
            manifest=manifest,
        )

        self.assertEqual(executable.manifest.outputs, outputs)

    def test_adjacent_regions_and_separate_coordinates_do_not_overlap(self) -> None:
        """Input checks preserve half-open bounds and physical coordinates."""

        manifest = CentExecutionManifest(
            inputs=(
                CentInputBinding(
                    name="first_row",
                    region=dram_region(column=0, scalar_count=4),
                ),
                CentInputBinding(
                    name="adjacent",
                    region=dram_region(column=4, scalar_count=4),
                ),
                CentInputBinding(
                    name="other_bank",
                    region=dram_region(bank=1, scalar_count=4),
                ),
            )
        )

        executable = CentExecutable(
            program=make_program(make_hardware()),
            manifest=manifest,
        )

        self.assertEqual(
            tuple(binding.name for binding in executable.manifest.inputs),
            ("first_row", "adjacent", "other_bank"),
        )

    def test_executable_rejects_out_of_bounds_regions(self) -> None:
        """Each raw region must fit completely in its hardware address space."""

        hardware = make_hardware()
        invalid_regions = (
            dram_region(column=7, scalar_count=2),
            CentSharedBufferRegion(
                address=CentSharedBufferAddress(slot=3),
                # Four slots with four lanes each provide lanes 0 through 15.
                # Slot 3 therefore has room for exactly four scalars, not five.
                scalar_count=5,
            ),
            CentGlobalBufferRegion(
                address=CentGlobalBufferAddress(channel=2, column=0),
                scalar_count=1,
            ),
            CentGlobalBufferRegion(
                address=CentGlobalBufferAddress(channel=0, column=11),
                scalar_count=2,
            ),
        )

        for index, region in enumerate(invalid_regions):
            with self.subTest(region=region):
                with self.assertRaises(ValueError):
                    CentExecutable(
                        program=make_program(hardware),
                        manifest=CentExecutionManifest(
                            inputs=(
                                CentInputBinding(name=f"input_{index}", region=region),
                            )
                        ),
                    )

    def test_global_buffer_uses_its_explicit_capacity(self) -> None:
        """Accept Global Buffer columns beyond the narrower DRAM row width."""

        manifest = CentExecutionManifest(
            outputs=(
                CentOutputBinding(
                    name="wide_global_output",
                    region=CentGlobalBufferRegion(
                        # The DRAM row ends at column 8, while this target's
                        # explicitly configured Global Buffer ends at column 12.
                        address=CentGlobalBufferAddress(channel=0, column=8),
                        scalar_count=4,
                    ),
                ),
            )
        )

        executable = CentExecutable(
            program=make_program(make_hardware()),
            manifest=manifest,
        )

        self.assertEqual(
            executable.manifest.outputs[0].region,
            CentGlobalBufferRegion(
                address=CentGlobalBufferAddress(channel=0, column=8),
                scalar_count=4,
            ),
        )


class RuntimeValidationHelperTests(unittest.TestCase):
    """Directly test nontrivial normalization and validation helpers."""

    def test_physical_span_normalizes_every_region_type(self) -> None:
        """Convert each typed region into its exact scalar interval."""

        hardware = make_hardware()
        cases = (
            (
                dram_region(channel=1, bank=2, row=1, column=3, scalar_count=2),
                _CentPhysicalSpan(
                    space=_CentStorageSpace.DRAM,
                    coordinates=(1, 2, 1),
                    start=3,
                    end=5,
                ),
            ),
            (
                CentSharedBufferRegion(
                    address=CentSharedBufferAddress(slot=2),
                    scalar_count=5,
                ),
                _CentPhysicalSpan(
                    space=_CentStorageSpace.SHARED_BUFFER,
                    coordinates=(),
                    # Two complete four-lane slots precede slot two.
                    start=8,
                    end=13,
                ),
            ),
            (
                CentGlobalBufferRegion(
                    address=CentGlobalBufferAddress(channel=1, column=4),
                    scalar_count=3,
                ),
                _CentPhysicalSpan(
                    space=_CentStorageSpace.GLOBAL_BUFFER,
                    coordinates=(1,),
                    start=4,
                    end=7,
                ),
            ),
        )

        for region, expected in cases:
            with self.subTest(region=region):
                self.assertEqual(_physical_span(region, hardware), expected)

    def test_physical_span_overlap_requires_same_region_and_intersection(self) -> None:
        """Use half-open intervals and complete physical coordinates."""

        base = _CentPhysicalSpan(
            space=_CentStorageSpace.DRAM,
            coordinates=(0, 0, 0),
            start=2,
            end=6,
        )

        self.assertTrue(
            base.overlaps(
                _CentPhysicalSpan(
                    space=_CentStorageSpace.DRAM,
                    coordinates=(0, 0, 0),
                    start=5,
                    end=7,
                )
            )
        )
        self.assertFalse(
            base.overlaps(
                _CentPhysicalSpan(
                    space=_CentStorageSpace.DRAM,
                    coordinates=(0, 0, 0),
                    start=6,
                    end=8,
                )
            )
        )
        self.assertFalse(
            base.overlaps(
                _CentPhysicalSpan(
                    space=_CentStorageSpace.DRAM,
                    coordinates=(0, 1, 0),
                    start=2,
                    end=6,
                )
            )
        )

    def test_unique_name_helper_scopes_names_to_one_direction(self) -> None:
        """Reject a duplicate in one supplied binding sequence."""

        bindings = (
            CentInputBinding(name="same", region=dram_region()),
            CentInputBinding(name="same", region=dram_region(bank=1)),
        )

        with self.assertRaisesRegex(ValueError, "duplicate input binding name"):
            _validate_unique_names(bindings, category="input")

    def test_input_overlap_helper_reports_the_binding_names(self) -> None:
        """Reject intersecting normalized inputs with a useful error."""

        bindings = (
            CentInputBinding(
                name="left",
                region=dram_region(column=0, scalar_count=2),
            ),
            CentInputBinding(
                name="right",
                region=dram_region(column=1, scalar_count=2),
            ),
        )

        with self.assertRaisesRegex(ValueError, "left and right overlap"):
            _validate_no_input_overlaps(bindings, hardware=make_hardware())

    def test_region_hardware_helper_checks_each_storage_capacity(self) -> None:
        """Reject one out-of-range region in each independent address space."""

        hardware = make_hardware()
        invalid_regions = (
            dram_region(column=7, scalar_count=2),
            CentSharedBufferRegion(
                address=CentSharedBufferAddress(slot=3),
                scalar_count=5,
            ),
            CentGlobalBufferRegion(
                address=CentGlobalBufferAddress(channel=0, column=11),
                scalar_count=2,
            ),
        )

        for region in invalid_regions:
            with self.subTest(region=region):
                with self.assertRaises(ValueError):
                    _validate_region_hardware(region, hardware)


if __name__ == "__main__":
    unittest.main()
