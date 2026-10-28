"""Tests for DeviceCapability, DeviceSpec and the device spec YAML loader."""

from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from vllm_cent.activations import ActivationKind
from vllm_cent.cent import CentHardwareSpec
from vllm_cent.device import (
    CentArchitecture,
    DeviceCapability,
    DeviceSpec,
    MemorySpace,
    MemorySpaceSpec,
    TargetDescriptor,
    TraceFormat,
    load_device_spec,
    parse_device_spec,
    snapshot,
)
from vllm_cent.ir import DType, ExecutionDomain, Operation, TensorType, Value
from vllm_cent.ir.device import (
    CXLAllReduceOp,
    DeviceBroadcastOp,
    DeviceCopyOp,
    DeviceReshapeOp,
    DeviceTransferOp,
    PIMActivationOp,
    PIMCacheWriteOp,
    PIMEwmulOp,
    PIMGemvOp,
    PIMQKOp,
    PIMSVOp,
    PNMActivationOp,
    PNMAddOp,
    PNMDivOp,
    PNMExpOp,
    PNMMaxOp,
    PNMMulOp,
    PNMRMSNormOp,
    PNMRopeOp,
    PNMSubOp,
    PNMSumOp,
)
from vllm_cent.ir.semantic import AddOp

from tests.test_device_cent import PAPER

PIM = ExecutionDomain.PIM
PNM = ExecutionDomain.PNM
BF16 = DType.BF16

EXAMPLE = Path(__file__).resolve().parents[1] / "examples" / "cent" / "cent.yaml"

# A target that runs bf16 GEMV and PNM add, sigmoid at AFid 2 in PIM, and SiLU
# in PNM: enough to reach every branch of why_unsupported().
CAPABILITY = DeviceCapability(
    ops={
        PIMGemvOp: frozenset({BF16}),
        PNMAddOp: frozenset({BF16, DType.FP32}),
        PIMActivationOp: frozenset({BF16}),
        PNMActivationOp: frozenset({BF16}),
    },
    pim_activations={ActivationKind.SIGMOID: 2},
    pnm_activations=frozenset({ActivationKind.SILU}),
    pim_gemv_bias=False,
)


def _value(
    shape: tuple[int, ...], domain: ExecutionDomain, dtype: DType = BF16
) -> Value:
    """Create an unannotated, replicated value in ``domain``.

    Args:
        shape: Tensor shape.
        domain: Where the tensor lives.
        dtype: Element type.

    Returns:
        A new value.
    """

    return Value(type=TensorType(shape=shape, dtype=dtype, domain=domain))


def _gemv(bias: bool = False) -> PIMGemvOp:
    """Return a [1, 4] x [2, 4] GEMV on PIM, with a [2] bias if requested.

    Args:
        bias: Whether to pass the optional third operand.

    Returns:
        The op.
    """

    operands = [_value((1, 4), PIM), _value((2, 4), PIM)]
    if bias:
        operands.append(_value((2,), PIM))
    return PIMGemvOp(operands=tuple(operands))


def _example_document() -> dict[str, Any]:
    """Return the example YAML as a fresh, mutable document.

    Returns:
        The snapshot of the example spec, which has the same schema; tests
        edit their own copy to build invalid documents.
    """

    return copy.deepcopy(snapshot(load_device_spec(EXAMPLE)))


class DeviceCapabilityTest(unittest.TestCase):
    """Support is looked up by op class, dtype and op-specific parameters."""

    def test_supported_ops(self) -> None:
        """Listed ops in a listed dtype, with an available function, are supported.

        Views are supported although ``CAPABILITY`` does not list them.
        """

        ops = {
            "gemv": _gemv(),
            "fp32 add": PNMAddOp(
                operands=(_value((4,), PNM, DType.FP32), _value((4,), PNM, DType.FP32))
            ),
            "sigmoid in PIM": PIMActivationOp(
                operands=(_value((4,), PIM),), function=ActivationKind.SIGMOID
            ),
            "silu in PNM": PNMActivationOp(
                operands=(_value((4,), PNM),), function=ActivationKind.SILU
            ),
            "view": DeviceReshapeOp(operands=(_value((2, 2), PNM),), shape=(4,)),
        }
        for name, op in ops.items():
            with self.subTest(op=name):
                self.assertIsNone(CAPABILITY.why_unsupported(op))

    def test_unsupported_ops_name_the_missing_capability(self) -> None:
        """Each check of why_unsupported() gives its own reason."""

        cases = {
            "op not listed": (
                PNMMulOp(operands=(_value((4,), PNM), _value((4,), PNM))),
                "pnm.mul is not supported",
            ),
            "dtype not listed": (
                PNMAddOp(
                    operands=(
                        _value((4,), PNM, DType.FP16),
                        _value((4,), PNM, DType.FP16),
                    )
                ),
                "pnm.add does not support fp16",
            ),
            "no PIM lookup table": (
                PIMActivationOp(
                    operands=(_value((4,), PIM),), function=ActivationKind.TANH
                ),
                "pim.activation has no lookup table for tanh",
            ),
            "not computed in PNM": (
                PNMActivationOp(
                    operands=(_value((4,), PNM),), function=ActivationKind.GELU_ERF
                ),
                "pnm.activation cannot compute gelu_erf",
            ),
            "GEMV bias": (_gemv(bias=True), "pim.gemv cannot add a bias"),
        }
        for name, (op, reason) in cases.items():
            with self.subTest(case=name):
                self.assertEqual(CAPABILITY.why_unsupported(op), reason)

    def test_gemv_bias_when_enabled(self) -> None:
        """The same biased GEMV is supported once ``pim_gemv_bias`` is set."""

        capability = DeviceCapability(
            ops={PIMGemvOp: frozenset({BF16})},
            pim_activations={},
            pnm_activations=frozenset(),
            pim_gemv_bias=True,
        )
        self.assertIsNone(capability.why_unsupported(_gemv(bias=True)))

    def test_function_id(self) -> None:
        """The AFid of a loaded table; a missing table raises KeyError."""

        self.assertEqual(CAPABILITY.function_id(ActivationKind.SIGMOID), 2)
        with self.assertRaisesRegex(KeyError, "no PIM lookup table for relu"):
            CAPABILITY.function_id(ActivationKind.RELU)

    def test_collections_are_frozen_copies(self) -> None:
        """Changing the caller's dict afterwards does not change the capability."""

        ops: dict[type[Operation], frozenset[DType]] = {PIMGemvOp: frozenset({BF16})}
        capability = DeviceCapability(
            ops=ops,
            pim_activations={},
            pnm_activations=frozenset(),
            pim_gemv_bias=False,
        )
        ops[PNMAddOp] = frozenset({BF16})
        self.assertEqual(dict(capability.ops), {PIMGemvOp: frozenset({BF16})})
        with self.assertRaises(TypeError):
            capability.ops[PNMAddOp] = frozenset({BF16})  # type: ignore[index]

    def test_rejections(self) -> None:
        """Semantic ops, views, empty dtypes and bad AFids are configuration errors."""

        cases: dict[str, tuple[dict[str, Any], str]] = {
            "semantic op": ({"ops": {AddOp: {BF16}}}, "aloi.add is not a Device IR op"),
            "view": (
                {"ops": {DeviceReshapeOp: {BF16}}},
                "device.reshape is a view and always supported",
            ),
            "no dtype": ({"ops": {PIMGemvOp: set()}}, "pim.gemv lists no dtype"),
            "negative AFid": (
                {"pim_activations": {ActivationKind.SIGMOID: -1}},
                "AFid values must be >= 0",
            ),
            "shared AFid": (
                {
                    "pim_activations": {
                        ActivationKind.SIGMOID: 0,
                        ActivationKind.TANH: 0,
                    }
                },
                "two PIM activations share one AFid",
            ),
        }
        defaults: dict[str, Any] = {
            "ops": {},
            "pim_activations": {},
            "pnm_activations": frozenset(),
            "pim_gemv_bias": False,
        }
        for name, (fields, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                DeviceCapability(**{**defaults, **fields})


class DeviceSpecTest(unittest.TestCase):
    """A device spec names its profile and looks up derived storage."""

    def setUp(self) -> None:
        """Build a paper-scale CENT spec with the test capability."""

        self.spec = DeviceSpec(
            name="test",
            target=TargetDescriptor("cent_aim", TraceFormat.AIM_TEXT_TRACE),
            architecture=PAPER,
            capabilities=CAPABILITY,
        )

    def test_memory_space(self) -> None:
        """The Shared Buffer comes from the architecture: one 64 KiB buffer."""

        self.assertEqual(
            self.spec.memory_space(MemorySpace.PNM_SB),
            MemorySpaceSpec(
                space=MemorySpace.PNM_SB,
                instance_size=65536,
                instances=1,
                access_size=32,
            ),
        )

    def test_missing_memory_space(self) -> None:
        """CENT has no PNM register space configured, so the lookup fails."""

        with self.assertRaisesRegex(KeyError, "has no memory space 'pnm.gpr'"):
            self.spec.memory_space(MemorySpace.PNM_GPR)

    def test_empty_names(self) -> None:
        """Neither the profile nor the target may be unnamed."""

        with self.assertRaisesRegex(ValueError, "device spec name"):
            DeviceSpec(
                name="",
                target=self.spec.target,
                architecture=PAPER,
                capabilities=CAPABILITY,
            )
        with self.assertRaisesRegex(ValueError, "target_id"):
            TargetDescriptor("", TraceFormat.AIM_TEXT_TRACE)


class LoadDeviceSpecTest(unittest.TestCase):
    """The example YAML loads into the paper-scale CENT device."""

    def test_example(self) -> None:
        """Every section of examples/cent/cent.yaml, as typed objects.

        The PNM softmax is commented out in the file, so it is not listed;
        views are never listed.
        """

        pim_ops = (PIMGemvOp, PIMEwmulOp, PIMActivationOp, PIMQKOp, PIMSVOp)
        pnm_ops = (
            PNMAddOp,
            PNMSubOp,
            PNMMulOp,
            PNMDivOp,
            PNMExpOp,
            PNMMaxOp,
            PNMSumOp,
            PNMRMSNormOp,
            PNMRopeOp,
        )
        other_ops = (
            PIMCacheWriteOp,
            CXLAllReduceOp,
            DeviceTransferOp,
            DeviceCopyOp,
            DeviceBroadcastOp,
        )
        self.assertEqual(
            load_device_spec(EXAMPLE),
            DeviceSpec(
                name="cent_paper",
                target=TargetDescriptor("cent_aim", TraceFormat.AIM_TEXT_TRACE),
                architecture=PAPER,
                capabilities=DeviceCapability(
                    ops={
                        op: frozenset({BF16}) for op in (*pim_ops, *pnm_ops, *other_ops)
                    },
                    pim_activations={ActivationKind.SIGMOID: 0},
                    pnm_activations=frozenset(),
                    pim_gemv_bias=False,
                ),
            ),
        )

    def test_example_gives_the_backend_paper_spec(self) -> None:
        """Bytes to bf16 counts, with sigmoid's configured AFid.

        This is the same CentHardwareSpec the compiler tests construct by
        hand, so the YAML can replace it.
        """

        spec = load_device_spec(EXAMPLE)
        assert isinstance(spec.architecture, CentArchitecture)
        self.assertEqual(
            spec.architecture.to_cent_hardware(
                BF16, spec.capabilities.function_id(ActivationKind.SIGMOID)
            ),
            CentHardwareSpec(
                num_channels=32,
                num_banks=16,
                dram_rows=16384,
                dram_columns=1024,
                burst_length=16,
                accumulator_slots_per_bank=32,
                sigmoid_activation_function_id=0,
                shared_buffer_slots=2048,
                global_buffer_columns=1024,
            ),
        )

    def test_document_must_be_a_mapping(self) -> None:
        """A YAML list at the top level is rejected with the file name."""

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "list.yaml"
            path.write_text("- 1\n")
            with self.assertRaisesRegex(ValueError, "list.yaml must be a mapping"):
                load_device_spec(path)


class SnapshotTest(unittest.TestCase):
    """A snapshot is a plain, sorted document that reads back to the same spec."""

    def test_round_trip_through_json(self) -> None:
        """Reading a JSON-encoded snapshot gives an equal spec."""

        spec = load_device_spec(EXAMPLE)
        document = json.loads(json.dumps(snapshot(spec)))
        self.assertEqual(parse_device_spec(document), spec)

    def test_layout(self) -> None:
        """Ops grouped by prefix and sorted; enums written as their values."""

        spec = DeviceSpec(
            name="small",
            target=TargetDescriptor("cent_aim", TraceFormat.AIM_TEXT_TRACE),
            architecture=PAPER,
            capabilities=CAPABILITY,
        )
        self.assertEqual(
            snapshot(spec),
            {
                "schema_version": "0.1",
                "name": "small",
                "target": {"target_id": "cent_aim", "trace_format": "aim_text_trace"},
                "architecture": {
                    "type": "cent",
                    "channels": 32,
                    "banks_per_channel": 16,
                    "banks_per_group": 4,
                    "rows_per_bank": 16384,
                    "row_size": 2048,
                    "burst_size": 32,
                    "global_buffer_size": 2048,
                    "shared_buffer_size": 65536,
                    "accumulators_per_bank": 32,
                },
                "capabilities": {
                    "ops": {
                        "pim": {"activation": ["bf16"], "gemv": ["bf16"]},
                        "pnm": {"activation": ["bf16"], "add": ["bf16", "fp32"]},
                    },
                    "pim_activations": {"sigmoid": 2},
                    "pnm_activations": ["silu"],
                    "pim_gemv_bias": False,
                },
            },
        )


class ParseDeviceSpecErrorTest(unittest.TestCase):
    """Every mistake is reported with the dotted path of the bad entry."""

    def test_rejections(self) -> None:
        """Start from the valid example and break one entry per case."""

        def without(path: str) -> dict[str, Any]:
            """Return the example with the entry at the dotted ``path`` removed."""

            document = _example_document()
            *parents, key = path.split(".")
            section = document
            for parent in parents:
                section = section[parent]
            del section[key]
            return document

        def with_value(path: str, value: object) -> dict[str, Any]:
            """Return the example with the entry at the dotted ``path`` set."""

            document = _example_document()
            *parents, key = path.split(".")
            section = document
            for parent in parents:
                section = section[parent]
            section[key] = value
            return document

        cases = {
            "unknown section": (with_value("extra", {}), r"sections: \['extra'\]"),
            "old schema": (with_value("schema_version", "0.0"), "schema_version"),
            "no name": (without("name"), "name must be a non-empty string"),
            "no target_id": (without("target.target_id"), "target.target_id"),
            "unknown trace format": (
                with_value("target.trace_format", "vcd"),
                "target.trace_format must be one of",
            ),
            "unknown architecture": (
                with_value("architecture.type", "hbm"),
                "architecture.type must be one of",
            ),
            "misspelled field": (
                with_value("architecture.chanels", 32),
                r"section 'architecture': \['chanels'\]",
            ),
            "missing field": (
                without("architecture.row_size"),
                r"architecture is missing \['row_size'\]",
            ),
            "non-int field": (
                with_value("architecture.channels", "32"),
                "architecture.channels must be a positive int",
            ),
            # Valid ints that break a CentArchitecture rule still fail.
            "invalid geometry": (
                with_value("architecture.channels", 33),
                "channels must be between 1 and 32",
            ),
            "unknown namespace": (
                with_value("capabilities.ops.gpu", {}),
                r"section 'capabilities.ops': \['gpu'\]",
            ),
            "unknown op": (
                with_value("capabilities.ops.pim.gemm", ["bf16"]),
                "capabilities.ops.pim.gemm: .*pim.gemm",
            ),
            "listed view": (
                with_value("capabilities.ops.device.reshape", ["bf16"]),
                "views are always supported",
            ),
            "unknown dtype": (
                with_value("capabilities.ops.pim.gemv", ["bfloat"]),
                "capabilities.ops.pim.gemv: unknown dtype 'bfloat'",
            ),
            "dtypes not a list": (
                with_value("capabilities.ops.pim.gemv", "bf16"),
                "capabilities.ops.pim.gemv must be a list of names",
            ),
            "unknown PIM activation": (
                with_value("capabilities.pim_activations", {"swish": 1}),
                "capabilities.pim_activations.swish: unknown activation",
            ),
            "negative AFid": (
                with_value("capabilities.pim_activations", {"sigmoid": -1}),
                "capabilities.pim_activations.sigmoid must be an int >= 0",
            ),
            "unknown PNM activation": (
                with_value("capabilities.pnm_activations", ["swish"]),
                "capabilities.pnm_activations: unknown activation 'swish'",
            ),
            "bias not a bool": (
                with_value("capabilities.pim_gemv_bias", "no"),
                "pim_gemv_bias must be true or false",
            ),
        }
        for name, (document, message) in cases.items():
            with self.subTest(case=name), self.assertRaisesRegex(ValueError, message):
                parse_device_spec(document)


if __name__ == "__main__":
    unittest.main()
