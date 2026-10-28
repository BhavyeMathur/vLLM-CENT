"""Describe a target device: its geometry, storage and supported operations.

A ``DeviceSpec`` is what lowering and mapping know about the hardware:

- ``architecture``: the target's geometry, from which the sizes of its kinds
  of storage (``memory_spaces()``) are derived, so they are never written
  twice;
- ``capabilities``: which Device IR ops the target runs, for which element
  types, and the few op-specific parameters (activation functions, a GEMV
  bias);
- ``target``: which backend and output format the target uses.

Device specs are written in YAML and read by ``device.loader``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from types import MappingProxyType

from ..activations import ActivationKind
from ..ir.base import Operation
from ..ir.device import PIMActivationOp, PIMGemvOp, PNMActivationOp, ViewOp
from ..ir.types import DType, ExecutionDomain

__all__ = [
    "Architecture",
    "DeviceCapability",
    "DeviceSpec",
    "MemorySpace",
    "MemorySpaceSpec",
    "TargetDescriptor",
    "TraceFormat",
]

# Name prefixes of Device IR ops: an op whose NAME starts with one of these,
# followed by a dot, is a device op (``pim.gemv``); others are semantic.
DEVICE_OP_NAMESPACES = frozenset({"pim", "pnm", "cxl", "device"})


class MemorySpace(StrEnum):
    """Where a tensor is stored.

    Members:
        PIM_DRAM: Tensor being stored in DRAM rows, Read and Write using normal
            RD/WR commands
        PIM_GB: Tensor temporarily residing in Global Buffer (GB), Write only
            using WR_GB commands. No way to read it other than being consumed
            by the PU
        PNM_GPR: Tensor being stored in PNM General Purpose Register
    """

    PIM_DRAM = "pim.dram"
    PIM_GB = "pim.gb"
    PIM_ACC = "pim.acc"
    PNM_GPR = "pnm.gpr"
    PNM_SB = "pnm.sb"

    @property
    def domain(self) -> ExecutionDomain:
        """The execution domain this storage belongs to: the part before the dot."""
        return ExecutionDomain(self.value.partition(".")[0])


@dataclass(frozen=True, slots=True)
class MemorySpaceSpec:
    """The size of one kind of storage on a device.

    A device usually has many copies of each kind of storage, e.g. one DRAM
    bank per processing unit, so the size is given per copy. Sizes are in
    bytes, not values.

    Attributes:
        space: Which kind of storage this is.
        instance_size: Size of one copy, in bytes, e.g. one bank.
        instances: Number of copies on the device.
        access_size: Size of one access, in bytes, e.g. a 32-byte burst.
    """

    space: MemorySpace
    instance_size: int
    instances: int
    access_size: int

    @property
    def domain(self) -> ExecutionDomain:
        """The execution domain this storage belongs to."""
        return self.space.domain

    def capacity_values(self, dtype: DType) -> int:
        """How many elements of ``dtype`` one copy of the storage holds."""
        return self.instance_size * 8 // dtype.bits


class Architecture(ABC):
    """A target family's geometry; it derives the storage the target offers.

    Subclasses (e.g. ``CentArchitecture``) hold the family's counts and byte
    sizes and compute ``memory_spaces()`` from them, so capacities are never
    configured separately from the geometry they follow from.
    """

    __slots__ = ()

    @abstractmethod
    def memory_spaces(self) -> tuple[MemorySpaceSpec, ...]:
        """Return the kinds of storage this geometry provides, with their sizes."""


class TraceFormat(StrEnum):
    """How a backend serializes the target instructions it emits.

    Members:
        AIM_TEXT_TRACE: The textual AiM command stream read by aim_simulator.
    """

    AIM_TEXT_TRACE = "aim_text_trace"


@dataclass(frozen=True, slots=True)
class TargetDescriptor:
    """Identify the backend that compiles for a device and what it emits.

    Attributes:
        target_id: Name of the target backend, e.g. ``"cent_aim"``.
        trace_format: Format of the instruction payloads the backend writes.
    """

    target_id: str
    trace_format: TraceFormat

    def __post_init__(self) -> None:
        """Require a target name.

        Raises:
            ValueError: If ``target_id`` is empty.
        """

        if not self.target_id:
            raise ValueError("target_id must not be empty")


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceCapability:
    """Which Device IR ops a target runs, and with which parameters.

    Most ops only need their element types: ``ops`` maps each supported op
    class to the dtypes it accepts, and an op absent from ``ops`` is not
    supported. View ops (``ViewOp``) emit no instruction and are always
    supported. A few ops have parameters of their own, kept as fields here:
    the activation functions available in PIM (with the ``AFid`` the PIM
    lookup table uses for each) and in PNM, and whether a PIM GEMV can add a
    bias.

    Attributes:
        ops: Supported element types of each supported device op class.
        pim_activations: Activation functions loaded in the PIM lookup
            tables, mapped to their ``AFid``. The paper does not publish
            these numbers; they are target configuration.
        pnm_activations: Activation functions the PNM units can compute.
        pim_gemv_bias: Whether ``pim.gemv`` can add a bias (e.g. by
            initializing the accumulators with ``WR_BIAS``).
    """

    # TODO(implementation kind): Whether a supported op runs on real hardware
    # or only as a software reference service (with no physical encoding)
    # will be declared by each emitter of the Step 4 target backend, not
    # here. That backend should also check, when a spec is loaded, that every
    # op listed in ``ops`` has an emitter.

    ops: Mapping[type[Operation], frozenset[DType]]
    pim_activations: Mapping[ActivationKind, int]
    pnm_activations: frozenset[ActivationKind]
    pim_gemv_bias: bool

    def __post_init__(self) -> None:
        """Freeze the collections and check the configuration.

        Raises:
            ValueError: If ``ops`` names a semantic or view op, or an op with no
                dtype, or two activation functions share one ``AFid``, or an
                ``AFid`` is negative.
        """

        # The dataclass is frozen, so object.__setattr__ is the way to replace
        # the caller's (possibly mutable) collections with read-only copies.
        object.__setattr__(
            self,
            "ops",
            MappingProxyType(
                {op: frozenset(dtypes) for op, dtypes in self.ops.items()}
            ),
        )
        object.__setattr__(
            self, "pim_activations", MappingProxyType(dict(self.pim_activations))
        )
        object.__setattr__(self, "pnm_activations", frozenset(self.pnm_activations))

        for op, dtypes in self.ops.items():
            if op.NAME.partition(".")[0] not in DEVICE_OP_NAMESPACES:
                raise ValueError(f"{op.NAME} is not a Device IR op")
            if issubclass(op, ViewOp):
                raise ValueError(f"{op.NAME} is a view and always supported")
            if not dtypes:
                raise ValueError(f"{op.NAME} lists no dtype")
        function_ids = list(self.pim_activations.values())
        if any(function_id < 0 for function_id in function_ids):
            raise ValueError(f"AFid values must be >= 0, got {function_ids}")
        if len(set(function_ids)) != len(function_ids):
            raise ValueError(f"two PIM activations share one AFid: {function_ids}")

    def why_unsupported(self, op: Operation) -> str | None:
        """Tell why the target cannot run ``op``, or None if it can.

        The op's element type is the dtype of its first result (its first
        operand if it has none): device ops compute in the type they
        produce. Integer index operands, such as RoPE positions, do not
        count.

        Args:
            op: A Device IR op instance, as a lowering recipe built it.

        Returns:
            None if supported; otherwise a sentence naming the op and the
            missing capability.
        """

        if isinstance(op, ViewOp):
            return None
        dtypes = self.ops.get(type(op))
        if dtypes is None:
            return f"{op.NAME} is not supported"
        dtype = (op.results[0] if op.results else op.operands[0]).type.dtype
        if dtype not in dtypes:
            return f"{op.NAME} does not support {dtype}"
        if isinstance(op, PIMActivationOp) and op.function not in self.pim_activations:
            return f"pim.activation has no lookup table for {op.function.value}"
        if isinstance(op, PNMActivationOp) and op.function not in self.pnm_activations:
            return f"pnm.activation cannot compute {op.function.value}"
        if isinstance(op, PIMGemvOp) and op.bias is not None and not self.pim_gemv_bias:
            return "pim.gemv cannot add a bias"
        return None

    def function_id(self, function: ActivationKind) -> int:
        """Return the ``AFid`` of an activation in the PIM lookup tables.

        Args:
            function: An activation function, e.g. sigmoid for
                ``CentArchitecture.to_cent_hardware``.

        Returns:
            Its configured ``AFid``.

        Raises:
            KeyError: If the PIM lookup tables do not hold ``function``.
        """

        if function not in self.pim_activations:
            raise KeyError(f"no PIM lookup table for {function.value}")
        return self.pim_activations[function]


@dataclass(frozen=True, slots=True)
class DeviceSpec:
    """Everything lowering and mapping know about one target device.

    Attributes:
        name: Name of this device profile, e.g. ``"cent_paper"``.
        target: The backend that compiles for it and its output format.
        architecture: Geometry, and the storage derived from it.
        capabilities: Supported Device IR ops, element types and parameters.
    """

    name: str
    target: TargetDescriptor
    architecture: Architecture
    capabilities: DeviceCapability

    def __post_init__(self) -> None:
        """Require a profile name.

        Raises:
            ValueError: If ``name`` is empty.
        """

        if not self.name:
            raise ValueError("device spec name must not be empty")

    def memory_space(self, space: MemorySpace) -> MemorySpaceSpec:
        """Look up the size of one kind of storage.

        Args:
            space: The kind of storage.

        Returns:
            Its sizes, derived from the architecture.

        Raises:
            KeyError: If this device has no such storage.
        """

        for spec in self.architecture.memory_spaces():
            if spec.space is space:
                return spec
        raise KeyError(f"device {self.name!r} has no memory space {space.value!r}")
