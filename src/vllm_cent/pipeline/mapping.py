"""Choose where tensors will live and check that storage reuse is safe.

A mapping names a storage object for every tensor and gives an execution order
for the operations. Objects are still named buffers, not numeric addresses.
Two tensors may share an object only when overwriting the old contents cannot
affect a later calculation or a requested output.

The baseline mapper makes a simple deterministic choice. A caller or external
optimizer can supply a different choice, but every plan uses the same verifier.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from ..cent import CentHardwareSpec
from .ir import (
    DeviceGraph,
    DeviceOperationKind,
    DeviceRankGraph,
    NameValue,
    verify_device,
)

__all__ = [
    "AccumulatorPlacement",
    "CentTargetProfile",
    "ExecutionContract",
    "MappingPlan",
    "MappingProvider",
    "MappingRequest",
    "ValuePlacement",
    "baseline_mapping",
    "verify_mapping",
]


class ExecutionContract(Enum):
    """Choose whether unresolved physical operations may use declared reference rules.

    Members:
        STANDARD: Use only the existing instruction subset. New accumulator and
            scalar reference operations are not available.
        REFERENCE: Permit explicit accumulator selection, scalar packing, and
            normalization and nonlinear math. These local rules are not a physical instruction ABI.
    """

    STANDARD = "standard"
    REFERENCE = "reference"


@dataclass(frozen=True, slots=True, kw_only=True)
class AccumulatorPlacement:
    """Choose which accumulator a matrix operation owns while it executes.

    Attributes:
        operation: Name of a linear or attention operation in the device graph.
        register: Accumulator index used in each active bank. Operations execute
            sequentially, so different operations may reuse this index.
    """

    operation: str
    register: int


@dataclass(frozen=True, slots=True, kw_only=True)
class CentTargetProfile:
    """Supply the hardware limits used when placing tensors and emitting commands.

    The earlier Device IR only says which operations must be supported. This
    profile adds memory capacity, burst width, and command limits. The simulator
    uses Python-float reference arithmetic; this profile does not specify the
    rounding rules needed to claim exact physical-hardware results.

    Attributes:
        name: Human-readable target name for diagnostics.
        hardware: Channel, bank, row, register, and buffer sizes for this device.
        max_accumulate_slots: Largest number of Shared Buffer slots to process
            in one ACC instruction. A longer addition is split into chunks.
        execution_contract: ``STANDARD`` keeps the existing instruction subset;
            ``REFERENCE`` explicitly opts into local accumulator, packing, and
            scalar rules whose physical encoding is not established.
    """

    name: str
    hardware: CentHardwareSpec
    max_accumulate_slots: int
    execution_contract: ExecutionContract = ExecutionContract.STANDARD

    def __post_init__(self) -> None:
        """Require a target name and an ACC limit that allows at least one slot.

        Raises:
            ValueError: If ``name`` is empty or ``max_accumulate_slots`` is less than one.
        """
        if not self.name or self.max_accumulate_slots < 1:
            raise ValueError("target name and positive max_accumulate_slots required")


@dataclass(frozen=True, slots=True, kw_only=True)
class ValuePlacement:
    """Assign one named tensor to a named Shared Buffer object.

    Assigning multiple tensors to the same object reuses its memory.
    The ``verify_mapping()`` function checks that this reuse preserves needed
    values. Numeric slot addresses are assigned in a later pass.

    Attributes:
        value: Tensor name from the Device IR.
        storage: Name of the buffer object that will hold this tensor.

    Examples:
        With ``value="z"`` and ``storage="x_buffer"``, tensor ``z`` occupies the
        object named ``x_buffer``. Assigning ``x`` to that object too is safe only
        if its original value is no longer needed when ``z`` overwrites it.
    """

    value: str
    storage: str


@dataclass(frozen=True, slots=True, kw_only=True)
class MappingPlan:
    """Record the execution order and which tensors share storage.

    All tensors use the Shared Buffer at operation boundaries. Arithmetic
    replaces the left operand's storage when safe; otherwise target lowering
    first copies it to the result buffer. Copy and reshape may reuse their
    sole input's storage after its last use. Linear, RMSNorm, and attention require their
    inputs to remain separate from their result throughout the operation.
    Multiplication stages operands in
    exclusive DRAM scratch during code generation.
    The plan does not yet choose scratch addresses, combined operations,
    or communication between devices.

    Attributes:
        placements: One assignment for every graph input and operation result.
            Repeated storage names mean intentional reuse of the same memory.
        schedule: Every operation name exactly once, with each input's producer
            before the operation that reads it.
        accumulators: One explicit register assignment for each linear or attention operation.
            Other operations do not request accumulator storage.
    """

    placements: tuple[ValuePlacement, ...]
    schedule: tuple[str, ...]
    accumulators: tuple[AccumulatorPlacement, ...] = ()


@dataclass(frozen=True, slots=True, kw_only=True)
class MappingRequest:
    """Give a mapper the work to place, hardware limits, and caller requirements.

    Attributes:
        graph: Device operations to place, including the earlier graph and the
            caller's rule for splitting it across logical devices.
        target: Physical device sizes and the maximum ACC command length.
        constraints: Optional tensor-to-storage assignments that must be kept.
            The mapper chooses assignments for the remaining tensors.
    """

    graph: DeviceGraph
    target: CentTargetProfile
    constraints: tuple[ValuePlacement, ...] = ()


class MappingProvider(Protocol):
    """Define the callable interface used by baseline and external mappers.

    A provider accepts a ``MappingRequest`` and returns a proposed ``MappingPlan``. The
    compiler verifies that plan before emitting instructions. This interface
    does not prescribe how an optimizer scores or searches for plans.
    """

    def __call__(self, request: MappingRequest) -> MappingPlan:
        """Choose operation order and tensor storage for the supplied work.

        Args:
            request: Device graph, hardware limits, and assignments the caller requires.

        Returns:
            Proposed schedule and storage assignments, still subject to verification.
        """
        ...


def _rank(request: MappingRequest) -> DeviceRankGraph:
    """Check the graph and select the only device this backend can execute.

    Args:
        request: Device graph to validate and physical target to compile for.

    Returns:
        Work assigned to logical device zero, including a valid identity graph.

    Raises:
        ValueError: If the graph is invalid or does not contain
            exactly one rank on device zero.
        NotImplementedError: If verifying the graph reaches an unsupported operation
            or capability.
    """
    verify_device(request.graph)
    if len(request.graph.ranks) != 1 or request.graph.ranks[0].device_id != 0:
        # TODO(runtime): Multi-rank IR is verified, but CXL routing and collective
        # execution require a runtime ABI before code generation can support it.
        raise ValueError("CENT code generation requires one rank on device zero")
    rank = request.graph.ranks[0]
    return rank


def _values(rank: DeviceRankGraph) -> tuple[NameValue, ...]:
    """List every tensor that needs a storage assignment.

    Args:
        rank: Device-local input tensors and operations.

    Returns:
        Inputs first, followed by each operation's result in operation order.
    """
    return rank.inputs + tuple(op.output for op in rank.operations)


def _slots(value: NameValue, target: CentTargetProfile) -> int:
    """Round a tensor's element count up to complete Shared Buffer slots.

    The runtime must fill unused lanes with zeros before an instruction reads them.

    Args:
        value: Tensor whose dimensions determine the number of elements.
        target: Hardware description supplying the number of lanes in one slot.

    Returns:
        Number of consecutive slots required, including any partly used last slot.

    Examples:
        For five elements and two lanes per slot, reserve three slots. The runtime
        writes zero into the unused sixth lane.
    """
    count = 1
    for dimension in value.type.shape:
        count *= dimension
    width = target.hardware.burst_length
    return (count + width - 1) // width


def _last_uses(rank: DeviceRankGraph, schedule: tuple[str, ...]) -> dict[str, int]:
    """Find the final operation that needs each tensor's current contents.

    A buffer can be overwritten after this last read. Requested outputs must
    survive all operations because the runtime extracts them only at the end.

    Args:
        rank: Device graph containing tensor uses and requested outputs.
        schedule: Valid operation-name order; indices in this tuple measure time.

    Returns:
        Tensor name to last-use index. Unused inputs get -1. An unused result gets
        its own producing operation's index. Requested outputs get ``len(schedule)``,
        one step beyond the last operation, to keep their storage live through exit.
    """
    # -1 means an input has no readers and may be reused at the first operation.
    last = {value.name: -1 for value in _values(rank)}
    operations = {op.name: op for op in rank.operations}
    for index, name in enumerate(schedule):
        op = operations[name]
        last[op.output.name] = index
        for operand in op.inputs:
            last[operand] = index
    # Extraction happens after the final command, so returned values cannot
    # be reclaimed merely because no later operation reads them.
    for name in rank.outputs:
        last[name] = len(schedule)
    return last


def verify_mapping(request: MappingRequest, plan: MappingPlan) -> None:
    """Check that a proposed schedule and storage reuse preserve every needed value.

    Both a manually supplied plan and a mapper's result pass through this check.
    Nothing in the plan or request is modified.

    Args:
        request: Device graph, hardware limits, and required tensor placements.
        plan: Complete operation schedule and tensor-to-storage assignments to check.

    Raises:
        ValueError: If operations or placements are missing or repeated, an input
            is read before its producer, required placements are ignored, buffers
            exceed capacity, reused sizes differ, or reuse overwrites needed data.
        NotImplementedError: If graph verification reaches an unsupported operation
            or capability.

    Examples:
        For ``z = x + y``, the plan may put ``z`` in the buffer for ``x`` if nobody
        needs the original ``x`` afterward. If ``x`` is also a requested output,
        that placement is rejected.
    """
    rank = _rank(request)
    values = _values(rank)
    reference_kinds = (
        DeviceOperationKind.LINEAR,
        DeviceOperationKind.RMS_NORM,
        DeviceOperationKind.ACTIVATION,
        DeviceOperationKind.SOFTMAX,
        DeviceOperationKind.ATTENTION,
    )
    if (
        any(op.kind in reference_kinds for op in rank.operations)
        and request.target.execution_contract is not ExecutionContract.REFERENCE
    ):
        raise ValueError(
            "This operation requires the explicit REFERENCE execution contract"
        )
    required_accumulators = {
        op.name
        for op in rank.operations
        if op.kind in (DeviceOperationKind.LINEAR, DeviceOperationKind.ATTENTION)
    }
    accumulator_map = {item.operation: item.register for item in plan.accumulators}
    if (
        len(accumulator_map) != len(plan.accumulators)
        or set(accumulator_map) != required_accumulators
    ):
        raise ValueError(
            "accumulator placements must cover every linear or attention operation exactly once"
        )
    if any(
        not 0 <= register < request.target.hardware.accumulator_slots_per_bank
        for register in accumulator_map.values()
    ):
        raise ValueError("accumulator register is outside target capacity")
    operations = {op.name: op for op in rank.operations}
    if len(plan.schedule) != len(operations) or set(plan.schedule) != set(operations):
        raise ValueError("schedule must name every operation exactly once")
    placements = {p.value: p.storage for p in plan.placements}
    if (
        len(placements) != len(plan.placements)
        or set(placements) != {value.name for value in values}
        or any(not storage for storage in placements.values())
    ):
        raise ValueError("placements must cover each value exactly once")
    constrained: set[str] = set()
    for constraint in request.constraints:
        if constraint.value in constrained:
            raise ValueError("duplicate placement constraint")
        constrained.add(constraint.value)
        if placements.get(constraint.value) != constraint.storage:
            raise ValueError("mapping violates placement constraint")
    # Check order separately from storage: every read must have a preceding
    # producer even if the chosen buffer would otherwise be safe to reuse.
    available = {value.name for value in rank.inputs}
    for name in plan.schedule:
        op = operations[name]
        if not set(op.inputs) <= available:
            raise ValueError("schedule violates data dependencies")
        available.add(op.output.name)
    # Count each storage object once. Several tensor names may reuse that
    # object over time, but they must agree on its size.
    sizes: dict[str, int] = {}
    for value in values:
        storage = placements[value.name]
        size = _slots(value, request.target)
        if storage in sizes and sizes[storage] != size:
            raise ValueError("aliased storage must have identical slot counts")
        sizes[storage] = size
    if sum(sizes.values()) > request.target.hardware.shared_buffer_slots:
        raise ValueError("mapping exceeds Shared Buffer capacity")

    # Inputs are loaded together before instruction zero, including inputs that
    # will first be consumed much later. They may never alias at entry.
    live: dict[str, str] = {}
    for value in rank.inputs:
        storage = placements[value.name]
        if storage in live:
            raise ValueError("input storage aliases another live input")
        live[storage] = value.name
    last = _last_uses(rank, plan.schedule)
    for index, name in enumerate(plan.schedule):
        op = operations[name]
        for operand in op.inputs:
            if live.get(placements[operand]) != operand:
                raise ValueError("operand storage no longer contains its live value")
        output_storage = placements[op.output.name]
        occupant = live.get(output_storage)
        # Arithmetic updates its destination in place. If the
        # result shares the left input's buffer, its final read can happen in
        # this instruction. If it shares only the right input's buffer, copying
        # the left input into the destination would erase the right input before
        # arithmetic reads it. Reject that placement even at its last use.
        # Matrix products, normalization, and attention revisit their inputs, so
        # even a last-use input cannot be overwritten by those schedules.
        # Unary copy/reshape preserves flat values. Nonlinear reference commands
        # snapshot the logical input before writing, so their last-use handoff
        # is also safe without changing shape or alias checks.
        if occupant is not None:
            legal_handoff = (
                occupant == op.inputs[0]
                and last[occupant] == index
                and op.kind
                not in (
                    DeviceOperationKind.LINEAR,
                    DeviceOperationKind.RMS_NORM,
                    DeviceOperationKind.ATTENTION,
                )
            )
            expired = last[occupant] < index
            if not legal_handoff and not expired:
                raise ValueError("result storage aliases a live value")
        live[output_storage] = op.output.name


def _fresh_storage(preferred: str, reserved: set[str]) -> str:
    """Choose a buffer name that does not collide with an existing name.

    Args:
        preferred: Tensor name to use as the buffer name if it is available.
        reserved: Buffer names already used or reserved by caller requirements.

    Returns:
        The preferred name, or its first free numbered form.

    Examples:
        If ``x`` and ``x.0`` are taken, a preferred name of ``x`` becomes ``x.1``.
    """
    if preferred not in reserved:
        return preferred
    suffix = 0
    while f"{preferred}.{suffix}" in reserved:
        suffix += 1
    return f"{preferred}.{suffix}"


def baseline_mapping(request: MappingRequest) -> MappingPlan:
    """Run operations in graph order and reuse buffers once their data is no longer needed.

    For each result, first honor a caller-supplied placement. Otherwise, reuse
    the left operand's buffer if this is its last read. If that is not possible,
    reuse another expired buffer of the same size, or create a new buffer name.

    Last-use indices tell us when to put buffers on size-specific free lists,
    so the mapper does not search the whole graph for every result. This is a
    simple baseline, not a cost model or a complete solver for constraints.

    Args:
        request: Device work, hardware limits, and any fixed tensor placements.

    Returns:
        A complete plan checked by ``verify_mapping``, with no numeric addresses yet.

    Raises:
        ValueError: If the resulting plan cannot satisfy the requirements or fit
            in memory. A different manual schedule may still admit a valid plan.
        NotImplementedError: If graph verification reaches an unsupported operation
            or capability.
    """
    rank = _rank(request)
    schedule = tuple(op.name for op in rank.operations)
    last = _last_uses(rank, schedule)
    # Reserve manual names before creating defaults. A caller can request a
    # buffer called y for x without accidentally putting input y there too.
    forced = {p.value: p.storage for p in request.constraints}
    reserved = set(forced.values())
    placements: dict[str, str] = {}
    owners: dict[str, str] = {}
    sizes: dict[str, int] = {}
    # owners tracks current contents; expirations schedules when those contents
    # may be discarded. free groups available objects by their slot count.
    expirations: dict[int, list[tuple[str, str]]] = {}
    free: dict[int, list[str]] = {}
    for value in rank.inputs:
        storage = forced.get(value.name)
        if storage is None:
            storage = _fresh_storage(value.name, reserved)
        reserved.add(storage)
        placements[value.name] = storage
        owners[storage] = value.name
        sizes[storage] = _slots(value, request.target)
        expirations.setdefault(last[value.name] + 1, []).append((storage, value.name))
    for index, op in enumerate(rank.operations):
        # If x was last read at step 2, its buffer becomes available at step 3.
        # But step 2 may already have replaced x with z in that buffer. Check
        # the current owner so x's old expiration does not discard the new z.
        for storage, owner in expirations.get(index, ()):
            if owners[storage] == owner:
                free.setdefault(sizes[storage], []).append(storage)
        size = _slots(op.output, request.target)
        left = op.inputs[0]
        if op.output.name in forced:
            storage = forced[op.output.name]
        elif (
            last[left] == index
            and sizes[placements[left]] == size
            and op.kind
            not in (
                DeviceOperationKind.LINEAR,
                DeviceOperationKind.RMS_NORM,
                DeviceOperationKind.ATTENTION,
            )
        ):
            storage = placements[left]
        else:
            candidates = free.setdefault(size, [])
            # A fixed placement may have reused a buffer that is still listed
            # as free. Ignore that old entry while its new contents are needed.
            while candidates and last[owners[candidates[-1]]] >= index:
                candidates.pop()
            storage = (
                candidates.pop()
                if candidates
                else _fresh_storage(op.output.name, reserved)
            )
        reserved.add(storage)
        placements[op.output.name] = storage
        owners[storage] = op.output.name
        sizes[storage] = size
        expirations.setdefault(last[op.output.name] + 1, []).append(
            (storage, op.output.name)
        )
    plan = MappingPlan(
        placements=tuple(
            ValuePlacement(value=v.name, storage=placements[v.name])
            for v in _values(rank)
        ),
        schedule=schedule,
        accumulators=tuple(
            AccumulatorPlacement(
                operation=op.name,
                register=index % request.target.hardware.accumulator_slots_per_bank,
            )
            for index, op in enumerate(rank.operations)
            if op.kind in (DeviceOperationKind.LINEAR, DeviceOperationKind.ATTENTION)
        ),
    )
    verify_mapping(request, plan)
    return plan
