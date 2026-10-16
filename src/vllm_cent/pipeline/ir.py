"""Describe tensor calculations and split them into work for each device.

The compiler keeps three versions of the graph here: the original calculation,
one copy for each device's portion of the tensors, and the operations each
device must support. These are intermediate representations (IRs): structured
Python objects that later passes turn into instructions.

No stage in this module chooses physical addresses. Operations create new
named results and do not modify their inputs. Adapters must reject cache
updates, shared-storage mutation, communication, and control flow because these
objects cannot describe them yet. Addition, multiplication, copy, and contiguous
reshape, linear transforms, normalization, activations, softmax, and attention reach
supported Device IR. Other operation types can supply their
own shape checks without changing the graph container or its verifier.
"""

from dataclasses import dataclass, replace
from enum import Enum
from .operations.activation import ActivationKind, ActivationOperation, SoftmaxOperation
from .operations.attention import AttentionOperation
from .operations.transformer import RmsNormOperation
from .operations.base import Operation, ScalarType, TensorType, Value
from .operations.tensor import (
    AddOperation,
    CopyOperation,
    MultiplyOperation,
    LinearOperation,
    ReshapeOperation,
)

__all__ = [
    "CapabilityProfile",
    "DeviceActivationOperation",
    "DeviceAttentionOperation",
    "DeviceGraph",
    "DeviceOperation",
    "DeviceOperationKind",
    "DeviceRankGraph",
    "DeviceRmsNormOperation",
    "MemorySpace",
    "Operation",
    "ParallelPlan",
    "RankGraph",
    "ScalarType",
    "SemanticGraph",
    "ShardedGraph",
    "ShardedValue",
    "TensorType",
    "Value",
    "legalize",
    "shard",
    "verify_device",
    "verify_semantic",
    "verify_sharded",
]


@dataclass(frozen=True, slots=True)
class SemanticGraph:
    """Describe a tensor calculation without choosing how hardware runs it.

    The graph records named tensors, the calculations that produce new tensors,
    and the results the caller wants back. It describes the calculation itself;
    later compiler passes decide how to split it across devices and where to store
    the data.

    Attributes:
        inputs: Named tensors the caller must supply, in argument order.
        operations: Calculations in order. Each operation can read only the graph
            inputs or results produced earlier in this tuple.
        outputs: Names of tensors returned to the caller, in result order.

    Examples:
        For ``z = x + y`` followed by ``out = z + y``, ``inputs`` contains
        ``x`` and ``y``, ``operations`` contains the two additions in that order,
        and ``outputs`` can contain ``out``. Listing ``x`` in ``outputs`` too
        means its original value must remain available.
    """

    inputs: tuple[Value, ...]
    operations: tuple[Operation, ...]
    outputs: tuple[str, ...]


def verify_semantic(graph: SemanticGraph) -> None:
    """Check that every calculation refers to existing, correctly shaped tensors.

    Walk the operations in order, remembering which tensors have been defined.
    An operation can read an earlier result, but not a future result or an
    unknown name. Each result must have a new name and the expected shape.
    This verifies the graph's description; it does not run any tensor arithmetic.

    Args:
        graph: Input tensors, ordered operations, and requested outputs to check.
            The graph is not modified.

    Raises:
        ValueError: If names repeat or are missing, inputs are used before they
            exist, or a concrete operation rejects its input/output types.
    """
    # Only names already in this table may be read. Adding results as we go
    # makes a forward reference fail at the operation that tries to use it.
    values: dict[str, TensorType] = {}
    operation_names: set[str] = set()
    for value in graph.inputs:
        _define(values, value)
    for operation in graph.operations:
        if not operation.name or operation.name in operation_names:
            raise ValueError("Operation names must be nonempty and unique")
        operation_names.add(operation.name)
        if any(name not in values for name in operation.inputs):
            raise ValueError(f"Undefined operand in {operation.name}")
        # The graph owns naming and dependencies. The concrete operation owns
        # its mathematical rules, so a new operation needs no branch here.
        operation.verify(tuple(values[name] for name in operation.inputs))
        _define(values, operation.output)
    if (
        not graph.outputs
        or len(set(graph.outputs)) != len(graph.outputs)
        or any(name not in values for name in graph.outputs)
    ):
        raise ValueError("Graph outputs must name unique defined values")


def _define(values: dict[str, TensorType], value: Value) -> None:
    """Remember a tensor's type after checking that its name is unused.

    Args:
        values: Lookup table of previously defined tensor names and types.
            This function adds one entry to it.
        value: New graph input or operation result to record.

    Raises:
        ValueError: If the name is empty or already in the table.
    """
    if not value.name or value.name in values:
        raise ValueError("SSA value names must be nonempty and unique")
    values[value.name] = value.type


@dataclass(frozen=True, slots=True)
class ParallelPlan:
    """Describe how to split the graph's tensors across logical devices.

    The ``shard()`` pass applies this choice; this object only records it.
    With one device, tensors stay whole. Physical channels, banks, and addresses
    are selected later.

    The same equal split applies to every tensor. Multi-device splitting
    supports equal-shape addition, multiplication, and copy. Reshape needs an
    ownership transformation that is not implemented. Code generation requires one rank
    on logical device zero. This is not yet a general plan for model parallelism.

    Attributes:
        device_ids: Devices that receive the pieces, in order. A device's
            position in this tuple is its rank. These IDs do not select physical
            channels.
        shard_axis: Tensor dimension to split equally. For a matrix, ``0`` splits
            rows and ``1`` splits columns. Ignored when there is only one device.

    Examples:
        For an eight-element vector, ``device_ids=(0, 1)`` and ``shard_axis=0``
        assign entries ``[0:4]`` to device 0 and ``[4:8]`` to device 1.
        With ``device_ids=(3, 7)``, rank 0 uses device 3 and rank 1 uses device 7.
    """

    device_ids: tuple[int, ...] = (0,)
    shard_axis: int = 0

    def __post_init__(self) -> None:
        """Check device IDs and the axis before any graph is supplied.

        Whether the axis exists and divides each tensor evenly is checked later,
        when ``shard()`` knows the shapes.

        Raises:
            ValueError: If there are no devices, a device ID repeats or is negative,
                or the axis is negative.
        """
        if not self.device_ids or len(set(self.device_ids)) != len(self.device_ids):
            raise ValueError("Logical device ids must be nonempty and unique")
        if self.shard_axis < 0 or any(device < 0 for device in self.device_ids):
            raise ValueError("Device ids and shard axis must be nonnegative")


@dataclass(frozen=True, slots=True)
class ShardedValue:
    """Record which part of a full tensor belongs to one device.

    The record connects a local tensor to the full tensor it came from.
    Its interval includes ``start`` and excludes ``stop``, like a Python slice.

    Attributes:
        global_value: Name and type of the complete tensor before splitting.
        local_value: Same tensor name, with the smaller shape used on this device.
        axis: Dimension that was split. The value ``-1`` marks an unsplit tensor on
            a single device.
        start: First coordinate owned along the split dimension; zero if unsplit.
        stop: First coordinate after the owned part. For an unsplit tensor this
            instead records its total element count.

    Examples:
        For the second half of an eight-element vector, ``global_value`` has
        ``shape=(8,)``, ``local_value`` has ``shape=(4,)``, and the interval is
        ``start=4``, ``stop=8``.
    """

    global_value: Value
    local_value: Value
    axis: int
    start: int
    stop: int


@dataclass(frozen=True, slots=True)
class RankGraph:
    """Collect the tensor pieces and calculations assigned to one device.

    Attributes:
        rank: Position of this device in ``ParallelPlan``.device_ids.
        device_id: Logical device ID at that position.
        graph: Calculations with input and result shapes changed to this device's
            portions. Tensor names still match the original graph.
        values: Records connecting each local tensor to its part of the full
            tensor. Inputs come first, followed by operation results.
    """

    rank: int
    device_id: int
    graph: SemanticGraph
    values: tuple[ShardedValue, ...]


@dataclass(frozen=True, slots=True)
class ShardedGraph:
    """Keep the original calculation together with its per-device versions.

    Retaining the original lets a verifier check that no part was lost or
    assigned twice when tensors were split.

    Attributes:
        source: Full graph before splitting.
        parallel: Caller-supplied rule for splitting the tensors.
        shards: One graph per device, in the order given by ``parallel.device_ids``.
    """

    source: SemanticGraph
    parallel: ParallelPlan
    shards: tuple[RankGraph, ...]


def _partition(value: Value, plan: ParallelPlan, rank: int) -> ShardedValue:
    """Find one device's piece of a tensor without moving any data.

    Args:
        value: Name and shape of the complete tensor.
        plan: Participating device IDs and the dimension to split equally.
        rank: This device's position in ``plan.device_ids``, supplied by ``shard()``.

    Returns:
        The original and local tensor types plus the interval this device owns.

    Raises:
        ValueError: If the selected dimension is absent or its length cannot be
            divided equally among the devices.

    Examples:
        With two devices and eight entries, rank 1 gets ``start=4`` and ``stop=8``.
    """
    count = len(plan.device_ids)
    if count == 1:
        return ShardedValue(value, value, -1, 0, value.type.numel)
    axis = plan.shard_axis
    if axis >= len(value.type.shape) or value.type.shape[axis] % count:
        raise ValueError("Shard axis must exist and divide evenly across devices")
    # Only the chosen dimension shrinks. For a (4, 6) matrix split along
    # axis 1 across two devices, both local matrices have shape (4, 3).
    extent = value.type.shape[axis] // count
    shape = (*value.type.shape[:axis], extent, *value.type.shape[axis + 1 :])
    local = replace(value, type=TensorType(shape, value.type.dtype))
    # Equal consecutive intervals tile the full dimension without gaps:
    # rank 0 owns [0:extent], rank 1 owns [extent:2*extent], and so on.
    return ShardedValue(value, local, axis, rank * extent, (rank + 1) * extent)


def shard(graph: SemanticGraph, parallel: ParallelPlan) -> ShardedGraph:
    """Build a version of the calculation for each device's tensor pieces.

    Each device receives operations over its own tensor pieces. Supported
    addition, multiplication, and copy need no communication because each output
    element depends only on corresponding input elements. A single-device plan leaves all shapes unchanged.

    Args:
        graph: Full calculation before splitting, with no physical addresses.
        parallel: Caller-supplied device IDs and dimension to divide equally.

    Returns:
        Original graph plus one local graph and tensor-ownership records per device.

    Raises:
        ValueError: If the graph is invalid or a tensor cannot be split as requested.
        NotImplementedError: If a multi-device graph contains an operation other
            than addition, multiplication, or copy. In particular, reshape does
            not yet define how output coordinates preserve device ownership.

    Examples:
        For ``z = x + y``, split ``x``, ``y``, and ``z`` in the same way, then
        give every device an addition over its own pieces.
    """
    verify_semantic(graph)
    if len(parallel.device_ids) > 1 and any(
        not isinstance(operation, (AddOperation, MultiplyOperation, CopyOperation))
        for operation in graph.operations
    ):
        raise NotImplementedError(
            "Multi-rank sharding supports only addition, multiplication, and copy"
        )
    global_values = (
        *graph.inputs,
        *(operation.output for operation in graph.operations),
    )
    ranks: list[RankGraph] = []
    for rank, device_id in enumerate(parallel.device_ids):
        values = tuple(_partition(value, parallel, rank) for value in global_values)
        local_values = {value.local_value.name: value.local_value for value in values}
        # Every operand keeps its name; only its shape changes. Reusing the
        # original operation order preserves which results feed later work.
        local = SemanticGraph(
            tuple(local_values[value.name] for value in graph.inputs),
            tuple(
                replace(operation, output=local_values[operation.output.name])
                for operation in graph.operations
            ),
            graph.outputs,
        )
        verify_semantic(local)
        ranks.append(RankGraph(rank, device_id, local, values))
    return ShardedGraph(graph, parallel, tuple(ranks))


def verify_sharded(graph: ShardedGraph) -> None:
    """Check that device graphs contain exactly the pieces the plan assigns.

    Rebuild the expected split from the original graph and compare every field.
    This catches missing ranks, incorrect intervals, and local operations that
    were changed after splitting.

    Args:
        graph: Original graph, parallel plan, and per-device results to compare.

    Raises:
        ValueError: If a device, owned interval, tensor shape, or operation differs
            from the split required by the plan, or the source graph is invalid.
        NotImplementedError: If the source needs unsupported multi-device operations.
    """
    expected = shard(graph.source, graph.parallel)
    if graph != expected:
        raise ValueError(
            "Sharded graph violates rank coverage or local tensor semantics"
        )


class DeviceOperationKind(Enum):
    """Identify operations that device components can perform.

    Members:
        PNM_ADD: Add matching tensor elements in processing-near-memory (PNM)
            logic, using operands in the device's Shared Buffer.
        PIM_MULTIPLY: Multiply matching elements using processing-in-memory
            bank arithmetic. Inputs and results use the Shared Buffer at this
            stage; target lowering supplies the required DRAM staging transfers.
        COPY: Produce the same tensor values under a new name. Mapping can
            avoid a physical copy when memory reuse preserves all needed values.
        RESHAPE: Change dimensions without reordering flattened tensor elements.
            Memory reuse or a copy is selected later by mapping.
        LINEAR: Batched input-row dot products against weight rows. Execution
            requires an explicit reference accumulator and packing contract.
        ATTENTION: Supplied-tensor scaled dot products, softmax, and weighted values.
        ACTIVATION: Apply the selected scalar nonlinear formula to each element.
        SOFTMAX: Normalize each last-dimension row into probabilities.
        RMS_NORM: Per-row square mean, reciprocal root, and learned scaling.
            Execution requires the reference scalar and broadcast contract.
    """

    PNM_ADD = "pnm_add"
    PIM_MULTIPLY = "pim_multiply"
    COPY = "copy"
    RESHAPE = "reshape"
    LINEAR = "linear"
    RMS_NORM = "rms_norm"
    ACTIVATION = "activation"
    SOFTMAX = "softmax"
    ATTENTION = "attention"


class MemorySpace(Enum):
    """Name a kind of storage before choosing an address in it.

    Members:
        SHARED_BUFFER: The buffer shared by components within one CENT device.
            This name says where data must reside, not which slots it occupies.
    """

    SHARED_BUFFER = "shared_buffer"


@dataclass(frozen=True, slots=True)
class CapabilityProfile:
    """List the component operations available on the chosen target.

    The legalization pass uses this list to decide whether it can implement a
    requested calculation. This profile has
    no channel counts, memory capacities, or addresses; those enter later.

    Attributes:
        supported_operations: Device operation kinds that the backend can lower.
            An empty set rejects every operation; the default enables addition,
            multiplication, copy, reshape, linear transforms, normalization,
            nonlinear functions, and supplied-tensor attention. Target mapping separately requires reference
            opt-in for accumulator, scalar, and nonlinear computations.

    Examples:
        An ``AddOperation`` requires ``DeviceOperationKind.PNM_ADD`` in
        ``supported_operations``.
    """

    supported_operations: frozenset[DeviceOperationKind] = frozenset(
        {
            DeviceOperationKind.PNM_ADD,
            DeviceOperationKind.PIM_MULTIPLY,
            DeviceOperationKind.COPY,
            DeviceOperationKind.RESHAPE,
            DeviceOperationKind.LINEAR,
            DeviceOperationKind.RMS_NORM,
            DeviceOperationKind.ACTIVATION,
            DeviceOperationKind.SOFTMAX,
            DeviceOperationKind.ATTENTION,
        }
    )


@dataclass(frozen=True, slots=True)
class DeviceOperation:
    """Describe a calculation in terms of an available device component.

    Legalization selects a supported operation and the kind of storage its
    operands need. Dependencies record which earlier operations must finish
    before this operation can read their results.

    Attributes:
        name: ``Operation`` name carried forward from the semantic graph.
        kind: Device operation selected to implement the calculation.
        inputs: Names of this device's operand tensors, in calculation order.
        output: Name and shape of the tensor produced on this device.
        dependencies: Earlier operations that produce these inputs, listed once
            each in input order. Caller-supplied inputs need no producer operation.
        memory_space: Kind of storage required for the inputs and result.
        block: Original model-block label retained for diagnostics.

    Examples:
        For ``z = x + y``, legalization selects ``DeviceOperationKind.PNM_ADD``
        and requires Shared Buffer operands. If ``x`` came from an earlier
        operation, that operation is listed in ``dependencies``.
    """

    name: str
    kind: DeviceOperationKind
    inputs: tuple[str, ...]
    output: Value
    dependencies: tuple[str, ...]
    memory_space: MemorySpace = MemorySpace.SHARED_BUFFER
    block: str = ""


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceAttentionOperation(DeviceOperation):
    """Retain attention position and scaling choices for target lowering.

    Attributes:
        name: Original operation name.
        kind: ``DeviceOperationKind.ATTENTION``.
        inputs: Query, key, value, and optional additive mask names.
        output: Four-dimensional attention result type and name.
        dependencies: Operations producing the operands.
        memory_space: Shared Buffer storage at graph boundaries.
        block: Source model-block label.
        causal: Whether later keys are masked.
        query_position_offset: Position of query row zero, measured in tokens.
        scale: Explicit multiplier, or ``None`` for the inverse square root of width.
    """

    causal: bool
    query_position_offset: int
    scale: float | None


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceActivationOperation(DeviceOperation):
    """Retain the scalar function chosen by the source activation.

    Attributes:
        name: Original semantic operation name.
        kind: ``DeviceOperationKind.ACTIVATION``.
        inputs: One input tensor name.
        output: Result name with the unchanged input type.
        dependencies: Operations producing the input.
        memory_space: Shared Buffer storage at operation boundaries.
        block: Optional source model-block label.
        function: Required scalar formula and GELU approximation choice.
    """

    function: ActivationKind


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceRmsNormOperation(DeviceOperation):
    """Retain the normalization constant for explicit reference lowering.

    Attributes:
        name: Operation name retained from the semantic graph.
        kind: ``DeviceOperationKind.RMS_NORM``.
        inputs: Input and learned scale tensor names.
        output: Normalized result name and unchanged input type.
        dependencies: Operations producing the input tensors.
        memory_space: Shared Buffer storage at operation boundaries.
        block: Original model-block label for diagnostics.
        epsilon: Constant added to the mean square before its reciprocal square
            root; retained from the validated semantic normalization operation.
    """

    epsilon: float


@dataclass(frozen=True, slots=True)
class DeviceRankGraph:
    """Collect device-supported calculations for one logical device.

    Attributes:
        rank: Position of this device in the caller's parallel plan.
        device_id: Logical ID assigned to that position.
        inputs: Input tensors for this device, with local shapes and argument order.
        operations: Device operations ordered after the operations they depend on.
        outputs: Local tensor names returned by this device, in result order.
    """

    rank: int
    device_id: int
    inputs: tuple[Value, ...]
    operations: tuple[DeviceOperation, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeviceGraph:
    """Keep the device-supported work and the graph it was derived from.

    No memory addresses have been chosen yet. Mapping will decide which storage
    objects hold these tensors and whether their memory can be reused.

    Attributes:
        source: Per-device semantic graphs and their original parallel plan.
        capabilities: Available operations used when choosing device operations.
        ranks: Work for every device, in the same order as the parallel plan.
    """

    source: ShardedGraph
    capabilities: CapabilityProfile
    ranks: tuple[DeviceRankGraph, ...]


def legalize(graph: ShardedGraph, capabilities: CapabilityProfile) -> DeviceGraph:
    """Turn each requested calculation into an operation the device supports.

    Legalization selects tensor arithmetic, copying, reshaping, normalization,
    nonlinear functions, and attention when the required capabilities are present.
    Unsupported calculations are
    rejected rather than emitted as incomplete commands. No address is chosen.
    Shared Buffer inputs and outputs are the boundary even for multiplication,
    whose later target commands stage operands through DRAM banks.

    Args:
        graph: Per-device graphs produced by splitting the original calculation.
        capabilities: Device operation kinds available to implement each calculation.

    Returns:
        Device operations with their required memory kind and the names of any
        earlier operations that must produce their inputs.

    Raises:
        ValueError: If the per-device graphs do not match the original graph and plan.
        NotImplementedError: If a calculation has no implementation or the target
            lacks the operation needed to implement it.
    """
    verify_sharded(graph)
    ranks: list[DeviceRankGraph] = []
    for rank in graph.shards:
        producers: dict[str, str] = {}
        operations: list[DeviceOperation] = []
        for operation in rank.graph.operations:
            if isinstance(operation, AddOperation):
                kind = DeviceOperationKind.PNM_ADD
            elif isinstance(operation, MultiplyOperation):
                kind = DeviceOperationKind.PIM_MULTIPLY
            elif isinstance(operation, CopyOperation):
                kind = DeviceOperationKind.COPY
            elif isinstance(operation, ReshapeOperation):
                kind = DeviceOperationKind.RESHAPE
            elif isinstance(operation, LinearOperation):
                kind = DeviceOperationKind.LINEAR
            elif isinstance(operation, AttentionOperation):
                kind = DeviceOperationKind.ATTENTION
            elif isinstance(operation, ActivationOperation):
                kind = DeviceOperationKind.ACTIVATION
            elif isinstance(operation, SoftmaxOperation):
                kind = DeviceOperationKind.SOFTMAX
            elif isinstance(operation, RmsNormOperation):
                kind = DeviceOperationKind.RMS_NORM
            else:
                raise NotImplementedError(
                    f"No device legalization for {type(operation).__name__}"
                )
            if kind not in capabilities.supported_operations:
                raise NotImplementedError(
                    f"Target capability {kind.name} is unavailable"
                )
            # Graph inputs arrive from the caller and have no producer here.
            # dict.fromkeys removes repeated producers while keeping input order;
            # z + z needs to wait for z's producer only once.
            dependencies = tuple(
                dict.fromkeys(
                    producers[name] for name in operation.inputs if name in producers
                )
            )
            lowered = DeviceOperation(
                operation.name,
                kind,
                operation.inputs,
                operation.output,
                dependencies,
                block=operation.block,
            )
            if isinstance(operation, RmsNormOperation):
                lowered = DeviceRmsNormOperation(
                    name=operation.name,
                    kind=kind,
                    inputs=operation.inputs,
                    output=operation.output,
                    dependencies=dependencies,
                    block=operation.block,
                    epsilon=operation.epsilon,
                )
            if isinstance(operation, ActivationOperation):
                lowered = DeviceActivationOperation(
                    name=operation.name,
                    kind=kind,
                    inputs=operation.inputs,
                    output=operation.output,
                    dependencies=dependencies,
                    block=operation.block,
                    function=operation.function,
                )
            if isinstance(operation, AttentionOperation):
                lowered = DeviceAttentionOperation(
                    name=operation.name,
                    kind=kind,
                    inputs=operation.inputs,
                    output=operation.output,
                    dependencies=dependencies,
                    block=operation.block,
                    causal=operation.causal,
                    query_position_offset=operation.query_position_offset,
                    scale=operation.scale,
                )
            operations.append(lowered)
            producers[operation.output.name] = operation.name
        ranks.append(
            DeviceRankGraph(
                rank.rank,
                rank.device_id,
                rank.graph.inputs,
                tuple(operations),
                rank.graph.outputs,
            )
        )
    return DeviceGraph(graph, capabilities, tuple(ranks))


def verify_device(graph: DeviceGraph) -> None:
    """Check that device operations still implement the per-device graphs.

    Re-run legalization and compare its result. Changing a tensor shape, dropping
    a dependency, or selecting a different memory kind cannot pass this check.

    Args:
        graph: Device operations together with their source and capability profile.

    Raises:
        ValueError: If an operation, input/output shape, memory kind, or dependency
            differs from the expected result, or the source graph is invalid.
        NotImplementedError: If a required calculation or capability is unsupported.
    """
    expected = legalize(graph.source, graph.capabilities)
    if graph != expected:
        raise ValueError("Device graph violates legalization or dependency invariants")
