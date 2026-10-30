"""Choose device operations for an IR function, before any address exists.

The compiler's operations live in ``vllm_cent.ir``: values there are Python
objects, and a function may have unnamed intermediate results. This module is
the boundary to the CENT backend. ``legalize()`` gives every value a name (the
same name the IR printer shows), checks that the backend supports each
operation, and records the device operation kind and the storage each operand
needs. Mapping and target lowering then refer to tensors and operations by
these names.

No stage in this module chooses physical addresses. Additions,
multiplications, copies, order-preserving reshapes, linear transforms, RMS
normalization, activations, softmax, rotary embedding, and fused attention
reach the device form. Other IR operations, such as KV-cache updates, the
decomposed attention steps, and all-reduce, are rejected with
``NotImplementedError``.
"""

from dataclasses import dataclass
from enum import Enum

from ..activations import ActivationKind
from ..ir import (
    ActivationOp,
    AddOp,
    AttentionOp,
    CopyOp,
    DType,
    Function,
    LinearOp,
    MergeHeadsOp,
    MulOp,
    Operation,
    ReshapeOp,
    RMSNormOp,
    RopeOp,
    SoftmaxOp,
    SplitHeadsOp,
    TensorType,
    Value,
    value_names,
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
    "DeviceRopeOperation",
    "MemorySpace",
    "NameValue",
    "legalize",
    "verify_device",
]

# Machine epsilon of fp32, 2**-23: the distance from 1.0 to the next float.
# An RMSNorm without an explicit eps uses it, as torch.nn.RMSNorm(eps=None)
# does for fp32 inputs.
_FP32_EPSILON = 2.0**-23


@dataclass(frozen=True, slots=True)
class NameValue:
    """Give a tensor a name so device operations can refer to it.

    IR values are identified by object; device work is identified by name,
    because schedules and storage plans are written as names. A name is
    defined only once in a device graph.

    Attributes:
        name: Tensor name, unique within this graph.
        type: Tensor dimensions and element format. The actual data arrives later.

    Examples:
        For ``z = x + y``, ``x`` and ``y`` name the inputs and ``z`` names a new result.
    """

    name: str
    type: TensorType


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
        ROPE: Rotate head vectors by angles looked up per token position.
            Execution requires the reference rotary-embedding service.
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
    ROPE = "rope"


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
            nonlinear functions, rotary embedding, and supplied-tensor
            attention. Target mapping separately requires reference
            opt-in for accumulator, scalar, and nonlinear computations.

    Examples:
        An ``AddOp`` requires ``DeviceOperationKind.PNM_ADD`` in
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
            DeviceOperationKind.ROPE,
        }
    )


@dataclass(frozen=True, slots=True)
class DeviceOperation:
    """Describe a calculation in terms of an available device component.

    Legalization selects a supported operation and the kind of storage its
    operands need. Dependencies record which earlier operations must finish
    before this operation can read their results.

    Attributes:
        name: Name of the source operation's result. One name identifies both
            the operation (in schedules) and the tensor it produces.
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
    output: NameValue
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


@dataclass(frozen=True, slots=True, kw_only=True)
class DeviceRopeOperation(DeviceOperation):
    """Retain how a rotary embedding pairs the rotated elements.

    Attributes:
        name: Name of the rotated result.
        kind: ``DeviceOperationKind.ROPE``.
        inputs: Head vectors, positions, and angle table names, in that order.
        output: Rotated result name with the unchanged head-vector type.
        dependencies: Operations producing the inputs.
        memory_space: Shared Buffer storage at operation boundaries.
        block: Source model-block label.
        is_neox_style: True pairs element ``i`` with ``i + rotary_dim // 2``;
            False pairs ``2i`` with ``2i + 1``.
    """

    is_neox_style: bool


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
    inputs: tuple[NameValue, ...]
    operations: tuple[DeviceOperation, ...]
    outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DeviceGraph:
    """Keep the device-supported work and the graph it was derived from.

    No memory addresses have been chosen yet. Mapping will decide which storage
    objects hold these tensors and whether their memory can be reused.

    Attributes:
        source: The IR function this device work was selected for.
        capabilities: Available operations used when choosing device operations.
        ranks: Work for every device, in the same order as the parallel plan.
    """

    source: Function
    capabilities: CapabilityProfile
    ranks: tuple[DeviceRankGraph, ...]


def legalize(function: Function, capabilities: CapabilityProfile) -> DeviceGraph:
    """Turn each IR operation into an operation the device supports.

    First give every value its printed name. Then, for each operation in
    order, select the device operation kind, check that the target provides
    it, and record which earlier operations produce its inputs. Unsupported
    calculations are rejected rather than emitted as incomplete commands. No
    address is chosen. Shared Buffer inputs and outputs are the boundary even
    for multiplication, whose later target commands stage operands through
    DRAM banks.

    The function runs on one device. A tensor-parallel function can be
    legalized too: its shapes are already one rank's local shapes, and each
    rank would run the same program on its own pieces.

    Args:
        function: Verified IR function with static shapes. Its values may carry
            semantic axes and placements; legalization ignores them.
        capabilities: Device operation kinds available to implement each calculation.

    Returns:
        Device operations with their required memory kind and the names of any
        earlier operations that must produce their inputs, for device zero.

    Raises:
        ValueError: If the function is invalid, an argument's name is missing
            or repeated, a tensor has a zero-length dimension, or the results
            are empty or repeat a value.
        NotImplementedError: If a tensor is not fp32, a calculation has no
            device implementation, or the target lacks the operation needed to
            implement it.
    """
    # TODO(multi-device): Target lowering accepts exactly one rank on logical
    # device zero. Running a tensor-parallel function needs one rank graph per
    # device plus all-reduce lowering; neither exists yet.
    #
    # Names come from the printer's rules, so a device graph uses the same
    # names as the IR dump of the function it came from.
    function.verify()
    names = value_names(function)
    # Report an unsupported operation before an unsupported tensor type: RoPE's
    # int64 positions would otherwise hide that RoPE itself has no lowering.
    kinds = [_device_kind(op, names[op.results[0]]) for op in function.operations]
    _check_backend_types(function, names)
    if not function.results or len(set(function.results)) != len(function.results):
        raise ValueError("Function results must name unique values")
    # Argument names are the compiled program's input names, which callers
    # bind values to. The printer would quietly rename a repeated name (x,
    # x_1) or number a missing one, so require each name to survive as given.
    for argument in function.operands:
        if not argument.name or names[argument] != argument.name:
            raise ValueError(
                f"Function arguments need unique, non-empty names; "
                f"{argument.name!r} would become {names[argument]!r}"
            )

    producers: dict[str, str] = {}
    operations: list[DeviceOperation] = []
    for op, kind in zip(function.operations, kinds, strict=True):
        # Every supported operation produces exactly one result, so the
        # result's name also names the device operation.
        name = names[op.results[0]]
        if kind not in capabilities.supported_operations:
            raise NotImplementedError(f"Target capability {kind.name} is unavailable")
        inputs = tuple(names[operand] for operand in op.operands)
        # Graph inputs arrive from the caller and have no producer here.
        # dict.fromkeys removes repeated producers while keeping input order;
        # z + z needs to wait for z's producer only once.
        dependencies = tuple(
            dict.fromkeys(producers[input] for input in inputs if input in producers)
        )
        operations.append(_device_operation(op, name, kind, inputs, dependencies))
        producers[name] = name
    rank = DeviceRankGraph(
        0,
        0,
        tuple(NameValue(names[value], value.type) for value in function.operands),
        tuple(operations),
        tuple(names[value] for value in function.results),
    )
    return DeviceGraph(function, capabilities, (rank,))


def _check_backend_types(function: Function, names: dict[Value, str]) -> None:
    """Reject tensors the reference backend cannot store or compute with.

    Args:
        function: Function whose arguments and results are checked.
        names: Printed name of every value, for the error message.

    Raises:
        NotImplementedError: If a tensor is not fp32, other than RoPE's
            integer positions. The functional simulator computes with Python
            floats; other formats have no lowering yet.
        ValueError: If a tensor has a zero-length dimension and so no element
            to store.
    """
    values = (
        *function.operands,
        *(result for op in function.operations for result in op.results),
    )
    # A RoPE position only selects a table row. It is stored as a scalar value,
    # which holds every integer below 2**24 exactly, so integer dtypes are safe.
    positions = {op.positions for op in function.operations if isinstance(op, RopeOp)}
    for value in values:
        # RopeOp itself requires the positions to be int32 or int64.
        if value not in positions and value.type.dtype is not DType.FP32:
            raise NotImplementedError(
                f"The CENT pipeline supports fp32 tensors only; "
                f"{names[value]} is {value.type.dtype}"
            )
        if any(dim == 0 for dim in value.type.shape):
            raise ValueError(f"Tensor dimensions must be positive: {names[value]}")


def _device_kind(op: Operation, name: str) -> DeviceOperationKind:
    """Select the device operation kind that implements an IR operation.

    Args:
        op: The IR operation.
        name: Its result name, for the error message.

    Returns:
        The kind whose target lowering computes the same result.

    Raises:
        NotImplementedError: If no device operation implements ``op``.
    """
    if isinstance(op, AddOp):
        return DeviceOperationKind.PNM_ADD
    if isinstance(op, MulOp):
        return DeviceOperationKind.PIM_MULTIPLY
    if isinstance(op, CopyOp):
        return DeviceOperationKind.COPY
    # split_heads and merge_heads only regroup the last dimensions, so they
    # keep the flattened element order, like any reshape.
    if isinstance(op, (ReshapeOp, SplitHeadsOp, MergeHeadsOp)):
        return DeviceOperationKind.RESHAPE
    if isinstance(op, LinearOp):
        # TODO(linear bias): The reference GEMV lowering computes x @ W.T
        # only. A bias needs either an addition after it or a bias-aware
        # accumulator initialization (CENT's WR_BIAS); neither is wired up.
        if op.bias is not None:
            raise NotImplementedError(
                f"No device legalization for {op.NAME} with a bias ({name})"
            )
        return DeviceOperationKind.LINEAR
    if isinstance(op, AttentionOp):
        return DeviceOperationKind.ATTENTION
    if isinstance(op, ActivationOp):
        return DeviceOperationKind.ACTIVATION
    if isinstance(op, SoftmaxOp):
        return DeviceOperationKind.SOFTMAX
    if isinstance(op, RMSNormOp):
        # TODO(unscaled RMSNorm): The reference RMSNorm lowering multiplies by
        # a weight tensor. Llama 4's QK-norm has none; lowering it needs either
        # a weightless variant of the emitter or a constant ones vector.
        if op.weight is None:
            raise NotImplementedError(
                f"No device legalization for {op.NAME} without a weight ({name})"
            )
        return DeviceOperationKind.RMS_NORM
    if isinstance(op, RopeOp):
        return DeviceOperationKind.ROPE
    raise NotImplementedError(f"No device legalization for {op.NAME} ({name})")


def _device_operation(
    op: Operation,
    name: str,
    kind: DeviceOperationKind,
    inputs: tuple[str, ...],
    dependencies: tuple[str, ...],
) -> DeviceOperation:
    """Build the device operation for ``op``, with the parameters its kind needs.

    Args:
        op: The IR operation being legalized.
        name: Name of its result; it names the device operation too.
        kind: Device operation kind selected for ``op``.
        inputs: Names of its operands, in operand order.
        dependencies: Earlier device operations producing those operands.

    Returns:
        A plain ``DeviceOperation``, or a subclass that also keeps the
        normalization constant, activation formula, or attention options
        that target lowering needs.
    """
    output = NameValue(name, op.results[0].type)
    # The name hint is the module path, e.g. "mlp.gate_proj"; its parent
    # path, "mlp", labels the model block in diagnostics.
    block = (op.name_hint or "").rpartition(".")[0]
    if isinstance(op, RMSNormOp):
        return DeviceRmsNormOperation(
            name=name,
            kind=kind,
            inputs=inputs,
            output=output,
            dependencies=dependencies,
            block=block,
            epsilon=_FP32_EPSILON if op.eps is None else op.eps,
        )
    if isinstance(op, ActivationOp):
        return DeviceActivationOperation(
            name=name,
            kind=kind,
            inputs=inputs,
            output=output,
            dependencies=dependencies,
            block=block,
            function=op.function,
        )
    if isinstance(op, RopeOp):
        return DeviceRopeOperation(
            name=name,
            kind=kind,
            inputs=inputs,
            output=output,
            dependencies=dependencies,
            block=block,
            is_neox_style=op.is_neox_style,
        )
    if isinstance(op, AttentionOp):
        return DeviceAttentionOperation(
            name=name,
            kind=kind,
            inputs=inputs,
            output=output,
            dependencies=dependencies,
            block=block,
            causal=op.causal,
            query_position_offset=op.query_position_offset,
            scale=op.scale,
        )
    return DeviceOperation(name, kind, inputs, output, dependencies, block=block)


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
