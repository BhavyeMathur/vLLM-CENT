"""Turn a storage plan into CENT instructions in two steps.

First describe tensor calculations and their staging needs using buffer names.
Then assign those buffers numeric Shared Buffer slots and fill in instruction
operands. Keeping these steps separate makes it possible to inspect required
data movement before addresses obscure which tensors are being used.
"""

from dataclasses import dataclass
from enum import Enum, auto
from math import sqrt

from ..cent import (
    Accumulate,
    CentMemoryAddress,
    CentProgram,
    CentSharedBufferAddress,
    ReadSingleBank,
    WriteSingleBank,
)
from ..cent.instructions import CentInstruction
from ..cent.reference import ReferenceActivation, ReferenceSoftmax
from ..cent.utils import ceil_div
from ..runtime import (
    CentExecutable,
    CentExecutionManifest,
    CentInputBinding,
    CentOutputBinding,
    CentSharedBufferRegion,
)
from .ir import (
    DeviceOperationKind,
    DeviceRmsNormOperation,
    DeviceActivationOperation,
    DeviceAttentionOperation,
)
from .attention_codegen import (
    AttentionCommand,
    _attention_workspace_slots,
    _emit_attention,
)
from .nonlinear import ActivationCommand, SoftmaxCommand
from .reference_codegen import (
    LinearCommand,
    _LinearBuffers,
    RmsNormCommand,
    ReferenceWorkspace,
    _workspace_slots,
    _emit_linear,
    _emit_rms_norm,
    _emit_multiply_chunk,
)
from .mapping import (
    MappingPlan,
    MappingRequest,
    _rank,
    _slots,
    _values,
    _fresh_storage,
    verify_mapping,
)

__all__ = [
    "AllocatedStorage",
    "TargetCommand",
    "TargetCommandKind",
    "TargetProgram",
    "allocate_storage",
    "emit_program",
    "lower_target",
]


class TargetCommandKind(Enum):
    """Choose which action a command performs on its named buffers.

    Members:
        COPY: Copy every occupied slot, including zero padding, to another buffer.
            Code generation uses a temporary DRAM row for this transfer.
        ACCUMULATE: Add each source element to the corresponding destination
            element, replacing the destination. Both inputs must already be set.
        MULTIPLY: Multiply source and destination elements, replacing the
            destination. Emission stages both operands in private DRAM scratch.
    """

    COPY = auto()
    ACCUMULATE = auto()
    MULTIPLY = auto()


@dataclass(frozen=True, slots=True, kw_only=True)
class TargetCommand:
    """Describe a CENT action using buffer names instead of numeric addresses.

    The named buffers receive actual Shared Buffer addresses in a later pass.

    Attributes:
        kind: Whether to copy, add, or multiply source values into the destination.
        source: Name of the buffer to read, including its padded lanes.
        destination: Buffer to write. Arithmetic also reads its previous contents.
        slot_count: Number of complete slots to process. This includes the last
            slot even when only some of its lanes contain tensor elements.
        operation: Name of the Device IR operation this command implements,
            so diagnostics can point back to the requested calculation.

    Examples:
        A ``COPY`` from ``x_buffer`` to ``z_buffer`` preserves ``x`` before
        ``ACCUMULATE`` adds ``y_buffer`` into ``z_buffer``.
    """

    kind: TargetCommandKind
    source: str
    destination: str
    slot_count: int
    operation: str


@dataclass(frozen=True, slots=True, kw_only=True)
class TargetProgram:
    """Keep the chosen commands beside the mapping they must implement.

    The request and mapping let the verifier rebuild the expected commands.
    That prevents a hand-built program from dropping a copy that protects a
    still-needed input.

    Attributes:
        request: Original device calculations and physical target limits.
        mapping: Chosen operation order and tensor-to-buffer assignments.
        commands: Copies, arithmetic, or explicit matrix/normalization/attention plans in
            execution order, using buffer names rather than numeric addresses.
        reference_workspace: Reusable temporary Shared Buffer reservation for
            matrix, normalization, and attention stages; ``None`` if none needs it.
    """

    request: MappingRequest
    mapping: MappingPlan
    commands: tuple[
        TargetCommand
        | LinearCommand
        | RmsNormCommand
        | ActivationCommand
        | SoftmaxCommand
        | AttentionCommand,
        ...,
    ]
    reference_workspace: ReferenceWorkspace | None = None


@dataclass(frozen=True, slots=True, kw_only=True)
class AllocatedStorage:
    """Record the concrete slots assigned to one named Shared Buffer object.

    Several tensors may use this same object at different times, as chosen by
    the mapping pass.

    Attributes:
        name: Buffer name used in the mapping and target commands.
        start: Address of the object's first Shared Buffer slot.
        slot_count: Number of consecutive complete slots reserved for the object.

    Examples:
        For ``start.slot=3`` and ``slot_count=2``, the object occupies slots 3 and 4.
    """

    name: str
    start: CentSharedBufferAddress
    slot_count: int


def lower_target(request: MappingRequest, mapping: MappingPlan) -> TargetProgram:
    """Choose copies and arithmetic that implement the supplied mapping.

    Addition and multiplication replace a destination holding the left input. If
    the mapping gives the result a separate buffer, copy the left input there
    first. If they share a buffer, no copy is needed. The mapping verifier has
    already checked that overwriting that buffer will not destroy a needed value.
    Copy and reshape preserve flat element order, so they need only a copy when
    their input and output have distinct storage. A same-storage unary operation
    emits no commands; its output name and shape remain in the runtime contract.

    Args:
        request: Device calculations, hardware limits, and required placements.
        mapping: Proposed operation order and tensor-to-buffer assignments from
            either a caller or a mapper. This function checks but does not change it.

    Returns:
        Target commands using buffer names, together with their request and mapping.

    Raises:
        ValueError: If the mapping has invalid ordering, storage reuse, or capacity.
        NotImplementedError: If the source needs an unsupported device operation.
    """
    verify_mapping(request, mapping)
    rank = request.graph.ranks[0]
    locations = {p.value: p.storage for p in mapping.placements}
    operations = {op.name: op for op in rank.operations}
    commands: list[
        TargetCommand
        | LinearCommand
        | RmsNormCommand
        | ActivationCommand
        | SoftmaxCommand
        | AttentionCommand
    ] = []
    values = {value.name: value for value in _values(rank)}
    registers = {item.operation: item.register for item in mapping.accumulators}
    workspace_slots = 0
    for name in mapping.schedule:
        op = operations[name]
        left = locations[op.inputs[0]]
        destination = locations[op.output.name]
        count = _slots(op.output, request.target)
        if op.kind is DeviceOperationKind.LINEAR:
            source_type = values[op.inputs[0]].type
            input_width = source_type.shape[-1]
            width = request.target.hardware.burst_length
            linear = LinearCommand(
                source=left,
                weights=locations[op.inputs[1]],
                destination=destination,
                input_width=input_width,
                output_width=op.output.type.shape[-1],
                batch_count=source_type.numel // input_width,
                accumulation_register=registers[name],
                tile_width=min(
                    request.target.hardware.dram_columns,
                    request.target.hardware.global_buffer_columns,
                    ceil_div(input_width, width) * width,
                ),
                channel=0,
                bank_count=request.target.hardware.num_banks,
                operation=name,
            )
            commands.append(linear)
            workspace_slots = max(
                workspace_slots, _workspace_slots(linear, request.target.hardware)
            )
            continue
        if isinstance(op, DeviceRmsNormOperation):
            source_type = values[op.inputs[0]].type
            norm = RmsNormCommand(
                source=left,
                scale=locations[op.inputs[1]],
                destination=destination,
                width=source_type.shape[-1],
                batch_count=source_type.numel // source_type.shape[-1],
                epsilon=op.epsilon,
                tile_slots=request.target.hardware.dram_columns
                // request.target.hardware.burst_length,
                operation=name,
            )
            commands.append(norm)
            workspace_slots = max(
                workspace_slots, _workspace_slots(norm, request.target.hardware)
            )
            continue
        if isinstance(op, DeviceAttentionOperation):
            batch, heads, queries, query_width = values[op.inputs[0]].type.shape
            _, kv_heads, keys, _ = values[op.inputs[1]].type.shape
            value_width = values[op.inputs[2]].type.shape[-1]
            hardware = request.target.hardware
            width = hardware.burst_length
            tile_limit = min(hardware.dram_columns, hardware.global_buffer_columns)
            attention = AttentionCommand(
                query=left,
                key=locations[op.inputs[1]],
                value=locations[op.inputs[2]],
                mask=locations[op.inputs[3]] if len(op.inputs) == 4 else None,
                shared_mask=len(op.inputs) == 4
                and len(values[op.inputs[3]].type.shape) == 2,
                destination=destination,
                batch_count=batch,
                query_heads=heads,
                kv_heads=kv_heads,
                query_count=queries,
                key_count=keys,
                query_width=query_width,
                value_width=value_width,
                scale=op.scale if op.scale is not None else 1 / sqrt(query_width),
                causal=op.causal,
                query_position_offset=op.query_position_offset,
                accumulation_register=registers[name],
                query_tile_width=min(tile_limit, ceil_div(query_width, width) * width),
                value_tile_width=min(tile_limit, ceil_div(keys, width) * width),
                channel=0,
                bank_count=hardware.num_banks,
                operation=name,
            )
            commands.append(attention)
            workspace_slots = max(
                workspace_slots, _attention_workspace_slots(attention, hardware)
            )
            continue
        if isinstance(op, DeviceActivationOperation):
            commands.append(
                ActivationCommand(
                    source=left,
                    destination=destination,
                    value_count=op.output.type.numel,
                    function=op.function,
                    operation=name,
                )
            )
            continue
        if op.kind is DeviceOperationKind.SOFTMAX:
            row_width = op.output.type.shape[-1]
            commands.append(
                SoftmaxCommand(
                    source=left,
                    destination=destination,
                    row_count=op.output.type.numel // row_width,
                    row_width=row_width,
                    operation=name,
                )
            )
            continue
        # Unary operations preserve the flat values. Binary operations first
        # place the left input in their destructive destination. Either case
        # needs a copy only when mapping selected distinct storage objects.
        if destination != left:
            commands.append(
                TargetCommand(
                    kind=TargetCommandKind.COPY,
                    source=left,
                    destination=destination,
                    slot_count=count,
                    operation=name,
                )
            )
        if op.kind in (DeviceOperationKind.COPY, DeviceOperationKind.RESHAPE):
            continue
        if op.kind is DeviceOperationKind.PNM_ADD:
            kind = TargetCommandKind.ACCUMULATE
        elif op.kind is DeviceOperationKind.PIM_MULTIPLY:
            kind = TargetCommandKind.MULTIPLY
        else:
            raise NotImplementedError(f"No target lowering for {op.kind}")
        commands.append(
            TargetCommand(
                kind=kind,
                source=locations[op.inputs[1]],
                destination=destination,
                slot_count=count,
                operation=name,
            )
        )
    workspace = None
    if workspace_slots:
        workspace = ReferenceWorkspace(
            name=_fresh_storage("__reference_workspace", set(locations.values())),
            slot_count=workspace_slots,
        )
    return TargetProgram(
        request=request,
        mapping=mapping,
        commands=tuple(commands),
        reference_workspace=workspace,
    )


def _verify_target(program: TargetProgram) -> None:
    """Check that the commands match the work and mapping they came from.

    Args:
        program: Target commands with the original request and mapping. It may
            have been built by a caller rather than by ``lower_target()``.

    Raises:
        ValueError: If a required copy is missing, a command is changed, or the
            underlying mapping is invalid.
        NotImplementedError: If the source needs an unsupported device operation.
    """
    expected = lower_target(program.request, program.mapping)
    if program != expected:
        raise ValueError("Target IR commands do not implement the validated mapping")


def allocate_storage(program: TargetProgram) -> tuple[AllocatedStorage, ...]:
    """Give each distinct buffer its own consecutive Shared Buffer slots.

    The mapping has already decided which tensors reuse the same buffer name.
    This pass honors that choice: it assigns an address once per name and places
    different names in non-overlapping intervals, starting at slot zero.

    Args:
        program: Named target commands and their mapping, checked before allocation.

    Returns:
        Buffer names, start addresses, and slot counts in the order the names first
        appear in ``mapping.placements``, followed by any reference workspace.
        Capacity includes both graph tensors and this temporary working space.

    Raises:
        ValueError: If commands disagree with the mapping, or the mapping is invalid.
        NotImplementedError: If the source needs an unsupported device operation.
    """
    _verify_target(program)
    values = {v.name: v for v in _values(program.request.graph.ranks[0])}
    allocated: dict[str, AllocatedStorage] = {}
    next_slot = 0
    for placement in program.mapping.placements:
        # Repeated names are deliberate reuse chosen by mapping, not requests
        # for additional storage. Allocate the shared object only once.
        if placement.storage in allocated:
            continue
        count = _slots(values[placement.value], program.request.target)
        allocated[placement.storage] = AllocatedStorage(
            name=placement.storage,
            start=CentSharedBufferAddress(slot=next_slot),
            slot_count=count,
        )
        next_slot += count
    if program.reference_workspace is not None:
        workspace = program.reference_workspace
        allocated[workspace.name] = AllocatedStorage(
            name=workspace.name,
            start=CentSharedBufferAddress(slot=next_slot),
            slot_count=workspace.slot_count,
        )
        next_slot += workspace.slot_count
    if next_slot > program.request.target.hardware.shared_buffer_slots:
        raise ValueError(
            "graph tensors and reference workspace exceed Shared Buffer capacity"
        )
    return tuple(allocated.values())


def _emit_commands(
    program: TargetProgram,
    allocations: tuple[AllocatedStorage, ...],
) -> tuple[CentInstruction, ...]:
    """Fill in addresses and split long commands into supported chunks.

    A copy travels through one temporary DRAM row, so each chunk must fit in that
    row. Multiplication uses the same row in each bank of channel zero and
    initializes both operands in every four-bank group before ``EW_MUL``. An
    addition is split at the target's ACC slot limit. Chunk offsets are
    added to both Shared Buffer addresses so later chunks continue where the
    previous ones stopped.

    Args:
        program: Verified target commands with buffer names and physical limits.
        allocations: Buffer addresses produced by ``allocate_storage()`` for this program.

    Returns:
        Concrete CENT instructions in execution order. Each copy becomes one or
        more write/read pairs. Products stage bank operands, multiply, and read
        back the first group. Additions become one or more ``ACC`` instructions.
    """
    # Build the typed address lookup once. Rebuilding it for every reference
    # command would multiply graph size by the number of allocated objects.
    addresses = {item.name: item.start for item in allocations}
    target = program.request.target
    hardware = target.hardware
    row_slots = hardware.dram_columns // hardware.burst_length
    instructions: list[CentInstruction] = []
    for command in program.commands:
        if isinstance(command, (AttentionCommand, LinearCommand, RmsNormCommand)):
            workspace = program.reference_workspace
            if workspace is None:
                raise ValueError("reference command requires an allocated workspace")
            workspace_address = addresses[workspace.name]
            if isinstance(command, LinearCommand):
                emitted = _emit_linear(
                    command,
                    hardware,
                    _LinearBuffers(
                        source=addresses[command.source],
                        weights=addresses[command.weights],
                        destination=addresses[command.destination],
                        workspace=workspace_address,
                    ),
                )
            elif isinstance(command, RmsNormCommand):
                emitted = _emit_rms_norm(
                    command, hardware, addresses, workspace_address
                )
            else:
                emitted = _emit_attention(
                    command, hardware, addresses, workspace_address
                )
            instructions.extend(emitted)
            continue
        if isinstance(command, ActivationCommand):
            instructions.append(
                ReferenceActivation(
                    source=addresses[command.source],
                    destination=addresses[command.destination],
                    value_count=command.value_count,
                    function=command.function,
                )
            )
            continue
        if isinstance(command, SoftmaxCommand):
            instructions.append(
                ReferenceSoftmax(
                    source=addresses[command.source],
                    destination=addresses[command.destination],
                    row_count=command.row_count,
                    row_width=command.row_width,
                )
            )
            continue
        # Shared Buffer has no copy instruction. A preservation copy writes one
        # chunk to private DRAM scratch and immediately reads it back. Scratch is
        # exclusively owned by this artifact; no weights or KV rows exist in this
        # slice. Each chunk overwrites scratch, so only one row is needed.
        limit = (
            row_slots
            if command.kind is not TargetCommandKind.ACCUMULATE
            else target.max_accumulate_slots
        )
        # For five slots and a limit of two, process offsets 0, 2, and 4 with
        # lengths 2, 2, and 1. Every slot is visited exactly once.
        for offset in range(0, command.slot_count, limit):
            count = min(limit, command.slot_count - offset)
            source = CentSharedBufferAddress(
                slot=addresses[command.source].slot + offset
            )
            destination = CentSharedBufferAddress(
                slot=addresses[command.destination].slot + offset,
            )
            if command.kind is TargetCommandKind.COPY:
                scratch = CentMemoryAddress(channel=0, bank=0, row=0, column=0)
                instructions.extend(
                    (
                        WriteSingleBank(
                            address=scratch, operation_size=count, source=source
                        ),
                        ReadSingleBank(
                            address=scratch,
                            operation_size=count,
                            destination=destination,
                        ),
                    )
                )
            elif command.kind is TargetCommandKind.MULTIPLY:
                instructions.extend(
                    _emit_multiply_chunk(hardware, source, destination, count)
                )
            else:
                instructions.append(
                    Accumulate(
                        operation_size=count,
                        source=source,
                        destination=destination,
                    )
                )
    return tuple(instructions)


def emit_program(program: TargetProgram) -> CentExecutable:
    """Build the CENT program and describe where the runtime loads and reads data.

    The returned manifest records physical regions and scalar counts, including
    padding. ``CompiledArtifact`` adds the tensor shapes and handles that padding
    when callers supply or retrieve logical tensor values.

    Args:
        program: Named target commands, mapping, and original device calculations.

    Returns:
        A ``CentExecutable`` containing instructions and named input/output memory
        regions. It can be reused with different input values.

    Raises:
        ValueError: If commands, storage reuse, addresses, instructions, or input
            regions fail verification.
        NotImplementedError: If the source needs an unsupported device operation.
    """
    allocations = allocate_storage(program)
    storage = {item.name: item for item in allocations}
    places = {p.value: storage[p.storage] for p in program.mapping.placements}
    hardware = program.request.target.hardware
    rank = _rank(program.request)
    # Instructions operate on full slots, but the runtime counts scalar lanes.
    # Include padding here so the loader initializes every lane commands read.
    regions = {
        name: CentSharedBufferRegion(
            address=item.start,
            scalar_count=item.slot_count * hardware.burst_length,
        )
        for name, item in places.items()
    }
    manifest = CentExecutionManifest(
        inputs=tuple(
            CentInputBinding(name=v.name, region=regions[v.name]) for v in rank.inputs
        ),
        outputs=tuple(
            CentOutputBinding(name=name, region=regions[name]) for name in rank.outputs
        ),
    )
    return CentExecutable(
        program=CentProgram(
            hardware=hardware, instructions=_emit_commands(program, allocations)
        ),
        manifest=manifest,
    )
