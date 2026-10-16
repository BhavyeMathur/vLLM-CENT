"""Compile Python tensor graphs through separate, checked stages."""

from .artifact import CompiledArtifact, compile_target
from .ir import CapabilityProfile, ParallelPlan, SemanticGraph, legalize, shard
from .mapping import (
    AccumulatorPlacement,
    ExecutionContract,
    CentTargetProfile,
    MappingPlan,
    MappingProvider,
    MappingRequest,
    ValuePlacement,
    baseline_mapping,
    verify_mapping,
)
from .target import (
    TargetProgram,
    allocate_storage,
    emit_program,
    lower_target,
)

__all__ = [
    "AccumulatorPlacement",
    "CentTargetProfile",
    "CompiledArtifact",
    "ExecutionContract",
    "MappingPlan",
    "MappingProvider",
    "MappingRequest",
    "TargetProgram",
    "ValuePlacement",
    "allocate_storage",
    "baseline_mapping",
    "compile_graph",
    "compile_target",
    "emit_program",
    "lower_target",
    "verify_mapping",
]


def compile_graph(
    graph: SemanticGraph,
    target: CentTargetProfile,
    *,
    parallel: ParallelPlan | None = None,
    capabilities: CapabilityProfile | None = None,
    mapping: MappingPlan | None = None,
    mapper: MappingProvider = baseline_mapping,
    constraints: tuple[ValuePlacement, ...] = (),
) -> CompiledArtifact:
    """Turn a tensor calculation into a reusable CENT program, one stage at a time.

    Split the graph according to the caller's parallel plan, select supported
    device operations, choose buffer reuse and operation order, then assign
    addresses and emit instructions. Each boundary checks the previous result.
    Static addition, multiplication, copy, and contiguous reshape run on device
    zero. Linear transforms, RMS normalization, activations, softmax, and attention
    require the explicit reference execution contract. Tensor shapes are known at compile time;
    elementwise arithmetic does not broadcast.

    Args:
        graph: Tensor inputs, calculations, and outputs with shapes already known.
            It contains no hardware addresses.
        target: Hardware sizes and command limits used from mapping onward.
        parallel: Optional rule for splitting tensors across logical devices.
            ``None`` keeps every tensor whole on device zero.
        capabilities: Device operations available for implementing the graph.
            ``None`` enables all implemented device operations.
        mapping: Optional complete manual schedule and buffer assignment. When
            supplied, it is verified and ``mapper`` is not called.
        mapper: Callable that proposes a mapping when ``mapping`` is ``None``. The default
            reuses storage after its contents are no longer needed.
        constraints: Required tensor-to-buffer assignments. Both manual plans and
            mapper-produced plans must honor these assignments.

    Returns:
        Compiled artifact containing instructions, input/output shapes, physical
        regions, and the intermediate target commands for inspection.

    Raises:
        ValueError: If graph shapes, sharding, mapping, or target limits are invalid.
        NotImplementedError: If a calculation or required capability has no lowering.
    """
    # Sharding changes which tensor elements each device owns. Legalization
    # then chooses device-supported operations for those local calculations.
    device = legalize(
        shard(graph, ParallelPlan() if parallel is None else parallel),
        CapabilityProfile() if capabilities is None else capabilities,
    )
    # Physical limits enter here. Manual and generated plans share the same
    # target-lowering verifier, so neither can bypass storage-safety checks.
    request = MappingRequest(graph=device, target=target, constraints=constraints)
    selected = mapper(request) if mapping is None else mapping
    return compile_target(lower_target(request, selected))
