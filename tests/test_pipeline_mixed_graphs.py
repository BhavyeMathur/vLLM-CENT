"""Compare mixed tensor graphs with arithmetic evaluated independently on the host."""

import unittest
from random import Random

from vllm_cent.cent import CentHardwareSpec
from vllm_cent.ir import (
    AddOp,
    CopyOp,
    DType,
    Function,
    MulOp,
    Operation,
    ReshapeOp,
    TensorType,
    Value,
    value_names,
)
from vllm_cent.pipeline import (
    CentTargetProfile,
    MappingPlan,
    ValuePlacement,
    compile_graph,
)
from vllm_cent.runtime import CentNamedScalars

_CASE_COUNT = 120
_OPERATION_COUNT = 8


def _target() -> CentTargetProfile:
    """Provide small transfer chunks and enough capacity for distinct buffers.

    Returns:
        Two-lane slots and four-value rows on a device with two bank groups.
        Forty slots fit ten eight-element tensors even without memory reuse.
    """
    return CentTargetProfile(
        name="mixed-graph-oracle",
        hardware=CentHardwareSpec(
            num_channels=2,
            num_banks=8,
            dram_rows=1,
            dram_columns=4,
            burst_length=2,
            accumulator_slots_per_bank=2,
            sigmoid_activation_function_id=0,
            shared_buffer_slots=40,
        ),
        max_accumulate_slots=1,
    )


def _case(
    random: Random, case_index: int
) -> tuple[Function, tuple[CentNamedScalars, ...], tuple[CentNamedScalars, ...]]:
    """Build a bounded graph while separately calculating each tensor's values.

    Args:
        random: Seeded generator choosing operands, output names, and input data.
        case_index: Selects the input shape and rotates the operation sequence.

    Returns:
        Function, caller input values, and independently calculated requested
        outputs. The first input is always returned so its original contents
        must survive. Operation ``index`` names its result ``value{index}``.
    """
    # Cycling, rather than randomly sampling, guarantees scalar, odd-tail, and
    # multi-row coverage. A scalar and a one-element vector can reshape into
    # each other without changing their one flattened value.
    shapes = ((), (1,), (3,), (5,), (6,), (8,))
    tensor_type = TensorType(shape=shapes[case_index % len(shapes)], dtype=DType.FP32)
    input_names = ("x", "y")
    inputs = tuple(Value(type=tensor_type, name=name) for name in input_names)
    values = list(inputs)
    # Host values are keyed by IR value; only the requested outputs need names.
    host = {
        value: tuple(float(random.randrange(-2, 3)) for _ in range(tensor_type.numel))
        for value in inputs
    }
    bindings = tuple(
        CentNamedScalars(name=name, values=host[value])
        for name, value in zip(input_names, inputs, strict=True)
    )
    operations: list[Operation] = []
    choices: tuple[type[Operation], ...] = (AddOp, MulOp, CopyOp, ReshapeOp)
    for index in range(_OPERATION_COUNT):
        # Every graph includes each operation twice. Earlier results remain
        # eligible as operands, creating branches and longer-lived values.
        operation_type = choices[(index + case_index) % len(choices)]
        left = random.choice(values)
        name = f"value{index}"
        operation: Operation
        if operation_type is AddOp or operation_type is MulOp:
            right = random.choice(
                [value for value in values if value.type == left.type]
            )
            # The oracle uses ordinary host arithmetic and no compiler layouts,
            # instruction helpers, or functional-simulator kernels.
            if operation_type is AddOp:
                operation = AddOp(operands=(left, right), name_hint=name)
                expected = tuple(
                    a + b for a, b in zip(host[left], host[right], strict=True)
                )
            else:
                operation = MulOp(operands=(left, right), name_hint=name)
                expected = tuple(
                    a * b for a, b in zip(host[left], host[right], strict=True)
                )
        elif operation_type is CopyOp:
            operation = CopyOp(operands=(left,), name_hint=name)
            expected = host[left]
        else:
            if left.type.numel == 1:
                shape: tuple[int, ...] = (1,) if left.type.shape == () else ()
            else:
                shape = (
                    (1, left.type.numel)
                    if len(left.type.shape) == 1
                    else (left.type.numel,)
                )
            operation = ReshapeOp(operands=(left,), shape=shape, name_hint=name)
            expected = host[left]
        operations.append(operation)
        output = operation.results[0]
        values.append(output)
        host[output] = expected
    # Returning both an intermediate and the final result tests preservation
    # even when that intermediate's other consumers have already finished.
    intermediate = random.choice(values[2:-1])
    results = (inputs[0], intermediate, values[-1])
    graph = Function(
        name="main", operands=inputs, operations=tuple(operations), results=results
    )
    names = value_names(graph)
    expected_outputs = tuple(
        CentNamedScalars(name=names[value], values=host[value]) for value in results
    )
    return graph, bindings, expected_outputs


def _separate_storage(graph: Function) -> MappingPlan:
    """Give every tensor a separate named buffer without using mapper internals.

    Args:
        graph: Ordered tensor calculations; arguments and results are named
            as the IR printer names them.

    Returns:
        Manual mapping in graph order, with no storage reuse. This deliberately
        exercises copies that the generated mapping may safely eliminate. Each
        schedule entry is an operation's result name, which also names the
        device operation.
    """
    names = value_names(graph)
    return MappingPlan(
        placements=tuple(
            ValuePlacement(value=name, storage=f"buffer_{name}")
            for name in names.values()
        ),
        schedule=tuple(names[operation.results[0]] for operation in graph.operations),
    )


class MixedGraphOracleTests(unittest.TestCase):
    """Exercise operation composition against values independent of compiler passes."""

    def test_generated_and_manual_mappings_match_host_arithmetic(self) -> None:
        """Preserve outputs through mixed graphs with both reused and distinct buffers."""
        random = Random(42)
        target = _target()
        for case_index in range(_CASE_COUNT):
            graph, bindings, expected = _case(random, case_index)
            expected_types = tuple(value.type for value in graph.results)
            for mapping in (None, _separate_storage(graph)):
                with self.subTest(case=case_index, manual=mapping is not None):
                    artifact = compile_graph(graph, target, mapping=mapping)
                    self.assertEqual(artifact.execute(bindings), expected)
                    self.assertEqual(
                        tuple(value.type for value in artifact.outputs),
                        expected_types,
                    )
