"""Compare mixed tensor graphs with arithmetic evaluated independently on the host."""

import unittest
from random import Random

from vllm_cent.cent import CentHardwareSpec
from vllm_cent.pipeline import (
    CentTargetProfile,
    MappingPlan,
    ValuePlacement,
    compile_graph,
)
from vllm_cent.pipeline.ir import SemanticGraph, TensorType, Value
from vllm_cent.pipeline.operations import (
    AddOperation,
    CopyOperation,
    MultiplyOperation,
    Operation,
    ReshapeOperation,
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
) -> tuple[SemanticGraph, tuple[CentNamedScalars, ...], tuple[CentNamedScalars, ...]]:
    """Build a bounded graph while separately calculating each tensor's values.

    Args:
        random: Seeded generator choosing operands, output names, and input data.
        case_index: Selects the input shape and rotates the operation sequence.

    Returns:
        Graph, caller input values, and independently calculated requested outputs.
        The first input is always returned so its original contents must survive.
    """
    # Cycling, rather than randomly sampling, guarantees scalar, odd-tail, and
    # multi-row coverage. A scalar and a one-element vector can reshape into
    # each other without changing their one flattened value.
    shapes = ((), (1,), (3,), (5,), (6,), (8,))
    tensor_type = TensorType(shapes[case_index % len(shapes)])
    inputs = (Value("x", tensor_type), Value("y", tensor_type))
    values = list(inputs)
    host = {
        name: tuple(float(random.randrange(-2, 3)) for _ in range(tensor_type.numel))
        for name in ("x", "y")
    }
    bindings = tuple(
        CentNamedScalars(name=value.name, values=host[value.name]) for value in inputs
    )
    operations: list[Operation] = []
    choices = (AddOperation, MultiplyOperation, CopyOperation, ReshapeOperation)
    for index in range(_OPERATION_COUNT):
        # Every graph includes each operation twice. Earlier results remain
        # eligible as operands, creating branches and longer-lived values.
        operation_type = choices[(index + case_index) % len(choices)]
        left = random.choice(values)
        operand_names = (left.name,)
        output_type = left.type
        if operation_type in (AddOperation, MultiplyOperation):
            right = random.choice(
                [value for value in values if value.type == left.type]
            )
            operand_names += (right.name,)
            # The oracle uses ordinary host arithmetic and no compiler layouts,
            # instruction helpers, or functional-simulator kernels.
            if operation_type is AddOperation:
                expected = tuple(
                    a + b
                    for a, b in zip(host[left.name], host[right.name], strict=True)
                )
            else:
                expected = tuple(
                    a * b
                    for a, b in zip(host[left.name], host[right.name], strict=True)
                )
        else:
            expected = host[left.name]
            if operation_type is ReshapeOperation:
                if left.type.numel == 1:
                    shape = (1,) if left.type.shape == () else ()
                else:
                    shape = (
                        (1, left.type.numel)
                        if len(left.type.shape) == 1
                        else (left.type.numel,)
                    )
                output_type = TensorType(shape)
        output = Value(f"value{index}", output_type)
        operations.append(
            operation_type(
                name=f"operation{index}", inputs=operand_names, output=output
            )
        )
        values.append(output)
        host[output.name] = expected
    # Returning both an intermediate and the final result tests preservation
    # even when that intermediate's other consumers have already finished.
    intermediate = random.choice(values[2:-1]).name
    output_names = ("x", intermediate, values[-1].name)
    graph = SemanticGraph(
        inputs=inputs, operations=tuple(operations), outputs=output_names
    )
    expected_outputs = tuple(
        CentNamedScalars(name=name, values=host[name]) for name in output_names
    )
    return graph, bindings, expected_outputs


def _separate_storage(graph: SemanticGraph) -> MappingPlan:
    """Give every tensor a separate named buffer without using mapper internals.

    Args:
        graph: Ordered tensor calculations with all inputs and results named.

    Returns:
        Manual mapping in graph order, with no storage reuse. This deliberately
        exercises copies that the generated mapping may safely eliminate.
    """
    values = (*graph.inputs, *(operation.output for operation in graph.operations))
    return MappingPlan(
        placements=tuple(
            ValuePlacement(value=value.name, storage=f"buffer_{value.name}")
            for value in values
        ),
        schedule=tuple(operation.name for operation in graph.operations),
    )


class MixedGraphOracleTests(unittest.TestCase):
    """Exercise operation composition against values independent of compiler passes."""

    def test_generated_and_manual_mappings_match_host_arithmetic(self) -> None:
        """Preserve outputs through mixed graphs with both reused and distinct buffers."""
        random = Random(42)
        target = _target()
        for case_index in range(_CASE_COUNT):
            graph, bindings, expected = _case(random, case_index)
            value_types = {value.name: value.type for value in graph.inputs}
            value_types.update(
                (op.output.name, op.output.type) for op in graph.operations
            )
            expected_types = tuple(value_types[name] for name in graph.outputs)
            for mapping in (None, _separate_storage(graph)):
                with self.subTest(case=case_index, manual=mapping is not None):
                    artifact = compile_graph(graph, target, mapping=mapping)
                    self.assertEqual(artifact.execute(bindings), expected)
                    self.assertEqual(
                        tuple(value.type for value in artifact.outputs),
                        expected_types,
                    )
