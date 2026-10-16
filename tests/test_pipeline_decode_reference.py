"""Check normalization and projections together against independent host math."""

import math
import unittest
from random import Random

from vllm_cent.cent import CentHardwareSpec
from vllm_cent.pipeline import (
    AccumulatorPlacement,
    CentTargetProfile,
    ExecutionContract,
    MappingPlan,
    ValuePlacement,
    compile_graph,
)
from vllm_cent.pipeline.ir import SemanticGraph, TensorType, Value
from vllm_cent.pipeline.operations import (
    AddOperation,
    LinearOperation,
    MultiplyOperation,
    RmsNormOperation,
)
from vllm_cent.runtime import CentNamedScalars
from vllm_cent.simulator import execute_functionally


def _target(burst: int, banks: int) -> CentTargetProfile:
    """Make small rows that force projections to continue across input tiles.

    Args:
        burst: Scalar lanes transferred together in each Shared Buffer slot.
        banks: Number of parallel weight rows available in one channel.

    Returns:
        Explicit reference target with two bursts per DRAM row and two registers.
    """
    return CentTargetProfile(
        name="decode-reference-oracle",
        hardware=CentHardwareSpec(
            num_channels=2,
            num_banks=banks,
            dram_rows=2,
            dram_columns=burst * 2,
            burst_length=burst,
            accumulator_slots_per_bank=2,
            sigmoid_activation_function_id=0,
            shared_buffer_slots=1024,
            global_buffer_columns=burst * 2,
        ),
        max_accumulate_slots=2,
        execution_contract=ExecutionContract.REFERENCE,
    )


def _graph(rows: int, width: int, outputs: int) -> SemanticGraph:
    """Describe normalization, projection, a residual addition, and squaring.

    Args:
        rows: Number of independent input vectors in the batch.
        width: Input width, also the learned normalization scale length.
        outputs: Number of projection weight rows and output columns.

    Returns:
        Graph returning its input and normalization intermediate as well as the
        final result, so later destructive arithmetic must preserve them.
    """
    source = TensorType((rows, width))
    projected = TensorType((rows, outputs))
    return SemanticGraph(
        inputs=(
            Value("x", source),
            Value("scale", TensorType((width,))),
            Value("weights", TensorType((outputs, width))),
            Value("residual", projected),
        ),
        operations=(
            RmsNormOperation(
                name="normalize",
                inputs=("x", "scale"),
                output=Value("normalized", source),
                epsilon=0.125,
            ),
            LinearOperation(
                name="project",
                inputs=("normalized", "weights"),
                output=Value("projection", projected),
            ),
            AddOperation(
                name="add",
                inputs=("projection", "residual"),
                output=Value("sum", projected),
            ),
            MultiplyOperation(
                name="square",
                inputs=("sum", "sum"),
                output=Value("result", projected),
            ),
        ),
        outputs=("result", "x", "normalized"),
    )


def _manual(graph: SemanticGraph) -> MappingPlan:
    """Place each tensor separately and explicitly use the second accumulator.

    Args:
        graph: Graph with a projection named ``project``.

    Returns:
        Caller-written plan using register one, independent of baseline choices.
    """
    values = (*graph.inputs, *(op.output for op in graph.operations))
    return MappingPlan(
        placements=tuple(
            ValuePlacement(value=value.name, storage=f"buffer_{value.name}")
            for value in values
        ),
        schedule=tuple(op.name for op in graph.operations),
        accumulators=(AccumulatorPlacement(operation="project", register=1),),
    )


class DecodeReferenceTests(unittest.TestCase):
    """Exercise composed dataflow, tails, preserved outputs, and reusable artifacts."""

    def test_composed_graph_matches_host_across_shapes_and_mappings(self) -> None:
        """Compare both mappings with row-wise host math and check physical padding."""
        random = Random(91)
        # Width seven crosses two-value and four-value scratch rows. Nine
        # output columns cross a four-bank group and leave inactive banks.
        cases = ((1, 1, 1), (2, 3, 5), (3, 7, 9))
        for burst, banks in ((1, 4), (2, 4), (4, 8)):
            for rows, width, columns in cases:
                graph = _graph(rows, width, columns)
                for mapping in (None, _manual(graph)):
                    with self.subTest(
                        burst=burst,
                        banks=banks,
                        shape=(rows, width, columns),
                        manual=mapping is not None,
                    ):
                        artifact = compile_graph(
                            graph, _target(burst, banks), mapping=mapping
                        )
                        # Rebind all inputs, including weights, without compiling again.
                        # The second run includes a zero row to test epsilon handling.
                        for run in range(2):
                            host = {
                                value.name: tuple(
                                    random.randrange(-8, 9) / 4
                                    for _ in range(value.type.numel)
                                )
                                for value in graph.inputs
                            }
                            if run:
                                host["x"] = (0.0,) * width + host["x"][width:]
                            normalized: list[float] = []
                            expected: list[float] = []
                            for row in range(rows):
                                x = host["x"][row * width : (row + 1) * width]
                                denominator = math.sqrt(
                                    math.fsum(v * v for v in x) / width + 0.125
                                )
                                norm = tuple(
                                    v * scale / denominator
                                    for v, scale in zip(x, host["scale"], strict=True)
                                )
                                normalized.extend(norm)
                                for column in range(columns):
                                    weight = host["weights"][
                                        column * width : (column + 1) * width
                                    ]
                                    projection = math.fsum(
                                        a * b for a, b in zip(norm, weight, strict=True)
                                    )
                                    expected.append(
                                        (
                                            projection
                                            + host["residual"][row * columns + column]
                                        )
                                        ** 2
                                    )
                            bindings = tuple(
                                CentNamedScalars(name=name, values=values)
                                for name, values in host.items()
                            )
                            actual = artifact.execute(bindings)
                            for output, wanted in zip(
                                actual, (expected, host["x"], normalized), strict=True
                            ):
                                self.assertEqual(len(output.values), len(wanted))
                                for observed, reference in zip(
                                    output.values, wanted, strict=True
                                ):
                                    self.assertTrue(
                                        math.isclose(
                                            observed,
                                            reference,
                                            rel_tol=1e-12,
                                            abs_tol=1e-12,
                                        ),
                                        (observed, reference),
                                    )
                            # Inspect the raw runtime outputs as well: logical slicing
                            # must not hide dirty padding left by a producer.
                            raw = execute_functionally(artifact.bind(bindings)).outputs
                            for value, output in zip(
                                artifact.outputs, raw, strict=True
                            ):
                                self.assertTrue(
                                    all(
                                        lane == 0.0
                                        for lane in output.values[value.type.numel :]
                                    )
                                )
