"""Check supplied-tensor attention against direct row-by-row host arithmetic."""

import math
import unittest
from dataclasses import replace

from vllm_cent.pipeline import AccumulatorPlacement, ExecutionContract, compile_graph
from vllm_cent.pipeline.ir import SemanticGraph, TensorType, Value
from vllm_cent.pipeline.operations import AttentionOperation
from vllm_cent.runtime import CentNamedScalars
from tests.test_pipeline_gemv_norm import target


def attention_graph(
    *, causal: bool = False, offset: int = 0, mask: bool = False
) -> SemanticGraph:
    """Build two query heads sharing one key/value head, with different value width.

    Args:
        causal: Whether each query can read only keys through its position.
        offset: Position assigned to the first query.
        mask: Whether to include a separate full query-head mask.

    Returns:
        Static graph retaining all original operands as observable outputs.
    """
    inputs = (
        Value("q", TensorType((1, 2, 2, 3))),
        Value("k", TensorType((1, 1, 3, 3))),
        Value("v", TensorType((1, 1, 3, 2))),
    )
    if mask:
        inputs += (Value("mask", TensorType((1, 2, 2, 3))),)
    return SemanticGraph(
        inputs=inputs,
        operations=(
            AttentionOperation(
                name="attention",
                inputs=tuple(value.name for value in inputs),
                output=Value("out", TensorType((1, 2, 2, 2))),
                causal=causal,
                query_position_offset=offset,
            ),
        ),
        outputs=("out", *(value.name for value in inputs)),
    )


class AttentionTests(unittest.TestCase):
    """Cover grouped heads, tail widths, masks, and explicit accumulator placement."""

    def test_gqa_causal_mask_and_register_selection(self) -> None:
        """Each head reads the shared KV head but its own query and additive-mask row."""
        q = tuple(float(i % 5 - 2) for i in range(12))
        k = tuple(float(i % 4 - 1) for i in range(9))
        v = (2.0, 3.0, -1.0, 4.0, 5.0, -2.0)
        mask = (0.0, -math.inf, 0.0, 0.0, 0.0, 0.0, -math.inf, 0.0, 0.0, 0.0, 1.0, 0.0)
        graph = attention_graph(causal=True, offset=1, mask=True)
        artifact = compile_graph(graph, target())
        artifact = compile_graph(
            graph,
            target(),
            mapping=replace(
                artifact.target_ir.mapping,
                accumulators=(AccumulatorPlacement(operation="attention", register=1),),
            ),
        )
        inputs = tuple(
            CentNamedScalars(name=name, values=values)
            for name, values in (("q", q), ("k", k), ("v", v), ("mask", mask))
        )
        actual = artifact.execute(inputs)
        expected = []
        for head in range(2):
            for query in range(2):
                logits = []
                for key in range(3):
                    dot = sum(
                        q[(head * 2 + query) * 3 + col] * k[key * 3 + col]
                        for col in range(3)
                    )
                    logits.append(
                        dot / math.sqrt(3) + mask[(head * 2 + query) * 3 + key]
                        if key <= query + 1
                        else -math.inf
                    )
                weights = [math.exp(value - max(logits)) for value in logits]
                total = sum(weights)
                expected.extend(
                    sum(weights[key] / total * v[key * 2 + col] for key in range(3))
                    for col in range(2)
                )
        for value, wanted in zip(actual[0].values, expected, strict=True):
            self.assertAlmostEqual(value, wanted)
        self.assertEqual(actual[1:], inputs)

    def test_validation_gate_and_alias(self) -> None:
        """Invalid grouped-head geometry and missing reference opt-in fail before emission."""
        graph = attention_graph()
        with self.assertRaisesRegex(ValueError, "REFERENCE"):
            compile_graph(
                graph, replace(target(), execution_contract=ExecutionContract.STANDARD)
            )
        operation = graph.operations[0]
        with self.assertRaises(ValueError):
            operation.verify(
                (
                    TensorType((1, 3, 2, 3)),
                    TensorType((1, 2, 3, 3)),
                    TensorType((1, 2, 3, 2)),
                )
            )
        for changes in ({"scale": math.nan}, {"query_position_offset": -1}):
            with self.assertRaises(ValueError):
                replace(operation, **changes)

    def test_non_square_causal_prefix_requires_explicit_offset(self) -> None:
        """One query sees key zero unless its supplied position admits later keys."""
        from vllm_cent.pipeline import ValuePlacement

        inputs = (
            Value("q", TensorType((1, 1, 1, 1))),
            Value("k", TensorType((1, 1, 5, 1))),
            Value("v", TensorType((1, 1, 5, 1))),
        )
        graph = SemanticGraph(
            inputs=inputs,
            operations=(
                AttentionOperation(
                    name="attention",
                    inputs=("q", "k", "v"),
                    output=Value("out", inputs[0].type),
                    causal=True,
                    scale=0.0,
                ),
            ),
            outputs=("out",),
        )
        values = (
            CentNamedScalars(name="q", values=(2.0,)),
            CentNamedScalars(name="k", values=(1.0, 2.0, 3.0, 4.0, 5.0)),
            CentNamedScalars(name="v", values=(1.0, 2.0, 3.0, 4.0, 5.0)),
        )
        for offset, expected in ((0, 1.0), (4, 3.0), (7, 3.0)):
            positioned = replace(
                graph,
                operations=(
                    replace(graph.operations[0], query_position_offset=offset),
                ),
            )
            self.assertAlmostEqual(
                compile_graph(positioned, target()).execute(values)[0].values[0],
                expected,
            )
        # The output fits the query's slot, but attention reads queries repeatedly.
        with self.assertRaisesRegex(ValueError, "live"):
            compile_graph(
                graph,
                target(),
                constraints=(
                    ValuePlacement(value="q", storage="same"),
                    ValuePlacement(value="out", storage="same"),
                ),
            )

    def test_workspace_capacity_and_mask_shapes(self) -> None:
        """Nested GEMV scratch counts toward capacity and unsupported masks fail early."""
        from vllm_cent.pipeline import compile_target

        graph = attention_graph()
        # Graph tensors occupy 18 slots; one KV head plus row and MAC work need 19.
        device = replace(
            target(), hardware=replace(target().hardware, shared_buffer_slots=37)
        )
        artifact = compile_graph(graph, device)
        self.assertEqual(artifact.target_ir.reference_workspace.slot_count, 19)
        with self.assertRaisesRegex(ValueError, "capacity"):
            compile_graph(
                graph,
                replace(
                    device, hardware=replace(device.hardware, shared_buffer_slots=36)
                ),
            )
        with self.assertRaises(ValueError):
            compile_target(replace(artifact.target_ir, reference_workspace=None))
        operation = graph.operations[0]
        types = tuple(value.type for value in graph.inputs)
        for invalid in (
            (),
            types[:2],
            (*types, TensorType((3,))),
            (TensorType((2, 3)), *types[1:]),
        ):
            with self.subTest(types=invalid), self.assertRaises(ValueError):
                operation.verify(invalid)
