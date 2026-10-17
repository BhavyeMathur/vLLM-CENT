"""Check supplied-query/key/value attention against independently evaluated equations."""

import math
import unittest
from dataclasses import replace
from random import Random

from vllm_cent.cent import CentHardwareSpec, MacAllBanks, render_text_program
from vllm_cent.cent.reference import ReferenceSoftmax
from vllm_cent.graphs import build_attention_graph
from vllm_cent.pipeline import (
    AccumulatorPlacement,
    CentTargetProfile,
    ExecutionContract,
    MappingPlan,
    ValuePlacement,
    compile_graph,
)
from vllm_cent.pipeline.ir import SemanticGraph, Value
from vllm_cent.pipeline.operations import AddOperation
from vllm_cent.runtime import CentNamedScalars
from vllm_cent.simulator import CentExecutionFault, execute_functionally


def _target(burst: int) -> CentTargetProfile:
    """Use short Global Buffer tiles and enough capacity for all supplied tensors.

    Args:
        burst: Scalar lanes per slot, varied to expose different tail boundaries.

    Returns:
        Explicit reference profile with a one-burst Global Buffer and four banks.
    """
    return CentTargetProfile(
        name="attention-oracle",
        execution_contract=ExecutionContract.REFERENCE,
        max_accumulate_slots=2,
        hardware=CentHardwareSpec(
            num_channels=2,
            num_banks=4,
            dram_rows=1,
            dram_columns=2 * burst,
            burst_length=burst,
            global_buffer_columns=burst,
            accumulator_slots_per_bank=2,
            sigmoid_activation_function_id=0,
            shared_buffer_slots=2048,
        ),
    )


def _manual(graph: SemanticGraph) -> MappingPlan:
    """Give each tensor separate storage and run attention on register one.

    Args:
        graph: One-operation attention graph with optional extra observed inputs.

    Returns:
        Valid complete mapping independent of the automatic mapper.
    """
    values = (*graph.inputs, graph.operations[0].output)
    return MappingPlan(
        placements=tuple(
            ValuePlacement(value=v.name, storage=f"buffer_{v.name}") for v in values
        ),
        schedule=("attention",),
        accumulators=(AccumulatorPlacement(operation="attention", register=1),),
    )


def _oracle(
    graph: SemanticGraph,
    data: dict[str, tuple[float, ...]],
    *,
    causal: bool,
    offset: int,
    scale: float | None,
) -> tuple[float, ...]:
    """Evaluate scaled attention directly without using compiler packing helpers.

    Args:
        graph: Declared shapes for query, key, value, and optional mask.
        data: Flattened values for those tensors in batch/head/sequence/lane order.
        causal: Whether each query may read only keys at or before its position.
        offset: Position of the first query in the supplied key sequence.
        scale: Multiplier for dot products, or ``None`` for reciprocal-root width.

    Returns:
        Flattened output retaining batch and query-head dimensions.
    """
    batch, heads, queries, width = graph.inputs[0].type.shape
    _, kv_heads, keys, _ = graph.inputs[1].type.shape
    value_width = graph.inputs[2].type.shape[-1]
    factor = 1 / math.sqrt(width) if scale is None else scale
    output: list[float] = []
    mask = data.get("mask")
    full_mask = len(graph.inputs) == 4 and len(graph.inputs[3].type.shape) == 4
    for b in range(batch):
        for h in range(heads):
            kv_head = h // (heads // kv_heads)
            for q in range(queries):
                query_start = ((b * heads + h) * queries + q) * width
                logits: list[float] = []
                for k in range(keys):
                    key_start = ((b * kv_heads + kv_head) * keys + k) * width
                    score = (
                        math.fsum(
                            data["query"][query_start + d] * data["key"][key_start + d]
                            for d in range(width)
                        )
                        * factor
                    )
                    if mask is not None:
                        mask_index = (
                            (((b * heads + h) * queries + q) * keys + k)
                            if full_mask
                            else q * keys + k
                        )
                        score += mask[mask_index]
                    if causal and k > offset + q:
                        score = -math.inf
                    logits.append(score)
                maximum = max(logits)
                numerators = [math.exp(logit - maximum) for logit in logits]
                denominator = math.fsum(numerators)
                for d in range(value_width):
                    output.append(
                        math.fsum(
                            numerators[k]
                            / denominator
                            * data["value"][
                                ((b * kv_heads + kv_head) * keys + k) * value_width + d
                            ]
                            for k in range(keys)
                        )
                    )
    return tuple(output)


class AttentionLayerTests(unittest.TestCase):
    """Exercise numerical attention, public shape contracts, masking, and reuse."""

    def test_builder_rejects_wrong_rank_before_output_inference(self) -> None:
        """An empty value shape reports a shape error rather than an index failure."""
        shapes = [(1, 1, 1, 2), (1, 1, 1, 2), (1, 1, 1, 3)]
        for index in range(3):
            invalid = shapes.copy()
            invalid[index] = ()
            with (
                self.subTest(index=index),
                self.assertRaisesRegex(ValueError, "rank four"),
            ):
                build_attention_graph(*invalid)

    def test_attention_matches_independent_oracle_across_heads_and_masks(self) -> None:
        """Exercise GQA, both mask layouts, rectangular causality, and non-square values."""
        random = Random(208)
        shapes = (
            ((1, 1, 1, 1), (1, 1, 1, 1), (1, 1, 1, 1)),
            ((2, 4, 3, 3), (2, 2, 5, 3), (2, 2, 5, 2)),
        )
        for query_shape, key_shape, value_shape in shapes:
            batch, heads, length, _ = query_shape
            keys = key_shape[2]
            for mask_mode, causal, offset, scale in (
                ("none", False, 0, None),
                ("shared", True, 0, 0.5),
                ("full", True, 2, None),
            ):
                mask_shape = (
                    None
                    if mask_mode == "none"
                    else (
                        (length, keys)
                        if mask_mode == "shared"
                        else (batch, heads, length, keys)
                    )
                )
                graph = build_attention_graph(
                    query_shape,
                    key_shape,
                    value_shape,
                    mask_shape=mask_shape,
                    causal=causal,
                    query_position_offset=offset,
                    scale=scale,
                )
                graph = replace(graph, outputs=("output", "query", "key", "value"))
                for burst, mapping in ((1, None), (2, _manual(graph)), (4, None)):
                    with self.subTest(shape=query_shape, mask=mask_mode, burst=burst):
                        artifact = compile_graph(graph, _target(burst), mapping=mapping)
                        for _ in range(2):
                            data = {
                                v.name: tuple(
                                    random.randrange(-6, 7) / 4
                                    for _ in range(v.type.numel)
                                )
                                for v in graph.inputs
                            }
                            if mask_shape is not None:
                                # Key zero stays available, so even the first
                                # causal query has a well-defined distribution.
                                data["mask"] = tuple(
                                    -math.inf
                                    if i % keys == keys - 1 and keys > 1
                                    else (i % 4) / 8
                                    for i in range(math.prod(mask_shape))
                                )
                            bindings = tuple(
                                CentNamedScalars(name=name, values=values)
                                for name, values in data.items()
                            )
                            actual = artifact.execute(bindings)
                            expected = _oracle(
                                graph, data, causal=causal, offset=offset, scale=scale
                            )
                            self.assertEqual(len(actual[0].values), len(expected))
                            for observed, wanted in zip(
                                actual[0].values, expected, strict=True
                            ):
                                self.assertTrue(
                                    math.isclose(
                                        observed, wanted, rel_tol=1e-12, abs_tol=1e-12
                                    ),
                                    (observed, wanted),
                                )
                            for output in actual[1:]:
                                self.assertEqual(output.values, data[output.name])
                            raw = execute_functionally(artifact.bind(bindings)).outputs
                            for value, output in zip(
                                artifact.outputs, raw, strict=True
                            ):
                                self.assertTrue(
                                    all(
                                        lane == 0
                                        for lane in output.values[value.type.numel :]
                                    )
                                )
                        instructions = artifact.executable.program.instructions
                        self.assertTrue(
                            any(isinstance(i, MacAllBanks) for i in instructions)
                        )
                        self.assertTrue(
                            any(isinstance(i, ReferenceSoftmax) for i in instructions)
                        )
                        with self.assertRaises((TypeError, ValueError)):
                            render_text_program(artifact.executable.program)

    def test_decode_offset_changes_visible_keys(self) -> None:
        """A single query sees key zero at offset zero and the whole prefix at offset two."""
        inputs = (
            CentNamedScalars(name="query", values=(0.0,)),
            CentNamedScalars(name="key", values=(1.0, 2.0, 3.0)),
            CentNamedScalars(name="value", values=(3.0, 6.0, 12.0)),
        )
        for offset, expected in ((0, 3.0), (2, 7.0)):
            graph = build_attention_graph(
                (1, 1, 1, 1),
                (1, 1, 3, 1),
                (1, 1, 3, 1),
                causal=True,
                query_position_offset=offset,
            )
            self.assertEqual(
                compile_graph(graph, _target(2)).execute(inputs),
                (CentNamedScalars(name="output", values=(expected,)),),
            )

    def test_all_masked_row_is_rejected(self) -> None:
        """Preserve the explicit softmax policy instead of silently producing zeros."""
        graph = build_attention_graph(
            (1, 1, 1, 1), (1, 1, 2, 1), (1, 1, 2, 1), mask_shape=(1, 2)
        )
        artifact = compile_graph(graph, _target(2))
        with self.assertRaises(CentExecutionFault):
            artifact.execute(
                (
                    CentNamedScalars(name="query", values=(1.0,)),
                    CentNamedScalars(name="key", values=(1.0, 2.0)),
                    CentNamedScalars(name="value", values=(3.0, 4.0)),
                    CentNamedScalars(name="mask", values=(-math.inf, -math.inf)),
                )
            )

    def test_attention_composes_with_producers_and_residual_consumers(self) -> None:
        """Consume a computed query and preserve attention output through an addition."""
        base = build_attention_graph((1, 2, 2, 3), (1, 1, 3, 3), (1, 1, 3, 3))
        query_type = base.inputs[0].type
        output_type = base.operations[0].output.type
        graph = SemanticGraph(
            inputs=(
                *base.inputs,
                Value("delta", query_type),
                Value("residual", output_type),
            ),
            operations=(
                AddOperation(
                    name="prepare_query",
                    inputs=("query", "delta"),
                    output=Value("prepared", query_type),
                ),
                replace(
                    base.operations[0],
                    inputs=("prepared", "key", "value"),
                    output=Value("attended", output_type),
                ),
                AddOperation(
                    name="add_residual",
                    inputs=("attended", "residual"),
                    output=Value("output", output_type),
                ),
            ),
            outputs=("output", "attended", "query"),
        )
        data = {
            v.name: tuple((i % 7 - 3) / 4 for i in range(v.type.numel))
            for v in graph.inputs
        }
        adjusted = dict(data)
        adjusted["query"] = tuple(
            a + b for a, b in zip(data["query"], data["delta"], strict=True)
        )
        attended = _oracle(base, adjusted, causal=False, offset=0, scale=None)
        expected = tuple(a + b for a, b in zip(attended, data["residual"], strict=True))
        artifact = compile_graph(graph, _target(2))
        outputs = artifact.execute(
            tuple(
                CentNamedScalars(name=name, values=values)
                for name, values in data.items()
            )
        )
        for output, wanted in zip(
            outputs, (expected, attended, data["query"]), strict=True
        ):
            for observed, reference in zip(output.values, wanted, strict=True):
                self.assertTrue(
                    math.isclose(observed, reference, rel_tol=1e-12, abs_tol=1e-12)
                )

    def test_builder_rejects_incompatible_shapes(self) -> None:
        """Reject mismatched batches, heads, widths, mask shapes, and invalid options."""
        q, k, v = (1, 4, 2, 3), (1, 2, 5, 3), (1, 2, 5, 2)
        for wrong_q, wrong_k, wrong_v in (
            (q, (2, 2, 5, 3), v),
            (q, (1, 3, 5, 3), v),
            (q, (1, 2, 5, 4), v),
            (q, k, (1, 2, 4, 2)),
        ):
            with (
                self.subTest(shapes=(wrong_q, wrong_k, wrong_v)),
                self.assertRaises(ValueError),
            ):
                build_attention_graph(wrong_q, wrong_k, wrong_v)
        for mask_shape in ((5,), (2, 4), (1, 1, 2, 5)):
            with self.subTest(mask=mask_shape), self.assertRaises(ValueError):
                build_attention_graph(q, k, v, mask_shape=mask_shape)
        for scale in (math.inf, math.nan):
            with self.subTest(scale=scale), self.assertRaises(ValueError):
                build_attention_graph(q, k, v, scale=scale)
        with self.assertRaises(ValueError):
            build_attention_graph(q, k, v, query_position_offset=-1)
