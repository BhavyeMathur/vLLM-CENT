"""Exercise one scratch reservation across alternating reference command families."""

import math
import unittest
from dataclasses import replace

from tests.test_pipeline_gemv_norm import target
from vllm_cent.cent import CentSharedBufferAddress
from vllm_cent.ir import (
    AttentionOp,
    DType,
    Function,
    LinearOp,
    RMSNormOp,
    TensorType,
    Value,
)
from vllm_cent.pipeline import compile_graph
from vllm_cent.pipeline.attention_codegen import (
    AttentionCommand,
    _attention_workspace,
    _AttentionWorkspace,
)
from vllm_cent.runtime import CentNamedScalars


def attention(
    query: tuple[float, ...], keys: tuple[float, ...], values: tuple[float, ...]
) -> tuple[float, ...]:
    """Evaluate two queries against three keys using an independent scalar oracle.

    Args:
        query: Two consecutive three-component query rows.
        keys: Three consecutive three-component key rows.
        values: Three consecutive three-component value rows.

    Returns:
        Two output rows, each containing three weighted value components.
    """
    result: list[float] = []
    for row in range(2):
        scores = [
            sum(
                query[row * 3 + component] * keys[key * 3 + component]
                for component in range(3)
            )
            / math.sqrt(3)
            for key in range(3)
        ]
        weights = [math.exp(score - max(scores)) for score in scores]
        total = sum(weights)
        result.extend(
            sum(weights[key] * values[key * 3 + component] / total for key in range(3))
            for component in range(3)
        )
    return tuple(result)


class WorkspaceCompositionTests(unittest.TestCase):
    """Keep intermediate outputs while repeatedly replacing shared scratch contents."""

    def test_attention_norm_linear_attention_share_only_scratch(self) -> None:
        """Different emitters reuse one region without clobbering preserved graph values."""
        query, key, value, scale, weight = (
            Value(type=TensorType(shape=shape, dtype=DType.FP32), name=name)
            for name, shape in (
                ("q", (1, 1, 2, 3)),
                ("k", (1, 1, 3, 3)),
                ("v", (1, 1, 3, 3)),
                # The learned norm scale takes the name the target would pick
                # first for its scratch region, so the target must pick another.
                ("__reference_workspace", (3,)),
                ("w", (3, 3)),
            )
        )
        first_attention = AttentionOp(operands=(query, key, value), name_hint="a")
        norm = RMSNormOp(
            operands=(first_attention.results[0], scale),
            eps=1e-5,
            name_hint="n",
        )
        linear = LinearOp(operands=(norm.results[0], weight), name_hint="z")
        last_attention = AttentionOp(
            operands=(linear.results[0], key, value), name_hint="out"
        )
        graph = Function(
            name="main",
            operands=(query, key, value, scale, weight),
            operations=(first_attention, norm, linear, last_attention),
            results=(
                first_attention.results[0],
                norm.results[0],
                linear.results[0],
                last_attention.results[0],
                scale,
                query,
            ),
        )
        q = (1.0, -2.0, 3.0, 2.0, 1.0, -1.0)
        keys = (1.0, 0.0, -1.0, 2.0, 1.0, 0.0, -1.0, 1.0, 2.0)
        values = (2.0, -1.0, 3.0, 1.0, 2.0, 4.0, -2.0, 3.0, 1.0)
        gamma = (1.0, 2.0, 0.5)
        weights = (1.0, 0.0, -1.0, 2.0, 1.0, 0.0, 0.0, 1.0, 1.0)
        first = attention(q, keys, values)
        normalized = tuple(
            first[row * 3 + c]
            * gamma[c]
            / math.sqrt(sum(x * x for x in first[row * 3 : row * 3 + 3]) / 3 + 1e-5)
            for row in range(2)
            for c in range(3)
        )
        linear = tuple(
            sum(normalized[row * 3 + c] * weights[out * 3 + c] for c in range(3))
            for row in range(2)
            for out in range(3)
        )
        expected = (
            first,
            normalized,
            linear,
            attention(linear, keys, values),
            gamma,
            q,
        )
        # Twenty input slots + twelve preserved result slots + twenty-two
        # temporary slots fit exactly; scratch is not allocated once per stage.
        profile = target()
        profile = replace(
            profile, hardware=replace(profile.hardware, shared_buffer_slots=54)
        )
        artifact = compile_graph(graph, profile)
        workspace = artifact.target_ir.reference_workspace
        self.assertIsNotNone(workspace)
        assert workspace is not None
        self.assertEqual(workspace.slot_count, 22)
        self.assertNotEqual(workspace.name, "__reference_workspace")
        first_command = artifact.target_ir.commands[0]
        assert isinstance(first_command, AttentionCommand)
        # At two lanes per slot, the logical scratch sizes 3,9,9,3,3
        # occupy 2,5,5,2,2 slots. Six matrix slots finish the 22-slot region.
        self.assertEqual(
            _attention_workspace(
                first_command, profile.hardware, CentSharedBufferAddress(slot=7)
            ),
            _AttentionWorkspace(
                query=CentSharedBufferAddress(slot=7),
                key=CentSharedBufferAddress(slot=9),
                value_transpose=CentSharedBufferAddress(slot=14),
                scores=CentSharedBufferAddress(slot=19),
                result=CentSharedBufferAddress(slot=21),
                matrix=CentSharedBufferAddress(slot=23),
                slot_count=22,
            ),
        )
        inputs = tuple(
            CentNamedScalars(name=name, values=data)
            for name, data in (
                ("q", q),
                ("k", keys),
                ("v", values),
                ("__reference_workspace", gamma),
                ("w", weights),
            )
        )
        for output, wanted in zip(artifact.execute(inputs), expected, strict=True):
            for actual, value in zip(output.values, wanted, strict=True):
                self.assertAlmostEqual(actual, value, places=12)
        with self.assertRaisesRegex(ValueError, "workspace exceed"):
            compile_graph(
                graph,
                replace(
                    profile, hardware=replace(profile.hardware, shared_buffer_slots=53)
                ),
            )
