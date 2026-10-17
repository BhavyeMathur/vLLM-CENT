"""Check a complete Llama feed-forward graph against independent host equations."""

import math
import unittest
from random import Random

from vllm_cent.cent import CentHardwareSpec
from vllm_cent.ir import Function, LinearOp, value_names
from vllm_cent.models.llama import LlamaModelSpec, build_feed_forward_graph
from vllm_cent.pipeline import (
    AccumulatorPlacement,
    CentTargetProfile,
    ExecutionContract,
    MappingPlan,
    ValuePlacement,
    compile_graph,
)
from vllm_cent.runtime import CentNamedScalars
from vllm_cent.simulator import execute_functionally


def _target() -> CentTargetProfile:
    """Provide small rows and two registers for a tiled feed-forward calculation.

    Returns:
        Explicit reference target with enough space for preserved intermediate outputs.
    """
    return CentTargetProfile(
        name="feed-forward-reference",
        execution_contract=ExecutionContract.REFERENCE,
        max_accumulate_slots=2,
        hardware=CentHardwareSpec(
            num_channels=1,
            num_banks=4,
            dram_rows=1,
            dram_columns=4,
            burst_length=2,
            accumulator_slots_per_bank=2,
            sigmoid_activation_function_id=0,
            shared_buffer_slots=512,
        ),
    )


def _with_results(graph: Function, *names: str) -> Function:
    """Return the same calculation with a different list of returned values.

    Args:
        graph: Function whose arguments and operations are reused unchanged.
        names: Printed names of the values to return, in result order. Each
            may name an argument or an operation result.

    Returns:
        Function sharing ``graph``'s values and operations, returning ``names``.
    """
    by_name = {name: value for value, name in value_names(graph).items()}
    return Function(
        name=graph.name,
        operands=graph.operands,
        operations=graph.operations,
        results=tuple(by_name[name] for name in names),
    )


def _manual(graph: Function) -> MappingPlan:
    """Assign distinct buffers and a nonzero register without baseline mapper helpers.

    Device operations are named after their results, so the schedule and the
    accumulator choices list result names such as ``feed_forward.gate``.

    Args:
        graph: Feed-forward operations in dependency order.

    Returns:
        Complete mapping that deliberately forgoes buffer reuse.
    """
    names = value_names(graph)
    return MappingPlan(
        placements=tuple(
            ValuePlacement(value=name, storage=f"buffer_{name}")
            for name in names.values()
        ),
        schedule=tuple(names[op.results[0]] for op in graph.operations),
        accumulators=tuple(
            AccumulatorPlacement(operation=names[op.results[0]], register=1)
            for op in graph.operations
            if isinstance(op, LinearOp)
        ),
    )


def _host_feed_forward(
    inputs: dict[str, tuple[float, ...]],
    hidden: int,
    intermediate: int,
    epsilon: float,
) -> tuple[float, ...]:
    """Calculate each row with Python equations independent of simulator services.

    Args:
        inputs: Flattened graph inputs, including row-major projection matrices.
        hidden: Input/output vector width.
        intermediate: Gate/up projection width.
        epsilon: Positive constant added to each mean square.

    Returns:
        Flattened residual output in input-row order.
    """
    result: list[float] = []
    for start in range(0, len(inputs["x"]), hidden):
        x = inputs["x"][start : start + hidden]
        denominator = math.sqrt(math.fsum(v * v for v in x) / hidden + epsilon)
        normalized = tuple(
            v * weight / denominator
            for v, weight in zip(x, inputs["norm_weight"], strict=True)
        )
        gated: list[float] = []
        for row in range(intermediate):
            gate = math.fsum(
                normalized[k] * inputs["gate_weight"][row * hidden + k]
                for k in range(hidden)
            )
            up = math.fsum(
                normalized[k] * inputs["up_weight"][row * hidden + k]
                for k in range(hidden)
            )
            # Test values stay moderate, so this independent direct sigmoid
            # formula does not share the simulator's stability branches.
            gated.append(gate / (1 + math.exp(-gate)) * up)
        for row in range(hidden):
            projection = math.fsum(
                gated[k] * inputs["down_weight"][row * intermediate + k]
                for k in range(intermediate)
            )
            result.append(x[row] + projection)
    return tuple(result)


class LlamaSemanticGraphTests(unittest.TestCase):
    """Check model composition, repeated data binding, liveness, and shape boundaries."""

    def test_feed_forward_matches_host_with_both_mappings(self) -> None:
        """Keep the residual and branching normalized input alive across projections."""
        random = Random(135)
        for hidden, intermediate, batch in ((1, 1, ()), (3, 5, (2,)), (5, 9, (2, 2))):
            model = LlamaModelSpec(
                hidden_size=hidden,
                num_attention_heads=1,
                num_kv_heads=1,
                intermediate_size=intermediate,
            )
            graph = build_feed_forward_graph(model, epsilon=0.125, batch_shape=batch)
            # Expose intermediate outputs to make storage preservation observable.
            # The default block label prefixes every result name.
            graph = _with_results(
                graph, "feed_forward.output", "x", "feed_forward.normalized"
            )
            for mapping in (None, _manual(graph)):
                with self.subTest(
                    shape=(hidden, intermediate, batch), manual=mapping is not None
                ):
                    artifact = compile_graph(graph, _target(), mapping=mapping)
                    for run in range(2):
                        inputs = {
                            v.name: tuple(
                                random.randrange(-4, 5) / 4 for _ in range(v.type.numel)
                            )
                            for v in graph.operands
                        }
                        if run:
                            inputs["x"] = (0.0,) * hidden + inputs["x"][hidden:]
                        bindings = tuple(
                            CentNamedScalars(name=name, values=values)
                            for name, values in inputs.items()
                        )
                        result = artifact.execute(bindings)
                        expected = _host_feed_forward(
                            inputs, hidden, intermediate, 0.125
                        )
                        for observed, wanted in zip(
                            result[0].values, expected, strict=True
                        ):
                            self.assertTrue(
                                math.isclose(
                                    observed, wanted, rel_tol=1e-12, abs_tol=1e-12
                                ),
                                (observed, wanted),
                            )
                        self.assertEqual(result[1].values, inputs["x"])
                        raw = execute_functionally(artifact.bind(bindings)).outputs
                        for value, output in zip(artifact.outputs, raw, strict=True):
                            self.assertTrue(
                                all(
                                    lane == 0.0
                                    for lane in output.values[value.type.numel :]
                                )
                            )

    def test_frontend_is_independent_of_hardware_and_checks_inputs(self) -> None:
        """Require valid shapes and epsilon before any target is selected."""
        model = LlamaModelSpec(
            hidden_size=3, num_attention_heads=1, num_kv_heads=1, intermediate_size=5
        )
        graph = build_feed_forward_graph(model, epsilon=1e-5, block="decoder.ffn")
        graph.verify()
        self.assertEqual(
            tuple(v.name for v in graph.operands),
            ("x", "norm_weight", "gate_weight", "up_weight", "down_weight"),
        )
        self.assertEqual(
            [v.type.shape for v in graph.operands],
            [(3,), (3,), (5, 3), (5, 3), (3, 5)],
        )
        # One operation per step of x + down(SiLU(gate(n)) * up(n)), n = RMSNorm(x).
        # Each result is named under the block label, as a module path would be.
        self.assertEqual(
            tuple(op.name_hint for op in graph.operations),
            (
                "decoder.ffn.normalized",
                "decoder.ffn.gate",
                "decoder.ffn.up",
                "decoder.ffn.activated_gate",
                "decoder.ffn.gated",
                "decoder.ffn.projected",
                "decoder.ffn.output",
            ),
        )
        names = value_names(graph)
        self.assertEqual(
            tuple(names[value] for value in graph.results), ("decoder.ffn.output",)
        )
        for epsilon in (0.0, -1.0, math.inf, math.nan):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                build_feed_forward_graph(model, epsilon=epsilon)
        with self.assertRaises(ValueError):
            build_feed_forward_graph(model, epsilon=1e-5, batch_shape=(0,))
