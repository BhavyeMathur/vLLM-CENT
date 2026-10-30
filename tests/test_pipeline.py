"""Run compiled graphs and test their storage, padding, and input checks."""

import unittest
from dataclasses import replace
from random import Random
from unittest.mock import Mock

from vllm_cent.cent import Accumulate, CentHardwareSpec
from vllm_cent.ir import AddOp, DType, Function, TensorType, Value
from vllm_cent.pipeline import (
    CentTargetProfile,
    MappingPlan,
    MappingRequest,
    ValuePlacement,
    baseline_mapping,
    compile_graph,
    lower_target,
    verify_mapping,
)
from vllm_cent.pipeline.ir import (
    CapabilityProfile,
    NameValue,
    legalize,
)
from vllm_cent.runtime import CentNamedScalars


def argument(name: str, shape: tuple[int, ...]) -> Value:
    """Build a named fp32 function argument.

    Args:
        name: Argument name, which also names the compiled program's input.
        shape: Tensor dimensions, outermost first.

    Returns:
        A new value suitable as a function operand.
    """
    return Value(type=TensorType(shape=shape, dtype=DType.FP32), name=name)


def graph(*, fanout: bool = False, shape: tuple[int, ...] = (3,)) -> Function:
    """Build z = x + y followed by out = z + y.

    Args:
        fanout: If True, also return x so its original contents must survive
            both additions. Otherwise x can be overwritten after its last read.
        shape: Dimensions shared by every tensor. The default three-element
            vector needs one trailing padding lane with two lanes per slot.

    Returns:
        Two-addition function returning out, and optionally x. Its operations
        and results are named ``z`` and ``out`` after their name hints.
    """
    x = argument("x", shape)
    y = argument("y", shape)
    first = AddOp(operands=(x, y), name_hint="z")
    second = AddOp(operands=(first.results[0], y), name_hint="out")
    return Function(
        name="main",
        operands=(x, y),
        operations=(first, second),
        results=(second.results[0], x) if fanout else second.results,
    )


def target(*, tile: int = 2, capacity: int = 16) -> CentTargetProfile:
    """Build a small reference target.

    Args:
        tile: Maximum slots per emitted ACC command.
        capacity: Available Shared Buffer slots.

    Returns:
        Target with two lanes per slot and four slots per DRAM row.
    """
    return CentTargetProfile(
        name="tiny",
        hardware=CentHardwareSpec(
            num_channels=1,
            num_banks=4,
            dram_rows=2,
            dram_columns=8,
            burst_length=2,
            accumulator_slots_per_bank=2,
            sigmoid_activation_function_id=0,
            shared_buffer_slots=capacity,
        ),
        max_accumulate_slots=tile,
    )


def request(*, fanout: bool = False) -> MappingRequest:
    """Build the device graph and target needed to test mapping independently.

    Args:
        fanout: Whether the graph also returns x, requiring its original contents
            to remain available through execution.

    Returns:
        Single-device mapping request for graph() on the small test target.
    """
    device = legalize(graph(fanout=fanout), CapabilityProfile())
    return MappingRequest(graph=device, target=target())


class PipelineTests(unittest.TestCase):
    """Exercise whole-graph planning, code generation, and request reuse."""

    def test_add_chain_reuses_dead_left_and_preserves_padding(self) -> None:
        """Two adds need two allocations and no copy when x dies at first use."""
        artifact = compile_graph(graph(), target())
        # x becomes z, then out, in the same buffer. Only the two ACC commands
        # remain because neither addition needs a preservation copy.
        self.assertEqual(len(artifact.executable.program.instructions), 2)
        self.assertTrue(
            all(
                isinstance(i, Accumulate)
                for i in artifact.executable.program.instructions
            )
        )
        values = (
            CentNamedScalars(name="x", values=(1.0, 2.0, 3.0)),
            CentNamedScalars(name="y", values=(10.0, 20.0, 30.0)),
        )
        self.assertEqual(
            artifact.execute(values),
            (CentNamedScalars(name="out", values=(21.0, 42.0, 63.0)),),
        )
        # Three logical values fill two two-lane slots, so the fourth lane is zero.
        packed = artifact.bind(values)
        self.assertEqual(packed.inputs[0].values, (1.0, 2.0, 3.0, 0.0))
        self.assertEqual(artifact.execute(values), artifact.execute(values))

    def test_fanout_copies_before_accumulating(self) -> None:
        """Preserve an input requested as output by performing addition in another buffer."""
        artifact = compile_graph(graph(fanout=True), target())
        values = (
            CentNamedScalars(name="x", values=(1.0, -2.0, 3.0)),
            CentNamedScalars(name="y", values=(2.0, 3.0, -4.0)),
        )
        self.assertEqual(
            artifact.execute(values),
            (
                CentNamedScalars(name="out", values=(5.0, 4.0, -5.0)),
                CentNamedScalars(name="x", values=(1.0, -2.0, 3.0)),
            ),
        )
        # Preserving x adds one DRAM write/read pair to the two ACC commands.
        self.assertEqual(len(artifact.executable.program.instructions), 4)

    def test_target_tiling_and_manual_mapping_match(self) -> None:
        """A stricter target splits two slots into separate ACC commands."""
        # IR functions compare by identity, and an artifact keeps its source
        # function, so both compilations must start from the same function.
        source = graph()
        original = compile_graph(source, target())
        tiled = compile_graph(source, target(tile=1))
        self.assertEqual(len(tiled.executable.program.instructions), 4)
        req = request()
        manual = baseline_mapping(req)
        self.assertEqual(compile_graph(source, target(), mapping=manual), original)

    def test_invalid_mapping_is_rejected(self) -> None:
        """Reject plans that overwrite a needed input, omit storage, or run uses first."""
        req = request(fanout=True)
        plan = baseline_mapping(req)
        bad = tuple(
            replace(p, storage="x") if p.value == "z" else p for p in plan.placements
        )
        with self.assertRaisesRegex(ValueError, "live|alias|storage"):
            verify_mapping(req, replace(plan, placements=bad))
        with self.assertRaisesRegex(ValueError, "placement"):
            verify_mapping(req, replace(plan, placements=plan.placements[:-1]))
        with self.assertRaisesRegex(ValueError, "schedule"):
            verify_mapping(req, replace(plan, schedule=tuple(reversed(plan.schedule))))

    def test_capacity_and_bindings_fail_before_execution(self) -> None:
        """Small targets and malformed logical inputs fail explicitly."""
        with self.assertRaisesRegex(ValueError, "capacity"):
            compile_graph(graph(), target(capacity=3))
        artifact = compile_graph(graph(), target())
        with self.assertRaisesRegex(ValueError, "input"):
            artifact.bind(())
        with self.assertRaisesRegex(ValueError, "count"):
            artifact.bind(
                (
                    CentNamedScalars(name="x", values=(1.0,)),
                    CentNamedScalars(name="y", values=(1.0, 2.0, 3.0)),
                )
            )

    def test_target_ir_has_symbolic_storage(self) -> None:
        """Keep buffer names in target commands until a later pass assigns slot addresses."""
        req = request()
        plan = baseline_mapping(req)
        symbolic = lower_target(req, plan)
        self.assertEqual(symbolic.mapping, plan)
        self.assertEqual(symbolic.commands[0].destination, "x")
        self.assertIsInstance(plan, MappingPlan)
        self.assertIsInstance(plan.placements[0], ValuePlacement)


class MappingBoundaryTests(unittest.TestCase):
    """Check manual placements and when buffers can safely hold new values."""

    def test_manual_constraints_and_external_provider(self) -> None:
        """A manual storage name can equal another input's logical name."""
        req = replace(request(), constraints=(ValuePlacement(value="x", storage="y"),))
        plan = baseline_mapping(req)
        self.assertEqual(plan.placements[0].storage, "y")
        self.assertEqual(plan.placements[1].storage, "y.0")
        provider = Mock(return_value=plan)
        artifact = compile_graph(
            graph(), target(), constraints=req.constraints, mapper=provider
        )
        self.assertEqual(artifact.target_ir.mapping, plan)
        provider.assert_called_once()
        invalid_constraints = (
            (ValuePlacement(value="absent", storage="x"),),
            (ValuePlacement(value="x", storage="x"),) * 2,
        )
        for constraints in invalid_constraints:
            with self.subTest(constraints=constraints), self.assertRaises(ValueError):
                baseline_mapping(replace(req, constraints=constraints))
        with self.assertRaisesRegex(ValueError, "constraint"):
            verify_mapping(req, baseline_mapping(request()))

    def test_dead_storage_and_self_add(self) -> None:
        """Self-add reads initialized lanes; later values may reuse expired space."""
        original = graph()
        x = original.operands[0]
        double = AddOp(operands=(x, x), name_hint="z")
        single = replace(original, operations=(double,), results=double.results)
        artifact = compile_graph(single, target())
        self.assertEqual(
            artifact.execute(
                (
                    CentNamedScalars(name="x", values=(1.0, 2.0, 3.0)),
                    CentNamedScalars(name="y", values=(99.0, 99.0, 99.0)),
                )
            ),
            (CentNamedScalars(name="z", values=(2.0, 4.0, 6.0)),),
        )
        # x is retained; unused y dies at entry and becomes z's destination.
        retained = replace(single, results=(double.results[0], x))
        plan = compile_graph(retained, target()).target_ir.mapping
        self.assertEqual(plan.placements[-1].storage, "y")

    def test_rejects_invalid_plan_forms(self) -> None:
        """Reject incomplete plans, repeated inputs sharing storage, and unsafe result reuse."""
        req = request()
        plan = baseline_mapping(req)
        forms = (
            replace(plan, schedule=("z", "z")),
            replace(plan, placements=(*plan.placements, plan.placements[0])),
            replace(
                plan,
                placements=tuple(
                    replace(p, storage="") if p.value == "z" else p
                    for p in plan.placements
                ),
            ),
            replace(
                plan,
                placements=tuple(
                    replace(p, storage="x") if p.value == "y" else p
                    for p in plan.placements
                ),
            ),
            replace(
                plan,
                placements=tuple(
                    replace(p, storage="y") if p.value == "z" else p
                    for p in plan.placements
                ),
            ),
        )
        for bad in forms:
            with self.subTest(plan=bad), self.assertRaises(ValueError):
                verify_mapping(req, bad)

    def test_forced_output_and_fresh_storage_collision(self) -> None:
        """Manual output placements are preserved and live names stay distinct."""
        req = replace(
            request(fanout=True),
            constraints=(
                ValuePlacement(value="x", storage="z"),
                ValuePlacement(value="out", storage="manual_result"),
            ),
        )
        plan = baseline_mapping(req)
        self.assertEqual(plan.placements[2].storage, "z.0")
        self.assertEqual(plan.placements[3].storage, "manual_result")

    def test_invalid_targets_and_distributed_execution(self) -> None:
        """Reject unsupported executable scopes before allocating storage."""
        with self.assertRaises(ValueError):
            replace(target(), name="")
        with self.assertRaises(ValueError):
            replace(target(), max_accumulate_slots=0)
        # Legalization only produces rank zero on device zero, so a device
        # graph claiming another device no longer matches its source.
        device = legalize(graph(), CapabilityProfile())
        remote = replace(device, ranks=(replace(device.ranks[0], device_id=1),))
        with self.assertRaisesRegex(ValueError, "legalization"):
            baseline_mapping(MappingRequest(graph=remote, target=target()))

    def test_identity_graph_runs_without_instructions(self) -> None:
        """A graph returning an input needs bindings but no device instructions."""
        original = graph()
        identity = replace(original, operations=(), results=original.operands[:1])
        artifact = compile_graph(identity, target())
        self.assertEqual(artifact.executable.program.instructions, ())
        values = (
            CentNamedScalars(name="x", values=(1.0, -2.0, 3.0)),
            CentNamedScalars(name="y", values=(9.0, 8.0, 7.0)),
        )
        self.assertEqual(artifact.execute(values), (values[0],))

    def test_artifact_and_target_tampering(self) -> None:
        """Reject missing preservation copies and mismatched shapes or compiled commands."""
        from vllm_cent.pipeline import allocate_storage

        artifact = compile_graph(graph(fanout=True), target())
        bad = replace(artifact.target_ir, commands=artifact.target_ir.commands[1:])
        with self.assertRaisesRegex(ValueError, "Target IR"):
            allocate_storage(bad)
        with self.assertRaisesRegex(ValueError, "version"):
            replace(artifact, version=2)
        with self.assertRaisesRegex(ValueError, "logical binding"):
            replace(artifact, outputs=artifact.inputs)
        other = compile_graph(graph(), target())
        with self.assertRaisesRegex(ValueError, "executable"):
            replace(artifact, executable=other.executable)

    def test_copy_tiles_across_scratch_row_and_rebinds(self) -> None:
        """Nine values need five slots, so a four-slot row needs two copy chunks."""
        large = graph(fanout=True, shape=(9,))
        artifact = compile_graph(large, target(tile=2))
        # Two W/R pairs plus two adds with three ACC tiles apiece.
        self.assertEqual(len(artifact.executable.program.instructions), 10)
        for scale in (1.0, -3.0):
            result = artifact.execute(
                (
                    CentNamedScalars(name="x", values=(scale,) * 9),
                    CentNamedScalars(name="y", values=(2.0 * scale,) * 9),
                )
            )
            self.assertEqual(result[0].values, (5.0 * scale,) * 9)
            self.assertEqual(result[1].values, (scale,) * 9)

    def test_slots_and_fresh_storage_helpers(self) -> None:
        """Slot rounding and symbolic-name generation handle boundary cases."""
        from vllm_cent.pipeline.mapping import _fresh_storage, _slots

        self.assertEqual(_fresh_storage("x", {"x", "x.0", "x.1"}), "x.2")
        self.assertEqual(_fresh_storage("x", set()), "x")
        # A scalar still occupies one slot. Six values with two lanes per
        # slot fill exactly three slots.
        scalar = TensorType(shape=(), dtype=DType.FP32)
        matrix = TensorType(shape=(2, 3), dtype=DType.FP32)
        self.assertEqual(_slots(NameValue(name="a", type=scalar), target()), 1)
        self.assertEqual(_slots(NameValue(name="a", type=matrix), target()), 3)


class GraphDataflowTests(unittest.TestCase):
    """Compare compiled graphs with direct Python sums across branches and reused buffers."""

    def test_branched_graphs_match_host_addition(self) -> None:
        """Retain arbitrary intermediates while storage is reused across branches."""
        # A fixed seed makes failures repeatable. The small integer inputs keep
        # every sum exact in reference math, so this checks data flow, not rounding.
        random = Random(17)
        for case in range(12):
            inputs = tuple(argument(f"input{i}", (5,)) for i in range(4))
            # Every value is looked up by its printed name, which for these
            # unique hints is the hint itself.
            values = {value.name: value for value in inputs}
            oracle = {
                name: tuple(float(random.randrange(-4, 5)) for _ in range(5))
                for name in values
            }
            bindings = tuple(
                CentNamedScalars(name=name, values=oracle[name]) for name in values
            )
            names = list(oracle)
            operations = []
            for index in range(20):
                left, right = random.choices(names, k=2)
                name = f"v{index}"
                operation = AddOp(
                    operands=(values[left], values[right]), name_hint=name
                )
                operations.append(operation)
                values[name] = operation.results[0]
                oracle[name] = tuple(
                    a + b for a, b in zip(oracle[left], oracle[right], strict=True)
                )
                names.append(name)
            outputs = tuple(random.sample(names, 4))
            source = Function(
                name="main",
                operands=inputs,
                operations=tuple(operations),
                results=tuple(values[name] for name in outputs),
            )
            with self.subTest(case=case):
                artifact = compile_graph(source, target(capacity=80))
                self.assertEqual(
                    artifact.execute(bindings),
                    tuple(
                        CentNamedScalars(name=name, values=oracle[name])
                        for name in outputs
                    ),
                )

    def test_manual_topological_order_is_obeyed(self) -> None:
        """Independent operations can run in either dependency-valid order."""
        source = graph()
        x, y = source.operands
        first = source.operations[0]
        second = AddOp(operands=(y, x), name_hint="out")
        source = replace(
            source,
            operations=(first, second),
            results=(first.results[0], second.results[0]),
        )
        device = legalize(source, CapabilityProfile())
        req = MappingRequest(graph=device, target=target())
        # Give every tensor its own buffer so this test isolates execution order
        # from storage reuse. Neither addition depends on the other anymore.
        plan = MappingPlan(
            placements=tuple(
                ValuePlacement(value=name, storage=name)
                for name in ("x", "y", "z", "out")
            ),
            # Schedules name each operation by its result.
            schedule=("out", "z"),
        )
        verify_mapping(req, plan)
        artifact = compile_graph(source, target(), mapping=plan)
        self.assertEqual(artifact.target_ir.commands[0].operation, "out")
        self.assertEqual(
            artifact.execute(
                (
                    CentNamedScalars(name="x", values=(1.0, 2.0, 3.0)),
                    CentNamedScalars(name="y", values=(4.0, 5.0, 6.0)),
                )
            ),
            (
                CentNamedScalars(name="z", values=(5.0, 7.0, 9.0)),
                CentNamedScalars(name="out", values=(5.0, 7.0, 9.0)),
            ),
        )
