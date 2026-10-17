"""Check graph shapes, per-device tensor pieces, and operation dependencies."""

import unittest
from dataclasses import replace

from vllm_cent.pipeline.ir import (
    CapabilityProfile,
    DeviceOperationKind,
    DeviceRmsNormOperation,
    ParallelPlan,
    SemanticGraph,
    TensorType,
    Value,
    legalize,
    shard,
    verify_device,
    verify_semantic,
    verify_sharded,
)
from vllm_cent.pipeline.operations import (
    AddOperation,
    LinearOperation,
    RmsNormOperation,
)


def add_graph() -> SemanticGraph:
    """Build z = x + y followed by out = z + x on two (8, 4) tensors.

    Returns:
        Graph whose second addition must wait for the first to produce z.
    """
    tensor = TensorType((8, 4))
    return SemanticGraph(
        (Value("x", tensor), Value("y", tensor)),
        (
            AddOperation(name="first", inputs=("x", "y"), output=Value("z", tensor)),
            AddOperation(name="second", inputs=("z", "x"), output=Value("out", tensor)),
        ),
        ("out",),
    )


class PipelineIrTests(unittest.TestCase):
    """Check graph validity and that splitting preserves every requested calculation."""

    def test_uniform_sharding_and_legalization(self) -> None:
        """Keep the full shapes while assigning each device its own four-row piece."""
        graph = add_graph()
        # Split eight rows into two groups of four. Logical IDs 2 and 7 show
        # that rank position, not the numeric device ID, determines the slice.
        partitioned = shard(graph, ParallelPlan((2, 7), 0))
        self.assertEqual(partitioned.source, graph)
        self.assertEqual([rank.device_id for rank in partitioned.shards], [2, 7])
        for rank in partitioned.shards:
            self.assertEqual(rank.graph.inputs[0].type, TensorType((4, 4)))
            self.assertEqual(rank.values[0].global_value, graph.inputs[0])
            self.assertEqual(
                (rank.values[0].start, rank.values[0].stop),
                (rank.rank * 4, (rank.rank + 1) * 4),
            )
        # The first add reads external inputs. The second reads z, so it must
        # wait for the first add even though x is already available.
        device = legalize(partitioned, CapabilityProfile())
        self.assertEqual(device.ranks[0].operations[0].dependencies, ())
        self.assertEqual(device.ranks[0].operations[1].dependencies, ("first",))
        self.assertEqual(
            device.ranks[0].operations[0].kind, DeviceOperationKind.PNM_ADD
        )
        verify_device(device)

    def test_ssa_and_dominance(self) -> None:
        """Reject missing input names and attempts to redefine an existing tensor."""
        graph = add_graph()
        invalid_operations = (
            replace(graph.operations[0], inputs=("missing", "y")),
            replace(graph.operations[0], output=graph.inputs[0]),
        )
        for operation in invalid_operations:
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                verify_semantic(replace(graph, operations=(operation,)))
        with self.assertRaises(ValueError):
            verify_semantic(replace(graph, outputs=("missing",)))

    def test_add_rejects_broadcast(self) -> None:
        """Require matching input and output tensor shapes."""
        graph = add_graph()
        with self.assertRaisesRegex(ValueError, "shape"):
            verify_semantic(
                replace(graph, inputs=(graph.inputs[0], Value("y", TensorType((4,)))))
            )

    def test_partition_validation(self) -> None:
        """Reject repeated devices, uneven splits, and an incorrectly shifted interval."""
        with self.assertRaises(ValueError):
            TensorType((0,))
        with self.assertRaises(ValueError):
            ParallelPlan((1, 1))
        with self.assertRaises(ValueError):
            # Eight rows cannot be split equally among three devices.
            shard(add_graph(), ParallelPlan((0, 1, 2)))
        graph = shard(add_graph(), ParallelPlan((0, 1)))
        bad_value = replace(graph.shards[0].values[0], start=1)
        bad_rank = replace(
            graph.shards[0], values=(bad_value, *graph.shards[0].values[1:])
        )
        with self.assertRaises(ValueError):
            verify_sharded(replace(graph, shards=(bad_rank, graph.shards[1])))

    def test_unsupported_legalization_is_explicit(self) -> None:
        """Require listed capabilities and preserve RMSNorm epsilon during legalization."""
        with self.assertRaises(NotImplementedError):
            legalize(shard(add_graph(), ParallelPlan()), CapabilityProfile(frozenset()))
        tensor = TensorType((8, 4))
        graph = SemanticGraph(
            (Value("x", tensor), Value("scale", TensorType((4,)))),
            (
                RmsNormOperation(
                    name="norm",
                    inputs=("x", "scale"),
                    output=Value("out", tensor),
                    epsilon=1e-05,
                ),
            ),
            ("out",),
        )
        verify_semantic(graph)
        device = legalize(shard(graph, ParallelPlan()), CapabilityProfile())
        operation = device.ranks[0].operations[0]
        self.assertIsInstance(operation, DeviceRmsNormOperation)
        self.assertEqual(operation.kind, DeviceOperationKind.RMS_NORM)
        self.assertEqual(operation.epsilon, 1e-05)
        with self.assertRaises(NotImplementedError):
            legalize(shard(graph, ParallelPlan()), CapabilityProfile(frozenset()))

    def test_device_dependencies_are_verified(self) -> None:
        """Reject a device operation that forgets to wait for its input producer."""
        graph = legalize(shard(add_graph(), ParallelPlan()), CapabilityProfile())
        rank = graph.ranks[0]
        bad_operation = replace(rank.operations[1], dependencies=())
        with self.assertRaises(ValueError):
            verify_device(
                replace(
                    graph,
                    ranks=(
                        replace(rank, operations=(rank.operations[0], bad_operation)),
                    ),
                )
            )

    def test_linear_shape_and_single_device_legalization(self) -> None:
        """Preserve Linear shapes while rejecting unsupported cross-device sharding."""
        graph = SemanticGraph(
            (Value("x", TensorType((2, 4))), Value("w", TensorType((3, 4)))),
            (
                LinearOperation(
                    name="linear",
                    inputs=("x", "w"),
                    output=Value("out", TensorType((2, 3))),
                ),
            ),
            ("out",),
        )
        verify_semantic(graph)
        device = legalize(shard(graph, ParallelPlan()), CapabilityProfile())
        self.assertEqual(device.ranks[0].operations[0].kind, DeviceOperationKind.LINEAR)
        self.assertEqual(device.ranks[0].operations[0].output.type, TensorType((2, 3)))
        with self.assertRaises(NotImplementedError):
            legalize(shard(graph, ParallelPlan()), CapabilityProfile(frozenset()))
        with self.assertRaises(NotImplementedError):
            shard(graph, ParallelPlan((0, 1)))
        with self.assertRaises(ValueError):
            verify_semantic(
                replace(
                    graph,
                    operations=(
                        replace(
                            graph.operations[0], output=Value("out", TensorType((2, 4)))
                        ),
                    ),
                )
            )

    def test_operation_identifiers_and_arity(self) -> None:
        """Reject repeated/empty operation names and additions missing an operand."""
        graph = add_graph()
        for operation in (
            replace(graph.operations[1], name="first"),
            replace(graph.operations[1], name=""),
            replace(graph.operations[1], inputs=("x",)),
        ):
            with self.subTest(operation=operation), self.assertRaises(ValueError):
                verify_semantic(
                    replace(graph, operations=(graph.operations[0], operation))
                )
        with self.assertRaises(ValueError):
            verify_semantic(replace(graph, inputs=(Value("", TensorType((8, 4))),)))

    def test_scalar_identity_and_axis_ownership(self) -> None:
        """Keep a scalar whole on one device and split matrix columns across two devices."""
        scalar = SemanticGraph((Value("scalar", TensorType(())),), (), ("scalar",))
        identity = shard(scalar, ParallelPlan())
        self.assertEqual(identity.shards[0].graph, scalar)
        self.assertEqual(identity.shards[0].values[0].axis, -1)
        self.assertEqual(identity.shards[0].values[0].stop, 1)
        # Axis 1 divides four columns into two; each device still has eight rows.
        divided = shard(add_graph(), ParallelPlan((0, 1), 1))
        self.assertEqual(divided.shards[0].graph.inputs[0].type, TensorType((8, 2)))
        self.assertEqual(divided.shards[1].values[0].start, 2)
        for plan in (ParallelPlan((0, 1), 2), ParallelPlan((0, 1), 0)):
            with self.assertRaises(ValueError):
                shard(scalar, plan)
        for devices, axis in (((), 0), ((-1,), 0), ((0,), -1)):
            with self.assertRaises(ValueError):
                ParallelPlan(devices, axis)

    def test_rank_count_and_device_signature(self) -> None:
        """Reject a missing device graph or an output name changed after legalization."""
        graph = shard(add_graph(), ParallelPlan((0, 1)))
        with self.assertRaises(ValueError):
            verify_sharded(replace(graph, shards=graph.shards[:1]))
        device = legalize(graph, CapabilityProfile())
        with self.assertRaises(ValueError):
            verify_device(
                replace(
                    device,
                    ranks=(replace(device.ranks[0], outputs=("x",)), device.ranks[1]),
                )
            )

    def test_normalization_epsilon_contract(self) -> None:
        """Require a finite positive RMSNorm epsilon and reject epsilon on addition."""
        tensor = TensorType((8, 4))
        norm = RmsNormOperation(
            name="norm",
            inputs=("x", "scale"),
            output=Value("out", tensor),
            epsilon=1e-05,
        )
        graph = SemanticGraph(
            (Value("x", tensor), Value("scale", TensorType((4,)))), (norm,), ("out",)
        )
        verify_semantic(graph)
        self.assertEqual(shard(graph, ParallelPlan()).shards[0].graph, graph)
        for epsilon in (0.0, -1.0, float("nan"), float("inf")):
            with (
                self.subTest(epsilon=epsilon),
                self.assertRaisesRegex(ValueError, "epsilon"),
            ):
                verify_semantic(
                    replace(graph, operations=(replace(norm, epsilon=epsilon),))
                )
        add = add_graph()
        with self.assertRaisesRegex(TypeError, "epsilon"):
            verify_semantic(
                replace(add, operations=(replace(add.operations[0], epsilon=1e-5),))
            )
        with self.assertRaisesRegex(ValueError, "unique"):
            verify_semantic(replace(add, outputs=("out", "out")))


class AdditionalDeviceOperationTests(unittest.TestCase):
    """Verify graph-to-device selection for movement, reshape, and multiplication."""

    def test_mixed_operations_preserve_shapes_and_dependencies(self) -> None:
        """A copy, square, and reshape retain their different meanings after legalization."""
        from vllm_cent.pipeline.operations import (
            CopyOperation,
            MultiplyOperation,
            ReshapeOperation,
        )

        tensor = TensorType((2, 3))
        source = SemanticGraph(
            inputs=(Value("x", tensor),),
            operations=(
                CopyOperation(
                    name="copy", inputs=("x",), output=Value("copied", tensor)
                ),
                MultiplyOperation(
                    name="square",
                    inputs=("copied", "copied"),
                    output=Value("squared", tensor),
                ),
                ReshapeOperation(
                    name="reshape",
                    inputs=("squared",),
                    output=Value("out", TensorType((6,))),
                ),
            ),
            outputs=("out",),
        )
        device = legalize(shard(source, ParallelPlan()), CapabilityProfile())
        self.assertEqual(
            tuple(op.kind for op in device.ranks[0].operations),
            (
                DeviceOperationKind.COPY,
                DeviceOperationKind.PIM_MULTIPLY,
                DeviceOperationKind.RESHAPE,
            ),
        )
        self.assertEqual(
            tuple(op.dependencies for op in device.ranks[0].operations),
            ((), ("copy",), ("square",)),
        )
        self.assertEqual(device.ranks[0].operations[-1].output.type, TensorType((6,)))
        verify_device(device)
        with self.assertRaises(NotImplementedError):
            shard(source, ParallelPlan((0, 1)))
        for excluded in (
            DeviceOperationKind.COPY,
            DeviceOperationKind.PIM_MULTIPLY,
            DeviceOperationKind.RESHAPE,
        ):
            capabilities = CapabilityProfile(
                frozenset(set(DeviceOperationKind) - {excluded})
            )
            with (
                self.subTest(excluded=excluded),
                self.assertRaises(NotImplementedError),
            ):
                legalize(shard(source, ParallelPlan()), capabilities)

    def test_pointwise_operations_split_without_communication(self) -> None:
        """Copy and multiply operate independently on corresponding row slices."""
        from vllm_cent.pipeline.operations import CopyOperation, MultiplyOperation

        tensor = TensorType((4, 3))
        graph = SemanticGraph(
            inputs=(Value("x", tensor),),
            operations=(
                CopyOperation(
                    name="copy", inputs=("x",), output=Value("copied", tensor)
                ),
                MultiplyOperation(
                    name="square",
                    inputs=("copied", "copied"),
                    output=Value("out", tensor),
                ),
            ),
            outputs=("out",),
        )
        device = legalize(shard(graph, ParallelPlan((0, 1))), CapabilityProfile())
        self.assertEqual(
            tuple(rank.operations[-1].output.type for rank in device.ranks),
            (TensorType((2, 3)), TensorType((2, 3))),
        )
        verify_device(device)
