"""Check that legalization names IR values and selects device operations."""

import unittest
from dataclasses import replace

from vllm_cent.activations import ActivationKind
from vllm_cent.ir import (
    ActivationOp,
    AddOp,
    AllReduceOp,
    CopyOp,
    DType,
    Function,
    GqaQKOp,
    LinearOp,
    MergeHeadsOp,
    MulOp,
    Operation,
    Partial,
    ReshapeOp,
    RMSNormOp,
    RopeOp,
    SplitHeadsOp,
    TensorType,
    Value,
    value_names,
)
from vllm_cent.pipeline.ir import (
    CapabilityProfile,
    DeviceActivationOperation,
    DeviceGraph,
    DeviceOperation,
    DeviceOperationKind,
    DeviceRankGraph,
    DeviceRmsNormOperation,
    DeviceRopeOperation,
    NameValue,
    legalize,
    verify_device,
)

# fp32 machine epsilon, 2**-23. Legalization substitutes it for an RMSNorm
# whose eps is None, matching torch.nn.RMSNorm(eps=None) on fp32 inputs.
FP32_EPSILON = 2.0**-23


def tensor(*shape: int, dtype: DType = DType.FP32) -> TensorType:
    """Build an unannotated tensor type.

    Args:
        *shape: Tensor dimensions, outermost first.
        dtype: Element format. The CENT pipeline accepts only fp32, so tests
            pass another format only to check that it is rejected.

    Returns:
        Tensor type with the requested shape and element format.
    """
    return TensorType(shape=shape, dtype=dtype)


def argument(name: str, *shape: int, dtype: DType = DType.FP32) -> Value:
    """Build a named function argument.

    Args:
        name: Argument name. It becomes the tensor's device name.
        *shape: Tensor dimensions, outermost first.
        dtype: Element format; fp32 unless a test checks rejection.

    Returns:
        A new value suitable as a function operand.
    """
    return Value(type=tensor(*shape, dtype=dtype), name=name)


def function(
    operands: tuple[Value, ...],
    operations: tuple[Operation, ...],
    results: tuple[Value, ...],
) -> Function:
    """Wrap arguments, operations, and results in a function named ``main``.

    Args:
        operands: Function arguments, in caller order.
        operations: Operations in definition order.
        results: Values returned to the caller, in output order.

    Returns:
        Function containing exactly the supplied pieces.
    """
    return Function(
        name="main", operands=operands, operations=operations, results=results
    )


def add_function() -> Function:
    """Build ``z = x + y`` followed by ``out = z + x`` on two (8, 4) tensors.

    Returns:
        Function whose second addition must wait for the first to produce
        ``z``. It returns only ``out``.
    """
    x = argument("x", 8, 4)
    y = argument("y", 8, 4)
    first = AddOp(operands=(x, y), name_hint="z")
    second = AddOp(operands=(first.results[0], x), name_hint="out")
    return function((x, y), (first, second), second.results)


def rank_operations(source: Function) -> tuple[DeviceOperation, ...]:
    """Legalize ``source`` with every capability and return its device operations.

    Args:
        source: Function to legalize.

    Returns:
        Device operations of the only rank, in function order.
    """
    return legalize(source, CapabilityProfile()).ranks[0].operations


class LegalizeTests(unittest.TestCase):
    """Check device operation selection, naming, and dependency recording."""

    def test_add_chain_produces_complete_device_graph(self) -> None:
        """Two additions become two PNM additions on rank zero of device zero.

        The first addition reads only caller inputs, so it waits for nothing.
        The second reads ``z``, so it must wait for the operation named ``z``
        even though ``x`` is already available.
        """
        source = add_function()
        capabilities = CapabilityProfile()
        matrix = tensor(8, 4)
        expected = DeviceGraph(
            source=source,
            capabilities=capabilities,
            ranks=(
                DeviceRankGraph(
                    rank=0,
                    device_id=0,
                    inputs=(NameValue("x", matrix), NameValue("y", matrix)),
                    operations=(
                        DeviceOperation(
                            name="z",
                            kind=DeviceOperationKind.PNM_ADD,
                            inputs=("x", "y"),
                            output=NameValue("z", matrix),
                            dependencies=(),
                        ),
                        DeviceOperation(
                            name="out",
                            kind=DeviceOperationKind.PNM_ADD,
                            inputs=("z", "x"),
                            output=NameValue("out", matrix),
                            dependencies=("z",),
                        ),
                    ),
                    outputs=("out",),
                ),
            ),
        )
        device = legalize(source, capabilities)
        self.assertEqual(device, expected)
        verify_device(device)

    def test_device_names_equal_printer_names(self) -> None:
        """Arguments, hints, numbers, and duplicate suffixes follow ``value_names``.

        The first addition has no name hint, so the printer numbers it ``0``.
        The next two operations both ask for ``h``; the second becomes
        ``h_1`` because the first already claimed ``h``.
        """
        x = argument("x", 2)
        y = argument("y", 2)
        unnamed = AddOp(operands=(x, y))
        hinted = MulOp(operands=(unnamed.results[0], x), name_hint="h")
        repeated = AddOp(operands=(hinted.results[0], y), name_hint="h")
        source = function((x, y), (unnamed, hinted, repeated), repeated.results)
        names = value_names(source)
        self.assertEqual(
            [names[op.results[0]] for op in source.operations], ["0", "h", "h_1"]
        )
        vector = tensor(2)
        self.assertEqual(
            rank_operations(source),
            (
                DeviceOperation(
                    name="0",
                    kind=DeviceOperationKind.PNM_ADD,
                    inputs=("x", "y"),
                    output=NameValue("0", vector),
                    dependencies=(),
                ),
                DeviceOperation(
                    name="h",
                    kind=DeviceOperationKind.PIM_MULTIPLY,
                    inputs=("0", "x"),
                    output=NameValue("h", vector),
                    dependencies=("0",),
                ),
                DeviceOperation(
                    name="h_1",
                    kind=DeviceOperationKind.PNM_ADD,
                    inputs=("h", "y"),
                    output=NameValue("h_1", vector),
                    dependencies=("h",),
                ),
            ),
        )
        rank = legalize(source, CapabilityProfile()).ranks[0]
        self.assertEqual(rank.outputs, ("h_1",))

    def test_block_is_parent_path_of_name_hint(self) -> None:
        """A dotted module path labels its parent block; a plain hint has none.

        ``mlp.gate_proj`` belongs to block ``mlp``. Only the last component is
        removed, so ``layers.0.mlp.up`` belongs to ``layers.0.mlp``. A hint
        without a dot, and an operation without a hint, have an empty block.
        """
        x = argument("x", 2)
        gate = CopyOp(operands=(x,), name_hint="mlp.gate_proj")
        plain = CopyOp(operands=(gate.results[0],), name_hint="gate")
        nested = CopyOp(operands=(plain.results[0],), name_hint="layers.0.mlp.up")
        unnamed = CopyOp(operands=(nested.results[0],))
        source = function((x,), (gate, plain, nested, unnamed), unnamed.results)
        operations = rank_operations(source)
        self.assertEqual(
            [(op.name, op.block) for op in operations],
            [
                ("mlp.gate_proj", "mlp"),
                ("gate", ""),
                ("layers.0.mlp.up", "layers.0.mlp"),
                ("0", ""),
            ],
        )

    def test_rms_norm_epsilon_default_and_explicit(self) -> None:
        """RMSNorm without eps uses fp32 machine epsilon; an explicit eps is kept."""
        for eps, expected_epsilon in ((None, FP32_EPSILON), (1e-5, 1e-5)):
            with self.subTest(eps=eps):
                x = argument("x", 8, 4)
                scale = argument("scale", 4)
                norm = RMSNormOp(operands=(x, scale), eps=eps, name_hint="norm.out")
                source = function((x, scale), (norm,), norm.results)
                self.assertEqual(
                    rank_operations(source),
                    (
                        DeviceRmsNormOperation(
                            name="norm.out",
                            kind=DeviceOperationKind.RMS_NORM,
                            inputs=("x", "scale"),
                            output=NameValue("norm.out", tensor(8, 4)),
                            dependencies=(),
                            block="norm",
                            epsilon=expected_epsilon,
                        ),
                    ),
                )

    def test_rope_keeps_its_pairing_and_integer_positions(self) -> None:
        """RoPE becomes ROPE; its int64 positions are allowed, unlike other tensors.

        x is (3, 2, 4): three tokens of two heads of width four. Each token
        has one position, so positions is (3,); the table has eight rows of
        four rotary values (two cosines, two sines).
        """
        for is_neox_style in (True, False):
            with self.subTest(is_neox_style=is_neox_style):
                x = argument("x", 3, 2, 4)
                positions = argument("positions", 3, dtype=DType.INT64)
                table = argument("table", 8, 4)
                rope = RopeOp(
                    operands=(x, positions, table),
                    is_neox_style=is_neox_style,
                    name_hint="attn.q",
                )
                source = function((x, positions, table), (rope,), rope.results)
                self.assertEqual(
                    rank_operations(source),
                    (
                        DeviceRopeOperation(
                            name="attn.q",
                            kind=DeviceOperationKind.ROPE,
                            inputs=("x", "positions", "table"),
                            output=NameValue("attn.q", tensor(3, 2, 4)),
                            dependencies=(),
                            block="attn",
                            is_neox_style=is_neox_style,
                        ),
                    ),
                )

    def test_integer_tensor_outside_rope_is_unsupported(self) -> None:
        """Only a RoPE position may be an integer; returning one directly is rejected."""
        positions = argument("positions", 3, dtype=DType.INT64)
        with self.assertRaisesRegex(NotImplementedError, "positions is int64"):
            legalize(function((positions,), (), (positions,)), CapabilityProfile())

    def test_activation_keeps_its_formula(self) -> None:
        """The selected activation formula reaches the device operation unchanged."""
        x = argument("x", 2, 3)
        activation = ActivationOp(
            operands=(x,), function=ActivationKind.GELU_TANH, name_hint="act"
        )
        source = function((x,), (activation,), activation.results)
        self.assertEqual(
            rank_operations(source),
            (
                DeviceActivationOperation(
                    name="act",
                    kind=DeviceOperationKind.ACTIVATION,
                    inputs=("x",),
                    output=NameValue("act", tensor(2, 3)),
                    dependencies=(),
                    function=ActivationKind.GELU_TANH,
                ),
            ),
        )

    def test_split_and_merge_heads_become_reshapes(self) -> None:
        """Head regrouping keeps flattened element order, so it is a reshape.

        Splitting the last dimension of 8 into heads of width 4 gives two
        heads, (1, 1, 2, 4). Merging them restores (1, 1, 8).
        """
        x = argument("x", 1, 1, 8)
        split = SplitHeadsOp(operands=(x,), head_dim=4, name_hint="heads")
        merge = MergeHeadsOp(operands=(split.results[0],), name_hint="merged")
        source = function((x,), (split, merge), merge.results)
        self.assertEqual(
            rank_operations(source),
            (
                DeviceOperation(
                    name="heads",
                    kind=DeviceOperationKind.RESHAPE,
                    inputs=("x",),
                    output=NameValue("heads", tensor(1, 1, 2, 4)),
                    dependencies=(),
                ),
                DeviceOperation(
                    name="merged",
                    kind=DeviceOperationKind.RESHAPE,
                    inputs=("heads",),
                    output=NameValue("merged", tensor(1, 1, 8)),
                    dependencies=("heads",),
                ),
            ),
        )

    def test_linear_keeps_output_shape(self) -> None:
        """Two input rows against three weight rows produce a (2, 3) result."""
        x = argument("x", 2, 4)
        weight = argument("w", 3, 4)
        linear = LinearOp(operands=(x, weight), name_hint="out")
        source = function((x, weight), (linear,), linear.results)
        self.assertEqual(
            rank_operations(source),
            (
                DeviceOperation(
                    name="out",
                    kind=DeviceOperationKind.LINEAR,
                    inputs=("x", "w"),
                    output=NameValue("out", tensor(2, 3)),
                    dependencies=(),
                ),
            ),
        )

    def test_mixed_operations_preserve_shapes_and_dependencies(self) -> None:
        """A copy, square, and reshape keep their different device kinds.

        The square reads ``copied`` twice but waits for its producer once.
        """
        x = argument("x", 2, 3)
        copy = CopyOp(operands=(x,), name_hint="copied")
        square = MulOp(operands=(copy.results[0], copy.results[0]), name_hint="squared")
        reshape = ReshapeOp(operands=(square.results[0],), shape=(6,), name_hint="out")
        source = function((x,), (copy, square, reshape), reshape.results)
        matrix = tensor(2, 3)
        device = legalize(source, CapabilityProfile())
        self.assertEqual(
            device.ranks[0].operations,
            (
                DeviceOperation(
                    name="copied",
                    kind=DeviceOperationKind.COPY,
                    inputs=("x",),
                    output=NameValue("copied", matrix),
                    dependencies=(),
                ),
                DeviceOperation(
                    name="squared",
                    kind=DeviceOperationKind.PIM_MULTIPLY,
                    inputs=("copied", "copied"),
                    output=NameValue("squared", matrix),
                    dependencies=("copied",),
                ),
                DeviceOperation(
                    name="out",
                    kind=DeviceOperationKind.RESHAPE,
                    inputs=("squared",),
                    output=NameValue("out", tensor(6)),
                    dependencies=("squared",),
                ),
            ),
        )
        verify_device(device)
        # Removing any one kind from the target rejects the whole function.
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
                self.assertRaisesRegex(NotImplementedError, excluded.name),
            ):
                legalize(source, capabilities)

    def test_dependencies_list_each_producer_once_in_input_order(self) -> None:
        """``z = y + y`` waits for ``y`` once; ``c = b + a`` lists ``b`` first.

        ``a`` and ``b`` are produced in that order, but ``c`` reads ``b`` as
        its first input, so ``b`` comes first in its dependencies.
        """
        x = argument("x", 2)
        y = AddOp(operands=(x, x), name_hint="y")
        z = AddOp(operands=(y.results[0], y.results[0]), name_hint="z")
        a = AddOp(operands=(x, x), name_hint="a")
        b = AddOp(operands=(x, x), name_hint="b")
        c = AddOp(operands=(b.results[0], a.results[0]), name_hint="c")
        source = function((x,), (y, z, a, b, c), (z.results[0], c.results[0]))
        self.assertEqual(
            [(op.name, op.inputs, op.dependencies) for op in rank_operations(source)],
            [
                ("y", ("x", "x"), ()),
                ("z", ("y", "y"), ("y",)),
                ("a", ("x", "x"), ()),
                ("b", ("x", "x"), ()),
                ("c", ("b", "a"), ("b", "a")),
            ],
        )

    def test_argument_may_be_returned_directly(self) -> None:
        """An identity function has no device operations and returns its input."""
        x = argument("x", 3)
        rank = legalize(function((x,), (), (x,)), CapabilityProfile()).ranks[0]
        self.assertEqual(
            rank,
            DeviceRankGraph(
                rank=0,
                device_id=0,
                inputs=(NameValue("x", tensor(3)),),
                operations=(),
                outputs=("x",),
            ),
        )


class LegalizeRejectionTests(unittest.TestCase):
    """Check that legalization rejects work the CENT backend cannot run."""

    def test_non_fp32_tensor_is_unsupported(self) -> None:
        """The functional simulator computes only fp32 tensors."""
        for dtype in (DType.FP16, DType.BF16):
            with self.subTest(dtype=dtype):
                x = argument("x", 4, dtype=dtype)
                y = argument("y", 4, dtype=dtype)
                add = AddOp(operands=(x, y), name_hint="z")
                with self.assertRaisesRegex(NotImplementedError, "fp32"):
                    legalize(function((x, y), (add,), add.results), CapabilityProfile())

    def test_zero_length_dimension_is_rejected(self) -> None:
        """``TensorType`` allows a zero dimension, but the device has nothing to store."""
        x = argument("x", 0, 4)
        y = argument("y", 0, 4)
        add = AddOp(operands=(x, y), name_hint="z")
        with self.assertRaisesRegex(ValueError, "positive"):
            legalize(function((x, y), (add,), add.results), CapabilityProfile())

    def test_unsupported_operations_name_the_operation(self) -> None:
        """All-reduce and decomposed attention steps have no device lowering.

        The query (1, 1, 2, 4) holds two heads of width four, and the key
        cache (1, 3, 1, 4) holds three positions of one shared KV head.
        """
        # All-reduce accepts only a partial sum, one rank's share of a total.
        partial = Value(
            type=TensorType(shape=(4,), dtype=DType.FP32, placement=Partial()),
            name="x",
        )
        reduce = AllReduceOp(operands=(partial,), name_hint="r")
        query = argument("q", 1, 1, 2, 4)
        keys = argument("k", 1, 3, 1, 4)
        scores = GqaQKOp(operands=(query, keys), scale=0.5, name_hint="s")
        cases = (
            ("aloi.all_reduce", function((partial,), (reduce,), reduce.results)),
            ("aloi.gqa_qk", function((query, keys), (scores,), scores.results)),
        )
        for op_name, source in cases:
            with (
                self.subTest(op=op_name),
                self.assertRaisesRegex(NotImplementedError, op_name),
            ):
                legalize(source, CapabilityProfile())

    def test_results_must_be_present_and_unique(self) -> None:
        """A function returning nothing, or the same value twice, is rejected."""
        source = add_function()
        out = source.results[0]
        for results in ((), (out, out)):
            with (
                self.subTest(results=len(results)),
                self.assertRaisesRegex(ValueError, "unique"),
            ):
                legalize(replace(source, results=results), CapabilityProfile())

    def test_arguments_keep_their_names(self) -> None:
        """Argument names are the program's input names, so none may change.

        The printer would rename a second ``x`` to ``x_1`` and number an
        unnamed or empty-named argument ``0``; a caller binding inputs by the
        names it wrote would then miss them.
        """
        cases = {
            "repeated": ("x", "x", "'x' would become 'x_1'"),
            "empty": ("", "y", "'' would become '0'"),
            "missing": (None, "y", "None would become '0'"),
        }
        for label, (first_name, second_name, message) in cases.items():
            first = Value(type=tensor(2), name=first_name)
            second = Value(type=tensor(2), name=second_name)
            add = AddOp(operands=(first, second), name_hint="z")
            with (
                self.subTest(case=label),
                self.assertRaisesRegex(ValueError, message),
            ):
                legalize(
                    function((first, second), (add,), add.results),
                    CapabilityProfile(),
                )

    def test_missing_capability_is_unsupported(self) -> None:
        """A target without PNM addition cannot run an addition."""
        with self.assertRaisesRegex(NotImplementedError, "PNM_ADD"):
            legalize(add_function(), CapabilityProfile(frozenset()))
        x = argument("x", 8, 4)
        scale = argument("scale", 4)
        norm = RMSNormOp(operands=(x, scale), eps=1e-5, name_hint="out")
        with self.assertRaisesRegex(NotImplementedError, "RMS_NORM"):
            legalize(
                function((x, scale), (norm,), norm.results),
                CapabilityProfile(frozenset()),
            )

    def test_invalid_function_is_rejected(self) -> None:
        """An operation reading a value that is not defined in the function fails."""
        source = add_function()
        stray = argument("stray", 8, 4)
        add = AddOp(operands=(stray, source.operands[0]), name_hint="bad")
        with self.assertRaisesRegex(ValueError, "unavailable"):
            legalize(
                replace(source, operations=(add,), results=add.results),
                CapabilityProfile(),
            )


class VerifyDeviceTests(unittest.TestCase):
    """Check that tampered device graphs no longer match their source."""

    def test_missing_dependency_is_rejected(self) -> None:
        """A device operation that forgets to wait for its input producer fails."""
        graph = legalize(add_function(), CapabilityProfile())
        rank = graph.ranks[0]
        bad_operation = replace(rank.operations[1], dependencies=())
        bad_rank = replace(rank, operations=(rank.operations[0], bad_operation))
        with self.assertRaisesRegex(ValueError, "legalization"):
            verify_device(replace(graph, ranks=(bad_rank,)))

    def test_changed_output_is_rejected(self) -> None:
        """Returning ``x`` instead of ``out`` no longer matches the source function."""
        graph = legalize(add_function(), CapabilityProfile())
        bad_rank = replace(graph.ranks[0], outputs=("x",))
        with self.assertRaisesRegex(ValueError, "legalization"):
            verify_device(replace(graph, ranks=(bad_rank,)))

    def test_missing_rank_is_rejected(self) -> None:
        """A device graph must keep the one rank legalization produced."""
        graph = legalize(add_function(), CapabilityProfile())
        with self.assertRaisesRegex(ValueError, "legalization"):
            verify_device(replace(graph, ranks=()))


if __name__ == "__main__":
    unittest.main()
