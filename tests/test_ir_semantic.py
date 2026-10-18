"""Tests for the generic semantic ops: linear, norm, all_reduce, elementwise, reshape."""

from __future__ import annotations

import unittest
from collections.abc import Callable
from dataclasses import FrozenInstanceError
from math import prod

from vllm_cent.activations import ActivationKind
from vllm_cent.ir.base import Function, Operation, Value
from vllm_cent.ir.printer import print_function
from vllm_cent.ir.semantic import (
    ActivationOp,
    AddOp,
    AllReduceOp,
    CopyOp,
    LinearOp,
    MulOp,
    ReshapeOp,
    RMSNormOp,
    SoftmaxOp,
)
from vllm_cent.ir.types import (
    DType,
    Partial,
    Placement,
    Replicate,
    Shard,
    TensorRole,
    TensorType,
    packed_axis,
)

# Llama2-70B decode dimensions. One decode step processes one token of one
# request, so activations are [batch=1, seq_len=1, features].
HIDDEN_SIZE = 8192  # 64 query heads x head_dim 128
KV_WIDTH = 1024  # 8 KV heads x head_dim 128
INTERMEDIATE_SIZE = 28672


def _value(*shape: int, dtype: DType = DType.FP16, name: str | None = None) -> Value:
    """Create a value of the given shape.

    Args:
        *shape: Tensor dimensions.
        dtype: Element type, fp16 unless a test needs a mismatch.
        name: Optional name hint.

    Returns:
        A new value.
    """

    return Value(type=TensorType(shape=shape, dtype=dtype), name=name)


def _annotated(
    shape: tuple[int, ...],
    axes: tuple[str, ...],
    role: TensorRole = TensorRole.ACTIVATION,
) -> Value:
    """Create an fp16 value with semantic annotations.

    Args:
        shape: Tensor dimensions.
        axes: One axis name per dimension.
        role: Tensor role.

    Returns:
        A new value.
    """

    return Value(
        type=TensorType(shape=shape, dtype=DType.FP16, role=role, axes=axes)
    )


# Axis names as AnnotateRolesAndAxes would attach them for Llama2-70B. The
# query projection's output packs 64 heads x 128 elements into one 8192 dim.
TOKEN_AXES = ("batch", "seq_len")
HIDDEN = "hidden"
Q_PROJ_OUT = packed_axis("q_head", "head_dim")

# 8-way tensor parallelism splits the 64 query heads into 8 per rank, so every
# q_head*head_dim dimension shrinks from 8192 to 8 x 128 = 1024 on each rank.
TP = 8
LOCAL_HEADS_WIDTH = HIDDEN_SIZE // TP


def _placed(
    shape: tuple[int, ...],
    axes: tuple[str, ...],
    placement: Placement,
    global_shape: tuple[int, ...] | None = None,
    role: TensorRole = TensorRole.ACTIVATION,
) -> Value:
    """Create an annotated fp16 value as one tensor-parallel rank sees it.

    Args:
        shape: Local shape.
        axes: One axis name per dimension.
        placement: Which part of the tensor the rank holds.
        global_shape: Whole-tensor shape; required for a shard.
        role: Tensor role.

    Returns:
        A new value.
    """

    return Value(
        type=TensorType(
            shape=shape,
            dtype=DType.FP16,
            role=role,
            axes=axes,
            global_shape=global_shape,
            placement=placement,
        )
    )


def _replicated_hidden() -> Value:
    """Return the residual stream x: whole on every rank."""

    return _placed((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN), Replicate())


def _sharded_heads() -> Value:
    """Return an activation over q_head*head_dim, 8 heads per rank.

    This is what q_proj produces, and what the attention output that feeds
    o_proj looks like.
    """

    return _placed(
        (1, 1, LOCAL_HEADS_WIDTH),
        (*TOKEN_AXES, Q_PROJ_OUT),
        Shard("q_head"),
        global_shape=(1, 1, HIDDEN_SIZE),
    )


def _wq_shard() -> Value:
    """Return Wq on one rank: its q_head rows (dim 0) are split."""

    return _placed(
        (LOCAL_HEADS_WIDTH, HIDDEN_SIZE),
        (Q_PROJ_OUT, HIDDEN),
        Shard("q_head"),
        global_shape=(HIDDEN_SIZE, HIDDEN_SIZE),
        role=TensorRole.WEIGHT,
    )


def _wo_shard() -> Value:
    """Return Wo on one rank: its q_head columns (dim 1) are split."""

    return _placed(
        (HIDDEN_SIZE, LOCAL_HEADS_WIDTH),
        (HIDDEN, Q_PROJ_OUT),
        Shard("q_head"),
        global_shape=(HIDDEN_SIZE, HIDDEN_SIZE),
        role=TensorRole.WEIGHT,
    )


def _replicated_weight(axes: tuple[str, ...]) -> Value:
    """Return a whole 8192 x 8192 weight with the given axes."""

    return _placed(
        (HIDDEN_SIZE, HIDDEN_SIZE), axes, Replicate(), role=TensorRole.WEIGHT
    )


class LinearOpTest(unittest.TestCase):
    """PyTorch nn.Linear semantics: weight is [out_features, in_features]."""

    def test_llama2_70b_projection_shapes(self) -> None:
        """Each projection maps [*, in_features] to [*, out_features]."""

        hidden = _value(1, 1, HIDDEN_SIZE)
        ffn = _value(1, 1, INTERMEDIATE_SIZE)
        cases = {
            # q_proj/o_proj: 8192 -> 8192.
            "q_proj": (hidden, (HIDDEN_SIZE, HIDDEN_SIZE), (1, 1, HIDDEN_SIZE)),
            # k_proj/v_proj: GQA keeps only 8 KV heads, so 8192 -> 1024.
            "k_proj": (hidden, (KV_WIDTH, HIDDEN_SIZE), (1, 1, KV_WIDTH)),
            # gate_proj/up_proj expand to the FFN width: 8192 -> 28672.
            "gate_proj": (
                hidden,
                (INTERMEDIATE_SIZE, HIDDEN_SIZE),
                (1, 1, INTERMEDIATE_SIZE),
            ),
            # down_proj contracts back to the hidden size: 28672 -> 8192.
            "down_proj": (ffn, (HIDDEN_SIZE, INTERMEDIATE_SIZE), (1, 1, HIDDEN_SIZE)),
        }
        for name, (x, weight_shape, expected) in cases.items():
            with self.subTest(projection=name):
                op = LinearOp(operands=(x, _value(*weight_shape)))
                self.assertEqual(
                    op.results[0].type,
                    TensorType(shape=expected, dtype=DType.FP16),
                )

    def test_named_accessors(self) -> None:
        """The x/weight accessors name operands; feature counts come from weight."""

        x = _value(1, 1, HIDDEN_SIZE)
        weight = _value(KV_WIDTH, HIDDEN_SIZE)
        op = LinearOp(operands=(x, weight))
        self.assertIs(op.x, x)
        self.assertIs(op.weight, weight)
        self.assertEqual(op.in_features, HIDDEN_SIZE)
        self.assertEqual(op.out_features, KV_WIDTH)

    def test_rank_one_input(self) -> None:
        """Any number of leading dimensions is allowed, including none."""

        op = LinearOp(operands=(_value(HIDDEN_SIZE), _value(KV_WIDTH, HIDDEN_SIZE)))
        self.assertEqual(op.results[0].type.shape, (KV_WIDTH,))

    def test_keeps_every_leading_dimension(self) -> None:
        """Only the last dimension changes: (2, 4, 6) times a (5, 6) weight is (2, 4, 5).

        The leading sizes 2 and 4 differ from 1, so a result that dropped or
        reordered them would have a different shape.
        """

        op = LinearOp(operands=(_value(2, 4, 6), _value(5, 6)))
        self.assertEqual(
            op.results[0].type, TensorType(shape=(2, 4, 5), dtype=DType.FP16)
        )

    def test_rejects_in_features_mismatch(self) -> None:
        """The last dimension of x must equal weight.shape[1]."""

        x = _value(1, 1, HIDDEN_SIZE)
        cases = {
            # Wrong input width.
            "narrow weight": (KV_WIDTH, 4096),
            # A transposed k_proj weight: its out_features (8192) happens to
            # equal x's width, which an [in, out] check would wrongly accept.
            "transposed weight": (HIDDEN_SIZE, KV_WIDTH),
        }
        for name, weight_shape in cases.items():
            with self.subTest(case=name), self.assertRaises(ValueError):
                LinearOp(operands=(x, _value(*weight_shape)))

    def test_rejects_weight_rank_other_than_two(self) -> None:
        """The weight is always a 2-D matrix."""

        x = _value(1, 1, HIDDEN_SIZE)
        for weight_shape in ((HIDDEN_SIZE,), (1, KV_WIDTH, HIDDEN_SIZE)):
            with self.subTest(shape=weight_shape), self.assertRaises(ValueError):
                LinearOp(operands=(x, _value(*weight_shape)))

    def test_rejects_scalar_input(self) -> None:
        """A rank-0 x has no in_features dimension."""

        with self.assertRaises(ValueError):
            LinearOp(operands=(_value(), _value(KV_WIDTH, HIDDEN_SIZE)))

    def test_rejects_dtype_mismatch(self) -> None:
        """Mixed precision is not modeled; both operands share one dtype."""

        with self.assertRaises(ValueError):
            LinearOp(
                operands=(
                    _value(1, 1, HIDDEN_SIZE),
                    _value(KV_WIDTH, HIDDEN_SIZE, dtype=DType.BF16),
                )
            )

    def test_has_no_attributes(self) -> None:
        """Llama projections have no bias and no other attribute."""

        op = LinearOp(operands=(_value(1, HIDDEN_SIZE), _value(KV_WIDTH, HIDDEN_SIZE)))
        self.assertEqual(op.attributes(), {})

    def test_rebuild_reinfers_result_type(self) -> None:
        """Swapping in a TP shard of the weight changes the result shape.

        With tensor parallelism of degree 8, each rank holds 1/8 of q_proj's
        output rows: [8192 / 8, 8192] = [1024, 8192].
        """

        x = _value(1, 1, HIDDEN_SIZE)
        full = LinearOp(operands=(x, _value(HIDDEN_SIZE, HIDDEN_SIZE)))
        shard = full.with_operands((x, _value(HIDDEN_SIZE // 8, HIDDEN_SIZE)))
        self.assertEqual(shard.results[0].type.shape, (1, 1, HIDDEN_SIZE // 8))


class LinearOpAxesTest(unittest.TestCase):
    """The result keeps x's leading axes and takes the weight's output axis."""

    def test_q_proj_produces_packed_head_axis(self) -> None:
        """hidden -> q_head*head_dim, contracting over hidden."""

        op = LinearOp(
            operands=(
                _annotated((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN)),
                _annotated(
                    (HIDDEN_SIZE, HIDDEN_SIZE), (Q_PROJ_OUT, HIDDEN), TensorRole.WEIGHT
                ),
            )
        )
        self.assertEqual(
            op.results[0].type,
            TensorType(
                shape=(1, 1, HIDDEN_SIZE),
                dtype=DType.FP16,
                axes=(*TOKEN_AXES, Q_PROJ_OUT),
            ),
        )

    def test_o_proj_contracts_over_packed_head_axis(self) -> None:
        """o_proj's weight has the head axis second: it sums over the heads.

        This is why TP sharding q_head splits o_proj's contraction and needs
        an all-reduce, while it splits q_proj's output and needs none.
        """

        op = LinearOp(
            operands=(
                _annotated((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, Q_PROJ_OUT)),
                _annotated(
                    (HIDDEN_SIZE, HIDDEN_SIZE), (HIDDEN, Q_PROJ_OUT), TensorRole.WEIGHT
                ),
            )
        )
        self.assertEqual(op.results[0].type.axes, (*TOKEN_AXES, HIDDEN))

    def test_rejects_contraction_axis_mismatch(self) -> None:
        """Feeding the hidden vector to o_proj: sizes match, meanings do not.

        Both hidden and q_head*head_dim are 8192 wide, so only the axis names
        can catch this.
        """

        with self.assertRaisesRegex(ValueError, "does not match"):
            LinearOp(
                operands=(
                    _annotated((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN)),
                    _annotated(
                        (HIDDEN_SIZE, HIDDEN_SIZE),
                        (HIDDEN, Q_PROJ_OUT),
                        TensorRole.WEIGHT,
                    ),
                )
            )

    def test_unannotated_operand_gives_unannotated_result(self) -> None:
        """Without both operands' axes the output axis is unknown."""

        annotated_x = _annotated((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN))
        op = LinearOp(operands=(annotated_x, _value(KV_WIDTH, HIDDEN_SIZE)))
        self.assertEqual(op.results[0].type.axes, ())

    def test_result_is_activation(self) -> None:
        """A linear output is computed, never a resident weight."""

        op = LinearOp(
            operands=(
                _annotated((1, HIDDEN_SIZE), ("batch", HIDDEN)),
                _annotated(
                    (KV_WIDTH, HIDDEN_SIZE), ("kv_out", HIDDEN), TensorRole.WEIGHT
                ),
            )
        )
        self.assertIs(op.results[0].type.role, TensorRole.ACTIVATION)


class LinearOpPlacementTest(unittest.TestCase):
    """Which part of the result each tensor-parallel rank holds."""

    def test_column_parallel_q_proj(self) -> None:
        """Whole x times Wq's own heads gives this rank's heads of q.

        Locally [1, 1, 8192] x [1024, 8192]^T = [1, 1, 1024]; the whole q is
        [1, 1, 8192].
        """

        op = LinearOp(operands=(_replicated_hidden(), _wq_shard()))
        self.assertEqual(op.results[0].type, _sharded_heads().type)

    def test_row_parallel_o_proj(self) -> None:
        """This rank's heads times Wo's matching columns is a partial sum.

        Locally [1, 1, 1024] x [8192, 1024]^T = [1, 1, 8192]: the full output
        shape, but summed over only 8 of the 64 heads.
        """

        op = LinearOp(operands=(_sharded_heads(), _wo_shard()))
        self.assertEqual(
            op.results[0].type,
            TensorType(
                shape=(1, 1, HIDDEN_SIZE),
                dtype=DType.FP16,
                axes=(*TOKEN_AXES, HIDDEN),
                placement=Partial(),
            ),
        )

    def test_unsupported_combinations(self) -> None:
        """Combinations that would need communication before the matmul."""

        partial_x = _placed(
            (1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN), Partial()
        )
        batch_split_x = _placed(
            (1, 1, HIDDEN_SIZE),
            (*TOKEN_AXES, HIDDEN),
            Shard("batch"),
            global_shape=(TP, 1, HIDDEN_SIZE),
        )
        cases = {
            # Wo split but its input whole: x would first need a local slice.
            "replicated x, row-split weight": (_replicated_hidden(), _wo_shard()),
            # Input split but Wo whole: x would first need an all_gather.
            "split x, replicated weight": (
                _sharded_heads(),
                _replicated_weight((HIDDEN, Q_PROJ_OUT)),
            ),
            # A partial sum must be all_reduced before it is used.
            "partial x": (partial_x, _replicated_weight((Q_PROJ_OUT, HIDDEN))),
            # Splitting batch rows is data parallelism, not modeled yet.
            "x split along batch": (
                batch_split_x,
                _replicated_weight((Q_PROJ_OUT, HIDDEN)),
            ),
        }
        for name, operands in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, "unsupported placements"):
                    LinearOp(operands=operands)

    def test_same_axis_on_weight_output_is_not_row_parallel(self) -> None:
        """Both split on q_head is not enough: it must be the weight's in dim.

        This weight's q_head is its out dimension (dim 0), so the matmul does
        not sum over the split axis and no rule applies.
        """

        other_heads = packed_axis("other_head", "head_dim")
        weight_split_on_out = _placed(
            (LOCAL_HEADS_WIDTH, LOCAL_HEADS_WIDTH),
            (Q_PROJ_OUT, other_heads),
            Shard("q_head"),
            global_shape=(HIDDEN_SIZE, LOCAL_HEADS_WIDTH),
        )
        with self.assertRaisesRegex(ValueError, "unsupported placements"):
            LinearOp(operands=(_sharded_heads(), weight_split_on_out))


class RMSNormOpTest(unittest.TestCase):
    """Normalization over the last dimension, scaled by a learned weight."""

    def test_result_type_equals_input_type(self) -> None:
        """RMSNorm never changes shape or dtype."""

        x = _value(1, 1, HIDDEN_SIZE)
        op = RMSNormOp(operands=(x, _value(HIDDEN_SIZE)), eps=1e-5)
        self.assertEqual(op.results[0].type, x.type)

    def test_named_accessors(self) -> None:
        """The x and weight accessors name the two operands."""

        x = _value(1, 1, HIDDEN_SIZE)
        weight = _value(HIDDEN_SIZE)
        op = RMSNormOp(operands=(x, weight), eps=1e-5)
        self.assertIs(op.x, x)
        self.assertIs(op.weight, weight)

    def test_eps_is_an_attribute(self) -> None:
        """The only attribute is eps; Llama2 uses 1e-5."""

        op = RMSNormOp(operands=(_value(1, HIDDEN_SIZE), _value(HIDDEN_SIZE)), eps=1e-5)
        self.assertEqual(op.attributes(), {"eps": 1e-5})

    def test_eps_defaults_to_none(self) -> None:
        """Omitting eps records None; the backend later picks a dtype default."""

        op = RMSNormOp(operands=(_value(1, HIDDEN_SIZE), _value(HIDDEN_SIZE)))
        self.assertEqual(op.attributes(), {"eps": None})

    def test_weight_is_optional(self) -> None:
        """Llama 4's QK-norm: per-head [1, 1, 5, 128] queries, no learned scale.

        The query is one rank's 5 of 40 heads; normalizing over head_dim
        keeps that split, and the op prints with one operand.
        """

        q = _placed(
            (1, 1, 5, 128),
            (*TOKEN_AXES, "q_head", "head_dim"),
            Shard("q_head"),
            global_shape=(1, 1, 40, 128),
        )
        op = RMSNormOp(operands=(q,), eps=1e-5, name_hint="qk_norm")
        self.assertIsNone(op.weight)
        self.assertEqual(op.results[0].type, q.type)
        x = _value(1, 4, name="x")
        self.assertEqual(
            _print_one(RMSNormOp(operands=(x,), eps=1e-5, name_hint="norm")),
            "%norm = aloi.rms_norm(%x) {eps = 1e-05} : tensor<1x4xfp16>",
        )
        with self.assertRaisesRegex(ValueError, "expects"):
            RMSNormOp(operands=(x, _value(4), _value(4)))

    def test_eps_cannot_change_after_construction(self) -> None:
        """Ops are frozen, so eps stays the value the result type was checked with."""

        op = RMSNormOp(operands=(_value(1, HIDDEN_SIZE), _value(HIDDEN_SIZE)), eps=1e-5)
        with self.assertRaises(FrozenInstanceError):
            op.eps = 1e-4  # type: ignore[misc]
        self.assertEqual(op.eps, 1e-5)

    def test_rejects_non_positive_eps(self) -> None:
        """The eps term keeps the square root away from zero, so it must be > 0."""

        for eps in (0.0, -1e-5):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
                RMSNormOp(
                    operands=(_value(1, HIDDEN_SIZE), _value(HIDDEN_SIZE)), eps=eps
                )

    def test_rejects_non_finite_eps(self) -> None:
        """A NaN or infinite eps would make every normalized value NaN or zero."""

        for eps in (float("nan"), float("inf"), -float("inf")):
            with self.subTest(eps=eps), self.assertRaisesRegex(ValueError, "finite"):
                RMSNormOp(
                    operands=(_value(1, HIDDEN_SIZE), _value(HIDDEN_SIZE)), eps=eps
                )

    def test_rejects_weight_size_mismatch(self) -> None:
        """There is one weight per element of the normalized dimension."""

        with self.assertRaises(ValueError):
            RMSNormOp(operands=(_value(1, HIDDEN_SIZE), _value(4096)), eps=1e-5)

    def test_rejects_weight_rank_other_than_one(self) -> None:
        """The weight is a vector over the last dimension."""

        with self.assertRaises(ValueError):
            RMSNormOp(
                operands=(_value(1, HIDDEN_SIZE), _value(1, HIDDEN_SIZE)), eps=1e-5
            )

    def test_rejects_dtype_mismatch(self) -> None:
        """Both operands share one dtype, like LinearOp's operands."""

        with self.assertRaises(ValueError):
            RMSNormOp(
                operands=(
                    _value(1, HIDDEN_SIZE),
                    _value(HIDDEN_SIZE, dtype=DType.BF16),
                ),
                eps=1e-5,
            )

    def test_rejects_scalar_input(self) -> None:
        """A rank-0 x has no dimension to normalize; expect a clear ValueError."""

        with self.assertRaises(ValueError):
            RMSNormOp(operands=(_value(), _value(HIDDEN_SIZE)), eps=1e-5)


class RMSNormOpAxesTest(unittest.TestCase):
    """The result has x's axes; the weight is indexed by x's last axis."""

    def test_result_keeps_x_axes(self) -> None:
        """Normalizing does not change what any dimension means."""

        x = _annotated((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN))
        weight = _annotated((HIDDEN_SIZE,), (HIDDEN,), TensorRole.WEIGHT)
        op = RMSNormOp(operands=(x, weight), eps=1e-5)
        self.assertEqual(op.results[0].type, x.type)

    def test_unannotated_weight_still_propagates_x_axes(self) -> None:
        """The result's axes come from x alone."""

        x = _annotated((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN))
        op = RMSNormOp(operands=(x, _value(HIDDEN_SIZE)), eps=1e-5)
        self.assertEqual(op.results[0].type.axes, (*TOKEN_AXES, HIDDEN))

    def test_rejects_weight_axis_mismatch(self) -> None:
        """A weight over a different axis cannot scale x's last dimension."""

        x = _annotated((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN))
        weight = _annotated((HIDDEN_SIZE,), (Q_PROJ_OUT,), TensorRole.WEIGHT)
        with self.assertRaisesRegex(ValueError, "do not match"):
            RMSNormOp(operands=(x, weight), eps=1e-5)

    def test_result_is_activation_even_for_weight_input(self) -> None:
        """Copying x's type must not copy a WEIGHT role into the result."""

        x = _annotated((HIDDEN_SIZE,), (HIDDEN,), TensorRole.WEIGHT)
        weight = _annotated((HIDDEN_SIZE,), (HIDDEN,), TensorRole.WEIGHT)
        op = RMSNormOp(operands=(x, weight), eps=1e-5)
        self.assertEqual(
            op.results[0].type,
            TensorType(shape=(HIDDEN_SIZE,), dtype=DType.FP16, axes=(HIDDEN,)),
        )


class RMSNormOpPlacementTest(unittest.TestCase):
    """RMSNorm needs whole rows and a whole weight on every rank."""

    def _gamma(self) -> Value:
        """Return the replicated [8192] RMSNorm weight."""

        return _placed((HIDDEN_SIZE,), (HIDDEN,), Replicate(), role=TensorRole.WEIGHT)

    def test_rejects_split_weight(self) -> None:
        """Each rank scales its rows by all 8192 weights."""

        split_gamma = _placed(
            (LOCAL_HEADS_WIDTH,),
            (HIDDEN,),
            Shard("hidden"),
            global_shape=(HIDDEN_SIZE,),
            role=TensorRole.WEIGHT,
        )
        with self.assertRaisesRegex(ValueError, "weight must be replicated"):
            RMSNormOp(operands=(_replicated_hidden(), split_gamma), eps=1e-5)

    def test_rejects_partial_x(self) -> None:
        """The mean square of a partial sum is not that of the sum."""

        partial_x = _placed((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN), Partial())
        with self.assertRaisesRegex(ValueError, "all_reduce it first"):
            RMSNormOp(operands=(partial_x, self._gamma()), eps=1e-5)

    def test_rejects_split_last_dimension(self) -> None:
        """No rank would hold a whole row to normalize."""

        split_x = _placed(
            (1, 1, LOCAL_HEADS_WIDTH),
            (*TOKEN_AXES, HIDDEN),
            Shard("hidden"),
            global_shape=(1, 1, HIDDEN_SIZE),
        )
        with self.assertRaisesRegex(ValueError, "needs the whole dimension"):
            RMSNormOp(operands=(split_x, self._gamma()), eps=1e-5)

    def test_split_leading_dimension_is_kept(self) -> None:
        """Rows split along batch are normalized independently on each rank.

        With 16 requests over 8 ranks, each rank normalizes 2 whole rows.
        """

        batch_split_x = _placed(
            (2, 1, HIDDEN_SIZE),
            (*TOKEN_AXES, HIDDEN),
            Shard("batch"),
            global_shape=(16, 1, HIDDEN_SIZE),
        )
        op = RMSNormOp(operands=(batch_split_x, self._gamma()), eps=1e-5)
        self.assertEqual(op.results[0].type, batch_split_x.type)


class AllReduceOpTest(unittest.TestCase):
    """aloi.all_reduce turns a partial sum into the replicated total."""

    def test_partial_becomes_replicated(self) -> None:
        """Only the placement changes; shape, dtype and axes are kept."""

        partial = _placed((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN), Partial())
        op = AllReduceOp(operands=(partial,))
        self.assertEqual(op.results[0].type, _replicated_hidden().type)
        self.assertIs(op.x, partial)
        self.assertEqual(op.NAME, "aloi.all_reduce")
        self.assertEqual(op.attributes(), {})

    def test_rejects_operand_that_is_not_partial(self) -> None:
        """Summing a whole tensor over 8 ranks would multiply it by 8."""

        for operand in (_replicated_hidden(), _sharded_heads()):
            with self.subTest(placement=operand.type.placement):
                with self.assertRaisesRegex(ValueError, "partial sum"):
                    AllReduceOp(operands=(operand,))


def _unary_ops() -> tuple[tuple[str, Callable[[Value], Operation]], ...]:
    """Return a builder for each one-operand op that keeps its operand's type.

    ActivationOp needs a ``function`` argument and the others do not, so a
    test loops over builders rather than op classes.

    Returns:
        ``(label, build)`` pairs; ``build(x)`` returns the op applied to x.
    """

    return (
        (
            "activation",
            lambda x: ActivationOp(operands=(x,), function=ActivationKind.SILU),
        ),
        ("softmax", lambda x: SoftmaxOp(operands=(x,))),
    )


def _print_one(op: Operation) -> str:
    """Return the printed line of ``op``, as the only op of a function.

    Args:
        op: An op whose operands are function arguments.

    Returns:
        The op's line, without indentation.
    """

    function = Function(
        name="f", operands=op.operands, operations=(op,), results=op.results
    )
    return print_function(function).splitlines()[1].strip()


class UnaryElementwiseOpTest(unittest.TestCase):
    """activation and softmax keep the operand's type, except the role."""

    def test_type_is_kept(self) -> None:
        """A sharded tensor stays sharded; a weight-typed input becomes activation.

        The shard is attention scores [1, 1, 8, 2048]: 8 of 64 heads per rank,
        split on dimension 2, so softmax still sees whole rows of 2048.
        """

        scores = _placed(
            (1, 1, 8, 2048),
            (*TOKEN_AXES, "q_head", "context"),
            Shard("q_head"),
            global_shape=(1, 1, 64, 2048),
        )
        for label, build in _unary_ops():
            with self.subTest(op=label):
                self.assertEqual(build(scores).results[0].type, scores.type)
                weight = _placed((4,), (HIDDEN,), Replicate(), role=TensorRole.WEIGHT)
                result = build(weight).results[0].type
                self.assertIs(result.role, TensorRole.ACTIVATION)

    def test_rejects_partial_sum(self) -> None:
        """Neither op is linear, so the ranks must add their parts first."""

        partial = _placed((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN), Partial())
        for label, build in _unary_ops():
            with self.subTest(op=label):
                with self.assertRaisesRegex(ValueError, "partial sum"):
                    build(partial)

    def test_softmax_needs_whole_last_dimension(self) -> None:
        """Softmax sums over the last dimension; an activation needs no other element."""

        split_last = _sharded_heads()  # q_head*head_dim split on the last dim
        with self.assertRaisesRegex(ValueError, "whole dimension"):
            SoftmaxOp(operands=(split_last,))
        with self.assertRaisesRegex(ValueError, "dimension to normalize"):
            SoftmaxOp(operands=(_value(),))
        activation = ActivationOp(operands=(split_last,), function=ActivationKind.SILU)
        self.assertEqual(activation.results[0].type, split_last.type)


class ActivationOpTest(unittest.TestCase):
    """One op covers every activation formula; the kind is an attribute."""

    def test_every_kind_keeps_the_type(self) -> None:
        """The formula does not change the shape, axes or placement rules."""

        x = _sharded_heads()
        for kind in ActivationKind:
            with self.subTest(kind=kind):
                op = ActivationOp(operands=(x,), function=kind)
                self.assertEqual(op.results[0].type, x.type)
                self.assertEqual(op.attributes(), {"function": kind})

    def test_prints_the_kind_without_quotes(self) -> None:
        """ActivationKind is a str enum; the printer writes its value bare."""

        op = ActivationOp(
            operands=(_value(1, 4, name="x"),),
            function=ActivationKind.GELU_TANH,
            name_hint="act",
        )
        self.assertEqual(
            _print_one(op),
            "%act = aloi.activation(%x) {function = gelu_tanh} : tensor<1x4xfp16>",
        )


class CopyOpTest(unittest.TestCase):
    """copy produces a new value with the operand's type."""

    def test_type_is_kept(self) -> None:
        """A shard stays a shard, and the result is an activation."""

        x = _sharded_heads()
        self.assertEqual(CopyOp(operands=(x,)).results[0].type, x.type)
        weight = _placed((4,), (HIDDEN,), Replicate(), role=TensorRole.WEIGHT)
        self.assertIs(
            CopyOp(operands=(weight,)).results[0].type.role, TensorRole.ACTIVATION
        )

    def test_accepts_partial_sum(self) -> None:
        """Copying is linear: copy(a) + copy(b) == copy(a + b)."""

        partial = _placed((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN), Partial())
        self.assertEqual(CopyOp(operands=(partial,)).results[0].type, partial.type)


class ReshapeOpTest(unittest.TestCase):
    """reshape reinterprets the same row-major elements with a new shape."""

    def test_same_element_count(self) -> None:
        """(2, 3) holds six elements, so (6,) and (3, 2) are both valid."""

        x = _value(2, 3)
        for shape in ((6,), (3, 2), (1, 2, 3)):
            with self.subTest(shape=shape):
                op = ReshapeOp(operands=(x,), shape=shape)
                self.assertEqual(
                    op.results[0].type, TensorType(shape=shape, dtype=DType.FP16)
                )

    def test_one_element_becomes_scalar(self) -> None:
        """A (1,) tensor and a scalar both hold one element, so either fits the other."""

        op = ReshapeOp(operands=(_value(1),), shape=())
        self.assertEqual(op.results[0].type, TensorType(shape=(), dtype=DType.FP16))

    def test_drops_axes(self) -> None:
        """[batch, seq_len, hidden] -> [batch, hidden] has no per-dim names."""

        x = _annotated((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN))
        result = ReshapeOp(operands=(x,), shape=(1, HIDDEN_SIZE)).results[0].type
        self.assertEqual(result.axes, ())

    def test_rejects_different_element_count(self) -> None:
        """Six elements cannot fill four or eight."""

        for shape in ((4,), (2, 4), (-2, -3)):
            with self.subTest(shape=shape):
                with self.assertRaisesRegex(ValueError, "cannot reshape"):
                    ReshapeOp(operands=(_value(2, 3),), shape=shape)

    def test_rejects_split_and_partial(self) -> None:
        """A reshape can merge the split dimension, so only whole tensors qualify."""

        partial = _placed((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN), Partial())
        for x in (_sharded_heads(), partial):
            with self.subTest(placement=x.type.placement):
                with self.assertRaisesRegex(ValueError, "only whole tensors"):
                    ReshapeOp(operands=(x,), shape=(prod(x.type.shape),))

    def test_prints_the_shape(self) -> None:
        """The shape attribute prints as a list."""

        x = _value(2, 3, name="x")
        op = ReshapeOp(operands=(x,), shape=(6,), name_hint="flat")
        self.assertEqual(
            _print_one(op), "%flat = aloi.reshape(%x) {shape = [6]} : tensor<6xfp16>"
        )


class BinaryElementwiseOpTest(unittest.TestCase):
    """add and mul combine two identically typed tensors."""

    def test_matching_operands(self) -> None:
        """The residual add of whole tensors and the SwiGLU product of shards."""

        residual = AddOp(operands=(_replicated_hidden(), _replicated_hidden()))
        self.assertEqual(residual.results[0].type, _replicated_hidden().type)
        gated = MulOp(operands=(_sharded_heads(), _sharded_heads()))
        self.assertEqual(gated.results[0].type, _sharded_heads().type)
        self.assertEqual((AddOp.NAME, MulOp.NAME), ("aloi.add", "aloi.mul"))

    def test_same_value_twice_and_scalars(self) -> None:
        """Squaring reads one value twice, and two scalars combine into a scalar."""

        x = _value(2, 3)
        scalar = _value()
        for op_class in (AddOp, MulOp):
            with self.subTest(op=op_class.NAME):
                self.assertEqual(op_class(operands=(x, x)).results[0].type, x.type)
                self.assertEqual(
                    op_class(operands=(scalar, _value())).results[0].type,
                    scalar.type,
                )

    def test_rejects_normalization_epsilon(self) -> None:
        """Only RMSNormOp owns eps; passing it to an addition is a TypeError."""

        with self.assertRaisesRegex(TypeError, "eps"):
            AddOp(operands=(_value(3), _value(3)), eps=1e-5)  # type: ignore[call-arg]

    def test_unannotated_operand_gives_unannotated_result(self) -> None:
        """Axes are only kept when both operands have them."""

        op = AddOp(operands=(_replicated_hidden(), _value(1, 1, HIDDEN_SIZE)))
        self.assertEqual(op.results[0].type.axes, ())

    def test_rejections(self) -> None:
        """No broadcasting, no dtype mixing, no mismatched axes or placements."""

        partial = _placed((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, HIDDEN), Partial())
        cases = {
            "shapes differ": (_value(1, HIDDEN_SIZE), _value(1, 1, HIDDEN_SIZE)),
            "dtypes differ": (
                _value(1, HIDDEN_SIZE),
                _value(1, HIDDEN_SIZE, dtype=DType.BF16),
            ),
            "axes differ": (
                _replicated_hidden(),
                _placed((1, 1, HIDDEN_SIZE), (*TOKEN_AXES, Q_PROJ_OUT), Replicate()),
            ),
            "placements differ": (
                _sharded_heads(),
                _placed(
                    (1, 1, LOCAL_HEADS_WIDTH), (*TOKEN_AXES, Q_PROJ_OUT), Replicate()
                ),
            ),
            "partial sums": (partial, partial),
        }
        for name, operands in cases.items():
            for op_class in (AddOp, MulOp):
                with self.subTest(case=name, op=op_class.NAME):
                    with self.assertRaises(ValueError):
                        op_class(operands=operands)


if __name__ == "__main__":
    unittest.main()
