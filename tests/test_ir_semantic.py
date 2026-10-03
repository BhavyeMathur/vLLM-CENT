"""Tests for the semantic dialect: aloi.linear, aloi.rms_norm, aloi.all_reduce."""

from __future__ import annotations

import unittest

from vllm_cent.ir.base import Value
from vllm_cent.ir.semantic import AllReduceOp, LinearOp, RMSNormOp
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

    def test_rejects_non_positive_eps(self) -> None:
        """The eps term keeps the square root away from zero, so it must be > 0."""

        for eps in (0.0, -1e-5):
            with self.subTest(eps=eps), self.assertRaises(ValueError):
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


if __name__ == "__main__":
    unittest.main()
