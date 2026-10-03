"""Tests for the semantic dialect: aloi.linear and aloi.rms_norm."""

from __future__ import annotations

import unittest

from vllm_cent.ir.base import Value
from vllm_cent.ir.semantic import LinearOp, RMSNormOp
from vllm_cent.ir.types import DType, TensorType

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


if __name__ == "__main__":
    unittest.main()
