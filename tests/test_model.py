import unittest

from vllm_cent import LlamaModelSpec


class LlamaModelSpecTests(unittest.TestCase):
    """Test Llama model dimensions."""

    def test_derives_head_size_for_grouped_query_attention(self) -> None:
        """Derive head width from hidden width and query heads."""

        # A width of 64 over four query heads produces 16-value heads. Two KV
        # heads mean that each KV head is shared by two query heads.
        model = LlamaModelSpec(
            hidden_size=64,
            num_attention_heads=4,
            num_kv_heads=2,
            intermediate_size=128,
        )

        self.assertEqual(model.head_size, 16)

    def test_accepts_multi_head_attention(self) -> None:
        """Accept ordinary multi-head attention."""

        # Ordinary multi-head attention has one KV head per query head.
        model = LlamaModelSpec(
            hidden_size=64,
            num_attention_heads=4,
            num_kv_heads=4,
            intermediate_size=128,
        )

        self.assertEqual(model.num_attention_heads, model.num_kv_heads)

    def test_rejects_non_positive_dimensions(self) -> None:
        """Reject zero and negative model dimensions."""

        # Each tuple makes one required dimension zero or negative.
        invalid_dimensions = (
            (0, 1, 1, 1),
            (1, 0, 1, 1),
            (1, 1, 0, 1),
            (1, 1, 1, 0),
            (-1, 1, 1, 1),
        )

        for hidden, query_heads, kv_heads, intermediate in invalid_dimensions:
            with self.subTest(
                hidden=hidden,
                query_heads=query_heads,
                kv_heads=kv_heads,
                intermediate=intermediate,
            ):
                with self.assertRaises(ValueError):
                    LlamaModelSpec(
                        hidden_size=hidden,
                        num_attention_heads=query_heads,
                        num_kv_heads=kv_heads,
                        intermediate_size=intermediate,
                    )

    def test_rejects_fractional_head_width(self) -> None:
        """Require equal-width query heads."""

        # Ten hidden values cannot be split evenly between three query heads.
        with self.assertRaisesRegex(ValueError, "hidden_size must be divisible"):
            LlamaModelSpec(
                hidden_size=10,
                num_attention_heads=3,
                num_kv_heads=1,
                intermediate_size=16,
            )

    def test_rejects_uneven_grouped_query_attention(self) -> None:
        """Require equal-sized grouped-query attention groups."""

        # Six query heads cannot share four KV heads in equal-size groups.
        with self.assertRaisesRegex(
            ValueError, "num_attention_heads must be divisible"
        ):
            LlamaModelSpec(
                hidden_size=24,
                num_attention_heads=6,
                num_kv_heads=4,
                intermediate_size=32,
            )


if __name__ == "__main__":
    unittest.main()
