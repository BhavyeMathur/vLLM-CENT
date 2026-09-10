"""Describe the Llama sizes needed to compile one block."""

from dataclasses import dataclass

from ..base import ModelSpec

__all__ = ["LlamaModelSpec"]

# TODO(model): Add the Llama settings needed for numerical correctness.
#
# RMSNorm epsilon and RoPE theta/scaling are known missing inputs. Data type,
# activation choice, and optional projection biases still need design decisions.


@dataclass(frozen=True, slots=True, kw_only=True)
class LlamaModelSpec(ModelSpec):
    """Model sizes that determine the shape of one Llama block.

    Attributes:
        hidden_size: Values in each input and output vector. It must divide
            evenly among the query heads.
        num_attention_heads: Query heads in self-attention. The count must be a
            multiple of ``num_kv_heads`` so heads can be shared evenly.
        num_kv_heads: Distinct key and value heads. Several query heads may
            share one KV head.
        intermediate_size: Values produced inside the feed-forward layer.
    """

    hidden_size: int
    num_attention_heads: int
    num_kv_heads: int
    intermediate_size: int

    def __post_init__(self) -> None:
        """Check that the model sizes form complete attention heads.

        Raises:
            ValueError: If a size is empty or heads cannot be divided evenly.
        """

        if self.hidden_size < 1:
            raise ValueError("hidden_size must be at least 1")
        if self.num_attention_heads < 1:
            raise ValueError("num_attention_heads must be at least 1")
        if self.num_kv_heads < 1:
            raise ValueError("num_kv_heads must be at least 1")
        if self.intermediate_size < 1:
            raise ValueError("intermediate_size must be at least 1")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_kv_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_kv_heads")

    @property
    def head_size(self) -> int:
        """Return the number of values in one attention head.

        Returns:
            ``hidden_size`` divided by ``num_attention_heads``.
        """

        return self.hidden_size // self.num_attention_heads
