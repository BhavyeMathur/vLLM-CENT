"""A toy Llama decoder block written with ALOI custom ops.

The block grows one layer at a time as the compiler learns more ops. Today it
holds only the attention input norm and the query projection::

    x -> input_layernorm -> q_proj -> q

Every layer calls an ``aloi::*`` custom op instead of ``nn.RMSNorm`` or
``nn.Linear``. ``torch.export`` keeps a custom op as one graph node, while the
built-in layers may be decomposed into ``pow``/``mean``/``mm`` and lose their
meaning.

``build_meta_block`` creates the parameters on the meta device: they carry a
shape and dtype but no storage, so the real 70B dimensions export without
allocating memory.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from vllm_cent.frontend import custom_ops

from .config import LlamaConfig

__all__ = [
    "ToyLinear",
    "ToyLlama70BBlock",
    "ToyRMSNorm",
    "build_meta_block",
    "example_inputs",
]


class ToyRMSNorm(nn.Module):
    """RMSNorm over the last dimension, with a learned per-feature scale.

    Attributes:
        weight: Scale for each feature, shape ``[hidden_size]``.
        eps: Constant added to the mean square before the square root.
    """

    def __init__(self, hidden_size: int, eps: float) -> None:
        """Create the layer.

        Args:
            hidden_size: Length of the normalized last dimension.
            eps: Constant added to the mean square before the square root.
        """

        super().__init__()
        # Same initial value as nn.RMSNorm: every feature is scaled by 1.
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        """Normalize ``x`` and scale it by ``weight``.

        Args:
            x: Activations, shape ``[*, hidden_size]``.

        Returns:
            A tensor with the same shape and dtype as ``x``.
        """

        # @custom_op turns the function into a CustomOpDef object, and calling
        # that object is typed as returning Any. The annotation tells the type
        # checker that the result is a Tensor again.
        normalized: Tensor = custom_ops.rms_norm(x, self.weight, self.eps)
        return normalized


class ToyLinear(nn.Module):
    """Linear layer without bias; no Llama projection has a bias.

    Attributes:
        weight: Shape ``[out_features, in_features]``, the ``nn.Linear``
            layout. The layer computes ``x @ weight.T``.
    """

    def __init__(self, in_features: int, out_features: int) -> None:
        """Create the layer.

        Args:
            in_features: Length of each input vector.
            out_features: Length of each output vector.
        """

        super().__init__()
        # torch.empty leaves the values uninitialized, like nn.Linear before
        # reset_parameters. On the meta device there are no values at all;
        # a numeric test fills the weight itself.
        self.weight = nn.Parameter(torch.empty(out_features, in_features))

    def forward(self, x: Tensor) -> Tensor:
        """Compute ``x @ weight.T``.

        Args:
            x: Activations, shape ``[*, in_features]``.

        Returns:
            A tensor of shape ``[*, out_features]``.
        """

        # Annotated for the same reason as in ToyRMSNorm.forward.
        projected: Tensor = custom_ops.linear(x, self.weight)
        return projected


class ToyLlama70BBlock(nn.Module):
    """The part of one Llama decoder block that ALOI can compile so far.

    Submodule names match Hugging Face ``LlamaDecoderLayer``, so parameter
    names read like checkpoint keys: ``input_layernorm.weight`` and
    ``q_proj.weight``. ``q_proj`` sits directly on the block instead of under
    ``self_attn`` to keep the names in the IR short.

    Attributes:
        config: Model dimensions the layers were built from.
        input_layernorm: RMSNorm applied before attention.
        q_proj: Query projection, weight ``[num_heads * head_dim, hidden]``.
    """

    def __init__(self, config: LlamaConfig) -> None:
        """Create the layers.

        Args:
            config: Model dimensions.
        """

        super().__init__()
        self.config = config
        self.input_layernorm = ToyRMSNorm(config.hidden_size, config.rms_norm_eps)
        # One head_dim-long query per head. For Llama2-70B this is
        # 64 * 128 = 8192, the same as hidden_size, but that is not true of
        # every model, so the size is derived from the heads.
        self.q_proj = ToyLinear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
        )

    def forward(self, x: Tensor) -> Tensor:
        """Compute the query projection of the normalized input.

        Args:
            x: Activations, shape ``[batch, seq_len, hidden_size]``.

        Returns:
            Queries for all heads, shape
            ``[batch, seq_len, num_attention_heads * head_dim]``.
        """

        # Calling a submodule goes through nn.Module.__call__, which is typed
        # as returning Any; the annotations restore Tensor.
        hidden_states: Tensor = self.input_layernorm(x)
        query: Tensor = self.q_proj(hidden_states)
        return query


def build_meta_block(config: LlamaConfig) -> ToyLlama70BBlock:
    """Create a block whose parameters live on the meta device.

    Args:
        config: Model dimensions.

    Returns:
        The block, with parameters in ``config.dtype``.
    """

    # Every tensor created inside this context, including the parameters
    # made in __init__, is a meta tensor: shape and dtype, but no memory.
    with torch.device("meta"):
        block = ToyLlama70BBlock(config)
    # Parameters are created in the default dtype (fp32); convert them to the
    # model dtype afterwards, the same way a checkpoint loader would.
    return block.to(config.dtype)


def example_inputs(
    config: LlamaConfig,
    batch: int = 1,
    seq_len: int = 1,
) -> tuple[Tensor]:
    """Create meta sample inputs for ``torch.export``.

    The defaults describe decoding: one request producing one new token.

    Args:
        config: Model dimensions.
        batch: Number of requests.
        seq_len: Number of tokens per request.

    Returns:
        The ``forward`` arguments: ``x`` of shape
        ``[batch, seq_len, hidden_size]``.
    """

    x = torch.empty(
        batch,
        seq_len,
        config.hidden_size,
        dtype=config.dtype,
        device="meta",
    )
    return (x,)
