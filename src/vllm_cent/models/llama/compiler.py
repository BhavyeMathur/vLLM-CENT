"""Combine the steps that compile one Llama transformer block."""

from ...cent import (
    Accumulate,
    CentProgram,
    CentProgramBuilder,
    CentSharedBufferAddress,
    ceil_div,
)
from ...request import CompileRequest
from .attention import (
    _lower_kv_cache_update,
    _lower_output_gemv,
    _lower_rotary_embedding,
    _lower_score_gemv,
    _lower_softmax,
)
from .feed_forward import _lower_silu_product
from .linear import _lower_weight_gemv
from .normalization import _lower_rms_norm
from .planning import _create_context, _plan_memory

__all__ = ["compile_llama_transformer_block"]


def _lower_residual_add(
    builder: CentProgramBuilder, value_count: int
) -> None:
    """Add one vector to another vector of the same length.

    Args:
        builder: Program builder that receives the addition instruction.
        value_count: Number of values in each vector.

    Raises:
        ValueError: If ``value_count`` is less than 1.
    """

    # TODO(dataflow): Define where both input vectors are stored.
    #
    # ACC reads two Shared Buffer ranges. The code chooses adjacent ranges, but
    # does not yet copy the residual and new output into them.

    # TODO(ISA): Confirm how CENT performs residual addition.
    #
    # ACC can add vectors, but Figure 10 assigns residual addition to RISC-V.

    builder.append(
        Accumulate(
            operation_size=ceil_div(
                value_count, builder.hardware.burst_length
            ),
            destination=CentSharedBufferAddress(slot=0),
            # Store the second vector after the first so their buffer ranges do
            # not overlap.
            source=CentSharedBufferAddress(
                slot=ceil_div(value_count, builder.hardware.burst_length)
            ),
        )
    )


def compile_llama_transformer_block(request: CompileRequest) -> CentProgram:
    """Build instructions for one token passing through one Llama block.

    Args:
        request: Model sizes, CENT hardware, channel placement, and current
            token count.

    Returns:
        Structurally valid CENT program. TODOs mark missing numerical steps and
        runtime data connections.

    Raises:
        ValueError: If the model does not fit the selected hardware layout.
    """

    # TODO(runtime): Connect model tensors to program addresses.
    #
    # The request provides shapes, but not values. The program also does not say
    # where the input, weights, intermediate values, and output are stored.

    # Calculate shared sizes first, then reserve a separate DRAM range for each
    # tensor. This reports an oversized model before instructions are emitted.
    context = _create_context(request)
    layout = _plan_memory(context)
    builder = CentProgramBuilder(context.hardware, context.placement)

    # Attention first normalizes input x. Multiplying that value by three weight
    # matrices produces the query (Q), key (K), and value (V) vectors. K and V
    # are narrower when several query heads share one KV head.
    _lower_rms_norm(builder, context, layout.x, layout.x_copy, layout.sa_norm)
    _lower_weight_gemv(
        builder,
        layout.wq,
        context.model.hidden_size,
        context.model.hidden_size,
        context.hardware.accumulator_slots_per_bank,
    )
    _lower_weight_gemv(
        builder,
        layout.wk,
        context.model.hidden_size,
        context.kv_width,
        context.hardware.accumulator_slots_per_bank,
    )
    _lower_weight_gemv(
        builder,
        layout.wv,
        context.model.hidden_size,
        context.kv_width,
        context.hardware.accumulator_slots_per_bank,
    )

    # TODO(dataflow): Save Q, K, and V in separate locations.
    #
    # All three matrix-vector multiplications reuse the same result slots. Later
    # steps need three different tensors.

    # RoPE adds token-position information to Q and K. K and V then enter their
    # caches. Q times the key cache produces scores; softmax converts scores to
    # weights; and those weights combine the cached values.
    _lower_rotary_embedding(builder, context, layout)
    _lower_kv_cache_update(builder, context, layout)
    _lower_score_gemv(builder, context, layout)
    _lower_softmax(builder, context, layout)
    _lower_output_gemv(builder, context, layout)

    # WO combines the attention heads into one hidden-width vector. Adding the
    # original input creates the attention residual output.
    _lower_weight_gemv(
        builder,
        layout.wo,
        context.model.hidden_size,
        context.model.hidden_size,
        context.hardware.accumulator_slots_per_bank,
    )
    _lower_residual_add(builder, context.model.hidden_size)

    # The feed-forward layer normalizes that result. W1 and W3 expand it, SiLU
    # gates the expanded values, and W2 returns them to hidden width. The last
    # addition produces the block output.
    _lower_rms_norm(
        builder,
        context,
        layout.sa_copy,
        layout.sa_copy,
        layout.ffn_norm,
    )
    _lower_weight_gemv(
        builder,
        layout.w1,
        context.model.hidden_size,
        context.model.intermediate_size,
        context.hardware.accumulator_slots_per_bank,
        apply_activation=True,
    )
    _lower_weight_gemv(
        builder,
        layout.w3,
        context.model.hidden_size,
        context.model.intermediate_size,
        context.hardware.accumulator_slots_per_bank,
    )
    _lower_silu_product(builder, context, layout)
    _lower_weight_gemv(
        builder,
        layout.w2,
        context.model.intermediate_size,
        context.model.hidden_size,
        context.hardware.accumulator_slots_per_bank,
    )
    _lower_residual_add(builder, context.model.hidden_size)
    return builder.finish()
