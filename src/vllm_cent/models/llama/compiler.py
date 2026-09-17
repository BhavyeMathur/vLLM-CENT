"""Combine the steps that compile one Llama transformer block."""

from ...cent import CentProgram, CentProgramBuilder
from ...lowering import (
    lower_accumulate,
    lower_load_bank_group_vector,
    lower_rms_norm,
    lower_store_bank_group_vector,
    lower_weight_gemv,
)
from ...lowering.transformer import (
    lower_attention_output,
    lower_kv_cache_update,
    lower_rotary_embedding,
    lower_score_gemv,
    lower_softmax,
)
from ...request import CompileRequest
from .feed_forward import _lower_silu_product
from .planning import _LlamaCompilePlan, _create_compile_plan

__all__ = ["compile_llama_transformer_block"]


def _lower_self_attention(
    builder: CentProgramBuilder,
    plan: _LlamaCompilePlan,
) -> None:
    """Lower Llama's normalized self-attention and first residual.

    Args:
        builder: Program builder that receives the instructions.
        plan: Checked Llama dimensions and memory bindings.
    """

    operations = plan.self_attention

    # Attention first normalizes input x. Multiplying that value by three weight
    # matrices produces the query (Q), key (K), and value (V) vectors. K and V
    # are narrower when several query heads share one KV head.
    lower_rms_norm(
        builder,
        operations.normalization,
    )
    lower_weight_gemv(
        builder,
        operations.query_projection,
    )
    lower_weight_gemv(
        builder,
        operations.key_projection,
    )
    lower_weight_gemv(
        builder,
        operations.value_projection,
    )

    # TODO(dataflow): Repack the Q, K, and V accumulator results.
    #
    # RD_MAC produces one result slot per accumulator register. Attention uses
    # burst-packed vectors instead. The paper does not define the instruction
    # sequence or ordering that converts between those layouts. The repacker
    # must write every occupied slot and zero every non-logical lane before the
    # query, key, or value binding is valid.

    # RoPE adds token-position information to Q and K. K and V then enter their
    # caches. Q times the key cache produces scores; softmax converts scores to
    # weights; and those weights combine the cached values.
    lower_rotary_embedding(builder, operations.attention.rotary_embedding)
    lower_kv_cache_update(builder, operations.attention.kv_cache_update)
    lower_score_gemv(builder, operations.attention.score_gemv)
    lower_softmax(builder, operations.attention.softmax)
    lower_attention_output(builder, operations.attention.output)

    # WO combines the attention heads into one hidden-width vector. Adding the
    # original input creates the attention residual output.
    lower_weight_gemv(
        builder,
        operations.output_projection,
    )
    # TODO(dataflow): Repack the WO accumulator result into ``query``.
    #
    # The following residual addition needs a packed hidden-width vector. The
    # paper does not define how RD_MAC results become that packed vector. The
    # repacker must also overwrite its final padding lanes with zero.
    lower_accumulate(
        builder,
        operations.residual,
    )
    # The FFN needs most of the Shared Buffer for its expanded vector. Preserve
    # the attention residual in its planned DRAM workspace before reusing it.
    lower_store_bank_group_vector(
        builder,
        operations.store_residual,
    )


def _lower_feed_forward(
    builder: CentProgramBuilder,
    plan: _LlamaCompilePlan,
) -> None:
    """Lower Llama's normalized gated FFN and final residual.

    Args:
        builder: Program builder that receives the instructions.
        plan: Checked Llama dimensions and memory bindings.
    """

    operations = plan.feed_forward

    # The feed-forward layer normalizes that result. W1 and W3 expand it, SiLU
    # gates the expanded values, and W2 returns them to hidden width. The last
    # addition produces the block output.
    lower_rms_norm(
        builder,
        operations.normalization,
    )
    lower_weight_gemv(
        builder,
        operations.gate_projection,
    )
    lower_weight_gemv(
        builder,
        operations.up_projection,
    )
    _lower_silu_product(
        builder,
        operations.silu_product,
    )
    lower_weight_gemv(
        builder,
        operations.down_projection,
    )
    # TODO(dataflow): Repack the W2 accumulator result into ``normalized``.
    #
    # ACC consumes a packed vector, while RD_MAC returns accumulator slots.
    # The required conversion is the same unresolved boundary as Q, K, and V,
    # including the requirement to zero every padding lane.
    lower_load_bank_group_vector(
        builder,
        operations.load_residual,
    )
    lower_accumulate(
        builder,
        operations.residual,
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
        TypeError: If the request does not describe a Llama model.
        ValueError: If the model does not fit the selected hardware layout.
    """

    # TODO(runtime): Bind loaded model weights and request values.
    #
    # This compiler now names intermediate storage, but the future vLLM worker
    # must still load actual tensors into those addresses before execution. It
    # must pack logical vectors into complete slots and write explicit zeros in
    # every padding lane.

    plan = _create_compile_plan(request)
    builder = CentProgramBuilder(
        plan.context.hardware,
        plan.context.placement,
    )
    _lower_self_attention(builder, plan)
    _lower_feed_forward(builder, plan)
    return builder.finish()
