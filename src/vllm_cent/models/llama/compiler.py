"""Combine the steps that compile one Llama transformer block."""

from ...cent import CentProgram, CentProgramBuilder
from ...lowering import (
    CentDramRowRange,
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

    context = plan.context
    row_counts = plan.row_counts
    memory = plan.memory
    buffers = plan.buffers
    attention = plan.attention

    # Attention first normalizes input x. Multiplying that value by three weight
    # matrices produces the query (Q), key (K), and value (V) vectors. K and V
    # are narrower when several query heads share one KV head.
    lower_rms_norm(
        builder,
        input_rows=CentDramRowRange(
            start_row=memory.x, row_count=row_counts.x
        ),
        work_rows=CentDramRowRange(
            start_row=memory.x_copy, row_count=row_counts.x
        ),
        weight_rows=CentDramRowRange(
            start_row=memory.sa_norm, row_count=row_counts.x
        ),
        input_buffer=buffers.input,
        # The scale is consumed before the normalized result overwrites it.
        scale_buffer=buffers.normalized,
        partial_sum_buffer=buffers.value,
        output_buffer=buffers.normalized,
        value_count=context.model.hidden_size,
    )
    lower_weight_gemv(
        builder,
        weights=CentDramRowRange(
            start_row=memory.wq, row_count=row_counts.wq
        ),
        input_buffer=buffers.normalized,
        output_buffer=buffers.query_result,
        vector_size=context.model.hidden_size,
        output_size=context.model.hidden_size,
    )
    lower_weight_gemv(
        builder,
        weights=CentDramRowRange(
            start_row=memory.wk, row_count=row_counts.wk
        ),
        input_buffer=buffers.normalized,
        output_buffer=buffers.key_result,
        vector_size=context.model.hidden_size,
        output_size=context.kv_width,
    )
    lower_weight_gemv(
        builder,
        weights=CentDramRowRange(
            start_row=memory.wv, row_count=row_counts.wv
        ),
        input_buffer=buffers.normalized,
        output_buffer=buffers.value_result,
        vector_size=context.model.hidden_size,
        output_size=context.kv_width,
    )

    # TODO(dataflow): Repack the Q, K, and V accumulator results.
    #
    # RD_MAC produces one result slot per accumulator register. Attention uses
    # burst-packed vectors instead. The paper does not define the instruction
    # sequence or ordering that converts between those layouts.

    # RoPE adds token-position information to Q and K. K and V then enter their
    # caches. Q times the key cache produces scores; softmax converts scores to
    # weights; and those weights combine the cached values.
    lower_rotary_embedding(
        builder, attention.spec, attention.rows, attention.buffers
    )
    lower_kv_cache_update(
        builder, attention.spec, attention.rows, attention.buffers
    )
    lower_score_gemv(
        builder, attention.spec, attention.rows, attention.buffers
    )
    lower_softmax(
        builder, attention.spec, attention.rows, attention.buffers
    )
    lower_attention_output(
        builder, attention.spec, attention.rows, attention.buffers
    )

    # WO combines the attention heads into one hidden-width vector. Adding the
    # original input creates the attention residual output.
    lower_weight_gemv(
        builder,
        weights=CentDramRowRange(
            start_row=memory.wo, row_count=row_counts.wo
        ),
        input_buffer=buffers.normalized,
        output_buffer=buffers.query_result,
        vector_size=context.model.hidden_size,
        output_size=context.model.hidden_size,
    )
    # TODO(dataflow): Repack the WO accumulator result into ``query``.
    #
    # The following residual addition needs a packed hidden-width vector. The
    # paper does not define how RD_MAC results become that packed vector.
    lower_accumulate(
        builder,
        destination=buffers.query,
        source=buffers.input,
        value_count=context.model.hidden_size,
    )
    # The FFN needs most of the Shared Buffer for its expanded vector. Preserve
    # the attention residual in its planned DRAM workspace before reusing it.
    lower_store_bank_group_vector(
        builder,
        rows=CentDramRowRange(
            start_row=memory.sa,
            row_count=row_counts.hidden_vector,
        ),
        buffer=buffers.query,
        value_count=context.model.hidden_size,
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

    context = plan.context
    row_counts = plan.row_counts
    memory = plan.memory
    buffers = plan.buffers

    # The feed-forward layer normalizes that result. W1 and W3 expand it, SiLU
    # gates the expanded values, and W2 returns them to hidden width. The last
    # addition produces the block output.
    lower_rms_norm(
        builder,
        input_rows=CentDramRowRange(
            start_row=memory.sa_copy, row_count=row_counts.hidden_vector
        ),
        work_rows=CentDramRowRange(
            start_row=memory.sa_copy, row_count=row_counts.hidden_vector
        ),
        weight_rows=CentDramRowRange(
            start_row=memory.ffn_norm,
            row_count=row_counts.hidden_vector,
        ),
        input_buffer=buffers.query,
        scale_buffer=buffers.normalized,
        partial_sum_buffer=buffers.value,
        output_buffer=buffers.normalized,
        value_count=context.model.hidden_size,
    )
    lower_weight_gemv(
        builder,
        weights=CentDramRowRange(
            start_row=memory.w1, row_count=row_counts.w1
        ),
        input_buffer=buffers.normalized,
        output_buffer=buffers.ffn_gate,
        activated_output_buffer=buffers.ffn_gate_sigmoid,
        vector_size=context.model.hidden_size,
        output_size=context.model.intermediate_size,
    )
    lower_weight_gemv(
        builder,
        weights=CentDramRowRange(
            start_row=memory.w3, row_count=row_counts.w3
        ),
        input_buffer=buffers.normalized,
        output_buffer=buffers.ffn_up,
        vector_size=context.model.hidden_size,
        output_size=context.model.intermediate_size,
    )
    _lower_silu_product(
        builder,
        context,
        memory,
        buffers.ffn_product,
    )
    lower_weight_gemv(
        builder,
        weights=CentDramRowRange(
            start_row=memory.w2, row_count=row_counts.w2
        ),
        input_buffer=buffers.ffn_product,
        output_buffer=buffers.query_result,
        vector_size=context.model.intermediate_size,
        output_size=context.model.hidden_size,
    )
    # TODO(dataflow): Repack the W2 accumulator result into ``normalized``.
    #
    # ACC consumes a packed vector, while RD_MAC returns accumulator slots.
    # The required conversion is the same unresolved boundary as Q, K, and V.
    lower_load_bank_group_vector(
        builder,
        rows=CentDramRowRange(
            start_row=memory.sa,
            row_count=row_counts.hidden_vector,
        ),
        buffer=buffers.query,
        value_count=context.model.hidden_size,
    )
    lower_accumulate(
        builder,
        destination=buffers.normalized,
        source=buffers.query,
        value_count=context.model.hidden_size,
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

    # TODO(runtime): Bind loaded model weights and request values.
    #
    # This compiler now names intermediate storage, but the future vLLM worker
    # must still load actual tensors into those addresses before execution.

    plan = _create_compile_plan(request)
    builder = CentProgramBuilder(
        plan.context.hardware,
        plan.context.placement,
    )
    _lower_self_attention(builder, plan)
    _lower_feed_forward(builder, plan)
    return builder.finish()
