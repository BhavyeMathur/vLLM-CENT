"""Adaptor: fuse gqa_qk -> softmax -> gqa_sv into one attention op.

Models write attention decomposed (scores, softmax, weighted values): each
step has its own shape, axes, and tensor-parallel rule, and a cost model can
price it. The CENT pipeline currently lowers only the fused, batch/head/
sequence-order ``AttentionOp``, so ``compile_graph()`` runs this pass before
legalization to rewrite each chain it can convert.

TODO(modular attention): This adaptor exists to reuse the pipeline's
hand-written fused attention lowering. The goal is to lower the decomposed
ops one by one (score GEMV, softmax, value GEMV) and let fusion be an
optimization over them, not a single hand-written emitter. Remove the
adaptor once the backend lowers the decomposed ops.
"""

from ..ir import (
    AttentionOp,
    Function,
    GqaQKOp,
    GqaSVOp,
    Operation,
    Replicate,
    ReshapeOp,
    SoftmaxOp,
    Value,
)

__all__ = ["fuse_attention"]


def _order_preserving(shape: tuple[int, ...]) -> bool:
    """Tell whether swapping dimensions 1 and 2 keeps the row-major order.

    Token-first ``[batch, tokens, heads, width]`` and batch/head/sequence
    ``[batch, heads, tokens, width]`` list the same elements in the same
    order when either swapped dimension is 1; then a reshape converts them.

    Args:
        shape: A rank-4 shape.

    Returns:
        Whether a reshape is enough to change the layout.

    Examples:
        A decode query ``[1, 1, 8, 128]`` (one token) and a cache with one
        KV head ``[1, 2048, 1, 128]`` qualify; ``[1, 5, 2, 3]`` does not.
    """

    return shape[1] == 1 or shape[2] == 1


def _swap_middle(shape: tuple[int, ...]) -> tuple[int, ...]:
    """Return a rank-4 shape with dimensions 1 and 2 swapped.

    Args:
        shape: A rank-4 shape.

    Returns:
        ``(shape[0], shape[2], shape[1], shape[3])``.
    """

    return (shape[0], shape[2], shape[1], shape[3])


def _chain(function: Function, qk: GqaQKOp) -> tuple[SoftmaxOp, GqaSVOp] | None:
    """Return the softmax and gqa_sv that only consume this gqa_qk, if fusable.

    Args:
        function: The function containing ``qk``.
        qk: A score op.

    Returns:
        The rest of the chain, or None when it is not a private
        qk -> softmax -> sv chain of whole tensors whose layouts a reshape
        can convert.
    """

    scores = qk.results[0]
    score_users = function.users(scores)
    if len(score_users) != 1 or not isinstance(score_users[0], SoftmaxOp):
        return None
    softmax = score_users[0]
    probs = softmax.results[0]
    prob_users = function.users(probs)
    if len(prob_users) != 1 or not isinstance(prob_users[0], GqaSVOp):
        return None
    sv = prob_users[0]
    # The scores and probabilities disappear inside the fused op, so nothing
    # else may read or return them.
    if sv.probs is not probs or scores in function.results or probs in function.results:
        return None
    tensors = (qk.q, qk.k_cache, sv.v_cache, sv.results[0])
    # TODO(attention TP): With tensor parallelism the heads are split across
    # ranks. ReshapeOp and AttentionOp accept only whole tensors, so such
    # chains stay decomposed until the fused form gets a placement rule.
    if any(not isinstance(t.type.placement, Replicate) for t in tensors):
        return None
    # TODO(attention layout): The IR keeps the token-first layout, KV cache
    # [batch, context, kv_heads, width], while AttentionOp expects
    # [batch, kv_heads, context, width]. The IR has no transpose, so this
    # adaptor converts with reshapes only, and a reshape moves no data. That
    # works only when kv_heads or context is 1: a Llama 70B cache at tp=1,
    # [1, 2048, 8, 128], needs real data movement and stays decomposed. The
    # GQA grouping itself (64 query heads over 8 KV heads) is supported by
    # AttentionOp. Open question: lower gqa_qk / gqa_sv directly on the
    # token-first layout (the intended fix, see the module TODO), or teach the
    # fused lowering to read token-first caches. A per-step transpose of the
    # whole cache is not an acceptable fix.
    #
    # Swapping dimensions 1 and 2 by a reshape is only valid when one is 1.
    if not all(_order_preserving(t.type.shape) for t in tensors[:3]):
        return None
    return softmax, sv


def fuse_attention(function: Function) -> Function:
    """Replace each fusable gqa_qk -> softmax -> gqa_sv chain by AttentionOp.

    A chain becomes: reshapes of q, k_cache and v_cache to batch/head/
    sequence order, one AttentionOp (same scale, every cached key visible),
    and a reshape back to token-first order. The final reshape keeps the
    gqa_sv's name hint, so the attention result keeps its printed name.
    Chains that cannot be converted stay as they are; legalization then
    reports their gqa_qk.

    Args:
        function: Function to rewrite; it is not modified.

    Returns:
        A new function with the fusable chains replaced, or ``function``
        itself when there is nothing to fuse.

    Examples:
        One decode token with 4 query heads over 5 cached positions of one
        KV head: q ``[1, 1, 4, 2]`` becomes ``[1, 4, 1, 2]``, k ``[1, 5, 1, 2]``
        becomes ``[1, 1, 5, 2]``, and the ``[1, 4, 1, 3]`` attention result
        becomes ``[1, 1, 4, 3]`` again.
    """

    # Map each chain's gqa_qk to its softmax and gqa_sv, and mark those two as
    # consumed. Operations compare by identity, so they can be dictionary keys.
    chains: dict[Operation, tuple[SoftmaxOp, GqaSVOp]] = {}
    consumed: set[Operation] = set()
    for op in function.operations:
        if isinstance(op, GqaQKOp):
            chain = _chain(function, op)
            if chain is not None:
                chains[op] = chain
                consumed.update(chain)
    # Values and functions compare by identity; returning the same function
    # keeps two compilations of one unchanged function comparable.
    if not chains:
        return function

    # Rebuild the function, mapping every old value to its replacement.
    arguments = tuple(Value(type=arg.type, name=arg.name) for arg in function.operands)
    new_values: dict[Value, Value] = dict(
        zip(function.operands, arguments, strict=True)
    )
    operations: list[Operation] = []
    for op in function.operations:
        if op in consumed:
            continue
        if not isinstance(op, GqaQKOp) or op not in chains:
            rebuilt = op.with_operands(new_values[operand] for operand in op.operands)
            operations.append(rebuilt)
            new_values.update(zip(op.results, rebuilt.results, strict=True))
            continue

        _, sv = chains[op]
        hint = sv.name_hint or "attention"
        to_bhsd = tuple(
            ReshapeOp(
                operands=(new_values[t],),
                shape=_swap_middle(t.type.shape),
                name_hint=f"{hint}.{label}",
            )
            for t, label in (
                (op.q, "q_bhsd"),
                (op.k_cache, "k_bhsd"),
                (sv.v_cache, "v_bhsd"),
            )
        )
        # gqa_qk reads every cached position. In decode the cache already
        # holds the new token, so all positions are valid and no causal mask
        # is needed.
        attention = AttentionOp(
            operands=tuple(reshape.results[0] for reshape in to_bhsd),
            scale=op.scale,
            name_hint=f"{hint}.fused",
        )
        back = ReshapeOp(
            operands=attention.results,
            shape=_swap_middle(attention.results[0].type.shape),
            name_hint=hint,
        )
        operations.extend((*to_bhsd, attention, back))
        new_values[sv.results[0]] = back.results[0]

    return Function(
        name=function.name,
        operands=arguments,
        operations=tuple(operations),
        results=tuple(new_values[value] for value in function.results),
    )
