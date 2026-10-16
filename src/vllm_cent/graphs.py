"""Build reusable tensor graphs without selecting a hardware target.

These helpers describe complete calculations over caller-supplied tensors.
Compilation, input loading, and execution remain separate. Model-family graph
builders, such as Llama's feed-forward block, live in their model package.
"""

from .pipeline.ir import SemanticGraph, TensorType, Value, verify_semantic
from .pipeline.operations import AttentionOperation

__all__ = ["build_attention_graph"]


def build_attention_graph(
    query_shape: tuple[int, int, int, int],
    key_shape: tuple[int, int, int, int],
    value_shape: tuple[int, int, int, int],
    *,
    mask_shape: tuple[int, ...] | None = None,
    causal: bool = False,
    query_position_offset: int = 0,
    scale: float | None = None,
    block: str = "attention",
) -> SemanticGraph:
    """Describe scaled attention over already-projected query, key, and value tensors.

    The caller supplies ``query``, ``key``, and ``value`` in
    batch/head/sequence/feature order, plus ``mask`` when requested. The result
    keeps the query's batch, head, and sequence dimensions and uses the value
    tensor's feature width. Heads remain separate in the returned ``output``.
    There are no hidden projections, positional rotations, dropout, or cache
    updates. Input tensors are ordinary immutable values for each execution.

    Args:
        query_shape: ``(batch, query_heads, query_count, key_width)``.
        key_shape: ``(batch, kv_heads, key_count, key_width)``. Query heads must
            be an integer multiple of key/value heads. Consecutive query heads
            share each key/value head, as in grouped-query attention.
        value_shape: ``(batch, kv_heads, key_count, value_width)``. The value
            width may differ from the key width.
        mask_shape: ``None`` for no mask input, ``(query_count, key_count)`` for
            a shared additive mask, or ``(batch, query_heads, query_count,
            key_count)`` for separate masks. Finite entries are score biases;
            negative infinity hides an entry. Boolean masks and other implicit
            broadcast shapes are not supported.
        causal: If true, query row ``q`` can attend only to key indices at or
            below ``query_position_offset + q``. This intersects any supplied
            additive mask. If false, every unmasked key is available.
        query_position_offset: Nonnegative position of the first query within
            the supplied key sequence. Zero gives an upper-left causal mask;
            use ``key_count - query_count`` for an aligned suffix when nonnegative.
        scale: Finite multiplier applied to dot products before the mask.
            ``None`` selects ``1 / sqrt(key_width)``.
        block: Source label retained by the operation for diagnostics.

    Returns:
        Verified SemanticGraph with named inputs and one ``output`` tensor.
        Compile separately using ``ExecutionContract.REFERENCE``. Execution
        rejects a probability row with no available finite score.

    Raises:
        ValueError: If query/key/value shapes are not rank four, dimensions are
            nonpositive or incompatible, mask shape
            is unsupported, offset is negative, or explicit scale is non-finite.

    Examples:
        Query shape ``(1, 4, 2, 3)`` with keys ``(1, 2, 5, 3)`` and values
        ``(1, 2, 5, 7)`` produces ``(1, 4, 2, 7)``. Query heads zero and one
        use key/value head zero; query heads two and three use head one.

        A single causal query with five keys sees only key zero at offset zero.
        With offset four, it can see all five keys. The offset is explicit so
        a decoding caller cannot accidentally use a different causal alignment.
    """
    if any(len(shape) != 4 for shape in (query_shape, key_shape, value_shape)):
        raise ValueError("Attention query, key, and value shapes must have rank four")
    inputs: tuple[Value, ...] = (
        Value("query", TensorType(query_shape)),
        Value("key", TensorType(key_shape)),
        Value("value", TensorType(value_shape)),
    )
    if mask_shape is not None:
        inputs += (Value("mask", TensorType(mask_shape)),)
    graph = SemanticGraph(
        inputs=inputs,
        operations=(
            AttentionOperation(
                name="attention",
                inputs=tuple(value.name for value in inputs),
                output=Value(
                    "output", TensorType((*query_shape[:-1], value_shape[-1]))
                ),
                causal=causal,
                query_position_offset=query_position_offset,
                scale=scale,
                block=block,
            ),
        ),
        outputs=("output",),
    )
    verify_semantic(graph)
    return graph
