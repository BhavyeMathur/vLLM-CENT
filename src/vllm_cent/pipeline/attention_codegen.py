"""Expand supplied-tensor attention into two matrix products and row softmax.

For each query, the first matrix product measures its similarity to every key.
Scaling, masking, and softmax turn those scores into weights that sum to one.
The second product uses those weights to combine the value vectors. Only one
query's scores are kept at a time; queries sharing a key/value head reuse its
prepared buffers. Packing and nonlinear math use reference instructions whose
physical hardware encodings are not yet defined.
"""

from dataclasses import dataclass

from ..cent import CentHardwareSpec, CentSharedBufferAddress, ceil_div
from ..cent.instructions import CentInstruction
from ..cent.reference import (
    ReferenceAttentionScores,
    ReferenceCopy,
    ReferenceFill,
    ReferenceSoftmax,
    ReferenceStoreSlice,
    ReferenceTranspose,
)
from .reference_codegen import LinearCommand, _LinearBuffers, _emit_linear

__all__ = ["AttentionCommand"]


@dataclass(frozen=True, slots=True, kw_only=True)
class AttentionCommand:
    """Describe grouped attention and the hardware limits used by its dot products.

    Attributes:
        query: Shared Buffer object holding ``[B,Hq,L,D]`` queries.
        key: Object holding ``[B,Hkv,S,D]`` keys.
        value: Object holding ``[B,Hkv,S,Dv]`` values.
        mask: Optional additive-mask object.
        shared_mask: Whether the mask has shared ``[L,S]`` shape.
        destination: Object receiving ``[B,Hq,L,Dv]`` output.
        batch_count: Independent batches, or ``B``.
        query_heads: Query heads per batch, or ``Hq``.
        kv_heads: Key/value heads per batch, or ``Hkv``.
        query_count: Query rows per head, or ``L``.
        key_count: Key/value rows per head, or ``S``.
        query_width: Scalar components per query/key vector, or ``D``.
        value_width: Scalar components per value vector, or ``Dv``.
        scale: Resolved finite score multiplier.
        causal: Whether a query masks keys beyond its declared position.
        query_position_offset: Position of query zero, in tokens.
        accumulation_register: Mapped bank-local accumulator used by both products.
        query_tile_width: Scalar count in the largest query/key dot-product tile.
        value_tile_width: Scalar count in the largest probability/value tile.
        channel: Hardware channel used for temporary matrix tiles.
        bank_count: Banks used in each parallel dot-product group.
        operation: Source operation name retained for diagnostics.
    """

    query: str
    key: str
    value: str
    mask: str | None
    shared_mask: bool
    destination: str
    batch_count: int
    query_heads: int
    kv_heads: int
    query_count: int
    key_count: int
    query_width: int
    value_width: int
    scale: float
    causal: bool
    query_position_offset: int
    accumulation_register: int
    query_tile_width: int
    value_tile_width: int
    channel: int
    bank_count: int
    operation: str


@dataclass(frozen=True, slots=True, kw_only=True)
class _AttentionWorkspace:
    """Give each attention temporary a typed start address in one reservation.

    Attributes:
        query: Packed query vector reused for each query row.
        key: Packed key head reused across its query-head group.
        value_transpose: Transposed value head used as matrix-product weights.
        scores: One query's logits, then its normalized probabilities.
        result: One query's weighted value result.
        matrix: Tile and accumulator-readback scratch shared by both products.
        slot_count: Complete reservation size, including all padded temporaries.
    """

    query: CentSharedBufferAddress
    key: CentSharedBufferAddress
    value_transpose: CentSharedBufferAddress
    scores: CentSharedBufferAddress
    result: CentSharedBufferAddress
    matrix: CentSharedBufferAddress
    slot_count: int


def _attention_workspace(
    command: AttentionCommand,
    hardware: CentHardwareSpec,
    start: CentSharedBufferAddress,
) -> _AttentionWorkspace:
    """Lay out the temporaries once for both capacity checks and emission.

    Args:
        command: Attention dimensions and selected matrix tile widths.
        hardware: Scalar lanes per Shared Buffer slot.
        start: First slot available for the complete temporary reservation.

    Returns:
        Consecutive non-overlapping regions and their total occupied slot count.
    """
    width = hardware.burst_length
    cursor = start.slot

    def reserve(values: int) -> CentSharedBufferAddress:
        """Reserve complete slots for a scalar count and advance the cursor.

        Args:
            values: Number of scalar positions needed by this temporary.

        Returns:
            First slot of the newly reserved, padded region.
        """
        nonlocal cursor
        address = CentSharedBufferAddress(slot=cursor)
        cursor += ceil_div(values, width)
        return address

    query = reserve(command.query_width)
    key = reserve(command.key_count * command.query_width)
    value_transpose = reserve(command.value_width * command.key_count)
    scores = reserve(command.key_count)
    result = reserve(command.value_width)
    # Both products execute sequentially. Two equal tile buffers and one
    # padded bank readback region suffice for whichever product has wider tiles.
    matrix = reserve(
        2 * max(command.query_tile_width, command.value_tile_width)
        + ceil_div(command.bank_count, width) * width
    )
    return _AttentionWorkspace(
        query=query,
        key=key,
        value_transpose=value_transpose,
        scores=scores,
        result=result,
        matrix=matrix,
        slot_count=cursor - start.slot,
    )


def _attention_workspace_slots(
    command: AttentionCommand, hardware: CentHardwareSpec
) -> int:
    """Count attention temporaries without assigning an address.

    Args:
        command: Attention dimensions and matrix tile limits.
        hardware: Scalar lanes per slot.

    Returns:
        Peak additional Shared Buffer slots, independent of batch/query count.
    """
    return _attention_workspace(
        command, hardware, CentSharedBufferAddress(slot=0)
    ).slot_count


def _emit_attention(
    command: AttentionCommand,
    hardware: CentHardwareSpec,
    addresses: dict[str, CentSharedBufferAddress],
    workspace: CentSharedBufferAddress,
) -> tuple[CentInstruction, ...]:
    """Emit packed dot products, score masking, row softmax, and weighted values.

    Args:
        command: Verified attention choices and hardware tile limits.
        hardware: Geometry used by the existing matrix emitter.
        addresses: Numeric addresses for graph storage objects.
        workspace: First slot of the reserved temporary region.

    Returns:
        Reference packing and scalar primitives interleaved with explicit MACs.
    """
    local = _attention_workspace(command, hardware, workspace)
    output_count = (
        command.batch_count
        * command.query_heads
        * command.query_count
        * command.value_width
    )
    instructions: list[CentInstruction] = [
        ReferenceFill(
            destination=addresses[command.destination], value_count=output_count
        )
    ]
    score_product = LinearCommand(
        source="query",
        weights="key",
        destination="scores",
        input_width=command.query_width,
        output_width=command.key_count,
        batch_count=1,
        accumulation_register=command.accumulation_register,
        tile_width=command.query_tile_width,
        channel=command.channel,
        bank_count=command.bank_count,
        operation=command.operation,
    )
    value_product = LinearCommand(
        source="scores",
        weights="value_transpose",
        destination="result",
        input_width=command.key_count,
        output_width=command.value_width,
        batch_count=1,
        accumulation_register=command.accumulation_register,
        tile_width=command.value_tile_width,
        channel=command.channel,
        bank_count=command.bank_count,
        operation=command.operation,
    )
    # Consecutive query heads share a key/value head. With six query heads
    # and two KV heads, query heads 0..2 use KV head 0, and 3..5 use head 1.
    heads_per_group = command.query_heads // command.kv_heads
    for batch in range(command.batch_count):
        for kv_head in range(command.kv_heads):
            kv_index = batch * command.kv_heads + kv_head
            instructions.append(
                ReferenceCopy(
                    source=addresses[command.key],
                    destination=local.key,
                    source_offset=kv_index * command.key_count * command.query_width,
                    value_count=command.key_count * command.query_width,
                    destination_count=command.key_count * command.query_width,
                )
            )
            # The second GEMV needs a weight row for each output component.
            # Transposing one V head turns its S rows of Dv values into Dv rows of S.
            instructions.append(
                ReferenceTranspose(
                    source=addresses[command.value],
                    destination=local.value_transpose,
                    source_offset=kv_index * command.key_count * command.value_width,
                    row_count=command.key_count,
                    column_count=command.value_width,
                )
            )
            for group_head in range(heads_per_group):
                query_head = kv_head * heads_per_group + group_head
                for query in range(command.query_count):
                    # Flatten [batch, head, query] in row order. Multiplying
                    # this row index by the vector width locates its scalars.
                    query_index = (
                        batch * command.query_heads + query_head
                    ) * command.query_count + query
                    instructions.append(
                        ReferenceCopy(
                            source=addresses[command.query],
                            destination=local.query,
                            source_offset=query_index * command.query_width,
                            value_count=command.query_width,
                            destination_count=command.query_width,
                        )
                    )
                    instructions.extend(
                        _emit_linear(
                            score_product,
                            hardware,
                            _LinearBuffers(
                                source=local.query,
                                weights=local.key,
                                destination=local.scores,
                                workspace=local.matrix,
                            ),
                        )
                    )
                    # A shared mask starts over for each batch and head; a
                    # full mask follows the same flattened rows as Q.
                    mask_offset = 0
                    if command.mask is not None:
                        mask_offset = (
                            query if command.shared_mask else query_index
                        ) * command.key_count
                    instructions.append(
                        ReferenceAttentionScores(
                            source=local.scores,
                            destination=local.scores,
                            value_count=command.key_count,
                            scale=command.scale,
                            mask=addresses[command.mask]
                            if command.mask is not None
                            else None,
                            mask_offset=mask_offset,
                            causal_limit=command.query_position_offset + query
                            if command.causal
                            else None,
                        )
                    )
                    instructions.append(
                        ReferenceSoftmax(
                            source=local.scores,
                            destination=local.scores,
                            row_count=1,
                            row_width=command.key_count,
                        )
                    )
                    # Scores now hold probabilities. Multiplying them by
                    # the transposed values forms this query's output vector.
                    instructions.extend(
                        _emit_linear(
                            value_product,
                            hardware,
                            _LinearBuffers(
                                source=local.scores,
                                weights=local.value_transpose,
                                destination=local.result,
                                workspace=local.matrix,
                            ),
                        )
                    )
                    instructions.append(
                        ReferenceStoreSlice(
                            source=local.result,
                            destination=addresses[command.destination],
                            source_offset=0,
                            destination_offset=query_index * command.value_width,
                            value_count=command.value_width,
                            destination_count=output_count,
                        )
                    )
    return tuple(instructions)
