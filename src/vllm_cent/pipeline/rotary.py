"""Keep rotary-embedding operands visible until reference instruction emission."""

from dataclasses import dataclass

__all__ = ["RopeCommand"]


@dataclass(frozen=True, slots=True, kw_only=True)
class RopeCommand:
    """Rotate each token's head vectors by the table angles of its position.

    Attributes:
        source: Shared Buffer object holding ``[tokens, heads, head_dim]``
            values in row-major order.
        positions: Object holding one integer position per token.
        table: Object holding one row per position: ``rotary_dim // 2``
            cosines, then as many sines.
        destination: Result object; exact source reuse is safe at last use.
        token_count: Number of tokens (batch times sequence length).
        head_count: Head vectors per token.
        head_dim: Scalars in each head vector.
        rotary_dim: Leading scalars of each head vector that are rotated.
        position_count: Rows in the table.
        is_neox_style: Pairing of the rotated elements; see ``RopeOp``.
        operation: Source operation name retained for diagnostics.
    """

    source: str
    positions: str
    table: str
    destination: str
    token_count: int
    head_count: int
    head_dim: int
    rotary_dim: int
    position_count: int
    is_neox_style: bool
    operation: str
