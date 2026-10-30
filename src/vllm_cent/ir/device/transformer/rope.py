"""Rotary position embedding computed near memory."""

from __future__ import annotations

from dataclasses import dataclass

from ...semantic.transformer import RopeOp
from ..domain import _PNMOp

__all__ = [
    "PNMRopeOp",
]


@dataclass(frozen=True, eq=False, slots=True)
class PNMRopeOp(_PNMOp, RopeOp):
    """``aloi.rope``: rotate each head by its token's angles, x and table in PNM."""

    # TODO(RoPE table rows): Each token reads only the table row of its
    # position, yet this op takes the whole cos/sin table as an operand, so a
    # transfer before it would move every row (131072 x 128 for Llama 3.1).
    # A gather of the needed rows, before the transfer, is still to be
    # designed; until then the table must already be in PNM.
    NAME = "pnm.rope"
