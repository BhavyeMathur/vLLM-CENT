"""Grouped-query attention steps computed in the memory banks.

Both steps are batches of matrix-vector products whose matrices are the KV
cache: one matrix per (batch, KV head), shared by that head's group of query
heads. ``pim.qk`` sums over head_dim, like a GEMV over the cache's rows;
``pim.sv`` sums over the cached positions, the other direction of the same
row-per-position layout.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...semantic.transformer import GqaQKOp, GqaSVOp
from ..domain import _PIMOp

__all__ = [
    "PIMQKOp",
    "PIMSVOp",
]


@dataclass(frozen=True, eq=False, slots=True)
class PIMQKOp(_PIMOp, GqaQKOp):
    """``aloi.gqa_qk``: scaled scores of each query head against its KV head's keys."""

    NAME = "pim.qk"


@dataclass(frozen=True, eq=False, slots=True)
class PIMSVOp(_PIMOp, GqaSVOp):
    """``aloi.gqa_sv``: each query head's probabilities times its KV head's values."""

    # TODO(V cache layout): A bank sums along the rows it stores, but this
    # step sums over cached positions. With V stored one position per row,
    # the sum runs across rows; the ALOI reference mapping (tp_mapping.py)
    # instead spreads V along head_dim over the banks. Which layout CENT
    # uses, and so how this op lowers, is a mapping decision still to make.
    NAME = "pim.sv"
