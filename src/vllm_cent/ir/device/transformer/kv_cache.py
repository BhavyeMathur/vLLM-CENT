"""Writing new keys or values into a KV cache held in the memory banks."""

from __future__ import annotations

from dataclasses import dataclass

from ...semantic.transformer import KvUpdateOp
from ..domain import _PIMOp

__all__ = [
    "PIMCacheWriteOp",
]


@dataclass(frozen=True, eq=False, slots=True)
class PIMCacheWriteOp(_PIMOp, KvUpdateOp):
    """``aloi.kv_update``: write each new row of keys or values at its position.

    The cache and the new rows are in PIM memory. The positions (operand 2)
    only choose which rows are written, so they may be in any domain.
    """

    NAME = "pim.cache_write"
    INDEX_OPERANDS = frozenset({2})
