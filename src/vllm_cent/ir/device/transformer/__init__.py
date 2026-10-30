"""Device IR ops that only transformer-family models use.

Each implements a semantic op of ``ir.semantic.transformer`` on one
execution domain, with that op's type rules.
"""

from .attention import PIMQKOp, PIMSVOp
from .kv_cache import PIMCacheWriteOp
from .rope import PNMRopeOp

__all__ = [
    "PIMCacheWriteOp",
    "PIMQKOp",
    "PIMSVOp",
    "PNMRopeOp",
]
