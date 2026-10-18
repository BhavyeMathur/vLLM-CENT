"""Writing new keys or values into a KV cache.
"""

from __future__ import annotations

from dataclasses import dataclass

from ...base import Operation, Value
from ...types import Partial, Replicate, TensorType
from ._common import _ATTENTION_RANK, _POSITION_DTYPES

__all__ = [
    "KvUpdateOp",
]


@dataclass(frozen=True, eq=False, slots=True)
class KvUpdateOp(Operation):
    """Write this step's keys or values into a cache; return the new cache.

    Functional: the input cache is not modified. ``new[b, t]`` lands at
    ``cache[b, positions[b, t]]``.

    KvUpdateOp(operands=(cache, new, positions))
    """

    NAME = "aloi.kv_update"
    NUM_OPERANDS = 3

    @property
    def cache(self) -> Value:
        return self.operands[0]

    @property
    def new(self) -> Value:
        return self.operands[1]

    @property
    def positions(self) -> Value:
        return self.operands[2]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result has the cache's type, including its KV_CACHE role.

        Raises:
            ValueError: If the cache is not ``[batch, context, kv_heads,
                head_dim]``, new does not match it apart from its token
                count, positions do not give one integer per new token, the
                axes disagree, or the placements differ.
        """

        cache = self.cache.type
        new = self.new.type
        positions = self.positions.type
        if cache.rank != _ATTENTION_RANK or new.rank != _ATTENTION_RANK:
            raise ValueError(
                f"{self.NAME}: cache and new must be [batch, tokens, heads, "
                f"head_dim], got {cache.shape} and {new.shape}"
            )
        if cache.dtype != new.dtype:
            raise ValueError(f"{self.NAME}: cache and new dtypes differ")
        # Same batch, heads and head_dim; only the token count differs.
        if (cache.shape[0], *cache.shape[2:]) != (new.shape[0], *new.shape[2:]):
            raise ValueError(
                f"{self.NAME}: new {new.shape} does not fit cache {cache.shape}"
            )
        if positions.dtype not in _POSITION_DTYPES or positions.shape != new.shape[:2]:
            raise ValueError(
                f"{self.NAME}: positions must be integers of shape {new.shape[:2]}"
            )

        if cache.axes and new.axes:
            if (cache.axes[0], *cache.axes[2:]) != (new.axes[0], *new.axes[2:]):
                raise ValueError(
                    f"{self.NAME}: new axes {new.axes} do not match cache axes "
                    f"{cache.axes}"
                )
            if positions.axes and positions.axes != new.axes[:2]:
                raise ValueError(
                    f"{self.NAME}: positions axes {positions.axes} must equal "
                    f"{new.axes[:2]}"
                )

        # Each rank writes its own heads of the new token into its own heads
        # of the cache. Splitting the context dimension is rejected: the
        # position being written could belong to another rank.
        if not isinstance(positions.placement, Replicate):
            raise ValueError(f"{self.NAME}: positions must be replicated")
        if cache.placement != new.placement or isinstance(cache.placement, Partial):
            raise ValueError(
                f"{self.NAME}: unsupported placements cache={cache.placement}, "
                f"new={new.placement}"
            )
        if cache.sharded_dim == 1:
            raise ValueError(f"{self.NAME}: the cache's context dimension is split")
        return (cache,)
