"""Ops that combine values across tensor-parallel ranks.

They change a tensor's placement rather than its mathematical value;
the tensor-parallel pass inserts them.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..base import Operation, Value
from ..types import Partial, Replicate, TensorType

__all__ = [
    "AllReduceOp",
]


@dataclass(frozen=True, eq=False, slots=True)
class AllReduceOp(Operation):
    """Sum a partial tensor over all tensor-parallel ranks.

    Every rank contributes its partial sum and receives the total, so the
    result is replicated. Shape, dtype, role and axes are unchanged.

    AllReduceOp(operands=(x,))
    """

    NAME = "aloi.all_reduce"
    NUM_OPERANDS = 1

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result is x's type with a replicated placement.

        Raises:
            ValueError: If x is not a partial sum. Summing a replicated
                tensor over tp ranks would multiply it by tp, and summing a
                sharded one would add unrelated slices.
        """

        if not isinstance(self.x.type.placement, Partial):
            raise ValueError(
                f"{self.NAME}: operand must be a partial sum, "
                f"got {self.x.type.placement}"
            )
        return (replace(self.x.type, placement=Replicate()),)
