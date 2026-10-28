"""Device IR ops that move or reinterpret data, rather than compute.

``device.transfer`` is the only op that moves a tensor to another execution
domain. The other ops run in whatever domain their operand is in:

- views (``device.reshape``, ``device.split_heads``, ``device.merge_heads``)
  move no data; they read the same elements under another shape;
- ``device.copy`` and ``device.broadcast`` move data, but only within the
  operand's domain (e.g. between DRAM banks on PIM, or within the Shared
  Buffer on PNM). Which unit does the work follows from that domain.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ..base import Operation, Value
from ..semantic import BroadcastOp, CopyOp, ReshapeOp
from ..semantic.transformer import MergeHeadsOp, SplitHeadsOp
from ..types import ExecutionDomain, TensorType
from .domain import _SameDomainOp

__all__ = [
    "DeviceBroadcastOp",
    "DeviceCopyOp",
    "DeviceMergeHeadsOp",
    "DeviceReshapeOp",
    "DeviceSplitHeadsOp",
    "DeviceTransferOp",
]


@dataclass(frozen=True, eq=False, slots=True)
class DeviceTransferOp(Operation):
    """Move a tensor to another execution domain, e.g. a GEMV result to PNM.

    The values and the type are unchanged except for the domain. Each rank
    moves its own local tensor, so the placement is kept; moving a partial
    sum to PNM before ``cxl.all_reduce`` is the usual case.

    DeviceTransferOp(operands=(x,), to=ExecutionDomain.PNM)

    Attributes:
        to: The destination domain.
    """

    NAME = "device.transfer"
    NUM_OPERANDS = 1

    to: ExecutionDomain

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result is x's type in the destination domain.

        The role is kept: a weight moved to PNM is still the same weight.

        Raises:
            ValueError: If x has no domain (a Semantic IR value), or is
                already in the destination domain.
        """

        source = self.x.type.domain
        if source is None:
            raise ValueError(
                f"{self.NAME}: x has no domain; lower it to Device IR first"
            )
        if source is self.to:
            raise ValueError(f"{self.NAME}: x is already in {self.to}")
        return (replace(self.x.type, domain=self.to),)


@dataclass(frozen=True, eq=False, slots=True)
class DeviceCopyOp(_SameDomainOp, CopyOp):
    """``aloi.copy`` within the operand's domain."""

    NAME = "device.copy"


@dataclass(frozen=True, eq=False, slots=True)
class DeviceReshapeOp(_SameDomainOp, ReshapeOp):
    """``aloi.reshape`` as a view in the operand's domain; no data moves."""

    NAME = "device.reshape"


@dataclass(frozen=True, eq=False, slots=True)
class DeviceSplitHeadsOp(_SameDomainOp, SplitHeadsOp):
    """``aloi.split_heads`` as a view in the operand's domain; no data moves."""

    NAME = "device.split_heads"


@dataclass(frozen=True, eq=False, slots=True)
class DeviceMergeHeadsOp(_SameDomainOp, MergeHeadsOp):
    """``aloi.merge_heads`` as a view in the operand's domain; no data moves."""

    NAME = "device.merge_heads"


@dataclass(frozen=True, eq=False, slots=True)
class DeviceBroadcastOp(_SameDomainOp, BroadcastOp):
    """``aloi.broadcast`` within the operand's domain."""

    NAME = "device.broadcast"
