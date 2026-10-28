"""Execution-domain rules shared by Device IR ops.

A Device IR op reuses the type rules of the semantic op it implements and
adds one rule about where its tensors live (``TensorType.domain``). The
mixins here add that rule; a device op lists one before its semantic op,
e.g. ``class PIMGemvOp(_PIMOp, LinearOp)``.

- A compute op runs in one domain: every operand must already be there, and
  its results are there too.
- A same-domain op (view, copy, broadcast) runs wherever its operand is, and
  its result stays in that domain.

Only ``device.transfer`` moves a tensor to another domain.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

from ..base import Operation
from ..types import ExecutionDomain, TensorType

__all__: list[str] = []


class _DomainOp(Operation):
    """Mixin: every operand must be in ``DOMAIN``; results are put in ``DOMAIN``.

    Subclasses set ``DOMAIN`` and list this mixin before their semantic op,
    so that ``super().infer_result_types()`` runs the semantic op's rules.

    An op may list ``INDEX_OPERANDS``: operands that only say where to read
    or write, such as the cache positions of ``pim.cache_write``. The units
    do not compute on them, so they may be in any domain; they still need
    one, like every Device IR value.
    """

    # No instance attributes: without empty slots, every op instance of a
    # slotted dataclass that mixes this in would also get a __dict__.
    __slots__ = ()
    DOMAIN: ClassVar[ExecutionDomain]
    INDEX_OPERANDS: ClassVar[frozenset[int]] = frozenset()

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Check operand domains, then apply the semantic op's type rules.

        Raises:
            ValueError: If an operand has no domain (a Semantic IR value), a
                non-index operand is in another domain, or the semantic rules
                reject the operands.
        """

        # Check domains before the semantic rules, so a misplaced operand is
        # reported as such rather than as a confusing shape error.
        for index, operand in enumerate(self.operands):
            domain = operand.type.domain
            if domain is None:
                raise ValueError(
                    f"{self.NAME}: operand {index} has no domain; "
                    "lower it to Device IR first"
                )
            if index not in self.INDEX_OPERANDS and domain is not self.DOMAIN:
                raise ValueError(
                    f"{self.NAME}: operand {index} is in {domain}, "
                    f"not {self.DOMAIN}; insert device.transfer"
                )
        # At run time super() is the semantic op listed after the mixin (e.g.
        # LinearOp); the type checker only sees Operation's abstract method.
        semantic = super().infer_result_types()  # type: ignore[safe-super]
        return tuple(replace(result, domain=self.DOMAIN) for result in semantic)


class _PIMOp(_DomainOp):
    """Mixin for ops that compute in processing-in-memory units."""

    __slots__ = ()
    DOMAIN = ExecutionDomain.PIM


class _PNMOp(_DomainOp):
    """Mixin for ops that compute in processing-near-memory units."""

    __slots__ = ()
    DOMAIN = ExecutionDomain.PNM


class _SameDomainOp(Operation):
    """Mixin: the op runs where its operands are; results stay in that domain.

    For ops whose work does not depend on the domain: views move no data,
    and a copy or broadcast moves data only within the domain it is in.
    """

    __slots__ = ()

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Check that all operands share one domain, then apply the semantic rules.

        Raises:
            ValueError: If an operand has no domain (a Semantic IR value), the
                operands are in different domains, or the semantic rules
                reject the operands.
        """

        domain = self.operands[0].type.domain
        if domain is None:
            raise ValueError(
                f"{self.NAME}: operand 0 has no domain; lower it to Device IR first"
            )
        for index, operand in enumerate(self.operands):
            if operand.type.domain is not domain:
                raise ValueError(
                    f"{self.NAME}: operand {index} is in {operand.type.domain}, "
                    f"operand 0 in {domain}; insert device.transfer"
                )
        # See _DomainOp: super() is the semantic op listed after the mixin.
        semantic = super().infer_result_types()  # type: ignore[safe-super]
        return tuple(replace(result, domain=domain) for result in semantic)
