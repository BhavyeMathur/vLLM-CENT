"""Execution-domain rules shared by Device IR ops.

Every Device IR op derives from ``DeviceOp``. A device op reuses the type
rules of the semantic op it implements and adds one rule about where its
tensors live (``TensorType.domain``). The mixins here add that rule; a device
op lists one before its semantic op, e.g. ``class PIMGemvOp(_PIMOp, LinearOp)``.
``DeviceOp.SEMANTIC`` records which semantic op that is.

- A compute op runs in one domain: every operand must already be there, and
  its results are there too.
- A same-domain op (view, copy, broadcast) runs wherever its operand is, and
  its result stays in that domain.

Only ``device.transfer`` moves a tensor to another domain.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import ClassVar

from ..base import Operation, Value
from ..types import ExecutionDomain, TensorType

__all__ = ["DeviceOp"]


class DeviceOp(Operation):
    """Base of every Device IR op; records the semantic op it implements.

    A device op class inherits from device classes (``DeviceOp``, a domain
    mixin such as ``_PIMOp``, or another device op) and from at most one
    semantic op class, whose type rules it reuses. That semantic class is
    stored in ``SEMANTIC`` when the class is defined, so lowering can find
    which device ops implement a semantic op without a separate table.

    Attributes:
        INDEX_OPERANDS: Positions of operands that only say where to read or
            write, such as the cache positions of ``pim.cache_write``. The
            units do not compute on them, so they may be in any domain; they
            still need one, like every Device IR value.
        SEMANTIC: The semantic op class this device op implements, or
            ``None`` for an op with no semantic counterpart, such as
            ``device.transfer``, and for the mixins. It may be a class of
            shared type rules rather than the semantic op itself:
            ``pnm.softmax`` inherits ``SoftmaxRules``, not the decomposable
            ``SoftmaxOp``. A class that inherits only device classes keeps
            its parent's value.

    Examples:
        ``PIMGemvOp(_PIMOp, LinearOp)`` has ``SEMANTIC is LinearOp``;
        ``DeviceTransferOp(DeviceOp)`` has ``SEMANTIC is None``.
    """

    __slots__ = ()
    INDEX_OPERANDS: ClassVar[frozenset[int]] = frozenset()
    SEMANTIC: ClassVar[type[Operation] | None] = None

    @classmethod
    def operand_domain(cls, operands: Sequence[Value]) -> ExecutionDomain | None:
        """Return the domain this op needs its operands in.

        Lowering moves every other operand, except index operands, there
        before building the op. By default an op runs where its first
        operand already is (views, copies, broadcasts); ops that compute in
        one domain override this.

        Args:
            operands: The operands the op would be built from, in order.

        Returns:
            The required domain, or ``None`` if the first operand has no
            domain yet (a Semantic IR value).
        """

        return operands[0].type.domain

    def __init_subclass__(cls, **kwargs: object) -> None:
        """Register the op by name, then record its semantic base class.

        Only direct bases are inspected: a base that is an ``Operation`` but
        not a ``DeviceOp`` is the semantic op. ``@dataclass(slots=True)``
        creates the class a second time with the same bases, which gives the
        same result.

        Raises:
            TypeError: If the class inherits from two semantic op classes.
        """

        super().__init_subclass__(**kwargs)
        semantic = [
            base
            for base in cls.__bases__
            if issubclass(base, Operation) and not issubclass(base, DeviceOp)
        ]
        if len(semantic) > 1:
            names = [base.__name__ for base in semantic]
            raise TypeError(
                f"{cls.__name__} inherits several semantic ops {names}; "
                "a device op implements one"
            )
        if semantic:
            cls.SEMANTIC = semantic[0]


class _DomainOp(DeviceOp):
    """Mixin: every operand must be in ``DOMAIN``; results are put in ``DOMAIN``.

    Subclasses set ``DOMAIN`` and list this mixin before their semantic op,
    so that ``super().infer_result_types()`` runs the semantic op's rules.
    Operands listed in ``INDEX_OPERANDS`` may be in any domain.
    """

    # No instance attributes: without empty slots, every op instance of a
    # slotted dataclass that mixes this in would also get a __dict__.
    __slots__ = ()
    DOMAIN: ClassVar[ExecutionDomain]

    @classmethod
    def operand_domain(cls, operands: Sequence[Value]) -> ExecutionDomain:
        """Return ``DOMAIN``: the op computes there, wherever its operands are now.

        Args:
            operands: Unused; the domain is fixed by the op class.

        Returns:
            ``DOMAIN``.
        """

        return cls.DOMAIN

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


class _SameDomainOp(DeviceOp):
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
