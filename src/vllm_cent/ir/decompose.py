"""Replace composite semantic ops by the simpler ops that define them.

An op that implements ``Decomposable`` (e.g. ``aloi.softmax``) knows an
equivalent sequence of simpler semantic ops. This pass applies those
decompositions, for example before lowering to a target that has no unit for
the composite op but does have one for each of its parts.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

from .base import Decomposable, Function, Operation, Value, rebuild_function
from .passes import FunctionPass

__all__ = ["DecomposeOps"]


def _select_all(op: Operation) -> bool:
    """Select every decomposable op; the default of ``DecomposeOps``.

    Args:
        op: Any op.

    Returns:
        Always True.
    """

    return True


class DecomposeOps(FunctionPass):
    """Decompose composite ops such as ``aloi.softmax`` into simpler ops.

    The pass runs in rounds. Each round rebuilds the function and replaces
    every selected decomposable op by its decomposition. A decomposition may
    itself contain decomposable ops; the next round handles them. Rebuilding
    in whole rounds, rather than decomposing recursively inside one round,
    keeps every reference valid: ``rebuild_function`` remaps the results of
    the ops it replaces, but not of ops created inside a decomposition.
    """

    NAME = "decompose-ops"

    def __init__(
        self,
        select: Callable[[Operation], bool] = _select_all,
        max_rounds: int = 8,
    ) -> None:
        """Configure which ops to decompose.

        Args:
            select: Called with each decomposable op; True decomposes it. A
                lowering passes e.g. "the target has no native unit for this
                op"; the default decomposes everything.
            max_rounds: Most rounds to run. Nested decompositions need one
                round per level; exceeding the limit means two decompositions
                produce each other.

        Raises:
            ValueError: If ``max_rounds`` is below 1.
        """

        if max_rounds < 1:
            raise ValueError(f"max_rounds must be at least 1, got {max_rounds}")
        self._select = select
        self._max_rounds = max_rounds

    def run_on_function(self, function: Function) -> Function:
        """Decompose selected ops until none remain.

        Args:
            function: Function to rewrite; it is not modified.

        Returns:
            A function without selected decomposable ops, or ``function``
            itself when it has none.

        Raises:
            ValueError: If selected decomposable ops remain after
                ``max_rounds`` rounds.
        """

        for _ in range(self._max_rounds):
            if not self._remaining(function):
                return function
            # rebuild_function needs fresh argument values; keep their types
            # and names, as clone_function does.
            arguments = tuple(
                Value(type=argument.type, name=argument.name)
                for argument in function.operands
            )
            function = rebuild_function(function, arguments, rewrite=self._rewrite)
        remaining = self._remaining(function)
        if not remaining:
            return function
        raise ValueError(
            f"{self.NAME}: ops still decomposable after {self._max_rounds} "
            f"rounds, so decompositions produce each other: {remaining}"
        )

    def _remaining(self, function: Function) -> list[str]:
        """Return the names of the selected decomposable ops in ``function``.

        Args:
            function: Function to inspect.

        Returns:
            Each selected op's ``NAME``, in function order; empty when none.
        """

        return [op.NAME for op in function.operations if self._wants(op)]

    def _wants(self, op: Operation) -> bool:
        """Tell whether ``op`` is decomposable and selected.

        Args:
            op: Any op.

        Returns:
            Whether this pass replaces ``op``.
        """

        return isinstance(op, Decomposable) and self._select(op)

    def _rewrite(self, op: Operation) -> Sequence[Operation]:
        """Return op's decomposition if it is selected, else op itself.

        Used as ``rebuild_function``'s rewrite, which calls it with every
        rebuilt op.

        Args:
            op: A rebuilt op; its operands are already the new function's.

        Returns:
            The ops to emit in place of ``op``; the last one stands for it.
        """

        # isinstance (not _wants) so that the type checker knows op has
        # decompose().
        if isinstance(op, Decomposable) and self._select(op):
            return op.decompose()
        return (op,)
