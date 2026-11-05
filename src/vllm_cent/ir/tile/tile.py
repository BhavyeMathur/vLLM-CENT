"""Target-independent Tile IR ops: structured loops."""

from dataclasses import dataclass

from ..base import Block, Operation, Value
from ..printer import format_type
from ..types import DType, TensorType

# The type of a loop index: one integer.
INDEX = TensorType(shape=(), dtype=DType.INT32)


@dataclass(frozen=True, eq=False, slots=True)
class TileForOp(Operation):
    """Run ``body`` for index = 0 .. extent-1, carrying values between iterations.

    - ``operands``: the initial carried values, read by the first iteration.
    - ``body.arguments``: ``(index, *carried)``. In the first iteration
      ``carried[j]`` is ``operands[j]``; in every later one it is the
      previous iteration's ``body.results[j]``.
    - ``body.results``: one value per carried value, which becomes the next
      iteration's ``carried``. The carried values are fixed by the op: an
      iteration replaces each of them, it never adds or drops one.
    - ``results``: the last iteration's ``body.results``.

    Type checking does not unroll the loop. By induction, if the initial
    values have the carried types (the first iteration) and the body yields
    the carried types again (every next iteration), every iteration is well
    typed, whatever ``extent`` is. The ops inside the body checked their own
    types when they were built; which values the body may use is checked by
    ``Function.verify``, which knows the enclosing scope.
    """

    NAME = "tile.for"
    NUM_OPERANDS = None  # any number of carried values

    extent: int
    body: Block

    def blocks(self) -> tuple[Block, ...]:
        """The loop body."""
        return (self.body,)

    def attributes(self) -> dict[str, object]:
        """The extent; the body is printed as a nested block instead."""
        return {"extent": self.extent}

    @property
    def index(self) -> Value:
        """The iteration number, 0 .. extent-1, as the body sees it."""
        return self.body.arguments[0]

    @property
    def carried(self) -> tuple[Value, ...]:
        """The values one iteration passes to the next, as the body sees them."""
        return self.body.arguments[1:]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Check the carried types by induction; return the body's result types.

        Raises:
            ValueError: If ``extent`` is below 1, the body's arguments or
                results do not match the carried values in number, the index
                is not ``INDEX``, or a carried value changes type on the way
                in or around the loop.
        """

        if self.extent < 1:
            raise ValueError(
                f"{self.NAME}: extent must be at least 1, got {self.extent}"
            )

        count = len(self.operands)
        if len(self.body.arguments) != 1 + count:
            raise ValueError(
                f"{self.NAME}: {count} carried values need a body with "
                f"{1 + count} arguments (the index first), "
                f"got {len(self.body.arguments)}"
            )
        if len(self.body.results) != count:
            raise ValueError(
                f"{self.NAME}: {count} carried values need a body with "
                f"{count} results, got {len(self.body.results)}"
            )
        if self.index.type != INDEX:
            raise ValueError(
                f"{self.NAME}: the index must be {format_type(INDEX)}, "
                f"got {format_type(self.index.type)}"
            )

        for j, (initial, carried, yielded) in enumerate(
            zip(self.operands, self.carried, self.body.results, strict=True)
        ):
            # The first iteration reads the initial value as carried[j].
            if initial.type != carried.type:
                raise ValueError(
                    f"{self.NAME}: carried value {j} starts as "
                    f"{format_type(initial.type)}, but the body takes "
                    f"{format_type(carried.type)}"
                )
            # Every next iteration reads what the previous one yielded.
            if yielded.type != carried.type:
                raise ValueError(
                    f"{self.NAME}: carried value {j} is yielded as "
                    f"{format_type(yielded.type)}, but the body takes "
                    f"{format_type(carried.type)}"
                )

        return tuple(value.type for value in self.body.results)
