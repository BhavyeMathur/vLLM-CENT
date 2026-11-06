"""CENT Tile IR: tensor types that name a CENT storage, and the ops' shared rules.

A value's ``CentTensorType`` says where its data is (DRAM banks, a Global
Buffer, the MAC registers or the Shared Buffer); each ``cent.*`` op says
where it needs its operands, as ``_DomainOp`` does for Device IR domains.

A tile of a larger tensor is named by the whole tensor plus index operands:
``at[i]`` names the serial dimension of the tensor's layout that
``indices[i]`` selects. Only local rules are checked here (each index is an
``INDEX``, one name per index); whether the names exist in the layout, and
the tile's shape and position, are checked against the layout in Step 3.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise

from ..base import Operation, Value
from ..printer import format_type
from ..types import MemorySpace, TensorType
from .tile import INDEX

__all__ = ["CentTensorType", "CentTileOp"]


@dataclass(frozen=True)
class CentTensorType(TensorType):
    """A tensor in one kind of CENT storage, as CENT Tile IR ops see it.

    ``domain`` (Device IR's general PIM/PNM) still holds: ``memory_space``
    refines it to one CENT storage, e.g. a Global Buffer or the MAC
    registers. Left as None, the domain is taken from the memory space.

    Equality compares the memory space too, so ``tile.for`` rejects a carried
    value that enters in one space and is yielded in another, and
    ``dataclasses.replace`` keeps the space of the type it copies.
    """

    memory_space: MemorySpace = field(kw_only=True)

    def __post_init__(self) -> None:
        """Fill in the domain from the memory space, or check that they agree.

        Raises:
            ValueError: If ``domain`` is given and is not the memory space's.
        """

        super().__post_init__()
        if self.domain is None:
            object.__setattr__(self, "domain", self.memory_space.domain)
        elif self.domain is not self.memory_space.domain:
            raise ValueError(
                f"memory space {self.memory_space.value} belongs to domain "
                f"{self.memory_space.domain.value}, but the type says "
                f"{self.domain.value}"
            )

    def extra_annotations(self) -> tuple[str, ...]:
        """Print the memory space after the domain."""
        return (f"memory_space={self.memory_space.value}",)


@dataclass(frozen=True, eq=False, slots=True)
class CentTileOp(Operation):
    """Base of the CENT Tile IR ops: the channel mask they are issued with.

    ``channels`` lists the selected channels, ascending, e.g. ``(0, 1, 2)``;
    code generation turns it into the instruction's CHmask bits. Whether
    every channel exists on the device is not checked here: an op does not
    know its device (Step 3 or ``evaluate`` does).
    """

    channels: tuple[int, ...] = field(kw_only=True)

    def __post_init__(self) -> None:
        """Check the channel list before the operands' types.

        Raises:
            ValueError: If ``channels`` is empty, negative, unsorted or
                repeats a channel, or if ``infer_result_types`` rejects the
                operands.
        """

        object.__setattr__(self, "channels", tuple(self.channels))
        if not self.channels:
            raise ValueError(f"{self.NAME}: channels must select at least one")
        if self.channels[0] < 0:
            raise ValueError(
                f"{self.NAME}: channels must be non-negative, got {self.channels}"
            )
        # Strictly ascending: sorted, and no channel twice.
        if any(a >= b for a, b in pairwise(self.channels)):
            raise ValueError(
                f"{self.NAME}: channels must be ascending without repeats, "
                f"got {self.channels}"
            )
        super().__post_init__()


def _expect(
    op_name: str, what: str, value: Value, space: MemorySpace
) -> CentTensorType:
    """Return ``value``'s type if it is a ``CentTensorType`` in ``space``.

    Args:
        op_name: The op's NAME, for the message.
        what: The operand's role in the op, e.g. ``"weight"``.
        value: The operand.
        space: The memory space the op needs it in.

    Returns:
        The operand's type, narrowed to ``CentTensorType``.

    Raises:
        ValueError: If the operand has no memory space, or another one.
    """

    value_type = value.type
    if not isinstance(value_type, CentTensorType):
        raise ValueError(
            f"{op_name}: {what} must be in {space.value}, but "
            f"{format_type(value_type)} has no memory space"
        )
    if value_type.memory_space is not space:
        raise ValueError(
            f"{op_name}: {what} must be in {space.value}, "
            f"got {value_type.memory_space.value}"
        )
    return value_type


def _expect_indices(op_name: str, at: Sequence[str], indices: Sequence[Value]) -> None:
    """Check the index operands that select a tile, and the names in ``at``.

    Args:
        op_name: The op's NAME, for the message.
        at: The layout's serial dimension that each index selects.
        indices: The index operands.

    Raises:
        ValueError: If ``at`` and ``indices`` differ in length, a name is
            empty or repeated, or an index is not an ``INDEX``.
    """

    if len(at) != len(indices):
        raise ValueError(
            f"{op_name}: at names {len(at)} dimensions {list(at)}, "
            f"but {len(indices)} indices are given"
        )
    if any(not name for name in at) or len(set(at)) != len(at):
        raise ValueError(f"{op_name}: at must name distinct dimensions, got {list(at)}")
    for name, index in zip(at, indices, strict=True):
        if index.type != INDEX:
            raise ValueError(
                f"{op_name}: the index for {name!r} must be "
                f"{format_type(INDEX)}, got {format_type(index.type)}"
            )
