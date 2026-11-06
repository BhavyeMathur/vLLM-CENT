"""CENT Tile IR: tensor types that name a CENT storage, and the CENT ops.

A value's ``CentTensorType`` says where its data is (DRAM banks, a Global
Buffer, the MAC registers or the Shared Buffer); each ``cent.*`` op says
where it needs its operands, as ``_DomainOp`` does for Device IR domains.
The ops are named after the AiM instructions they become: ``cent.wr_bias``,
``cent.wr_gb``, ``cent.mac_abk`` and ``cent.rd_mac``.

A tile of a larger tensor is named by the whole tensor plus index operands:
``at[i]`` names the serial dimension of the tensor's layout that
``indices[i]`` selects.

An op checks only what it sees itself: memory spaces, dtypes, the channel
list, its indices, and that MAC registers have one row per channel. These
need more than one op, or a layout or device, and are checked later (Step 3):

- whether the names in ``at`` exist in the layout, and the tile's shape,
  position and opsize, e.g. that a ``wr_gb`` tile fills whole slots;
- the bank count, the second dimension of MAC registers;
- that ``mac_abk`` uses only channels that the ``wr_gb`` of its Global
  Buffer wrote: a value does not know the op that defined it;
- liveness: at most one live Global Buffer tile per channel, and one live
  MAC register value per bank (``accumulators_per_bank``);
- that ``Rd``/``Rs`` is aligned to the channel count, and that a
  ``stride = 0`` source is one slot.
"""

from abc import abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass, field
from itertools import pairwise
from typing import ClassVar

from ..base import Operation, Value
from ..printer import format_type
from ..types import MemorySpace, TensorType
from .tile import INDEX

__all__ = [
    "CentMacAbkOp",
    "CentRdMacOp",
    "CentTensorType",
    "CentTileOp",
    "CentWrBiasOp",
    "CentWrGbOp",
]


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


def _check_tile_shape(op_name: str, shape: tuple[int, ...]) -> None:
    """Check the ``shape`` attribute of an op that writes a new tile.

    Raises:
        ValueError: If the shape is empty or has a dimension below 1.
    """

    if not shape or any(dim < 1 for dim in shape):
        raise ValueError(
            f"{op_name}: shape must have positive dimensions, got {list(shape)}"
        )


def _check_mac_shape(op: CentTileOp, shape: tuple[int, ...]) -> None:
    """Check a MAC register tile's shape: ``(len(channels), banks)``.

    One row per selected channel, one column per bank. The bank count is
    the device's, so it is checked against the device later (Step 3).

    Raises:
        ValueError: If the shape is not two-dimensional, or its rows are
            not the selected channels.
    """

    _check_tile_shape(op.NAME, shape)
    if len(shape) != 2 or shape[0] != len(op.channels):
        raise ValueError(
            f"{op.NAME}: MAC registers of {len(op.channels)} channels have "
            f"shape [{len(op.channels)}, banks], got {list(shape)}"
        )


def _expect_same_dtype(op_name: str, **types: TensorType) -> None:
    """Check that the named operand types share one dtype.

    Raises:
        ValueError: If two of them differ, listing each operand's dtype.
    """

    if len({t.dtype for t in types.values()}) > 1:
        listed = ", ".join(f"{what} {t.dtype.value}" for what, t in types.items())
        raise ValueError(f"{op_name}: operands must share one dtype, got {listed}")


@dataclass(frozen=True, eq=False, slots=True)
class _TileRefOp(CentTileOp):
    """A CENT op whose operands are NUM_FIXED tensors, then one index per ``at``.

    The fixed operands are whole tensors. The indices select a tile of the
    tensor that the op reads or writes in pieces: ``at[i]`` names the
    serial dimension of its layout that ``indices[i]`` selects (see the
    module docstring). An op on whole tensors has ``at = ()``.

    Subclasses set ``NUM_FIXED`` and write ``_infer_tile_types``; the
    operand count and the indices are checked here first.
    """

    NUM_OPERANDS = None  # NUM_FIXED tensors, then any number of indices
    NUM_FIXED: ClassVar[int]

    at: tuple[str, ...] = field(kw_only=True)

    def __post_init__(self) -> None:
        """Store ``at`` as a tuple, then run the channel and type checks."""

        object.__setattr__(self, "at", tuple(self.at))
        super().__post_init__()

    @property
    def indices(self) -> tuple[Value, ...]:
        """The index operands, one per name in ``at``."""
        return self.operands[self.NUM_FIXED :]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Check the operand count and the indices, then the op's own rules.

        Raises:
            ValueError: If fewer than ``NUM_FIXED`` operands are given, the
                indices do not match ``at``, or ``_infer_tile_types``
                rejects the operands.
        """

        if len(self.operands) < self.NUM_FIXED:
            raise ValueError(
                f"{self.NAME} expects {self.NUM_FIXED} operands before its "
                f"indices, got {len(self.operands)}"
            )
        _expect_indices(self.NAME, self.at, self.indices)
        return self._infer_tile_types()

    @abstractmethod
    def _infer_tile_types(self) -> tuple[TensorType, ...]:
        """Check the fixed operands and return the result types."""


@dataclass(frozen=True, eq=False, slots=True)
class CentWrBiasOp(_TileRefOp):
    """``cent.wr_bias(src, *indices)``: load the MAC registers (``WR_BIAS``).

    Each selected channel copies one Shared Buffer slot into the MAC
    registers of its banks, one value per bank. Which slot (our convention;
    see ``examples/Tile_DSL/gemv.py``):

    - ``stride = 1``: channel ``c`` reads slot ``Rs + c``, where ``Rs`` is
      the first slot of the tile of ``src`` that the indices select. It
      restores partial sums that ``cent.rd_mac`` saved.
    - ``stride = 0``: every channel reads the same slot ``Rs``, e.g. a slot
      of zeros to start a sum.

    The result is the MAC register tile, shape ``(len(channels), banks)``.
    ``shape`` is given, not derived from ``src``: with ``stride = 0`` the
    source is one slot, and with ``stride = 1`` its tile's size depends on
    the layout, yet both must give the same register type.

    TODO(runtime): Nothing yet defines how a slot of zeros is filled (see
    ``cent/instructions/data_movement.py``).
    """

    NAME = "cent.wr_bias"
    NUM_FIXED = 1

    stride: int = field(kw_only=True)
    shape: tuple[int, ...] = field(kw_only=True)

    def __post_init__(self) -> None:
        """Store ``shape`` as a tuple, then run the other checks."""

        object.__setattr__(self, "shape", tuple(self.shape))
        super().__post_init__()

    @property
    def source(self) -> Value:
        """The Shared Buffer tensor read."""
        return self.operands[0]

    def _infer_tile_types(self) -> tuple[TensorType, ...]:
        """Return MAC registers of ``shape``, in the source's dtype.

        Raises:
            ValueError: If the source is not in the Shared Buffer, the
                stride is not 0 or 1, or ``shape`` is not
                ``(len(channels), banks)``.
        """

        source = _expect(self.NAME, "source", self.source, MemorySpace.PNM_SB)
        if self.stride not in (0, 1):
            raise ValueError(f"{self.NAME}: stride must be 0 or 1, got {self.stride}")
        _check_mac_shape(self, self.shape)
        return (
            CentTensorType(
                shape=self.shape, dtype=source.dtype, memory_space=MemorySpace.PIM_MAC
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class CentWrGbOp(_TileRefOp):
    """``cent.wr_gb(src, *indices)``: fill the Global Buffers (``WR_GB``).

    The tile of ``src`` that the indices select is broadcast from the Shared
    Buffer into the Global Buffer of every selected channel, so the result
    is one channel's copy: ``shape`` has no channel dimension. The
    instruction's opsize follows from ``shape`` when the op is expanded.
    """

    NAME = "cent.wr_gb"
    NUM_FIXED = 1

    shape: tuple[int, ...] = field(kw_only=True)

    def __post_init__(self) -> None:
        """Store ``shape`` as a tuple, then run the other checks."""

        object.__setattr__(self, "shape", tuple(self.shape))
        super().__post_init__()

    @property
    def source(self) -> Value:
        """The Shared Buffer tensor read."""
        return self.operands[0]

    def _infer_tile_types(self) -> tuple[TensorType, ...]:
        """Return a Global Buffer tile of ``shape``, in the source's dtype.

        Raises:
            ValueError: If the source is not in the Shared Buffer, or
                ``shape`` has a dimension below 1.
        """

        source = _expect(self.NAME, "source", self.source, MemorySpace.PNM_SB)
        _check_tile_shape(self.NAME, self.shape)
        return (
            CentTensorType(
                shape=self.shape, dtype=source.dtype, memory_space=MemorySpace.PIM_GB
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class CentMacAbkOp(_TileRefOp):
    """``cent.mac_abk(w, gb, acc, *indices)``: multiply-accumulate (``MAC_ABK``).

    Every bank of the selected channels multiplies the open row of its tile
    of ``w`` with its channel's Global Buffer and adds the products to its
    MAC register. The indices select the tile of ``w``.

    The result is the new value of the MAC registers, of ``acc``'s type, so
    a loop can carry it from one MAC to the next.
    """

    NAME = "cent.mac_abk"
    NUM_FIXED = 3

    @property
    def weight(self) -> Value:
        """The DRAM tensor whose tile is multiplied."""
        return self.operands[0]

    @property
    def global_buffer(self) -> Value:
        """The Global Buffer tile multiplied with each row."""
        return self.operands[1]

    @property
    def accumulator(self) -> Value:
        """The MAC registers added to."""
        return self.operands[2]

    def _infer_tile_types(self) -> tuple[TensorType, ...]:
        """Return ``acc``'s type.

        Raises:
            ValueError: If an operand is in the wrong memory space, the
                dtypes differ, or ``acc`` is not ``(len(channels), banks)``.
        """

        weight = _expect(self.NAME, "weight", self.weight, MemorySpace.PIM_DRAM)
        gb = _expect(self.NAME, "global buffer", self.global_buffer, MemorySpace.PIM_GB)
        acc = _expect(self.NAME, "accumulator", self.accumulator, MemorySpace.PIM_MAC)
        _expect_same_dtype(self.NAME, weight=weight, global_buffer=gb, accumulator=acc)
        _check_mac_shape(self, acc.shape)
        return (acc,)


@dataclass(frozen=True, eq=False, slots=True)
class CentRdMacOp(_TileRefOp):
    """``cent.rd_mac(acc, y, *indices)``: save the MAC registers (``RD_MAC``).

    Each selected channel writes its banks' MAC registers as one Shared
    Buffer slot: channel ``c`` writes slot ``Rd + c``, where ``Rd`` is the
    first slot of the tile of ``y`` that the indices select (our convention;
    see ``examples/Tile_DSL/gemv.py``).

    The Shared Buffer is state, so the result is a new version of ``y``, of
    ``y``'s type: later reads of the written tile use it, not ``y``.
    """

    NAME = "cent.rd_mac"
    NUM_FIXED = 2

    @property
    def accumulator(self) -> Value:
        """The MAC registers read."""
        return self.operands[0]

    @property
    def destination(self) -> Value:
        """The Shared Buffer tensor written."""
        return self.operands[1]

    def _infer_tile_types(self) -> tuple[TensorType, ...]:
        """Return the destination's type.

        Raises:
            ValueError: If an operand is in the wrong memory space, the
                dtypes differ, or ``acc`` is not ``(len(channels), banks)``.
        """

        acc = _expect(self.NAME, "accumulator", self.accumulator, MemorySpace.PIM_MAC)
        destination = _expect(
            self.NAME, "destination", self.destination, MemorySpace.PNM_SB
        )
        _expect_same_dtype(self.NAME, accumulator=acc, destination=destination)
        _check_mac_shape(self, acc.shape)
        return (destination,)
