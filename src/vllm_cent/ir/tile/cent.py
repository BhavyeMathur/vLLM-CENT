"""CENT Tile IR: tensor types that name a CENT storage, and the CENT ops.

A value's ``CentTensorType`` says where its data is (DRAM banks, a Global
Buffer, the MAC registers or the Shared Buffer); each ``cent.*`` op says
where it needs its operands, as ``_DomainOp`` does for Device IR domains.
The ops are named after the AiM instructions they become: ``cent.wr_bias``,
``cent.wr_gb``, ``cent.mac_abk`` and ``cent.rd_mac``; ``cent.wr_bias_zero``
is a ``WR_BIAS`` of zeros, which reads no Shared Buffer slot.

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
from ..types import DType, MemorySpace, TensorType
from .tile import INDEX

__all__ = [
    "CentCopyBkGbOp",
    "CentCopyGbBkOp",
    "CentEwaddOp",
    "CentEwmulOp",
    "CentGbConcatOp",
    "CentMacAbkOp",
    "CentMacBkBkOp",
    "CentPnmOp",
    "CentRdAfOp",
    "CentRdMacOp",
    "CentRdMemOp",
    "CentSyncOp",
    "CentTensorType",
    "CentTileOp",
    "CentWrBiasOp",
    "CentWrBiasZeroOp",
    "CentWrGbOp",
    "CentWrMemOp",
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


def _check_mac_shape(
    op: CentTileOp, shape: tuple[int, ...], *, subset: bool = False
) -> None:
    """Check a MAC register tile's shape: ``(len(channels), banks)``.

    One row per selected channel, one column per bank. The bank count is
    the device's, so it is checked against the device later (Step 3).
    With ``subset``, the op may use some of the tile's channels: the tile
    has at least one row per selected channel.

    Raises:
        ValueError: If the shape is not two-dimensional, or its rows are
            not the selected channels.
    """

    _check_tile_shape(op.NAME, shape)
    rows_ok = shape[0] >= len(op.channels) if subset else shape[0] == len(op.channels)
    if len(shape) != 2 or not rows_ok:
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
class CentWrBiasZeroOp(CentTileOp):
    """``cent.wr_bias_zero()``: clear the MAC registers (``WR_BIAS`` of zeros).

    Starts a sum at zero without a Shared Buffer slot of zeros: like a zero
    register, it reads no slot, so no tensor has to hold zeros. The AiM
    trace still names a GPR, which the simulator requires but does not use;
    the code generator writes 0, as CENT's own traces do.

    The result is the MAC register tile, shape ``(len(channels), banks)``,
    in ``dtype``.
    """

    NAME = "cent.wr_bias_zero"
    NUM_OPERANDS = 0

    shape: tuple[int, ...] = field(kw_only=True)
    dtype: DType = field(kw_only=True)

    def __post_init__(self) -> None:
        """Store ``shape`` as a tuple, then run the other checks."""

        object.__setattr__(self, "shape", tuple(self.shape))
        super().__post_init__()

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return MAC registers of ``shape`` in ``dtype``.

        Raises:
            ValueError: If ``shape`` is not ``(len(channels), banks)``.
        """

        _check_mac_shape(self, self.shape)
        return (
            CentTensorType(
                shape=self.shape,
                dtype=DType.normalize(self.dtype),
                memory_space=MemorySpace.PIM_MAC,
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
    """``cent.mac_abk(gb, w, acc, *indices)``: multiply-accumulate (``MAC_ABK``).

    ``acc += x @ w.T`` for one tile: every bank of the selected channels
    multiplies its channel's Global Buffer (``x``'s tile) with the open row
    of its tile of ``w`` and adds the products to its MAC register. The
    operands follow ``x @ w.T``: the input, then the weight; the indices
    select the tile of ``w``.

    The result is the new value of the MAC registers, of ``acc``'s type, so
    a loop can carry it from one MAC to the next. The op may select some of
    ``acc``'s channels, e.g. one MAC per operation size where the last
    channels hold a partial tile.
    """

    NAME = "cent.mac_abk"
    NUM_FIXED = 3

    @property
    def global_buffer(self) -> Value:
        """The Global Buffer tile multiplied with each row: the input."""
        return self.operands[0]

    @property
    def weight(self) -> Value:
        """The DRAM tensor whose tile is multiplied."""
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
        _check_mac_shape(self, acc.shape, subset=True)
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


def _expect_in(
    op_name: str, what: str, value: Value, *spaces: MemorySpace
) -> CentTensorType:
    """Return ``value``'s type if it is a ``CentTensorType`` in one of ``spaces``."""

    value_type = value.type
    if (
        not isinstance(value_type, CentTensorType)
        or value_type.memory_space not in spaces
    ):
        names = " or ".join(space.value for space in spaces)
        raise ValueError(
            f"{op_name}: {what} must be in {names}, got {format_type(value_type)}"
        )
    return value_type


def _check_start(
    op_name: str, whole: TensorType, part: TensorType, start: tuple[int, ...]
) -> None:
    """Check that ``part`` fits in ``whole`` at ``start``, aligned to its last dims."""

    if len(start) != whole.rank or part.rank > whole.rank:
        raise ValueError(
            f"{op_name}: start {list(start)} must give one offset per dimension of "
            f"{list(whole.shape)}, which must hold {list(part.shape)}"
        )
    lead = whole.rank - part.rank
    extents = (1,) * lead + part.shape
    for offset, extent, size in zip(start, extents, whole.shape, strict=True):
        if offset < 0 or offset + extent > size:
            raise ValueError(
                f"{op_name}: {list(part.shape)} at {list(start)} does not fit in "
                f"{list(whole.shape)}"
            )


@dataclass(frozen=True, eq=False, slots=True)
class CentWrMemOp(Operation):
    """``cent.wr_mem(src, dst) {start}``: write a Shared Buffer tensor into DRAM.

    Ordinary DRAM writes (``W MEM channel bank row``), one per burst of
    ``dst`` that the elements land in. ``src`` fills ``dst`` from ``start``,
    aligned to ``dst``'s last dimensions: a whole vector, or one position of
    a KV cache. The result is the new version of ``dst``.
    """

    NAME = "cent.wr_mem"
    NUM_OPERANDS = 2

    start: tuple[int, ...] = field(kw_only=True)

    def __post_init__(self) -> None:
        """Store ``start`` as a tuple, then check the operands."""
        object.__setattr__(self, "start", tuple(self.start))
        super().__post_init__()

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return ``dst``'s type."""
        src = _expect(self.NAME, "source", self.operands[0], MemorySpace.PNM_SB)
        dst = _expect(self.NAME, "destination", self.operands[1], MemorySpace.PIM_DRAM)
        _expect_same_dtype(self.NAME, source=src, destination=dst)
        _check_start(self.NAME, dst, src, self.start)
        return (dst,)


@dataclass(frozen=True, eq=False, slots=True)
class CentRdMemOp(Operation):
    """``cent.rd_mem(src, dst) {start}``: read DRAM into a Shared Buffer tensor.

    Ordinary DRAM reads (``R MEM channel bank row``), one per burst of
    ``src`` read from ``start``. The result is the new version of ``dst``.
    """

    NAME = "cent.rd_mem"
    NUM_OPERANDS = 2

    start: tuple[int, ...] = field(kw_only=True)

    def __post_init__(self) -> None:
        """Store ``start`` as a tuple, then check the operands."""
        object.__setattr__(self, "start", tuple(self.start))
        super().__post_init__()

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return ``dst``'s type."""
        src = _expect(self.NAME, "source", self.operands[0], MemorySpace.PIM_DRAM)
        dst = _expect(self.NAME, "destination", self.operands[1], MemorySpace.PNM_SB)
        _expect_same_dtype(self.NAME, source=src, destination=dst)
        _check_start(self.NAME, src, dst, self.start)
        return (dst,)


@dataclass(frozen=True, eq=False, slots=True)
class CentMacBkBkOp(CentTileOp):
    """``cent.mac_bk_bk(a, b, acc)``: multiply-accumulate between neighbor banks.

    ``MAC_ABK`` with configuration register 0 (broadcast) at 0: every pair
    of banks ``2b`` and ``2b + 1`` multiplies their rows (``a`` in the even
    bank, ``b`` in the odd one) and adds to the MAC register. CENT squares
    a vector this way for RMSNorm, with ``a`` and ``b`` copies of it.
    """

    NAME = "cent.mac_bk_bk"
    NUM_OPERANDS = 3

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return ``acc``'s type."""
        a = _expect(
            self.NAME, "even-bank operand", self.operands[0], MemorySpace.PIM_DRAM
        )
        b = _expect(
            self.NAME, "odd-bank operand", self.operands[1], MemorySpace.PIM_DRAM
        )
        acc = _expect(self.NAME, "accumulator", self.operands[2], MemorySpace.PIM_MAC)
        _expect_same_dtype(self.NAME, a=a, b=b, accumulator=acc)
        _check_mac_shape(self, acc.shape)
        return (acc,)


@dataclass(frozen=True, eq=False, slots=True)
class CentEwmulOp(CentTileOp):
    """``cent.ewmul(a, b, out)``: elementwise multiply in the banks (``EWMUL``).

    In each group of four banks, banks ``4b`` and ``4b + 1`` of one row are
    multiplied into bank ``4b + 2`` of the same row. The result is the new
    version of ``out``.
    """

    NAME = "cent.ewmul"
    NUM_OPERANDS = 3

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return ``out``'s type."""
        a = _expect(self.NAME, "first operand", self.operands[0], MemorySpace.PIM_DRAM)
        b = _expect(self.NAME, "second operand", self.operands[1], MemorySpace.PIM_DRAM)
        out = _expect(self.NAME, "result", self.operands[2], MemorySpace.PIM_DRAM)
        _expect_same_dtype(self.NAME, a=a, b=b, out=out)
        if not a.shape == b.shape == out.shape:
            raise ValueError(
                f"{self.NAME}: operands {list(a.shape)}, {list(b.shape)} and result "
                f"{list(out.shape)} must have one shape"
            )
        return (out,)


@dataclass(frozen=True, eq=False, slots=True)
class CentCopyBkGbOp(CentTileOp):
    """``cent.copy_bkgb(src) {bank, shape}``: copy one bank's row into the GB.

    ``COPY_BKGB``: in every selected channel, bank ``bank`` of ``src``'s row
    goes into the channel's Global Buffer. The result is that GB tile.
    """

    NAME = "cent.copy_bkgb"
    NUM_OPERANDS = 1

    bank: int = field(kw_only=True)
    shape: tuple[int, ...] = field(kw_only=True)

    def __post_init__(self) -> None:
        """Store ``shape`` as a tuple, then check the operands."""
        object.__setattr__(self, "shape", tuple(self.shape))
        super().__post_init__()

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return a Global Buffer tile of ``shape``."""
        src = _expect(self.NAME, "source", self.operands[0], MemorySpace.PIM_DRAM)
        if self.bank < 0:
            raise ValueError(f"{self.NAME}: bank must be non-negative, got {self.bank}")
        _check_tile_shape(self.NAME, self.shape)
        return (
            CentTensorType(
                shape=self.shape, dtype=src.dtype, memory_space=MemorySpace.PIM_GB
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class CentCopyGbBkOp(CentTileOp):
    """``cent.copy_gbbk(gb, dst) {bank}``: copy the GB into one bank's row.

    ``COPY_GBBK``: in every selected channel, the Global Buffer goes into
    bank ``bank`` of ``dst``'s row. The result is the new version of ``dst``.
    """

    NAME = "cent.copy_gbbk"
    NUM_OPERANDS = 2

    bank: int = field(kw_only=True)

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return ``dst``'s type."""
        gb = _expect(self.NAME, "global buffer", self.operands[0], MemorySpace.PIM_GB)
        dst = _expect(self.NAME, "destination", self.operands[1], MemorySpace.PIM_DRAM)
        _expect_same_dtype(self.NAME, global_buffer=gb, destination=dst)
        if self.bank < 0:
            raise ValueError(f"{self.NAME}: bank must be non-negative, got {self.bank}")
        return (dst,)


@dataclass(frozen=True, eq=False, slots=True)
class CentRdAfOp(_TileRefOp):
    """``cent.rd_af(acc, y, *indices)``: activate and read (``AF``, ``RD_AF``).

    Each selected channel applies the activation function to its banks' MAC
    registers (``AF``) and writes the results as one Shared Buffer slot
    (``RD_AF``), as ``cent.rd_mac`` does. The MAC registers keep their
    value, so ``cent.rd_mac`` may read them after. The result is the new
    version of ``y``.
    """

    NAME = "cent.rd_af"
    NUM_FIXED = 2

    @property
    def accumulator(self) -> Value:
        """The MAC registers activated."""
        return self.operands[0]

    @property
    def destination(self) -> Value:
        """The Shared Buffer tensor written."""
        return self.operands[1]

    def _infer_tile_types(self) -> tuple[TensorType, ...]:
        """Return the destination's type."""
        acc = _expect(self.NAME, "accumulator", self.accumulator, MemorySpace.PIM_MAC)
        destination = _expect(
            self.NAME, "destination", self.destination, MemorySpace.PNM_SB
        )
        _expect_same_dtype(self.NAME, accumulator=acc, destination=destination)
        _check_mac_shape(self, acc.shape)
        return (destination,)


@dataclass(frozen=True, eq=False, slots=True)
class CentSyncOp(Operation):
    """``cent.sync()``: wait for every issued command (``SYNC``)."""

    NAME = "cent.sync"
    NUM_OPERANDS = 0

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """No results."""
        return ()


@dataclass(frozen=True, eq=False, slots=True)
class CentEwaddOp(Operation):
    """``cent.ewadd(a, b)``: add ``b`` into ``a`` in the Shared Buffer (``EWADD``).

    CENT issues it for the residual adds; the AiM simulator executes it as
    a no-op. The result is the new version of ``a``.
    """

    NAME = "cent.ewadd"
    NUM_OPERANDS = 2

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return ``a``'s type."""
        a = _expect(self.NAME, "first operand", self.operands[0], MemorySpace.PNM_SB)
        b = _expect(self.NAME, "second operand", self.operands[1], MemorySpace.PNM_SB)
        if (a.shape, a.dtype) != (b.shape, b.dtype):
            raise ValueError(
                f"{self.NAME}: operands {format_type(a)} and {format_type(b)} differ"
            )
        return (a,)


@dataclass(frozen=True, eq=False, slots=True)
class CentGbConcatOp(Operation):
    """``cent.gb_concat(*gbs)``: Global Buffer tiles of disjoint channels as one.

    Issues nothing: it lets one ``cent.mac_abk`` read Global Buffers that
    different ``cent.wr_gb`` filled with different tiles, e.g. one context
    shard per channel group. The result has the largest operand shape.
    """

    NAME = "cent.gb_concat"
    NUM_OPERANDS = None

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return a Global Buffer tile of the largest operand shape."""
        if len(self.operands) < 2:
            raise ValueError(f"{self.NAME}: needs at least two Global Buffer tiles")
        types = [
            _expect(self.NAME, f"operand {i}", v, MemorySpace.PIM_GB)
            for i, v in enumerate(self.operands)
        ]
        _expect_same_dtype(
            self.NAME, **{f"operand_{i}": t for i, t in enumerate(types)}
        )
        largest = max(types, key=lambda t: t.numel)
        return (largest,)


@dataclass(frozen=True, eq=False, slots=True)
class CentPnmOp(Operation):
    """``cent.pnm(*inputs, into) {kind, cycles}``: PNM work, modeled analytically.

    CENT computes some steps on the processing-near-memory units (RISC-V
    cores and vector units by the Shared Buffer) and charges them with an
    analytical latency instead of simulating them. This op stands for such
    a step: it reads ``inputs`` and writes ``into``, issues no AiM command,
    and appears in the trace as a comment with its ``cycles``. The result
    is the new version of ``into``.
    """

    NAME = "cent.pnm"
    NUM_OPERANDS = None

    kind: str = field(kw_only=True)
    cycles: float = field(kw_only=True)

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Return ``into``'s type."""
        if not self.operands:
            raise ValueError(f"{self.NAME}: needs a destination")
        into = _expect(self.NAME, "destination", self.operands[-1], MemorySpace.PNM_SB)
        for i, value in enumerate(self.operands[:-1]):
            _expect_in(
                self.NAME, f"input {i}", value, MemorySpace.PNM_SB, MemorySpace.PIM_DRAM
            )
        if self.cycles < 0:
            raise ValueError(
                f"{self.NAME}: cycles must be non-negative, got {self.cycles}"
            )
        return (into,)
