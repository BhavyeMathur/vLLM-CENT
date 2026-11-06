"""Physical layout of CENT tensors in DRAM banks and the Shared Buffer.

A ``PhysicalLayout`` is a mixed-radix factorization of a tensor's logical
dimensions into temporal wave axes (``Serial``, ``Row``) and spatial hardware
axes (``Channel``, ``Bank``, ``Column`` in DRAM; ``Slot``, ``Lane`` in the
Shared Buffer). Within each tensor dimension, factors are listed from
outermost (slowest-changing, largest stride) to innermost (fastest-changing,
``stride == 1``).

For example, the GEMV/GEMM weight layout::

    split(
        n=(Serial("n_wave", 8), Channel(), Bank()),
        k=(Row("k_row", 4), Column()),
    )

factors ``n`` (4096) with strides ``(512, 16, 1)`` and ``k`` (4096) with
strides ``(1024, 1)``. Every logical element ``w[n, k]`` therefore maps to a
unique hardware coordinate::

    bank    = (n // 1) % 16
    channel = (n // 16) % 32
    n_wave  = (n // 512) % 8
    col     = (k // 1) % 1024
    k_row   = (k // 1024) % 4
    row     = base + n_wave * 4 + k_row

Fixing the wave indices ``(n_wave, k_row)`` selects one whole DRAM tile
(``DramTile(row, opsize)``); fixing all logical coordinates ``(n, k)``
locates one scalar element (``DramElement(channel, bank, row, col)``).

The channel number may be split over several ``Channel`` factors, e.g. to
give two channel groups different halves of the reduction axis of
``w^T`` ``(K, N)``::

    split(
        k=(Channel(2), Row("k_row", 4), Column()),
        n=(Serial("n_wave", 16), Channel(16), Bank()),
    )

The factors form the channel number in mixed radix, the first in layout
order being the most significant: ``channel = group * 16 + c``, so channels
0-15 hold the first half of ``k`` and channels 16-31 the second. A Shared
Buffer tile written by ``RD_MAC`` splits its ``Slot(of=Channel(...))``
factors the same way, so channel ``c`` still lands in slot ``Rd + c``.

A ``Mac`` factor lays a weight out for a bank MAC array wider than AiM's
one output: e.g. ``n=(Channel(), Bank(), Mac())`` sends consecutive outputs
to one bank's array, and ``Mac`` and ``Column`` share each row in mixed
radix. ``locate_element`` reports the array output as ``mac``. No CENT op
uses a ``Mac`` wider than 1 yet.

v0.21 restrictions: a DRAM tile occupies one row of each bank, from column
0; a Shared Buffer tile occupies consecutive slots; every DRAM layout spans
all channels and banks.
"""

from __future__ import annotations

import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from math import prod

from ..device.cent import CentArchitecture
from ..ir.base import Value
from ..ir.printer import format_type
from ..ir.tile.cent import CentTensorType
from ..ir.types import DType, MemorySpace, TensorType

__all__ = [
    "Bank",
    "Channel",
    "Column",
    "ColumnBlock",
    "DramElement",
    "DramTile",
    "Factor",
    "Lane",
    "LayoutSpec",
    "Mac",
    "PhysicalLayout",
    "ResolvedFactor",
    "Row",
    "SbElement",
    "SbTile",
    "Serial",
    "Slot",
    "allocate",
    "resolve",
    "split",
]


@dataclass(frozen=True, slots=True)
class Serial:
    """A named temporal wave axis selected by a loop index across tiles."""

    name: str
    extent: int

    def __post_init__(self) -> None:
        """Reject an empty name or an extent below 1."""
        if not self.name:
            raise ValueError("Serial factor name must not be empty")
        if self.extent < 1:
            raise ValueError(
                f"Serial({self.name!r}) extent must be at least 1, got {self.extent}"
            )


@dataclass(frozen=True, slots=True)
class Row:
    """A named temporal axis along DRAM rows (DRAM-only alias of ``Serial``)."""

    name: str
    extent: int

    def __post_init__(self) -> None:
        """Reject an empty name or an extent below 1."""
        if not self.name:
            raise ValueError("Row factor name must not be empty")
        if self.extent < 1:
            raise ValueError(
                f"Row({self.name!r}) extent must be at least 1, got {self.extent}"
            )


@dataclass(frozen=True, slots=True)
class Channel:
    """DRAM channels; several ``Channel`` factors split the channel number.

    ``Channel()`` alone spans every channel (``arch.channels``). Several
    factors form the channel number in mixed radix, the first in layout
    order being the most significant. Their extents multiply to
    ``arch.channels``; at most one may leave its extent out, to be inferred.
    """

    extent: int | None = None

    def __post_init__(self) -> None:
        """Reject an extent below 1."""
        if self.extent is not None and self.extent < 1:
            raise ValueError(f"Channel extent must be at least 1, got {self.extent}")


@dataclass(frozen=True, slots=True)
class Bank:
    """The banks within one DRAM channel.

    ``Bank()`` spans every bank (``arch.banks_per_channel``).
    ``Bank(every=k, offset=g)`` spans member ``g`` of each group of ``k``
    neighboring banks: coordinate ``b`` is bank ``k * b + g``. CENT keeps
    elementwise operands this way: ``EWMUL`` multiplies banks ``4b`` and
    ``4b + 1`` into ``4b + 2``, and a MAC between neighbors pairs ``2b`` with
    ``2b + 1``.
    """

    every: int = 1
    offset: int = 0

    def __post_init__(self) -> None:
        """Reject a group below 1 or an offset outside it."""
        if self.every < 1 or not 0 <= self.offset < self.every:
            raise ValueError(
                f"Bank needs every >= 1 and 0 <= offset < every, got "
                f"every={self.every}, offset={self.offset}"
            )

    def bank(self, coord: int) -> int:
        """The bank in a channel that coordinate ``coord`` names."""
        return self.every * coord + self.offset


@dataclass(frozen=True, slots=True)
class Column:
    """Contiguous elements along one DRAM row (inferred when ``None``)."""

    extent: int | None = None

    def __post_init__(self) -> None:
        """Reject an extent below 1."""
        if self.extent is not None and self.extent < 1:
            raise ValueError(f"Column extent must be at least 1, got {self.extent}")


@dataclass(frozen=True, slots=True)
class Slot:
    """Shared Buffer slots within one tile.

    ``Slot(of=Channel(...))`` gives one slot per channel, numbered as the
    ``Channel`` factors of a DRAM layout are: ``Slot(of=Channel())`` alone
    spans every channel, and several such factors split the channel number
    in mixed radix (see ``Channel``). Their slot offset within the tile is
    then the channel number, as ``RD_MAC`` writes it.
    """

    extent: int | None = None
    of: Channel | None = None

    def __post_init__(self) -> None:
        """Reject a bad ``of``, an extent below 1, or two different extents."""
        if self.of is not None and not isinstance(self.of, Channel):
            raise ValueError(f"Slot.of must be Channel() or None, got {self.of!r}")
        if self.extent is not None and self.extent < 1:
            raise ValueError(f"Slot extent must be at least 1, got {self.extent}")
        if (
            self.of is not None
            and self.extent is not None
            and self.of.extent is not None
            and self.extent != self.of.extent
        ):
            raise ValueError(
                f"Slot extent {self.extent} differs from its Channel extent "
                f"{self.of.extent}"
            )


@dataclass(frozen=True, slots=True)
class Lane:
    """Lanes within one 256-bit Shared Buffer burst (``of=Bank()`` binds to banks)."""

    of: Bank | None = None

    def __post_init__(self) -> None:
        """Reject an ``of`` that is not ``Bank()``."""
        if self.of is not None and not isinstance(self.of, Bank):
            raise ValueError(f"Lane.of must be Bank() or None, got {self.of!r}")


@dataclass(frozen=True, slots=True)
class Mac:
    """The outputs of one bank's MAC array (``arch.mac_array_width``).

    A DRAM tile factor: values that differ only along ``Mac`` feed different
    outputs of the same bank's MAC array in one MAC instruction, so they
    sit in the same row. Within the row, ``Mac`` and ``Column`` form the
    column address in mixed radix, in layout order. AiM's MAC array is one
    output wide, and no ``cent.*`` op supports a wider one yet: the
    verifier rejects a weight laid out with ``Mac`` wider than 1.
    """

    extent: int | None = None

    def __post_init__(self) -> None:
        """Reject an extent below 1."""
        if self.extent is not None and self.extent < 1:
            raise ValueError(f"Mac extent must be at least 1, got {self.extent}")


@dataclass(frozen=True, slots=True)
class ColumnBlock:
    """A named index factor that selects a block of columns within a row.

    Where ``Serial``/``Row`` select the DRAM row of a tile, a column block
    puts several tiles side by side in one row: the index adds
    ``index * (columns of one tile)`` to the column address. CENT keeps eight
    128-value keys per 1024-value row this way.
    """

    name: str
    extent: int

    def __post_init__(self) -> None:
        """Reject an empty name or an extent below 1."""
        if not self.name:
            raise ValueError("ColumnBlock factor name must not be empty")
        if self.extent < 1:
            raise ValueError(
                f"ColumnBlock({self.name!r}) extent must be at least 1, got "
                f"{self.extent}"
            )


Factor = Serial | Row | Channel | Bank | Column | ColumnBlock | Mac | Slot | Lane


@dataclass(frozen=True, slots=True)
class LayoutSpec:
    """Unresolved per-dimension factorizations, outermost factor first.

    Attributes:
        dims: The factors of each tensor dimension, outermost first.
        dim_names: A name per dimension, for messages.
        channels: For a DRAM layout, the consecutive channels it occupies;
            None for all of them. Its ``Channel`` factors then number the
            channels from ``channels.start``, e.g. K and V projections on
            disjoint channel groups.
    """

    dims: tuple[tuple[Factor, ...], ...]
    dim_names: tuple[str, ...] = ()
    channels: range | None = None

    def __post_init__(self) -> None:
        """Store the factors as tuples; check names and the channel range."""
        dims = tuple(tuple(d) for d in self.dims)
        object.__setattr__(self, "dims", dims)
        object.__setattr__(self, "dim_names", tuple(self.dim_names))
        if self.dim_names and len(self.dim_names) != len(dims):
            raise ValueError("dim_names must match dims in length")
        if self.channels is not None and (
            self.channels.step != 1 or len(self.channels) < 1 or self.channels.start < 0
        ):
            raise ValueError(
                f"channels must be a non-empty range of consecutive channels, "
                f"got {self.channels}"
            )


def split(*, channels: range | None = None, **dims: Sequence[Factor]) -> LayoutSpec:
    """Build a ``LayoutSpec`` from keyword arguments in tensor-dimension order.

    Args:
        channels: For a DRAM layout, the channels it occupies (default all).
        **dims: Each dimension's factors, outermost first.
    """

    return LayoutSpec(
        dims=tuple(tuple(factors) for factors in dims.values()),
        dim_names=tuple(dims.keys()),
        channels=channels,
    )


@dataclass(frozen=True, slots=True)
class ResolvedFactor:
    """One factor with its tensor dimension, concrete extent, and within-dim stride.

    ``stride`` is the product of all inner factors' extents within the same
    tensor dimension, so logical coordinate ``c`` along ``dim`` projects onto
    this factor as ``(c // stride) % extent``.
    """

    factor: Factor
    dim: int
    extent: int
    stride: int = 1

    @property
    def is_index(self) -> bool:
        """Whether an index selects this factor (``Serial``, ``Row``, ``ColumnBlock``)."""
        return isinstance(self.factor, (Serial, Row, ColumnBlock))

    @property
    def name(self) -> str | None:
        """The index name of an index factor, or ``None`` for tile factors."""
        if isinstance(self.factor, (Serial, Row, ColumnBlock)):
            return self.factor.name
        return None

    def extract(self, dim_coord: int) -> int:
        """Extract this factor's coordinate from the logical coordinate of ``dim``."""
        return (dim_coord // self.stride) % self.extent


@dataclass(frozen=True, slots=True)
class DramTile:
    """Physical location of one DRAM tile across all selected channels and banks.

    ``opsize`` covers the bank with the most elements; a partial tile may
    hold fewer in some channels (``PhysicalLayout.channel_opsizes``).
    """

    row: int
    opsize: int
    column: int = 0


@dataclass(frozen=True, slots=True)
class SbTile:
    """Physical location of one Shared Buffer tile."""

    first_slot: int
    slots: int


@dataclass(frozen=True, slots=True)
class DramElement:
    """Physical location of one scalar element in DRAM."""

    channel: int
    bank: int
    row: int
    col: int
    mac: int = 0  # which output of the bank's MAC array; 0 without a Mac factor


@dataclass(frozen=True, slots=True)
class SbElement:
    """Physical location of one scalar element in the Shared Buffer."""

    slot: int
    lane: int


@dataclass(frozen=True, slots=True)
class PhysicalLayout:
    """Resolved mixed-radix mapping from tensor coordinates to CENT storage.

    Built by ``resolve``, which derives every field but ``base`` from
    ``spec``, the tensor type and the architecture; ``allocate`` (or a
    mapping plan) sets ``base``. ``verify_kernel`` resolves ``spec`` again
    for the device it checks, so a layout made for another device or dtype
    is rejected.
    """

    space: MemorySpace
    factors: tuple[ResolvedFactor, ...]
    shape: tuple[int, ...]
    burst_values: int
    spec: LayoutSpec
    base: int = 0

    def __post_init__(self) -> None:
        """Store the factors and shape as tuples."""
        object.__setattr__(self, "factors", tuple(self.factors))
        object.__setattr__(self, "shape", tuple(self.shape))

    @property
    def index_factors(self) -> tuple[ResolvedFactor, ...]:
        """The ``Serial`` and ``Row`` factors in layout order."""
        return tuple(rf for rf in self.factors if rf.is_index)

    @property
    def channel_factors(self) -> tuple[ResolvedFactor, ...]:
        """The factors that form the channel number, most significant first.

        ``Channel`` factors in DRAM, ``Slot(of=Channel(...))`` factors in the
        Shared Buffer.
        """
        return tuple(rf for rf in self.factors if _is_channel_part(rf.factor))

    @property
    def first_channel(self) -> int:
        """The lowest channel the layout occupies (``LayoutSpec.channels``)."""
        return 0 if self.spec.channels is None else self.spec.channels.start

    def occupied_channels(self, arch_channels: int) -> range:
        """The channels the layout occupies on a device of ``arch_channels``."""
        return (
            range(arch_channels) if self.spec.channels is None else self.spec.channels
        )

    def channel_base(self, first_slot: int) -> int:
        """``Rd``/``Rs`` of a tile of one slot per channel.

        Channel ``c`` uses slot ``Rd + c``. A layout on channels from ``f``
        stores channel ``c`` at ``first_slot + c - f``, so ``Rd`` is
        ``first_slot - f``.
        """
        return first_slot - self.first_channel

    @property
    def numbered_factors(self) -> tuple[ResolvedFactor, ...]:
        """The index factors that number tiles: ``Serial`` and ``Row``."""
        return tuple(rf for rf in self.factors if isinstance(rf.factor, (Serial, Row)))

    @property
    def column_factors(self) -> tuple[ResolvedFactor, ...]:
        """The factors of the column address, high first: column blocks, Mac, Column."""
        return tuple(
            rf
            for rf in self.factors
            if isinstance(rf.factor, (ColumnBlock, Mac, Column))
        )

    @property
    def bank_factor(self) -> ResolvedFactor:
        """The ``Bank`` factor of a DRAM layout."""
        return next(rf for rf in self.factors if isinstance(rf.factor, Bank))

    @property
    def row_factors(self) -> tuple[ResolvedFactor, ...]:
        """The factors that form the column address in a DRAM row, high first.

        ``Mac`` and ``Column``, in layout order.
        """
        return tuple(rf for rf in self.factors if isinstance(rf.factor, (Mac, Column)))

    @property
    def mac_width(self) -> int:
        """The MAC array outputs a DRAM tile feeds per bank; 1 without ``Mac``."""
        return prod(rf.extent for rf in self.factors if isinstance(rf.factor, Mac))

    @property
    def tile_factors(self) -> tuple[ResolvedFactor, ...]:
        """The spatial hardware factors within one tile, in layout order."""
        return tuple(rf for rf in self.factors if not rf.is_index)

    @property
    def index_names(self) -> tuple[str, ...]:
        """Names of all index factors, in layout order."""
        return tuple(rf.name for rf in self.index_factors if rf.name is not None)

    @property
    def tile_shape(self) -> tuple[int, ...]:
        """Extents of the spatial tile factors, in layout order."""
        return tuple(rf.extent for rf in self.tile_factors)

    @property
    def slots_per_tile(self) -> int:
        """Number of Shared Buffer slots per tile (for ``PNM_SB`` layouts)."""
        return prod(
            rf.extent for rf in self.tile_factors if isinstance(rf.factor, Slot)
        )

    @property
    def needs_channel_alignment(self) -> bool:
        """Whether ``base`` must be aligned to ``arch.channels`` (``Slot(of=Channel())``)."""
        return any(
            isinstance(rf.factor, Slot) and rf.factor.of is not None
            for rf in self.tile_factors
        )

    def tile_count(self) -> int:
        """Number of rows (DRAM) or slot groups (Shared Buffer) the tiles use."""
        return prod(rf.extent for rf in self.numbered_factors)

    def tile_number(self, index: Mapping[str, int]) -> int:
        """Row-major linearization of ``index`` along ``index_factors``."""

        expected = self.index_names
        if set(index.keys()) != set(expected):
            raise ValueError(
                f"tile index keys {sorted(index.keys())} do not match "
                f"layout index factors {list(expected)}"
            )
        for rf in self.index_factors:
            assert rf.name is not None
            if not 0 <= index[rf.name] < rf.extent:
                raise ValueError(
                    f"index {rf.name!r} = {index[rf.name]} out of bounds for "
                    f"extent {rf.extent}"
                )
        number = 0
        for rf in self.numbered_factors:
            assert rf.name is not None
            number = number * rf.extent + index[rf.name]
        return number

    def footprint(self) -> int:
        """Rows per bank (DRAM) or slots (Shared Buffer) occupied by the whole tensor."""

        if self.space is MemorySpace.PIM_DRAM:
            return self.tile_count()
        return self.tile_count() * self.slots_per_tile

    def locate(self, index: Mapping[str, int]) -> DramTile | SbTile:
        """Return the physical tile address selected by ``index``.

        A last, partial tile (see ``resolve``) has a smaller ``opsize`` or
        fewer ``slots``: only the bursts that hold elements.
        """

        tile_num = self.tile_number(index)
        if self.space is MemorySpace.PIM_DRAM:
            # The column where the tile starts: its column blocks' digits.
            column = 0
            for rf in self.column_factors:
                digit = index[rf.name] if rf.name is not None else 0
                column = column * rf.extent + digit
            opsizes = self.channel_opsizes(index)
            return DramTile(
                row=self.base + tile_num,
                opsize=max(opsizes.values(), default=0),
                column=column,
            )
        valid = self._valid_extents(index)
        return SbTile(
            first_slot=self.base + tile_num * self.slots_per_tile,
            slots=prod(
                valid[rf] for rf in self.tile_factors if isinstance(rf.factor, Slot)
            ),
        )

    def occupied_cells(self) -> frozenset[tuple[int, int]]:
        """The (channel, bank) pairs a DRAM layout puts elements in, in any row."""

        channels = self.channel_factors
        bank_rf = self.bank_factor
        bank = bank_rf.factor
        assert isinstance(bank, Bank)
        cells = set()
        for digits in itertools.product(*(range(rf.extent) for rf in channels)):
            channel = 0
            for rf, digit in zip(channels, digits, strict=True):
                channel = channel * rf.extent + digit
            for b in range(bank_rf.extent):
                cells.add((self.first_channel + channel, bank.bank(b)))
        return frozenset(cells)

    def channel_opsizes(self, index: Mapping[str, int]) -> dict[int, int]:
        """The bursts of the tile in each channel: those of its fullest bank.

        A channel with no element of the tile has 0. Every channel of a whole
        tile has the same count; a partial tile (the last tile along an index
        factor, or along the outermost ``Channel`` factor) may have fewer.
        """

        opsizes: dict[int, int] = {}
        for (channel, _bank), values in self.cell_values(index).items():
            bursts = -(-values // self.burst_values)
            opsizes[channel] = max(opsizes.get(channel, 0), bursts)
        return opsizes

    def cell_values(self, index: Mapping[str, int]) -> dict[tuple[int, int], int]:
        """The tile's elements in each (channel, bank) of a DRAM layout."""

        offset = [0] * len(self.shape)
        for rf in self.index_factors:
            assert rf.name is not None
            offset[rf.dim] += index[rf.name] * rf.stride
        channels = self.channel_factors
        bank_rf = self.bank_factor
        bank = bank_rf.factor
        assert isinstance(bank, Bank)
        whole = all(
            prod(rf.extent for rf in self.factors if rf.dim == dim) == size
            for dim, size in enumerate(self.shape)
        )
        full = prod(rf.extent for rf in self.row_factors)
        cells: dict[tuple[int, int], int] = {}
        for digits in itertools.product(*(range(rf.extent) for rf in channels)):
            channel = 0
            for rf, digit in zip(channels, digits, strict=True):
                channel = channel * rf.extent + digit
            channel += self.first_channel
            for b in range(bank_rf.extent):
                if whole:
                    cells[(channel, bank.bank(b))] = full
                    continue
                position = list(offset)
                for rf, digit in zip(channels, digits, strict=True):
                    position[rf.dim] += digit * rf.stride
                position[bank_rf.dim] += b * bank_rf.stride
                cells[(channel, bank.bank(b))] = self._row_values(position)
        return cells

    def _row_values(self, position: Sequence[int]) -> int:
        """Elements in the row of one bank whose fixed coordinates are ``position``."""

        count = 1
        for dim, size in enumerate(self.shape):
            factors = [rf for rf in self.factors if rf.dim == dim]
            in_row = [rf for rf in factors if isinstance(rf.factor, (Mac, Column))]
            remaining = size - position[dim]
            if not in_row:
                count *= 1 if remaining > 0 else 0
            else:
                count *= _count_below(in_row, remaining)
        return count

    def _valid_extents(self, index: Mapping[str, int]) -> dict[ResolvedFactor, int]:
        """How many values of each tile factor hold elements in this tile.

        Every factor is full except in a dimension whose last index selects
        a partial tile: there a factor of stride ``s`` holds
        ``ceil(remaining / s)`` values, at most its extent.
        """

        valid = {rf: rf.extent for rf in self.factors}
        for dim, size in enumerate(self.shape):
            factors = [rf for rf in self.factors if rf.dim == dim]
            outer = factors[0]
            if outer.extent * outer.stride == size or outer.name is None:
                continue
            remaining = size - index[outer.name] * outer.stride
            if remaining >= outer.stride:
                continue
            for rf in factors[1:]:
                valid[rf] = min(rf.extent, -(-remaining // rf.stride))
        return valid

    def locate_element(self, coords: Sequence[int]) -> DramElement | SbElement:
        """Map one logical element coordinate ``(d_0, d_1, ...)`` to hardware storage."""

        if len(coords) != len(self.shape):
            raise ValueError(
                f"expected {len(self.shape)} coordinates for shape {self.shape}, "
                f"got {len(coords)}"
            )
        for d, (c, size) in enumerate(zip(coords, self.shape, strict=True)):
            if not 0 <= c < size:
                raise ValueError(
                    f"coordinate {c} along dimension {d} out of bounds for size {size}"
                )

        # Linearize the tile-numbering factors in appearance order.
        tile_num = 0
        for rf in self.numbered_factors:
            tile_num = tile_num * rf.extent + rf.extract(coords[rf.dim])

        if self.space is MemorySpace.PIM_DRAM:
            # The channel number, in mixed radix over the Channel factors.
            channel = 0
            for rf in self.channel_factors:
                channel = channel * rf.extent + rf.extract(coords[rf.dim])
            channel += self.first_channel
            # The column address, in mixed radix over column blocks, Mac and Column.
            col = 0
            for rf in self.column_factors:
                col = col * rf.extent + rf.extract(coords[rf.dim])
            bk_rf = self.bank_factor
            bank = bk_rf.factor
            assert isinstance(bank, Bank)
            mac = next(
                (
                    rf.extract(coords[rf.dim])
                    for rf in self.tile_factors
                    if isinstance(rf.factor, Mac)
                ),
                0,
            )
            return DramElement(
                channel=channel,
                bank=bank.bank(bk_rf.extract(coords[bk_rf.dim])),
                row=self.base + tile_num,
                col=col,
                mac=mac,
            )

        slot_in_tile = 0
        for rf in self.tile_factors:
            if isinstance(rf.factor, Slot):
                slot_in_tile = slot_in_tile * rf.extent + rf.extract(coords[rf.dim])
        lane_rf = next(rf for rf in self.tile_factors if isinstance(rf.factor, Lane))
        return SbElement(
            slot=self.base + tile_num * self.slots_per_tile + slot_in_tile,
            lane=lane_rf.extract(coords[lane_rf.dim]),
        )


def _count_below(factors: Sequence[ResolvedFactor], remaining: int) -> int:
    """Coordinate tuples of ``factors`` whose offset ``sum(c * stride)`` < ``remaining``."""

    if remaining <= 0:
        return 0
    if not factors:
        return 1
    outer, *inner = sorted(factors, key=lambda rf: rf.stride, reverse=True)
    if not inner:
        return min(outer.extent, -(-remaining // outer.stride))
    return sum(
        _count_below(inner, remaining - c * outer.stride) for c in range(outer.extent)
    )


def _is_channel_part(factor: Factor) -> bool:
    """Whether ``factor`` is one digit of the channel number."""
    return isinstance(factor, Channel) or (
        isinstance(factor, Slot) and factor.of is not None
    )


def _declared_channel_extent(factor: Channel | Slot) -> int | None:
    """The extent a channel factor declares, or None to infer it."""
    if isinstance(factor, Channel):
        return factor.extent
    assert factor.of is not None
    return factor.extent if factor.extent is not None else factor.of.extent


def _channel_extents(
    spec: LayoutSpec, space: MemorySpace, arch: CentArchitecture
) -> dict[tuple[int, int], int]:
    """Resolve the extents of the factors that form the channel number.

    Returns:
        The extent of each channel factor, keyed by (dimension, position).

    Raises:
        ValueError: If more than one channel factor omits its extent, the
            extents do not multiply to the layout's channel count, or the
            channel range does not fit the device or is not a DRAM layout.
    """

    total = arch.channels
    if spec.channels is not None:
        if spec.channels.stop > arch.channels:
            raise ValueError(
                f"channels {spec.channels} exceed the device's {arch.channels}"
            )
        total = len(spec.channels)

    kind = "Channel" if space is MemorySpace.PIM_DRAM else "Slot(of=Channel)"
    in_dram = space is MemorySpace.PIM_DRAM
    declared: dict[tuple[int, int], int | None] = {}
    for d, factors in enumerate(spec.dims):
        for i, f in enumerate(factors):
            # Only the space's own kind; a misplaced one is reported later.
            if not _is_channel_part(f) or isinstance(f, Channel) != in_dram:
                continue
            assert isinstance(f, (Channel, Slot))
            declared[(d, i)] = _declared_channel_extent(f)
    if not declared:
        return {}

    missing = [key for key, e in declared.items() if e is None]
    known = prod(e for e in declared.values() if e is not None)
    if len(missing) > 1:
        raise ValueError(
            f"{len(missing)} {kind} factors omit their extent; at most one may"
        )
    if missing:
        if total % known != 0:
            raise ValueError(
                f"{kind} extents {known} do not divide the {total} channels"
            )
        declared[missing[0]] = total // known
    extents = {key: int(e) for key, e in declared.items() if e is not None}
    if prod(extents.values()) != total:
        raise ValueError(
            f"{kind} extents {list(extents.values())} multiply to "
            f"{prod(extents.values())}, expected {total} channels"
        )
    return extents


def _values_in_bytes(what: str, byte_count: int, dtype: DType) -> int:
    bits = byte_count * 8
    if bits % dtype.bits != 0:
        raise ValueError(
            f"{what} ({byte_count} bytes) does not hold a whole number of {dtype.value} values"
        )
    return bits // dtype.bits


def resolve(
    spec: LayoutSpec, tensor: TensorType, arch: CentArchitecture
) -> PhysicalLayout:
    """Resolve factor extents and strides against ``tensor`` and ``arch``.

    Raises:
        ValueError: If ``tensor`` is not in ``pim.dram`` or ``pnm.sb``, the
            rank or factor products do not match ``tensor.shape``, factors
            violate storage or hardware rules, or more than one factor in a
            dimension omits its extent.
    """

    if not isinstance(tensor, CentTensorType) or tensor.memory_space not in (
        MemorySpace.PIM_DRAM,
        MemorySpace.PNM_SB,
    ):
        raise ValueError(
            f"PhysicalLayout requires a CentTensorType in pim.dram or pnm.sb, "
            f"got {format_type(tensor)}"
        )

    space = tensor.memory_space
    if len(spec.dims) != tensor.rank:
        raise ValueError(
            f"layout has {len(spec.dims)} dimensions, but tensor "
            f"{format_type(tensor)} has rank {tensor.rank}"
        )

    burst_values = _values_in_bytes("burst_size", arch.burst_size, tensor.dtype)
    row_values = _values_in_bytes("row_size", arch.row_size, tensor.dtype)

    channel_extents = _channel_extents(spec, space, arch)
    seen_names: set[str] = set()
    resolved_all: list[ResolvedFactor] = []

    for dim_idx, (dim_size, dim_factors) in enumerate(
        zip(tensor.shape, spec.dims, strict=True)
    ):
        dim_label = (
            f"dimension {spec.dim_names[dim_idx]!r} ({dim_idx})"
            if spec.dim_names
            else f"dimension {dim_idx}"
        )
        if not dim_factors:
            raise ValueError(f"{dim_label} has no layout factors")

        extents: list[int | None] = []
        for pos, f in enumerate(dim_factors):
            if isinstance(f, Row) and space is not MemorySpace.PIM_DRAM:
                raise ValueError(
                    f"Row({f.name!r}) is only allowed in pim.dram layouts, "
                    f"not {space.value}"
                )
            if isinstance(f, ColumnBlock) and space is not MemorySpace.PIM_DRAM:
                raise ValueError(
                    f"ColumnBlock({f.name!r}) is only allowed in pim.dram layouts"
                )
            if isinstance(f, (Serial, Row, ColumnBlock)):
                if f.name in seen_names:
                    raise ValueError(f"duplicate index factor name {f.name!r}")
                seen_names.add(f.name)
                extents.append(f.extent)
            elif isinstance(f, Channel):
                if space is not MemorySpace.PIM_DRAM:
                    raise ValueError(
                        f"Channel() is only allowed in pim.dram layouts, not {space.value}"
                    )
                extents.append(channel_extents[(dim_idx, pos)])
            elif isinstance(f, Bank):
                if space is not MemorySpace.PIM_DRAM:
                    raise ValueError(
                        f"Bank() is only allowed in pim.dram layouts, not {space.value}"
                    )
                if arch.banks_per_channel % f.every:
                    raise ValueError(
                        f"Bank(every={f.every}) does not divide the "
                        f"{arch.banks_per_channel} banks of a channel"
                    )
                extents.append(arch.banks_per_channel // f.every)
            elif isinstance(f, Column):
                if space is not MemorySpace.PIM_DRAM:
                    raise ValueError(
                        f"Column() is only allowed in pim.dram layouts, not {space.value}"
                    )
                extents.append(f.extent)
            elif isinstance(f, Mac):
                if space is not MemorySpace.PIM_DRAM:
                    raise ValueError(
                        f"Mac() is only allowed in pim.dram layouts, not {space.value}"
                    )
                if f.extent is not None and f.extent != arch.mac_array_width:
                    raise ValueError(
                        f"Mac extent {f.extent} does not match the device's MAC "
                        f"array width ({arch.mac_array_width})"
                    )
                extents.append(arch.mac_array_width)
            elif isinstance(f, Slot):
                if space is not MemorySpace.PNM_SB:
                    raise ValueError(
                        f"Slot() is only allowed in pnm.sb layouts, not {space.value}"
                    )
                if f.of is not None:
                    extents.append(channel_extents[(dim_idx, pos)])
                else:
                    extents.append(f.extent)
            elif isinstance(f, Lane):
                if space is not MemorySpace.PNM_SB:
                    raise ValueError(
                        f"Lane() is only allowed in pnm.sb layouts, not {space.value}"
                    )
                lane_extent = (
                    arch.banks_per_channel if f.of is not None else burst_values
                )
                if lane_extent != burst_values:
                    raise ValueError(
                        f"Lane extent ({lane_extent}) must equal values per burst "
                        f"({burst_values}) for {tensor.dtype.value}"
                    )
                extents.append(lane_extent)
            else:
                raise TypeError(f"unknown layout factor: {f!r}")

        missing = [i for i, e in enumerate(extents) if e is None]
        if len(missing) > 1:
            raise ValueError(
                f"{dim_label} has {len(missing)} factors with inferred extent (None); "
                "at most one is allowed per dimension"
            )
        known_product = prod(e for e in extents if e is not None)
        if missing:
            hole = missing[0]
            if dim_size % known_product != 0 or dim_size // known_product < 1:
                raise ValueError(
                    f"{dim_label} size {dim_size} is not divisible by factor product "
                    f"{known_product}"
                )
            extents[hole] = dim_size // known_product

        concrete_extents = [int(e) for e in extents if e is not None]
        total = prod(concrete_extents)
        # A partial last part: the outermost factor is an index factor (the
        # last tile is partial) or a Channel factor (the last channels hold
        # fewer elements, or none), and only its last value is partial.
        inner = total // concrete_extents[0]
        ragged = (
            total > dim_size
            and isinstance(dim_factors[0], (Serial, Row, ColumnBlock, Channel))
            and (concrete_extents[0] - 1) * inner < dim_size
        ) or (total > dim_size and isinstance(dim_factors[0], Channel) and dim_size > 0)
        if total != dim_size and not ragged:
            raise ValueError(
                f"{dim_label} factor extents {concrete_extents} multiply to "
                f"{total}, expected dimension size {dim_size} (or, with an "
                "index or Channel factor outermost, more by less than one part)"
            )

        # Compute mixed-radix strides from innermost (stride = 1) to outermost.
        strides = [1] * len(concrete_extents)
        running = 1
        for i in range(len(concrete_extents) - 1, -1, -1):
            strides[i] = running
            running *= concrete_extents[i]

        for f, ext, st in zip(dim_factors, concrete_extents, strides, strict=True):
            resolved_all.append(
                ResolvedFactor(factor=f, dim=dim_idx, extent=ext, stride=st)
            )

    if space is MemorySpace.PIM_DRAM:
        channels = [rf for rf in resolved_all if isinstance(rf.factor, Channel)]
        banks = [rf for rf in resolved_all if isinstance(rf.factor, Bank)]
        columns = [rf for rf in resolved_all if isinstance(rf.factor, Column)]
        macs = [rf for rf in resolved_all if isinstance(rf.factor, Mac)]
        if not channels or len(banks) != 1 or len(columns) != 1:
            raise ValueError(
                "pim.dram layout must contain at least one Channel(), and Bank() "
                f"and Column() exactly once each (got {len(channels)} Channel, "
                f"{len(banks)} Bank, {len(columns)} Column)"
            )
        if len(macs) > 1:
            raise ValueError(
                f"pim.dram layout may contain Mac() at most once, got {len(macs)}"
            )
        bk_rf, col_rf = banks[0], columns[0]
        if any(
            ch_rf.dim == bk_rf.dim and ch_rf.stride <= bk_rf.stride
            for ch_rf in channels
        ):
            raise ValueError(
                "within one dimension, Channel() must be outer to Bank() so "
                "consecutive elements fill a channel's banks before moving to "
                "the next channel"
            )
        if col_rf.stride != 1:
            raise ValueError(
                "Column() must be the innermost factor of its dimension (stride 1)"
            )
        blocks = prod(
            rf.extent for rf in resolved_all if isinstance(rf.factor, ColumnBlock)
        )
        in_row = blocks * col_rf.extent * (macs[0].extent if macs else 1)
        if in_row > row_values:
            what = (
                f"Mac x Column extent {macs[0].extent} x {col_rf.extent}"
                if macs
                else f"Column extent {col_rf.extent}"
            )
            if blocks > 1:
                what = f"{blocks} column blocks of {what}"
            raise ValueError(
                f"{what} exceeds one DRAM row ({row_values} "
                f"{tensor.dtype.value} values)"
            )
    else:
        slots = [rf for rf in resolved_all if isinstance(rf.factor, Slot)]
        lanes = [rf for rf in resolved_all if isinstance(rf.factor, Lane)]
        if not slots:
            raise ValueError("pnm.sb layout must contain at least one Slot() factor")
        if len(lanes) != 1:
            raise ValueError(
                f"pnm.sb layout must contain Lane() exactly once, got {len(lanes)}"
            )
        if resolved_all[-1] is not lanes[0]:
            raise ValueError(
                "Lane() must be the innermost factor of the last dimension"
            )

    return PhysicalLayout(
        space=space,
        factors=tuple(resolved_all),
        shape=tensor.shape,
        burst_values=burst_values,
        spec=spec,
    )


def allocate(
    layouts: Mapping[Value, PhysicalLayout],
    arch: CentArchitecture,
    fixed: frozenset[Value] = frozenset(),
) -> dict[Value, PhysicalLayout]:
    """Assign non-overlapping ``base`` rows (DRAM) and slots (Shared Buffer).

    Tensors are placed one after another in the order of ``layouts``.

    Shared Buffer layouts containing ``Slot(of=Channel())`` are placed so
    that every ``RD_MAC`` / ``WR_BIAS`` base slot ``Rd``/``Rs`` is a multiple
    of ``arch.channels`` (``PhysicalLayout.channel_base``). Layouts whose base
    is already set (``fixed``) keep it; the others are placed after them.

    Raises:
        ValueError: If total DRAM rows exceed ``arch.rows_per_bank`` or total
            Shared Buffer slots exceed ``arch.shared_buffer_size // arch.burst_size``.
    """

    max_rows = arch.rows_per_bank
    max_slots = arch.shared_buffer_size // arch.burst_size

    # Place the others after every fixed layout of the same space.
    def end(space: MemorySpace) -> int:
        return max(
            (
                layouts[v].base + layouts[v].footprint()
                for v in fixed
                if layouts[v].space is space
            ),
            default=0,
        )

    next_row = end(MemorySpace.PIM_DRAM)
    next_slot = end(MemorySpace.PNM_SB)
    allocated: dict[Value, PhysicalLayout] = {}

    for value, layout in layouts.items():
        if value in fixed:
            allocated[value] = layout
        elif layout.space is MemorySpace.PIM_DRAM:
            base = next_row
            next_row += layout.footprint()
            if next_row > max_rows:
                raise ValueError(
                    f"DRAM allocation requires {next_row} rows per bank, "
                    f"exceeding rows_per_bank ({max_rows})"
                )
            allocated[value] = replace(layout, base=base)
        elif layout.space is MemorySpace.PNM_SB:
            if layout.needs_channel_alignment:
                # The first slot is f + Rd, with Rd a multiple of the channels.
                c, f = arch.channels, layout.first_channel
                next_slot = -(-max(next_slot - f, 0) // c) * c + f
            base = next_slot
            next_slot += layout.footprint()
            if next_slot > max_slots:
                raise ValueError(
                    f"Shared Buffer allocation requires {next_slot} slots, "
                    f"exceeding capacity ({max_slots})"
                )
            allocated[value] = replace(layout, base=base)
        else:
            raise ValueError(f"cannot allocate layout in space {layout.space.value}")

    return allocated
