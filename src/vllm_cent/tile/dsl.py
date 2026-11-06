"""Tile DSL: write CENT Tile IR kernels as nested Python ``with`` blocks.

The DSL is a builder, not an interpreter: each call records one IR op, and
the Python body of a loop runs once, while the loop is recorded. It adds no
semantics of its own; every construct is one Tile IR structure:

- ``k.tensor(...)`` is a function argument in DRAM or the Shared Buffer, and
  ``k.bind_layout(t, split(...))`` gives it a ``PhysicalLayout``.
- ``with k.serial(name, extent, carry=...) as loop`` is one ``tile.for``:
  ``loop.index`` and ``loop.carried`` are its block arguments,
  ``loop.yield_(...)`` sets the block's results, and ``loop.results`` (after
  the ``with`` block) are the op's results.
- ``k.wr_bias``, ``k.wr_bias_zero``, ``k.wr_gb``, ``k.mac_abk`` and
  ``k.rd_mac`` each record one ``cent.*`` op; ``channels=`` is its channel
  mask.
- ``t.tile(n_wave=n)`` names one tile of ``t`` by the index factors of its
  layout. The DSL derives the shapes the ops need from the layout and the
  device, so a kernel never writes them.

``k.build()`` assembles the function, allocates the layouts in argument
order and runs ``verify_kernel``, so a kernel that builds is legal on its
device. See ``examples/Tile_DSL`` for whole kernels.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from math import prod
from types import TracebackType

from ..device.cent import CentArchitecture
from ..device.spec import DeviceSpec
from ..ir.base import Block, Function, Operation, Value
from ..ir.tile import (
    INDEX,
    CentCopyBkGbOp,
    CentCopyGbBkOp,
    CentEwaddOp,
    CentEwmulOp,
    CentGbConcatOp,
    CentMacAbkOp,
    CentMacBkBkOp,
    CentPnmOp,
    CentRdAfOp,
    CentRdMacOp,
    CentRdMemOp,
    CentSyncOp,
    CentTensorType,
    CentWrBiasOp,
    CentWrBiasZeroOp,
    CentWrGbOp,
    CentWrMemOp,
    TileConstOp,
    TileForOp,
)
from ..ir.types import DType, MemorySpace, TensorType
from .kernel import CentKernel
from .layout import LayoutSpec, PhysicalLayout, allocate, resolve
from .verify import verify_kernel

__all__ = ["Loop", "TileKernel", "TileRef", "TileValue"]


@dataclass(frozen=True, eq=False)
class TileValue:
    """An SSA value of a kernel being built, as the DSL hands it out.

    Attributes:
        value: The IR value.
        kernel: The kernel that defines it.
        root: For a DRAM or Shared Buffer value, the kernel argument whose
            storage holds it (a loop-carried or rewritten version of a
            tensor keeps its tensor's root); None otherwise.
        hint: The name its later versions print with.
    """

    value: Value
    kernel: TileKernel
    root: Value | None = None
    hint: str | None = None

    @property
    def type(self) -> TensorType:
        """The IR value's type."""
        return self.value.type

    def tile(self, **index: TileValue | int) -> TileRef:
        """Name the tile selected by one index per index factor.

        Args:
            **index: For each index factor of the tensor's layout (``Serial``
                or ``Row``), the loop index that selects it, or a constant
                (recorded as ``tile.const``) where the schedule is unrolled.

        Returns:
            The tile, to pass to a ``cent.*`` op.

        Raises:
            ValueError: If the value is not a laid-out tensor, or the names do
                not match the layout's index factors.
        """

        return self.kernel._tile_ref(self, index)


@dataclass(frozen=True, eq=False)
class TileRef:
    """One tile of a tensor: the tensor, and the index for each index factor."""

    tensor: TileValue
    at: tuple[str, ...]
    indices: tuple[Value, ...]


class Loop:
    """A ``tile.for`` being recorded; returned by ``TileKernel.serial``."""

    def __init__(
        self, kernel: TileKernel, name: str, extent: int, carry: Sequence[TileValue]
    ) -> None:
        """Create the loop's index and carried values; record nothing yet."""

        self._kernel = kernel
        self.name = name
        self.extent = extent
        self._initial = tuple(kernel._own(v) for v in carry)
        self.index = TileValue(Value(type=INDEX, name=name), kernel, hint=name)
        self.carried = tuple(
            TileValue(Value(type=v.type, name=v.hint), kernel, v.root, v.hint)
            for v in self._initial
        )
        self._operations: list[Operation] = []
        self._yields: tuple[Value, ...] | None = None
        self._results: tuple[TileValue, ...] | None = None

    def yield_(self, *values: TileValue) -> None:
        """Set what one iteration passes to the next: one value per carried.

        Raises:
            ValueError: If called twice, or with the wrong number of values.
        """

        if self._yields is not None:
            raise ValueError(f"loop {self.name!r} already yielded")
        if len(values) != len(self.carried):
            raise ValueError(
                f"loop {self.name!r} carries {len(self.carried)} values, "
                f"but yield_ got {len(values)}"
            )
        self._yields = tuple(self._kernel._own(v).value for v in values)

    @property
    def results(self) -> tuple[TileValue, ...]:
        """The values of the last iteration, available after the ``with`` block.

        Raises:
            RuntimeError: If the loop has not been recorded yet.
        """

        if self._results is None:
            raise RuntimeError(
                f"the results of loop {self.name!r} exist only after its with block"
            )
        return self._results

    def __enter__(self) -> Loop:
        """Start recording the loop body."""
        self._kernel._loops.append(self)
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Record the ``tile.for`` in the enclosing block.

        Raises:
            ValueError: If the loop carries values but never yielded.
        """

        popped = self._kernel._loops.pop()
        assert popped is self
        if exc_type is not None:
            return
        if self._yields is None:
            if self.carried:
                raise ValueError(
                    f"loop {self.name!r} carries {len(self.carried)} values "
                    "but never called yield_"
                )
            self._yields = ()
        op = TileForOp(
            operands=tuple(v.value for v in self._initial),
            extent=self.extent,
            body=Block(
                (self.index.value, *(v.value for v in self.carried)),
                tuple(self._operations),
                self._yields,
            ),
            # Results print as the tensor they carry (%y), not as the index.
            name_hint=self.carried[0].hint if self.carried else self.name,
        )
        self._kernel._record(op)
        self._results = tuple(
            TileValue(result, self._kernel, v.root, v.hint)
            for result, v in zip(op.results, self.carried, strict=True)
        )


class TileKernel:
    """Builder of one CENT Tile IR kernel for one device.

    Use it as a context manager, declare tensors and their layouts, record
    loops and ops, set the results with ``returns``, then ``build``.
    """

    def __init__(self, name: str, device: DeviceSpec | CentArchitecture) -> None:
        """Start an empty kernel.

        Args:
            name: The kernel function's name.
            device: The device spec, or its CENT architecture directly.

        Raises:
            TypeError: If the device is not a CENT device.
        """

        arch = device.architecture if isinstance(device, DeviceSpec) else device
        if not isinstance(arch, CentArchitecture):
            raise TypeError(
                f"Tile DSL kernels need a CENT architecture, got {type(arch).__name__}"
            )
        self.name = name
        self.arch = arch
        self._arguments: list[Value] = []
        self._layouts: dict[Value, PhysicalLayout] = {}
        self._fixed: set[Value] = set()
        self._top: list[Operation] = []
        self._loops: list[Loop] = []
        self._results: tuple[Value, ...] | None = None
        self._kernel: CentKernel | None = None

    def __enter__(self) -> TileKernel:
        """Return the builder."""
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Nothing to close: loops close their own ``with`` blocks."""

    # ------------------------------------------------------------------
    # Tensors and layouts
    # ------------------------------------------------------------------

    def tensor(
        self,
        name: str,
        shape: tuple[int, ...],
        dtype: DType | str,
        space: MemorySpace,
    ) -> TileValue:
        """Declare a kernel argument in DRAM or the Shared Buffer.

        Raises:
            ValueError: If the name is taken, or the space is not ``pim.dram``
                or ``pnm.sb`` (Global Buffer and MAC values are made by ops).
        """

        if space not in (MemorySpace.PIM_DRAM, MemorySpace.PNM_SB):
            raise ValueError(
                f"kernel tensor {name!r} must be in pim.dram or pnm.sb, got "
                f"{space.value}: Global Buffer and MAC values come from ops"
            )
        if any(argument.name == name for argument in self._arguments):
            raise ValueError(f"kernel already has a tensor named {name!r}")
        value = Value(
            type=CentTensorType(
                shape=tuple(shape), dtype=DType.normalize(dtype), memory_space=space
            ),
            name=name,
        )
        self._arguments.append(value)
        return TileValue(value, self, root=value, hint=name)

    def bind_layout(
        self, tensor: TileValue, spec: LayoutSpec, *, base: int | None = None
    ) -> PhysicalLayout:
        """Resolve ``spec`` for a kernel tensor on this device.

        Args:
            tensor: A tensor made by ``k.tensor()``.
            spec: Its layout.
            base: Its first DRAM row or Shared Buffer slot, where a mapping
                fixes it (e.g. a weight placed with the whole model);
                otherwise ``build`` places it after the fixed ones.

        Returns:
            The resolved layout; ``build`` sets its base unless given here.

        Raises:
            ValueError: If the value is not a kernel tensor, already has a
                layout, or ``resolve`` rejects the spec.
        """

        value = self._own(tensor).value
        if value not in self._arguments:
            raise ValueError("only tensors made by k.tensor() take a layout")
        if value in self._layouts:
            raise ValueError(f"tensor {value.name!r} already has a layout")
        layout = resolve(spec, value.type, self.arch)
        if base is not None:
            layout = replace(layout, base=base)
            self._fixed.add(value)
        self._layouts[value] = layout
        return layout

    def layout_of(self, tensor: TileValue) -> PhysicalLayout:
        """The layout of the kernel tensor that holds ``tensor``.

        Before ``build``, its ``base`` is the one ``bind_layout`` fixed, or 0
        if ``build`` is to place it; the built kernel's layouts have them all.

        Raises:
            ValueError: If it is no DRAM or Shared Buffer tensor, or has no
                layout yet.
        """
        return self._layout_of(self._own(tensor))

    def all_channels(self) -> tuple[int, ...]:
        """The channel mask that selects every channel of the device."""
        return tuple(range(self.arch.channels))

    # ------------------------------------------------------------------
    # Loops and ops
    # ------------------------------------------------------------------

    def serial(self, name: str, extent: int, carry: Sequence[TileValue] = ()) -> Loop:
        """A loop of ``extent`` iterations; use it as ``with ... as loop``."""
        return Loop(self, name, extent, carry)

    def wr_bias(
        self,
        source: TileValue | TileRef,
        *,
        stride: int,
        channels: Sequence[int],
        name: str | None = None,
    ) -> TileValue:
        """Load the MAC registers of ``channels`` from the Shared Buffer.

        ``stride=0``: every channel reads the one slot of ``source`` (a whole
        one-slot tensor, e.g. zeros). ``stride=1``: channel ``c`` reads slot
        ``Rs + c`` of the tile ``source`` names.
        """

        ref = self._ref(source)
        channels = tuple(channels)
        op = CentWrBiasOp(
            operands=(ref.tensor.value, *ref.indices),
            channels=channels,
            at=ref.at,
            stride=stride,
            shape=(len(channels), self.arch.banks_per_channel),
            name_hint=name or "acc",
        )
        return self._emit(op, hint=name or "acc")

    def wr_bias_zero(
        self,
        *,
        channels: Sequence[int],
        dtype: DType | str | None = None,
        name: str | None = None,
    ) -> TileValue:
        """Clear the MAC registers of ``channels``: a sum starts at zero.

        Reads no Shared Buffer slot, so no tensor of zeros is needed.

        Args:
            channels: The channel mask.
            dtype: The registers' type; by default the one dtype of the
                kernel's tensors.
            name: The name the result prints with.

        Raises:
            ValueError: If ``dtype`` is omitted and the kernel's tensors do
                not share one dtype.
        """

        if dtype is None:
            dtypes = {argument.type.dtype for argument in self._arguments}
            if len(dtypes) != 1:
                raise ValueError(
                    "wr_bias_zero needs dtype= when the kernel's tensors do not "
                    f"share one dtype, got {sorted(dtypes)}"
                )
            dtype = dtypes.pop()
        channels = tuple(channels)
        op = CentWrBiasZeroOp(
            operands=(),
            channels=channels,
            shape=(len(channels), self.arch.banks_per_channel),
            dtype=DType.normalize(dtype),
            name_hint=name or "acc",
        )
        return self._emit(op, hint=name or "acc")

    def wr_gb(
        self,
        source: TileValue | TileRef,
        *,
        channels: Sequence[int],
        name: str | None = None,
    ) -> TileValue:
        """Broadcast a Shared Buffer tile into the Global Buffer of ``channels``."""

        ref = self._ref(source)
        layout = self._layout_of(ref.tensor)
        op = CentWrGbOp(
            operands=(ref.tensor.value, *ref.indices),
            channels=tuple(channels),
            at=ref.at,
            shape=(prod(layout.tile_shape),),
            name_hint=name or "gb",
        )
        return self._emit(op, hint=name or "gb")

    def mac_abk(
        self,
        gb: TileValue,
        weight: TileValue | TileRef,
        acc: TileValue,
        *,
        channels: Sequence[int],
        name: str | None = None,
    ) -> TileValue:
        """``acc += x @ w.T`` for one tile: the Global Buffer (``x``) times ``w``.

        The arguments follow ``x @ w.T``: the input in the Global Buffer,
        then the weight tile, then the MAC registers added to.
        """

        ref = self._ref(weight)
        hint = name or self._own(acc).hint or "acc"
        op = CentMacAbkOp(
            operands=(
                self._own(gb).value,
                ref.tensor.value,
                self._own(acc).value,
                *ref.indices,
            ),
            channels=tuple(channels),
            at=ref.at,
            name_hint=hint,
        )
        return self._emit(op, hint=hint)

    def rd_mac(
        self,
        acc: TileValue,
        *,
        into: TileValue | TileRef,
        channels: Sequence[int],
        name: str | None = None,
    ) -> TileValue:
        """Save the MAC registers of ``channels`` into a Shared Buffer tile.

        Returns:
            The new version of the tensor ``into`` names.
        """

        ref = self._ref(into)
        hint = name or ref.tensor.hint
        op = CentRdMacOp(
            operands=(self._own(acc).value, ref.tensor.value, *ref.indices),
            channels=tuple(channels),
            at=ref.at,
            name_hint=hint,
        )
        return self._emit(op, root=ref.tensor.root, hint=hint)

    def rd_af(
        self,
        acc: TileValue,
        *,
        into: TileValue | TileRef,
        channels: Sequence[int],
        name: str | None = None,
    ) -> TileValue:
        """Activate the MAC registers and save the results (``AF``, ``RD_AF``).

        Returns:
            The new version of the tensor ``into`` names.
        """

        ref = self._ref(into)
        hint = name or ref.tensor.hint
        op = CentRdAfOp(
            operands=(self._own(acc).value, ref.tensor.value, *ref.indices),
            channels=tuple(channels),
            at=ref.at,
            name_hint=hint,
        )
        return self._emit(op, root=ref.tensor.root, hint=hint)

    def wr_mem(
        self,
        src: TileValue,
        *,
        into: TileValue,
        start: Sequence[int] | None = None,
        name: str | None = None,
    ) -> TileValue:
        """Write a Shared Buffer tensor into DRAM with ordinary writes (``W MEM``).

        Args:
            src: The Shared Buffer tensor written.
            into: The DRAM tensor written to.
            start: Where ``src`` starts in ``into``, aligned to its last
                dimensions; the origin by default.
            name: The name the result prints with.

        Returns:
            The new version of ``into``.
        """

        dst = self._own(into)
        op = CentWrMemOp(
            operands=(self._own(src).value, dst.value),
            start=tuple(start) if start is not None else (0,) * dst.type.rank,
            name_hint=name or dst.hint,
        )
        return self._emit(op, root=dst.root, hint=name or dst.hint)

    def rd_mem(
        self,
        src: TileValue,
        *,
        into: TileValue,
        start: Sequence[int] | None = None,
        name: str | None = None,
    ) -> TileValue:
        """Read DRAM into a Shared Buffer tensor with ordinary reads (``R MEM``).

        Returns:
            The new version of ``into``.
        """

        dst = self._own(into)
        source = self._own(src)
        op = CentRdMemOp(
            operands=(source.value, dst.value),
            start=tuple(start) if start is not None else (0,) * source.type.rank,
            name_hint=name or dst.hint,
        )
        return self._emit(op, root=dst.root, hint=name or dst.hint)

    def mac_bk_bk(
        self,
        a: TileValue,
        b: TileValue,
        acc: TileValue,
        *,
        channels: Sequence[int],
        name: str | None = None,
    ) -> TileValue:
        """Multiply-accumulate banks ``2b`` (``a``) and ``2b + 1`` (``b``)."""

        hint = name or self._own(acc).hint or "acc"
        op = CentMacBkBkOp(
            operands=(self._own(a).value, self._own(b).value, self._own(acc).value),
            channels=tuple(channels),
            name_hint=hint,
        )
        return self._emit(op, hint=hint)

    def ewmul(
        self,
        a: TileValue,
        b: TileValue,
        *,
        into: TileValue,
        channels: Sequence[int],
        name: str | None = None,
    ) -> TileValue:
        """Multiply banks ``4b`` (``a``) and ``4b + 1`` (``b``) into ``4b + 2``.

        Returns:
            The new version of ``into``.
        """

        out = self._own(into)
        op = CentEwmulOp(
            operands=(self._own(a).value, self._own(b).value, out.value),
            channels=tuple(channels),
            name_hint=name or out.hint,
        )
        return self._emit(op, root=out.root, hint=name or out.hint)

    def copy_bkgb(
        self,
        src: TileValue,
        *,
        bank: int,
        channels: Sequence[int],
        name: str | None = None,
    ) -> TileValue:
        """Copy bank ``bank`` of ``src``'s row into the Global Buffers."""

        source = self._own(src)
        layout = self._layout_of(source)
        values = prod(rf.extent for rf in layout.column_factors)
        op = CentCopyBkGbOp(
            operands=(source.value,),
            channels=tuple(channels),
            bank=bank,
            shape=(values,),
            name_hint=name or "gb",
        )
        return self._emit(op, hint=name or "gb")

    def copy_gbbk(
        self,
        gb: TileValue,
        *,
        into: TileValue,
        bank: int,
        channels: Sequence[int],
        name: str | None = None,
    ) -> TileValue:
        """Copy the Global Buffers into bank ``bank`` of ``into``'s row.

        Returns:
            The new version of ``into``.
        """

        out = self._own(into)
        op = CentCopyGbBkOp(
            operands=(self._own(gb).value, out.value),
            channels=tuple(channels),
            bank=bank,
            name_hint=name or out.hint,
        )
        return self._emit(op, root=out.root, hint=name or out.hint)

    def sync(self) -> None:
        """Wait for every issued command (``SYNC``)."""
        self._record(CentSyncOp(operands=()))

    def ewadd(
        self, a: TileValue, b: TileValue, *, name: str | None = None
    ) -> TileValue:
        """Add ``b`` into ``a`` in the Shared Buffer (``EWADD``).

        Returns:
            The new version of ``a``.
        """

        first = self._own(a)
        op = CentEwaddOp(
            operands=(first.value, self._own(b).value), name_hint=name or first.hint
        )
        return self._emit(op, root=first.root, hint=name or first.hint)

    def gb_concat(self, *gbs: TileValue, name: str | None = None) -> TileValue:
        """Global Buffer tiles of disjoint channels, as one operand of a MAC."""

        op = CentGbConcatOp(
            operands=tuple(self._own(gb).value for gb in gbs), name_hint=name or "gb"
        )
        return self._emit(op, hint=name or "gb")

    def pnm(
        self,
        kind: str,
        *inputs: TileValue,
        into: TileValue,
        cycles: float,
        name: str | None = None,
    ) -> TileValue:
        """PNM work CENT models analytically: a trace comment with ``cycles``.

        Returns:
            The new version of ``into``.
        """

        out = self._own(into)
        op = CentPnmOp(
            operands=(*(self._own(v).value for v in inputs), out.value),
            kind=kind,
            cycles=cycles,
            name_hint=name or out.hint,
        )
        return self._emit(op, root=out.root, hint=name or out.hint)

    def returns(self, *values: TileValue) -> None:
        """Set the kernel's results.

        Raises:
            ValueError: If called inside a loop or twice.
        """

        if self._loops:
            raise ValueError("returns() must be called outside every loop")
        if self._results is not None:
            raise ValueError("the kernel's results are already set")
        self._results = tuple(self._own(v).value for v in values)

    def build(self) -> CentKernel:
        """Assemble, allocate and verify the kernel; later calls return it again.

        Raises:
            ValueError: If a loop is still open, or ``verify_kernel`` rejects
                the kernel.
        """

        if self._loops:
            raise ValueError(f"loop {self._loops[-1].name!r} is still open")
        if self._kernel is None:
            function = Function(
                self.name,
                tuple(self._arguments),
                tuple(self._top),
                self._results or (),
            )
            # Allocate in argument order, so bases follow the declarations.
            ordered = {
                a: self._layouts[a] for a in self._arguments if a in self._layouts
            }
            kernel = CentKernel(
                function, allocate(ordered, self.arch, frozenset(self._fixed))
            )
            verify_kernel(kernel, self.arch)
            self._kernel = kernel
        return self._kernel

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _own(self, value: TileValue) -> TileValue:
        """Check that ``value`` belongs to this kernel."""

        if not isinstance(value, TileValue) or value.kernel is not self:
            raise ValueError(f"{value!r} is not a value of kernel {self.name!r}")
        return value

    def _layout_of(self, tensor: TileValue) -> PhysicalLayout:
        """The layout of the kernel tensor that holds ``tensor``."""

        if tensor.root is None:
            raise ValueError(
                "only DRAM and Shared Buffer tensors have tiles, got "
                f"{tensor.hint or 'a value'} in {tensor.type}"
            )
        layout = self._layouts.get(tensor.root)
        if layout is None:
            raise ValueError(
                f"bind a layout to tensor {tensor.root.name!r} before using its tiles"
            )
        return layout

    def _tile_ref(
        self, tensor: TileValue, index: dict[str, TileValue | int]
    ) -> TileRef:
        """Build the ``TileRef`` for ``tensor.tile(**index)``."""

        layout = self._layout_of(self._own(tensor))
        names = tuple(index)
        if set(names) != set(layout.index_names):
            raise ValueError(
                f"a tile of {tensor.hint!r} needs one index for each of "
                f"{list(layout.index_names)}, got {list(names)}"
            )
        indices = tuple(
            self._const(v) if isinstance(v, int) else self._own(v).value
            for v in index.values()
        )
        return TileRef(tensor, names, indices)

    def _const(self, value: int) -> Value:
        """Record ``tile.const`` in the innermost open block; return its index."""

        op = TileConstOp(operands=(), value=value, name_hint=f"c{value}")
        self._record(op)
        return op.results[0]

    def _ref(self, source: TileValue | TileRef) -> TileRef:
        """A whole tensor is the tile with no indices."""

        if isinstance(source, TileRef):
            return source
        return TileRef(self._own(source), (), ())

    def _record(self, op: Operation) -> None:
        """Append ``op`` to the innermost open block."""

        target = self._loops[-1]._operations if self._loops else self._top
        target.append(op)

    def _emit(
        self, op: Operation, root: Value | None = None, hint: str | None = None
    ) -> TileValue:
        """Record a one-result ``cent.*`` op and wrap its result."""

        self._record(op)
        return TileValue(op.results[0], self, root, hint)
