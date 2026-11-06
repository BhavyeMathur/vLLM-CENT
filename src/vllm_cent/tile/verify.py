"""Static legality verifier for CENT Tile IR kernels."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from math import prod

from ..device.cent import CentArchitecture
from ..ir.base import Block, Function, Operation, Value
from ..ir.printer import format_type, value_names
from ..ir.tile import (
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
    CentTileOp,
    CentWrBiasOp,
    CentWrBiasZeroOp,
    CentWrGbOp,
    CentWrMemOp,
    TileConstOp,
    TileForOp,
)
from ..ir.types import MemorySpace
from .kernel import CentKernel
from .layout import Bank, Column, Lane, PhysicalLayout, Slot, resolve

__all__ = ["verify_kernel"]


def _op_label(op: Operation) -> str:
    return op.NAME if op.name_hint is None else f"{op.NAME} ({op.name_hint})"


def verify_kernel(kernel: CentKernel, arch: CentArchitecture) -> None:
    """Check that ``kernel`` is well formed and legal on ``arch``.

    Verification runs in six stages:
    1. SSA scoping and ``tile.for`` induction types (``Function.verify``).
    2. Parameter memory spaces, layout completeness, alignment, and non-overlapping
       footprints within ``arch`` capacity (``_check_layouts``).
    3. Storage root aliasing, channel ownership, and loop index extents
       (``_analyze``).
    4. Per-operation layout, index bound, tile shape, and channel compatibility
       checks (``_check_ops``).
    5. Per-channel liveness limits for Global Buffers and MAC accumulators
       (``_check_liveness``).
    6. Linear SSA version checks on Shared Buffer tensors (``_check_sb_versions``).

    Raises:
        ValueError: At the first legality rule broken.
    """

    kernel.function.verify()
    names = value_names(kernel.function)
    _check_layouts(kernel, arch, names)
    facts = _analyze(kernel.function, names)
    _check_ops(kernel, facts, arch, names)
    _check_liveness(kernel.function, facts, arch, names)
    _check_sb_versions(kernel.function, names)


def _check_layouts(
    kernel: CentKernel, arch: CentArchitecture, names: dict[Value, str]
) -> None:
    func = kernel.function
    arg_set = set(func.arguments)

    for key in kernel.layouts:
        if key not in arg_set:
            label = names.get(key, key.name or "<unnamed>")
            raise ValueError(f"layout given for non-argument value %{label}")

    dram_intervals: list[tuple[int, int, frozenset[tuple[int, int]], str]] = []
    sb_intervals: list[tuple[int, int, str]] = []
    max_rows = arch.rows_per_bank
    max_slots = arch.shared_buffer_size // arch.burst_size

    for arg in func.arguments:
        name = f"%{names[arg]}"
        arg_type = arg.type
        if not isinstance(arg_type, CentTensorType) or arg_type.memory_space not in (
            MemorySpace.PIM_DRAM,
            MemorySpace.PNM_SB,
        ):
            raise ValueError(
                f"kernel argument {name} must be in pim.dram or pnm.sb, "
                f"got {format_type(arg_type)}"
            )
        if arg not in kernel.layouts:
            raise ValueError(f"kernel argument {name} is missing a PhysicalLayout")

        layout = kernel.layouts[arg]
        if layout.space is not arg_type.memory_space:
            raise ValueError(
                f"kernel argument {name} is in {arg_type.memory_space.value}, "
                f"but its layout is in {layout.space.value}"
            )
        # The layout must be what its spec gives for this argument's type on
        # this device: a layout resolved for another device or dtype would
        # put data on channels, banks or lanes that do not exist here.
        try:
            expected = resolve(layout.spec, arg_type, arch)
        except ValueError as error:
            raise ValueError(
                f"kernel argument {name}: its layout does not fit this device "
                f"and type: {error}"
            ) from error
        if replace(layout, base=0) != expected:
            raise ValueError(
                f"kernel argument {name}: its layout was resolved for another "
                "device or type"
            )
        if layout.base < 0:
            raise ValueError(
                f"kernel argument {name} layout base must be >= 0, got {layout.base}"
            )

        start = layout.base
        end = start + layout.footprint()
        if layout.space is MemorySpace.PIM_DRAM:
            if end > max_rows:
                raise ValueError(
                    f"kernel argument {name} occupies DRAM rows [{start}, {end}), "
                    f"exceeding rows_per_bank ({max_rows})"
                )
            # Two tensors may share rows on different banks, e.g. the
            # operands of EWMUL in banks 4b and 4b + 1 of one row.
            cells = layout.occupied_cells()
            for other_start, other_end, other_cells, other_name in dram_intervals:
                if (
                    max(start, other_start) < min(end, other_end)
                    and cells & other_cells
                ):
                    raise ValueError(
                        f"DRAM rows of {name} [{start}, {end}) overlap with "
                        f"{other_name} [{other_start}, {other_end}) in the same banks"
                    )
            dram_intervals.append((start, end, cells, name))
        else:
            if layout.needs_channel_alignment:
                # Every tile's Rd = first slot - first channel must be a
                # multiple of the channel count.
                rd = layout.channel_base(start)
                if (
                    rd < 0
                    or rd % arch.channels != 0
                    or (
                        layout.tile_count() > 1
                        and layout.slots_per_tile % arch.channels
                    )
                ):
                    raise ValueError(
                        f"kernel argument {name} layout requires base aligned to "
                        f"{arch.channels} channels, got base {start} (Rd {rd})"
                    )
            if end > max_slots:
                raise ValueError(
                    f"kernel argument {name} occupies Shared Buffer slots "
                    f"[{start}, {end}), exceeding capacity ({max_slots})"
                )
            for other_start, other_end, other_name in sb_intervals:
                if max(start, other_start) < min(end, other_end):
                    raise ValueError(
                        f"Shared Buffer slots of {name} [{start}, {end}) overlap "
                        f"with {other_name} [{other_start}, {other_end})"
                    )
            sb_intervals.append((start, end, name))

    for res in func.results:
        res_type = res.type
        if not isinstance(res_type, CentTensorType) or res_type.memory_space not in (
            MemorySpace.PIM_DRAM,
            MemorySpace.PNM_SB,
        ):
            raise ValueError(
                f"kernel return value %{names[res]} must be in pim.dram or pnm.sb, "
                f"got {format_type(res_type)}"
            )


@dataclass
class _Facts:
    root: dict[Value, Value] = field(default_factory=dict)
    channels: dict[Value, tuple[int, ...]] = field(default_factory=dict)
    extent: dict[Value, int] = field(default_factory=dict)


def _analyze(function: Function, names: dict[Value, str]) -> _Facts:
    facts = _Facts()
    for arg in function.arguments:
        facts.root[arg] = arg
    _analyze_ops(function.operations, facts, names)
    return facts


def _analyze_ops(
    operations: tuple[Operation, ...], facts: _Facts, names: dict[Value, str]
) -> None:
    for op in operations:
        label = _op_label(op)
        if isinstance(op, TileForOp):
            facts.extent[op.index] = op.extent
            for initial, carried in zip(op.operands, op.carried, strict=True):
                if initial in facts.root:
                    facts.root[carried] = facts.root[initial]
                if initial in facts.channels:
                    facts.channels[carried] = facts.channels[initial]

            _analyze_ops(op.body.operations, facts, names)

            for carried, yielded, result in zip(
                op.carried, op.body.results, op.results, strict=True
            ):
                if carried in facts.root:
                    c_root = facts.root[carried]
                    y_root = facts.root.get(yielded)
                    if y_root != c_root:
                        y_name = names[y_root] if y_root is not None else "<unknown>"
                        raise ValueError(
                            f"{label}: carried value %{names[carried]} belongs to "
                            f"storage of %{names[c_root]}, but body yields "
                            f"%{names[yielded]} from storage of %{y_name}"
                        )
                    facts.root[result] = c_root
                if carried in facts.channels:
                    c_ch = facts.channels[carried]
                    y_ch = facts.channels.get(yielded)
                    if y_ch != c_ch:
                        raise ValueError(
                            f"{label}: carried value %{names[carried]} has channels "
                            f"{list(c_ch)}, but body yields %{names[yielded]} with "
                            f"channels {list(y_ch or ())}"
                        )
                    facts.channels[result] = c_ch
        elif isinstance(op, TileConstOp):
            # A constant index c selects as if by a loop of c + 1 iterations,
            # so the loop-extent bound check also bounds constants.
            facts.extent[op.results[0]] = op.value + 1
        elif isinstance(op, (CentRdMacOp, CentRdAfOp)):
            facts.root[op.results[0]] = facts.root[op.destination]
        elif isinstance(op, (CentWrMemOp, CentRdMemOp, CentCopyGbBkOp)):
            facts.root[op.results[0]] = facts.root[op.operands[1]]
        elif isinstance(op, CentEwmulOp):
            facts.root[op.results[0]] = facts.root[op.operands[2]]
        elif isinstance(op, CentEwaddOp):
            facts.root[op.results[0]] = facts.root[op.operands[0]]
        elif isinstance(op, CentPnmOp):
            facts.root[op.results[0]] = facts.root[op.operands[-1]]
        elif isinstance(op, CentMacAbkOp):
            # The MAC registers keep the channels of the tile they update.
            facts.channels[op.results[0]] = facts.channels[op.accumulator]
        elif isinstance(
            op,
            (CentWrGbOp, CentWrBiasOp, CentWrBiasZeroOp, CentMacBkBkOp, CentCopyBkGbOp),
        ):
            facts.channels[op.results[0]] = op.channels
        elif isinstance(op, CentGbConcatOp):
            union: set[int] = set()
            for gb in op.operands:
                channels = set(facts.channels[gb])
                if union & channels:
                    raise ValueError(
                        f"{label}: Global Buffer tiles share channels "
                        f"{sorted(union & channels)}"
                    )
                union |= channels
            facts.channels[op.results[0]] = tuple(sorted(union))
        elif isinstance(op, CentSyncOp):
            pass
        else:
            raise ValueError(f"unsupported op in CENT kernel: {label}")


def _is_channel_bank_sb_tile(layout: PhysicalLayout) -> bool:
    """Whether a Shared Buffer tile is one slot per channel, one lane per bank.

    That is the shape ``RD_MAC`` writes and ``WR_BIAS`` (stride 1) reads:
    the tile's slots are only ``Slot(of=Channel(...))`` factors, whose
    offset is the channel number (``resolve`` makes them span every
    channel), and its lanes are ``Lane(of=Bank())``.
    """

    *slots, lane = layout.tile_factors
    return (
        bool(slots)
        and all(isinstance(rf.factor, Slot) and rf.factor.of for rf in slots)
        and isinstance(lane.factor, Lane)
        and lane.factor.of is not None
    )


def _check_tile_channels(
    label: str,
    op: CentTileOp,
    layout: PhysicalLayout,
    arch: CentArchitecture,
    tensor: str,
) -> None:
    """A one-slot-per-channel tile holds slots only for its own channels."""

    occupied = layout.occupied_channels(arch.channels)
    if not set(op.channels) <= set(occupied):
        raise ValueError(
            f"{label}: channels {list(op.channels)} have no slots in "
            f"%{tensor}, laid out on channels {occupied}"
        )


def _check_accumulator(
    label: str,
    op: CentMacAbkOp | CentRdMacOp | CentRdAfOp,
    facts: _Facts,
    arch: CentArchitecture,
    names: dict[Value, str],
) -> None:
    """Check that ``op`` uses MAC registers of its own channels.

    A MAC may update some channels of the registers (one MAC per operation
    size); reading them takes all of their channels.
    """

    acc_ch = facts.channels[op.accumulator]
    if isinstance(op, CentMacAbkOp):
        if not set(op.channels) <= set(acc_ch):
            raise ValueError(
                f"{label}: channels {list(op.channels)} are not all channels of "
                f"accumulator %{names[op.accumulator]} {list(acc_ch)}"
            )
    elif op.channels != acc_ch:
        raise ValueError(
            f"{label}: channels {list(op.channels)} do not match "
            f"accumulator %{names[op.accumulator]} channels {list(acc_ch)}"
        )
    expected = (len(acc_ch), arch.banks_per_channel)
    if op.accumulator.type.shape != expected:
        raise ValueError(
            f"{label}: accumulator shape {list(op.accumulator.type.shape)} "
            f"does not match {list(expected)}"
        )


def _row_layout(
    label: str, value: Value, kernel: CentKernel, facts: _Facts, names: dict[Value, str]
) -> PhysicalLayout:
    """The layout of a DRAM operand that a bank op uses whole: one row."""

    root = facts.root[value]
    layout = kernel.layouts[root]
    if layout.tile_count() != 1 or layout.index_factors:
        raise ValueError(
            f"{label}: %{names[root]} must be one DRAM row, without indices"
        )
    return layout


def _bank_member(layout: PhysicalLayout) -> tuple[int, int]:
    """``(every, offset)`` of a layout's ``Bank`` factor."""

    bank = layout.bank_factor.factor
    assert isinstance(bank, Bank)
    return bank.every, bank.offset


def _check_bank_op(
    op: CentMacBkBkOp | CentEwmulOp | CentCopyBkGbOp | CentCopyGbBkOp,
    kernel: CentKernel,
    facts: _Facts,
    arch: CentArchitecture,
    names: dict[Value, str],
) -> None:
    """Check the bank placement of a MAC between neighbors, EWMUL or copy.

    Their DRAM operands are whole rows; the instruction names only one row,
    so operands of one instruction share it, in the banks the instruction
    pairs: ``2b`` and ``2b + 1`` for the neighbor MAC, ``4b``, ``4b + 1``
    and ``4b + 2`` for EWMUL.
    """

    label = _op_label(op)
    if isinstance(op, CentMacBkBkOp):
        dram, members = op.operands[:2], [(2, 0), (2, 1)]
    elif isinstance(op, CentEwmulOp):
        group = arch.banks_per_group
        dram, members = op.operands, [(group, 0), (group, 1), (group, 2)]
    elif isinstance(op, CentCopyBkGbOp):
        dram, members = op.operands[:1], []
    else:
        dram, members = op.operands[1:], []
    layouts = [_row_layout(label, v, kernel, facts, names) for v in dram]
    if len({layout.base for layout in layouts}) != 1:
        raise ValueError(
            f"{label}: operands must share one DRAM row, got rows "
            f"{[layout.base for layout in layouts]}"
        )
    for value, layout, member in zip(dram, layouts, members, strict=False):
        if _bank_member(layout) != member:
            raise ValueError(
                f"{label}: %{names[facts.root[value]]} must be in banks "
                f"{member[0]}b + {member[1]}, got Bank(every={_bank_member(layout)[0]}, "
                f"offset={_bank_member(layout)[1]})"
            )
    for layout in layouts:
        occupied = layout.occupied_channels(arch.channels)
        if not set(op.channels) <= set(occupied):
            raise ValueError(
                f"{label}: channels {list(op.channels)} hold no part of an operand "
                f"(channels {occupied})"
            )
    if isinstance(op, (CentCopyBkGbOp, CentCopyGbBkOp)):
        every, offset = _bank_member(layouts[0])
        if op.bank >= arch.banks_per_channel or op.bank % every != offset:
            raise ValueError(
                f"{label}: bank {op.bank} holds no part of the operand "
                f"(Bank(every={every}, offset={offset}))"
            )
    if isinstance(op, CentCopyGbBkOp):
        gb_ch = facts.channels[op.operands[0]]
        if not set(op.channels) <= set(gb_ch):
            raise ValueError(
                f"{label}: channels {list(op.channels)} are not covered by Global "
                f"Buffer %{names[op.operands[0]]} channels {list(gb_ch)}"
            )
    if isinstance(op, CentMacBkBkOp) and facts.channels[op.operands[2]] != op.channels:
        raise ValueError(f"{label}: accumulator channels differ from the op's")


def _check_ops(
    kernel: CentKernel,
    facts: _Facts,
    arch: CentArchitecture,
    names: dict[Value, str],
) -> None:
    def walk(operations: tuple[Operation, ...]) -> None:
        for op in operations:
            label = _op_label(op)
            if isinstance(op, TileForOp):
                walk(op.body.operations)
                continue
            if isinstance(
                op,
                (
                    TileConstOp,
                    CentWrMemOp,
                    CentRdMemOp,
                    CentSyncOp,
                    CentEwaddOp,
                    CentPnmOp,
                ),
            ):
                # Their operand spaces and bounds are checked when built; the
                # tensors they name have layouts by _check_layouts.
                continue
            if isinstance(op, CentGbConcatOp):
                continue  # channels checked by _analyze

            assert isinstance(op, CentTileOp)
            if op.channels[-1] >= arch.channels:
                raise ValueError(
                    f"{label}: channel {op.channels[-1]} out of range for device "
                    f"with {arch.channels} channels"
                )

            if isinstance(
                op, (CentMacBkBkOp, CentEwmulOp, CentCopyBkGbOp, CentCopyGbBkOp)
            ):
                _check_bank_op(op, kernel, facts, arch, names)
                continue
            if isinstance(op, CentWrBiasZeroOp):
                # It reads no tensor: only its register tile needs checking.
                expected = (len(op.channels), arch.banks_per_channel)
                if op.shape != expected:
                    raise ValueError(
                        f"{label}: MAC register shape must be {list(expected)}, "
                        f"got {list(op.shape)}"
                    )
                continue
            if isinstance(op, (CentWrGbOp, CentWrBiasOp)):
                indexed_value = op.source
            elif isinstance(op, CentMacAbkOp):
                indexed_value = op.weight
            elif isinstance(op, (CentRdMacOp, CentRdAfOp)):
                indexed_value = op.destination
            else:
                raise ValueError(f"unsupported op in CENT kernel: {label}")

            root = facts.root[indexed_value]
            layout = kernel.layouts[root]

            if set(op.at) != set(layout.index_names):
                raise ValueError(
                    f"{label}: at={list(op.at)} does not match layout index "
                    f"factors {list(layout.index_names)} of %{names[root]}"
                )
            factor_by_name = {
                rf.name: rf for rf in layout.index_factors if rf.name is not None
            }
            for factor_name, idx_val in zip(op.at, op.indices, strict=True):
                if idx_val not in facts.extent:
                    raise ValueError(
                        f"{label}: index %{names[idx_val]} is not a loop index or a constant"
                    )
                loop_ext = facts.extent[idx_val]
                factor_ext = factor_by_name[factor_name].extent
                if loop_ext > factor_ext:
                    raise ValueError(
                        f"{label}: index %{names[idx_val]} for {factor_name!r} has "
                        f"loop extent {loop_ext}, exceeding factor extent {factor_ext}"
                    )

            if isinstance(op, CentWrGbOp):
                tile_vals = prod(layout.tile_shape)
                op_vals = prod(op.shape)
                if op_vals != tile_vals:
                    raise ValueError(
                        f"{label}: shape {list(op.shape)} ({op_vals} values) does "
                        f"not match source tile size ({tile_vals} values)"
                    )
                gb_cap = arch.global_buffer_size * 8 // op.source.type.dtype.bits
                if op_vals > gb_cap:
                    raise ValueError(
                        f"{label}: tile of {op_vals} values exceeds Global Buffer "
                        f"capacity ({gb_cap} values)"
                    )

            elif isinstance(op, CentWrBiasOp):
                if op.shape != (len(op.channels), arch.banks_per_channel):
                    raise ValueError(
                        f"{label}: MAC register shape must be "
                        f"[{len(op.channels)}, {arch.banks_per_channel}], "
                        f"got {list(op.shape)}"
                    )
                if op.stride == 0:
                    if layout.slots_per_tile != 1 or layout.tile_count() != 1:
                        raise ValueError(
                            f"{label}: stride=0 source %{names[root]} must be a "
                            f"single-slot tensor, got {layout.footprint()} slots"
                        )
                else:
                    _check_tile_channels(label, op, layout, arch, names[root])
                    if not _is_channel_bank_sb_tile(layout):
                        raise ValueError(
                            f"{label}: stride=1 source %{names[root]} tile factors "
                            "must be (Slot(of=Channel()), Lane(of=Bank()))"
                        )

            elif isinstance(op, CentMacAbkOp):
                # The layout can describe a wider MAC array, but this op (and
                # its MAC register tile, one value per bank) cannot use one.
                if layout.mac_width > 1:
                    raise ValueError(
                        f"{label}: weight %{names[root]} is laid out for a MAC "
                        f"array {layout.mac_width} outputs wide, but cent.mac_abk "
                        "supports only one output per bank"
                    )
                col_rf = next(
                    rf for rf in layout.tile_factors if isinstance(rf.factor, Column)
                )
                gb_vals = prod(op.global_buffer.type.shape)
                if col_rf.extent != gb_vals:
                    raise ValueError(
                        f"{label}: weight Column extent ({col_rf.extent}) does not "
                        f"match Global Buffer tile size ({gb_vals})"
                    )
                weight_channels = layout.occupied_channels(arch.channels)
                if not set(op.channels) <= set(weight_channels):
                    raise ValueError(
                        f"{label}: channels {list(op.channels)} are not all "
                        f"channels of weight %{names[root]} ({weight_channels})"
                    )
                gb_ch = facts.channels[op.global_buffer]
                if not set(op.channels).issubset(set(gb_ch)):
                    raise ValueError(
                        f"{label}: channels {list(op.channels)} are not covered by "
                        f"Global Buffer %{names[op.global_buffer]} channels {list(gb_ch)}"
                    )
                _check_accumulator(label, op, facts, arch, names)

            elif isinstance(op, (CentRdMacOp, CentRdAfOp)):
                _check_tile_channels(label, op, layout, arch, names[root])
                if not _is_channel_bank_sb_tile(layout):
                    raise ValueError(
                        f"{label}: destination %{names[root]} tile factors must be "
                        "(Slot(of=Channel()), Lane(of=Bank()))"
                    )
                _check_accumulator(label, op, facts, arch, names)

    walk(kernel.function.operations)


def _used_in_block(block: Block) -> set[Value]:
    used: set[Value] = set(block.results)
    for op in block.operations:
        used.update(op.operands)
        for inner in op.blocks():
            used.update(_used_in_block(inner))
    return used


def _is_tracked_resource(value: Value) -> bool:
    t = value.type
    return isinstance(t, CentTensorType) and t.memory_space in (
        MemorySpace.PIM_GB,
        MemorySpace.PIM_MAC,
    )


def _check_liveness(
    function: Function,
    facts: _Facts,
    arch: CentArchitecture,
    names: dict[Value, str],
) -> None:
    limits = {
        MemorySpace.PIM_GB: 1,
        MemorySpace.PIM_MAC: arch.accumulators_per_bank,
    }
    space_labels = {
        MemorySpace.PIM_GB: "Global Buffer",
        MemorySpace.PIM_MAC: "MAC",
    }

    def check_block(
        ops: tuple[Operation, ...],
        block_args: tuple[Value, ...],
        results: tuple[Value, ...],
        outside: set[Value],
    ) -> None:
        # Compute last_use index in 0..len(ops) for values defined or used here.
        # Unconsumed resource values stay live until the end of the block
        # (len(ops) + 1) because hardware registers are not freed without a reader.
        last_use: dict[Value, int] = {}
        for i, op in enumerate(ops):
            for operand in op.operands:
                last_use[operand] = i
            for b in op.blocks():
                for v in _used_in_block(b):
                    last_use[v] = i
        for r in results:
            last_use[r] = len(ops)

        live: set[Value] = set(outside)
        for arg in block_args:
            if _is_tracked_resource(arg):
                _add_live(live, arg, "block entry")

        for i, op in enumerate(ops):
            label = _op_label(op)
            for b in op.blocks():
                used_inside = _used_in_block(b)
                inner_outside = {
                    v
                    for v in live
                    if v in used_inside or last_use.get(v, len(ops) + 1) > i
                }
                check_block(b.operations, b.arguments, b.results, inner_outside)

            # Expire values whose last use is this op (and are not kept alive from outside).
            expired = {
                v
                for v in live
                if v not in outside and last_use.get(v, len(ops) + 1) == i
            }
            live -= expired

            for res in op.results:
                if _is_tracked_resource(res):
                    _add_live(live, res, label)

    def _add_live(live: set[Value], new_val: Value, where: str) -> None:
        assert isinstance(new_val.type, CentTensorType)
        space = new_val.type.memory_space
        limit = limits[space]
        for ch in facts.channels[new_val]:
            holders = [
                v
                for v in live
                if isinstance(v.type, CentTensorType)
                and v.type.memory_space is space
                and ch in facts.channels[v]
            ]
            if len(holders) >= limit:
                existing = ", ".join(f"%{names[v]}" for v in holders)
                raise ValueError(
                    f"{where}: channel {ch} already holds live "
                    f"{space_labels[space]} value {existing}"
                )
        live.add(new_val)

    check_block(function.operations, function.arguments, function.results, set())


def _is_sb_value(value: Value) -> bool:
    """Whether ``value`` is a version of stored state: Shared Buffer or DRAM."""
    t = value.type
    return isinstance(t, CentTensorType) and t.memory_space in (
        MemorySpace.PNM_SB,
        MemorySpace.PIM_DRAM,
    )


def _check_sb_versions(function: Function, names: dict[Value, str]) -> None:
    def check_block(
        ops: tuple[Operation, ...],
        block_args: tuple[Value, ...],
        results: tuple[Value, ...],
    ) -> None:
        local_sb = {v for v in block_args if _is_sb_value(v)}
        for op in ops:
            for r in op.results:
                if _is_sb_value(r):
                    local_sb.add(r)

        last_use: dict[Value, int] = {}
        use_counts_at: dict[tuple[Value, int], int] = {}

        for i, op in enumerate(ops):
            for operand in op.operands:
                if _is_sb_value(operand):
                    last_use[operand] = i
                    use_counts_at[(operand, i)] = use_counts_at.get((operand, i), 0) + 1
            for b in op.blocks():
                for v in _used_in_block(b):
                    if _is_sb_value(v):
                        last_use[v] = i
                        use_counts_at[(v, i)] = use_counts_at.get((v, i), 0) + 1
        for r in results:
            if _is_sb_value(r):
                last_use[r] = len(ops)

        consumed_at: dict[Value, int] = {}
        for i, op in enumerate(ops):
            label = _op_label(op)
            consuming_operands: list[Value] = []
            if isinstance(op, (CentRdMacOp, CentRdAfOp)):
                consuming_operands.append(op.destination)
            elif isinstance(op, (CentWrMemOp, CentRdMemOp, CentCopyGbBkOp)):
                consuming_operands.append(op.operands[1])
            elif isinstance(op, CentEwmulOp):
                consuming_operands.append(op.operands[2])
            elif isinstance(op, CentEwaddOp):
                consuming_operands.append(op.operands[0])
            elif isinstance(op, CentPnmOp):
                consuming_operands.append(op.operands[-1])
            elif isinstance(op, TileForOp):
                consuming_operands.extend(v for v in op.operands if _is_sb_value(v))

            for v in consuming_operands:
                if v not in local_sb:
                    raise ValueError(
                        f"{label}: cannot overwrite outer Shared Buffer value "
                        f"%{names[v]} inside a nested block without carrying it"
                    )
                if v in consumed_at:
                    raise ValueError(
                        f"{label}: Shared Buffer value %{names[v]} was already "
                        "consumed earlier"
                    )
                if use_counts_at.get((v, i), 0) > 1:
                    raise ValueError(
                        f"{label}: Shared Buffer value %{names[v]} is both "
                        "consumed and used again at the same op"
                    )
                if last_use.get(v, i) > i:
                    raise ValueError(
                        f"{label}: Shared Buffer value %{names[v]} is read or "
                        "yielded after being overwritten"
                    )
                consumed_at[v] = i

            for b in op.blocks():
                check_block(b.operations, b.arguments, b.results)

    check_block(function.operations, function.arguments, function.results)
