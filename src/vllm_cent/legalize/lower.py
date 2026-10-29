"""Lower Semantic IR to the Device IR ops a target device supports.

``LowerToDevice`` turns every semantic op (``aloi.*``) into device ops
(``pim.*``, ``pnm.*``, ``cxl.*``, ``device.*``) that the target's
``DeviceSpec`` lists, in two stages:

1. Decompose: a decomposable op with no supported device op of its own is
   replaced by its decomposition, e.g. ``aloi.softmax`` by max, sub, exp,
   sum and div when the target has no ``pnm.softmax``. ``DecomposeOps`` does
   this; this module only tells it which ops to decompose.
2. Lower: every remaining op becomes one supported device op, preceded by
   the ``device.transfer`` ops that move its operands to where it computes.
   An op that no supported device op implements is an error.

Where tensors start and end is decided by their role: weights and the KV
cache live in the DRAM banks (PIM), activations near memory (PNM).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import assert_never

from ..device.spec import DeviceSpec
from ..ir.base import (
    Function,
    Operation,
    Value,
    rebuild_function,
    registered_operations,
)
from ..ir.device import DeviceOp, DeviceTransferOp
from ..ir.pass_manager import FunctionPass
from ..ir.passes import DecomposeOps
from ..ir.types import ExecutionDomain, TensorRole

__all__ = ["LowerToDevice"]

# Values already moved to another domain in the function being lowered, with
# their moved copies: (value, destination) -> the transfer's result. A value
# read by several ops in one domain is then moved once, not once per reader.
_Moved = dict[tuple[Value, ExecutionDomain], Value]


def _default_domain(role: TensorRole) -> ExecutionDomain:
    """Return where a function argument or result of this role lives.

    Args:
        role: What kind of tensor the value is.

    Returns:
        PIM for weights and the KV cache, which stay resident in the DRAM
        banks; PNM for activations, which flow between ops near memory.
    """

    match role:
        case TensorRole.WEIGHT | TensorRole.KV_CACHE:
            return ExecutionDomain.PIM
        case TensorRole.ACTIVATION:
            return ExecutionDomain.PNM
    assert_never(role)


def _implementations(semantic: type[Operation]) -> tuple[type[DeviceOp], ...]:
    """Return the device ops that can replace a semantic op, sorted by name.

    A device op implements ``semantic`` when ``semantic`` is its recorded
    semantic base (``DeviceOp.SEMANTIC``) or a subclass of it. The subclass
    case covers ops that share only type rules: ``pnm.softmax`` records
    ``SoftmaxRules``, and ``aloi.softmax`` is a ``SoftmaxRules``. Ops with no
    semantic base, such as ``device.transfer``, implement nothing.

    The device ops are found among the registered op classes, so only
    imported device ops are candidates; importing this module imports them
    all through ``vllm_cent.ir.device``.

    Args:
        semantic: The class of a Semantic IR op, e.g. ``MulOp``.

    Returns:
        The implementing device op classes, ordered by ``NAME`` so the result
        does not depend on import order. Empty when no device op implements
        ``semantic``.

    Examples:
        ``MulOp`` gives ``(PIMEwmulOp, PNMMulOp)``: an elementwise product in
        the banks or near memory. ``AddOp`` gives only ``(PNMAddOp,)``; the
        other elementwise ops share a base class with it, but no device op
        records that shared base.
    """

    candidates = [
        op
        for op in registered_operations().values()
        if issubclass(op, DeviceOp)
        and op.SEMANTIC is not None
        # Not issubclass(op, semantic): pnm.softmax is not a SoftmaxOp, but
        # SoftmaxOp is a SoftmaxRules.
        and issubclass(semantic, op.SEMANTIC)
    ]
    return tuple(sorted(candidates, key=lambda op: op.NAME))


@dataclass(frozen=True, slots=True)
class _Lowering:
    """One way to replace a semantic op: move some operands, then compute.

    Attributes:
        transfers: New ``device.transfer`` ops that move operands into the
            device op's domain, in operand order.
        device_op: The device op that replaces the semantic op; it reads the
            transfers' results in place of the moved operands.
    """

    transfers: tuple[DeviceTransferOp, ...]
    device_op: DeviceOp

    def ops(self) -> tuple[DeviceOp, ...]:
        """Return the ops to emit: the transfers, then the device op last."""

        return (*self.transfers, self.device_op)


def _label(op: Operation) -> str:
    """Name an op in error messages: its NAME, plus its name_hint if any.

    Decomposed ops carry hints such as ``attn.softmax.exp``, which tell
    which original op they came from.

    Args:
        op: Any op.

    Returns:
        E.g. ``"aloi.exp (attn.softmax.exp)"``, or ``"aloi.exp"``.
    """

    return op.NAME if op.name_hint is None else f"{op.NAME} ({op.name_hint})"


class LowerToDevice(FunctionPass):
    """Replace every semantic op by device ops the target supports.

    For each op, in order of preference:

    1. A supported device op that implements it directly. When several are
       supported (``pim.ewmul`` and ``pnm.mul`` for a product), the one that
       needs the fewest ``device.transfer`` ops wins; on a tie, the PNM one.
    2. Its decomposition, if it is decomposable, lowered op by op.
    3. Otherwise an error that lists why each device op was rejected.

    Function arguments are placed by role (see ``_default_domain``) and
    results are moved back to the domain of their role, so the output of one
    lowered decoder layer is where the next layer expects its input.

    The input must be Semantic IR with decomposed attention: run this pass
    before ``FuseAttention``, because no device op implements the fused
    ``aloi.attention``.
    """

    NAME = "lower-to-device"

    def __init__(self, spec: DeviceSpec) -> None:
        """Lower for one target device.

        Args:
            spec: The target; its capabilities decide which device ops are
                allowed, including ``device.transfer`` for each dtype moved.
        """

        self._capabilities = spec.capabilities

    def run_on_function(self, function: Function) -> Function:
        """Lower one function; ``function`` itself is not modified.

        Args:
            function: Semantic IR: no argument has a domain and no op is a
                device op.

        Returns:
            The same computation in Device IR. Arguments keep their names and
            gain a domain; results are in the domain of their role.

        Raises:
            ValueError: If the input is not Semantic IR, if an op has neither
                a supported device op nor a decomposition, or if a needed
                ``device.transfer`` is not supported for a value's dtype.
        """

        _require_semantic(function)
        prepared = DecomposeOps(
            select=lambda op: not self._has_direct_implementation(op)
        ).run_on_function(function)
        arguments = tuple(
            Value(
                type=replace(argument.type, domain=_default_domain(argument.type.role)),
                name=argument.name,
            )
            for argument in prepared.operands
        )
        moved: _Moved = {}
        lowered = rebuild_function(
            prepared, arguments, rewrite=lambda op: self._lower_op(op, moved)
        )
        return self._place_results(lowered, moved)

    def _has_direct_implementation(self, op: Operation) -> bool:
        """Tell whether some supported device op can replace ``op`` directly.

        ``op`` still has Semantic IR operands, which have no domain. Each
        candidate is tried on stand-in operands of the same types, already
        in the domain the candidate needs, so only the op's own support is
        checked: its class, dtype and parameters. The transfers that real
        operands may need are checked later, in ``_lower_op``.

        Args:
            op: A semantic op, as ``DecomposeOps`` sees it.

        Returns:
            True if at least one candidate is supported.
        """

        # Place every operand by role first: same-domain candidates (views,
        # copies) take the domain of their first operand.
        placed = [
            Value(type=replace(v.type, domain=_default_domain(v.type.role)))
            for v in op.operands
        ]
        for device_class in _implementations(type(op)):
            domain = device_class.operand_domain(placed)
            stand_ins = [
                value
                if index in device_class.INDEX_OPERANDS
                else Value(type=replace(value.type, domain=domain))
                for index, value in enumerate(placed)
            ]
            if isinstance(self._build(device_class, op, stand_ins), DeviceOp):
                return True
        return False

    def _lower_op(self, op: Operation, moved: _Moved) -> tuple[DeviceOp, ...]:
        """Replace one semantic op by transfers and its best device op.

        Used as ``rebuild_function``'s rewrite: ``op`` is the semantic op
        rebuilt on the new function's values, so its operands are already
        Device IR values with a domain.

        Args:
            op: A semantic op whose operands are Device IR values.
            moved: Values moved by earlier ops of this function. Operands
                found here are reused instead of moved again; the transfers
                of the chosen lowering are added.

        Returns:
            The new transfers that move its operands, then the device op; the
            device op is last, so its results replace ``op``'s.

        Raises:
            ValueError: If no candidate is supported, naming the op (and its
                name_hint) and each candidate's reason.
        """

        lowerings: list[_Lowering] = []
        reasons: list[str] = []
        for device_class in _implementations(type(op)):
            lowering = self._lower_with(device_class, op, moved)
            if isinstance(lowering, str):
                reasons.append(lowering)
            else:
                lowerings.append(lowering)
        if not lowerings:
            why = "; ".join(reasons) if reasons else "no device op implements it"
            raise ValueError(f"{self.NAME}: cannot lower {_label(op)}: {why}")

        def cost(lowering: _Lowering) -> tuple[int, bool]:
            """Fewer transfers first; on a tie, PNM before PIM (False < True)."""

            result_domain = lowering.device_op.results[0].type.domain
            return (len(lowering.transfers), result_domain is not ExecutionDomain.PNM)

        # min() keeps the first of equal costs, i.e. the first by NAME.
        chosen = min(lowerings, key=cost)
        # Record the new copies only now: the rejected candidates' transfers
        # are never emitted, so later ops must not read their results.
        for transfer in chosen.transfers:
            moved[(transfer.x, transfer.to)] = transfer.results[0]
        return chosen.ops()

    def _lower_with(
        self, device_class: type[DeviceOp], op: Operation, moved: _Moved
    ) -> _Lowering | str:
        """Try one candidate: move the operands it needs moved, then build it.

        Args:
            device_class: A device op that implements ``op``'s class.
            op: A semantic op whose operands are Device IR values.
            moved: Values already moved by earlier ops; an operand found here
                is read from its copy, with no new transfer. Not modified.

        Returns:
            The new transfers and the device op, or a sentence saying why
            this candidate cannot be used.
        """

        domain = device_class.operand_domain(op.operands)
        if domain is None:
            return f"{device_class.NAME}: operand 0 has no domain"
        transfers: list[DeviceTransferOp] = []
        operands: list[Value] = []
        for index, operand in enumerate(op.operands):
            if index in device_class.INDEX_OPERANDS or operand.type.domain is domain:
                operands.append(operand)
                continue
            copy = moved.get((operand, domain))
            if copy is not None:
                operands.append(copy)
                continue
            transfer = DeviceTransferOp(operands=(operand,), to=domain)
            reason = self._capabilities.why_unsupported(transfer)
            if reason is not None:
                return (
                    f"{device_class.NAME} needs operand {index} moved to "
                    f"{domain}, but {reason}"
                )
            transfers.append(transfer)
            operands.append(transfer.results[0])

        built = self._build(device_class, op, operands)
        if isinstance(built, str):
            return built
        return _Lowering(transfers=tuple(transfers), device_op=built)

    def _build(
        self, device_class: type[DeviceOp], op: Operation, operands: list[Value]
    ) -> DeviceOp | str:
        """Build ``device_class`` with ``op``'s attributes and check support.

        Args:
            device_class: A device op that implements ``op``'s class.
            op: The semantic op being replaced; its attributes (e.g. eps,
                the activation function) and name_hint are copied.
            operands: The device op's operands, already in its domain.

        Returns:
            The device op, or why it cannot be used: its type rules reject
            the operands, or the target does not support it.
        """

        try:
            # A device op inherits its semantic op's fields, so the semantic
            # op's attributes are exactly the device op's keyword arguments.
            device_op = device_class(
                operands=tuple(operands),
                name_hint=op.name_hint,
                **op.attributes(),
            )
        except ValueError as error:
            return str(error)
        reason = self._capabilities.why_unsupported(device_op)
        return device_op if reason is None else reason

    def _place_results(self, function: Function, moved: _Moved) -> Function:
        """Move each result to the domain of its role, e.g. an activation to PNM.

        Args:
            function: A lowered function.
            moved: Values the function already moved; a result found here is
                returned from its existing copy.

        Returns:
            ``function`` itself if every result is already in place;
            otherwise a copy with transfers appended and returned instead.

        Raises:
            ValueError: If a needed transfer is not supported.
        """

        transfers: list[DeviceTransferOp] = []
        results: list[Value] = []
        for result in function.results:
            domain = _default_domain(result.type.role)
            if result.type.domain is domain:
                results.append(result)
                continue
            copy = moved.get((result, domain))
            if copy is not None:
                results.append(copy)
                continue
            # Name the moved value after its producer, e.g. %down_proj_1.
            producer = function.defining_op(result)
            transfer = DeviceTransferOp(
                operands=(result,),
                to=domain,
                name_hint=None if producer is None else producer.name_hint,
            )
            reason = self._capabilities.why_unsupported(transfer)
            if reason is not None:
                raise ValueError(
                    f"{self.NAME}: cannot move a result of {function.name} "
                    f"to {domain}: {reason}"
                )
            transfers.append(transfer)
            results.append(transfer.results[0])
        if not transfers:
            return function
        return Function(
            name=function.name,
            operands=function.operands,
            operations=(*function.operations, *transfers),
            results=tuple(results),
        )


def _require_semantic(function: Function) -> None:
    """Check that ``function`` is Semantic IR, the input ``LowerToDevice`` expects.

    Args:
        function: The function about to be lowered.

    Raises:
        ValueError: If an argument already has a domain or an op is already
            a device op, i.e. the function was lowered before.
    """

    placed = [a.name for a in function.operands if a.type.domain is not None]
    if placed:
        raise ValueError(
            f"{LowerToDevice.NAME}: arguments {placed} of {function.name} "
            "already have a domain; the input must be Semantic IR"
        )
    device_ops = [op.NAME for op in function.operations if isinstance(op, DeviceOp)]
    if device_ops:
        raise ValueError(
            f"{LowerToDevice.NAME}: {function.name} already contains device "
            f"ops {device_ops}; the input must be Semantic IR"
        )
