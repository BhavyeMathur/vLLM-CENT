"""Translate a ``torch.export`` program into ALOI IR.

The importer walks the exported FX graph once, in order, and replaces every
node with ALOI IR: placeholders become function arguments, ``aloi::*``
custom-op calls become semantic ops, and the output node becomes the
function's results. It accepts static shapes only.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
from torch._ops import OpOverload
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx import Node

# Imported for its side effect: registering torch.ops.aloi.*, which the
# converter table below refers to.
from . import custom_ops  # noqa: F401
from ..ir.base import Function, Module, Operation, Value
from ..ir.printer import format_type
from ..ir.semantic import LinearOp, RMSNormOp
from ..ir.types import DType, TensorType

__all__ = ["import_exported_program"]


def _tensor_type(node: Node) -> TensorType:
    """Return the IR type of the tensor that ``node`` produces.

    Reads ``node.meta["val"]``, the fake tensor torch.export records for
    every node.

    Raises:
        NotImplementedError: If the node does not produce a tensor, or has a
            dynamic dimension.
        ValueError: If the dtype has no ALOI equivalent.
    """

    val = node.meta.get("val")
    if not isinstance(val, torch.Tensor):
        raise NotImplementedError(f"node {node.name} does not produce a tensor")

    # A dynamic dimension is a SymInt, not an int. Check before building the
    # TensorType: its int(dim) would silently turn the SymInt into the example
    # size (e.g. 4), and the compiled program would be wrong without an error.
    shape: list[int] = []
    for dim in val.shape:
        if not isinstance(dim, int):
            raise NotImplementedError(
                f"node {node.name} has dynamic dimension {dim}; "
                "only static shapes are supported"
            )
        shape.append(dim)

    return TensorType(shape=tuple(shape), dtype=DType.normalize(val.dtype))


def _argument_names(program: ExportedProgram) -> dict[str, str]:
    """Map each placeholder name to the name its IR argument should get.

    A parameter placeholder is called p_q_proj_weight in the graph; the IR
    uses its FQN q_proj.weight instead, the same key as in state_dict and
    checkpoints. A user input keeps its forward() parameter name, e.g. x.

    Raises:
        NotImplementedError: For input kinds the importer does not handle
            yet, such as BUFFER or CONSTANT_TENSOR.
    """

    names: dict[str, str] = {}

    for spec in program.graph_signature.input_specs:
        # spec.kind is an InputKind enum member, not a string, so compare
        # with "is" against the enum.
        if spec.kind is InputKind.PARAMETER and spec.target is not None:
            names[spec.arg.name] = spec.target
        elif spec.kind is InputKind.USER_INPUT:
            names[spec.arg.name] = spec.arg.name
        else:
            raise NotImplementedError(
                f"unsupported input {spec.arg.name} of kind {spec.kind.name}"
            )

    return names


def _module_path(node: Node) -> str | None:
    """Return the path of the innermost nn.Module that created ``node``.

    torch.export records the module call stack of every op, outermost first,
    as ``{key: (path, class name)}``. The innermost path, e.g. ``q_proj``,
    becomes the result's ``name_hint``.

    Returns:
        The module path, or None when there is none to use: placeholders have
        no stack, and an op called directly in the root ``forward`` has the
        empty path ``""``. The printer then numbers the result instead.
    """

    stack = node.meta.get("nn_module_stack")
    if not stack:
        return None
    path, _ = list(stack.values())[-1]
    return path or None


def _value(env: dict[Node, Value], arg: object) -> Value:
    """Return the IR value that stands for the FX node ``arg``.

    FX stores op arguments in an untyped tuple that can hold nodes, floats,
    lists or None. Tensor operands must be nodes already translated into
    ``env``; anything else means the graph does not have the shape the
    converter expects.

    Raises:
        TypeError: If ``arg`` is not an FX node, e.g. a constant in a
            tensor position.
    """

    if not isinstance(arg, Node):
        raise TypeError(f"expected a tensor operand, got {arg!r}")
    # Nodes are visited in topological order, so every operand has already
    # been translated; a KeyError here would be an importer bug.
    return env[arg]


# A converter turns one FX call into one IR op. It receives the node, the
# node-to-value map for looking up operands, and the result's name hint
# (Operation is frozen, so the hint must be given at construction).
_Converter = Callable[[Node, dict[Node, Value], str | None], Operation]


def _convert_linear(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aloi::linear(x, weight)`` into a ``LinearOp``.

    Args:
        node: The FX call.
        env: FX node to IR value map.
        name_hint: Name hint for the result.

    Returns:
        The new op.
    """

    x, weight = node.args
    return LinearOp(operands=(_value(env, x), _value(env, weight)), name_hint=name_hint)


def _convert_rms_norm(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aloi::rms_norm(x, weight, eps=None)`` into an ``RMSNormOp``.

    Args:
        node: The FX call.
        env: FX node to IR value map.
        name_hint: Name hint for the result.

    Returns:
        The new op.

    Raises:
        TypeError: If ``eps`` is neither a float nor None.
    """

    # When the caller omits eps, torch.export drops it: args has 2 entries.
    x, weight, *rest = node.args
    eps = rest[0] if rest else None
    if eps is not None and not isinstance(eps, float):
        raise TypeError(f"rms_norm eps must be a float or None, got {eps!r}")
    return RMSNormOp(
        operands=(_value(env, x), _value(env, weight)),
        eps=eps,
        name_hint=name_hint,
    )


# Supporting a new op means one converter and one entry here.
_CONVERTERS: dict[OpOverload, _Converter] = {
    torch.ops.aloi.linear.default: _convert_linear,
    torch.ops.aloi.rms_norm.default: _convert_rms_norm,
}


def import_exported_program(program: ExportedProgram, module_name: str) -> Module:
    """Translate an exported PyTorch program into an ALOI IR module.

    The FX graph is already a topologically ordered SSA list, so one walk
    over its nodes is enough. ``env`` maps every visited FX node to the IR
    value that replaces it, like the old-to-new map in ``clone_function``.

    Args:
        program: Result of ``export_model`` or ``torch.export.load``.
        module_name: Name of the returned module.

    Returns:
        A verified module with one function, ``forward``.

    Raises:
        NotImplementedError: For graph features the importer does not
            handle yet: mutated outputs, buffers, unknown ops, other node
            kinds.
        ValueError: If an IR op infers a different type than PyTorch did.
    """

    # A model that mutates a buffer or input (e.g. a KV cache update) gets
    # extra outputs mixed into the output node. Reject them for now instead
    # of silently returning them as ordinary results.
    for spec in program.graph_signature.output_specs:
        if spec.kind is not OutputKind.USER_OUTPUT:
            raise NotImplementedError(
                f"unsupported output {spec.arg.name} of kind {spec.kind.name}"
            )

    names = _argument_names(program)
    env: dict[Node, Value] = {}
    arguments: list[Value] = []
    operations: list[Operation] = []
    results: list[Value] = []

    for node in program.graph.nodes:
        # A placeholder is a graph input: parameters first, then user inputs.
        # Keep the graph's order; the function signature follows it.
        if node.op == "placeholder":
            argument = Value(type=_tensor_type(node), name=names[node.name])
            env[node] = argument
            arguments.append(argument)

        elif node.op == "call_function":
            converter = _CONVERTERS.get(node.target)
            if converter is None:
                raise NotImplementedError(
                    f"no ALOI converter for {node.target} (node {node.name})"
                )
            op = converter(node, env, _module_path(node))

            # Every op converted so far has exactly one result. Unpacking
            # (instead of results[0]) fails loudly if that ever changes.
            (result,) = op.results

            # PyTorch's fake kernel and the IR op's infer_result_types are two
            # independent shape inferences. If they disagree, one of them has
            # the op's semantics wrong; report it here, not in a later stage.
            expected = _tensor_type(node)
            if result.type != expected:
                raise ValueError(
                    f"{op.NAME} (node {node.name}) infers "
                    f"{format_type(result.type)}, but PyTorch says "
                    f"{format_type(expected)}"
                )

            env[node] = result
            operations.append(op)

        elif node.op == "output":
            # The output node has a single argument: the tuple of returned
            # nodes, e.g. args == ((linear,),).
            (outputs,) = node.args
            results = [_value(env, output) for output in outputs]

        else:
            # get_attr, call_module, call_method do not appear in the graphs
            # we export today.
            raise NotImplementedError(
                f"unsupported FX node kind {node.op!r} (node {node.name})"
            )

    function = Function(
        name="forward",
        operands=tuple(arguments),
        operations=tuple(operations),
        results=tuple(results),
    )
    module = Module(name=module_name, functions=(function,))
    # Guarantee that the importer never hands broken IR to the passes.
    module.verify()
    return module
