"""Translate a ``torch.export`` program into ALOI IR.

The importer walks the exported FX graph once, in order, and replaces every
node with ALOI IR: placeholders become function arguments, ``aloi::*``
custom-op calls become semantic ops, and the output node becomes the
function's results. It accepts static shapes only.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Iterable, Mapping

import torch
from torch._ops import OpOverload
from torch.export import ExportedProgram
from torch.export.graph_signature import InputKind, OutputKind
from torch.fx import Node

from ..activations import ActivationKind
from ..ir.base import Function, Module, Operation, Value
from ..ir.printer import format_type
from ..ir.semantic import (
    ActivationOp,
    AddOp,
    CopyOp,
    LinearOp,
    MulOp,
    ReshapeOp,
    RMSNormOp,
    SoftmaxOp,
)
from ..ir.semantic.transformer import (
    AttnTemperatureOp,
    GqaQKOp,
    GqaSVOp,
    KvUpdateOp,
    MergeHeadsOp,
    MoeExpertsOp,
    RopeOp,
    SplitHeadsOp,
    TopKSigmoidOp,
)
from ..ir.types import DType, TensorType

# Imported for its side effect: registering torch.ops.aloi.*, which the
# converter table below refers to.
from . import aloi  # noqa: F401

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
            yet, such as CONSTANT_TENSOR.
    """

    names: dict[str, str] = {}

    for spec in program.graph_signature.input_specs:
        # spec.kind is an InputKind enum member, not a string, so compare
        # with "is" against the enum.
        # Parameters and buffers (e.g. a RoPE cos/sin table) are named by
        # FQN, the key they have in the model's state.
        if (
            spec.kind in (InputKind.PARAMETER, InputKind.BUFFER)
            and spec.target is not None
        ):
            names[spec.arg.name] = spec.target
        elif spec.kind is InputKind.USER_INPUT:
            names[spec.arg.name] = spec.arg.name
        else:
            raise NotImplementedError(
                f"unsupported input {spec.arg.name} of kind {spec.kind.name}"
            )

    return names


def _module_path(node: Node) -> str:
    """Return the path of the innermost nn.Module that created ``node``.

    torch.export records the module call stack of every op, outermost first,
    as ``{key: (path, class name)}``.

    Returns:
        The innermost path, e.g. ``self_attn.q_proj``. Placeholders have no
        stack, and an op called in the root ``forward`` has path ``""``.
    """

    stack = node.meta.get("nn_module_stack")
    if not stack:
        return ""
    path, _ = list(stack.values())[-1]
    return str(path)


def _name_hints(
    nodes: Iterable[Node], parameters: Mapping[str, str]
) -> dict[Node, str]:
    """Choose a readable result name for every op call.

    - A call that reads exactly one parameter is named after the module that
      owns it: ``aloi.linear(x, self.q_proj.weight)`` gives
      ``self_attn.q_proj``, ``aloi.rms_norm(x, self.input_layernorm.weight,
      eps)`` gives ``input_layernorm``.
    - Otherwise, the only such call inside a module is named after the
      module; several calls inside one module's forward get the module path
      plus FX's unique node name (``self_attn.rope``, ``self_attn.rope_1``);
      calls in the root forward get the node name alone (``add_1``).

    Args:
        nodes: The graph's nodes.
        parameters: Parameter placeholder name to parameter FQN.

    Returns:
        A hint for every call_function node.
    """

    hints: dict[Node, str] = {}
    paths: dict[Node, str] = {}
    for node in nodes:
        if node.op != "call_function":
            continue
        owners = [
            parameters[operand.name].rpartition(".")[0]
            for operand in node.all_input_nodes
            if operand.name in parameters
        ]
        # A root-level parameter such as "w" has no owning module ("").
        if len(owners) == 1 and owners[0]:
            hints[node] = owners[0]
        else:
            paths[node] = _module_path(node)

    calls_per_path = Counter(paths.values())
    for node, path in paths.items():
        if not path:
            hints[node] = node.name
        elif calls_per_path[path] == 1:
            hints[node] = path
        else:
            hints[node] = f"{path}.{node.name}"
    return hints


def _arguments(node: Node) -> dict[str, object]:
    """Return every schema argument of an op call by name, defaults filled in.

    torch.export drops trailing optional arguments the caller omitted (rope
    without Llama 3 scaling has 3 arguments, not 7), and keyword-only ones such
    as add's ``alpha`` live in ``node.kwargs``. Going through the op's schema
    lets converters read every argument by name.

    Args:
        node: A call_function node whose target is an OpOverload.

    Returns:
        Argument name to value.

    Raises:
        TypeError: If a required argument is missing.
    """

    target = node.target
    assert isinstance(target, OpOverload)
    values: dict[str, object] = {}
    for index, argument in enumerate(target._schema.arguments):
        if index < len(node.args):
            values[argument.name] = node.args[index]
        elif argument.name in node.kwargs:
            values[argument.name] = node.kwargs[argument.name]
        elif argument.has_default_value():
            values[argument.name] = argument.default_value
        else:
            raise TypeError(f"{target}: missing argument {argument.name!r}")
    return values


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


def _float(name: str, value: object) -> float:
    """Return a float attribute.

    An int is accepted: a schema ``float`` may be called with ``10000``.

    Raises:
        TypeError: If the value is not a number.
    """

    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{name} must be a number, got {value!r}")
    return float(value)


def _optional_float(name: str, value: object) -> float | None:
    """Return an optional float attribute; see _float."""

    return None if value is None else _float(name, value)


def _int(name: str, value: object) -> int:
    """Return an int attribute.

    Raises:
        TypeError: If the value is not an int.
    """

    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int, got {value!r}")
    return value


# A converter turns one FX call into one IR op. It receives the node, the
# node-to-value map for looking up operands, and the result's name hint
# (Operation is frozen, so the hint must be given at construction).
_Converter = Callable[[Node, dict[Node, Value], str | None], Operation]


def _tensor_op(op_class: type[Operation], *operand_names: str) -> _Converter:
    """Make a converter for an op whose arguments are all tensor operands.

    Args:
        op_class: IR op to build.
        *operand_names: Schema argument names, in operand order.

    Returns:
        The converter.
    """

    def convert(node: Node, env: dict[Node, Value], name_hint: str | None) -> Operation:
        """Build ``op_class`` from the named tensor arguments."""

        arguments = _arguments(node)
        operands = tuple(_value(env, arguments[name]) for name in operand_names)
        return op_class(operands=operands, name_hint=name_hint)

    return convert


def _convert_linear(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aloi::linear(x, weight, bias=None)`` into a ``LinearOp``.

    A None bias gives the two-operand form.
    """

    arguments = _arguments(node)
    bias = arguments["bias"]
    operands: tuple[Value, ...] = (
        _value(env, arguments["x"]),
        _value(env, arguments["weight"]),
    )
    if bias is not None:
        operands += (_value(env, bias),)
    return LinearOp(operands=operands, name_hint=name_hint)


def _convert_rms_norm(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aloi::rms_norm(x, weight, eps=None)`` into an ``RMSNormOp``.

    A None weight (``elementwise_affine=False``) gives the one-operand form.
    """

    arguments = _arguments(node)
    weight = arguments["weight"]
    operands: tuple[Value, ...] = (_value(env, arguments["x"]),)
    if weight is not None:
        operands += (_value(env, weight),)
    return RMSNormOp(
        operands=operands,
        eps=_optional_float("eps", arguments["eps"]),
        name_hint=name_hint,
    )


def _convert_split_heads(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aloi::split_heads(x, head_dim)`` into a ``SplitHeadsOp``."""

    arguments = _arguments(node)
    return SplitHeadsOp(
        operands=(_value(env, arguments["x"]),),
        head_dim=_int("head_dim", arguments["head_dim"]),
        name_hint=name_hint,
    )


def _convert_rope(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aloi::rope(x, positions, cos_sin_cache, is_neox_style)``."""

    arguments = _arguments(node)
    is_neox_style = arguments["is_neox_style"]
    if not isinstance(is_neox_style, bool):
        raise TypeError(f"is_neox_style must be a bool, got {is_neox_style!r}")
    return RopeOp(
        operands=(
            _value(env, arguments["x"]),
            _value(env, arguments["positions"]),
            _value(env, arguments["cos_sin_cache"]),
        ),
        is_neox_style=is_neox_style,
        name_hint=name_hint,
    )


def _activation(function: ActivationKind) -> _Converter:
    """Make a converter for a one-operand aten activation.

    Args:
        function: The formula this aten op computes.
    """

    def convert(node: Node, env: dict[Node, Value], name_hint: str | None) -> Operation:
        """Build an ``ActivationOp`` that applies ``function``."""

        arguments = _arguments(node)
        return ActivationOp(
            operands=(_value(env, arguments["self"]),),
            function=function,
            name_hint=name_hint,
        )

    return convert


# aten.gelu's ``approximate`` string -> formula. Any other string is a
# formula the IR does not have.
_GELU_FORMULAS: dict[str, ActivationKind] = {
    "none": ActivationKind.GELU_ERF,
    "tanh": ActivationKind.GELU_TANH,
}


def _convert_gelu(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aten.gelu(self, *, approximate="none")``.

    The ``approximate`` string picks the formula, so one aten op becomes
    either of two ActivationOp kinds.

    Raises:
        NotImplementedError: If ``approximate`` names another formula.
    """

    arguments = _arguments(node)
    approximate = arguments["approximate"]
    # _arguments returns plain objects; only a str can be a key of the table.
    function = _GELU_FORMULAS.get(approximate) if isinstance(approximate, str) else None
    if function is None:
        raise NotImplementedError(f"gelu with approximate={approximate!r}")
    return ActivationOp(
        operands=(_value(env, arguments["self"]),),
        function=function,
        name_hint=name_hint,
    )


def _convert_clone(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aten.clone(self, *, memory_format=None)`` into a ``CopyOp``.

    Raises:
        NotImplementedError: If ``memory_format`` reorders the elements in
            memory (e.g. channels_last); CopyOp keeps the row-major order.
    """

    arguments = _arguments(node)
    memory_format = arguments["memory_format"]
    if memory_format not in (None, torch.contiguous_format):
        raise NotImplementedError(f"clone with memory_format={memory_format}")
    return CopyOp(operands=(_value(env, arguments["self"]),), name_hint=name_hint)


def _convert_reshape(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aten.view(self, size)`` or ``aten.reshape(self, shape)``.

    The target shape comes from the node's recorded result type, not from the
    size argument: the argument may contain -1 (``x.view(-1, 4)``), while the
    recorded shape is fully resolved.
    """

    arguments = _arguments(node)
    return ReshapeOp(
        operands=(_value(env, arguments["self"]),),
        shape=_tensor_type(node).shape,
        name_hint=name_hint,
    )


def _convert_attn_temperature(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aloi::attn_temperature(x, positions, floor_scale, attn_scale)``."""

    arguments = _arguments(node)
    return AttnTemperatureOp(
        operands=(_value(env, arguments["x"]), _value(env, arguments["positions"])),
        floor_scale=_float("floor_scale", arguments["floor_scale"]),
        attn_scale=_float("attn_scale", arguments["attn_scale"]),
        name_hint=name_hint,
    )


def _convert_topk_sigmoid(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aloi::topk_sigmoid(logits, top_k)`` into a ``TopKSigmoidOp``."""

    arguments = _arguments(node)
    return TopKSigmoidOp(
        operands=(_value(env, arguments["logits"]),),
        top_k=_int("top_k", arguments["top_k"]),
        name_hint=name_hint,
    )


def _convert_moe_experts(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aloi::moe_experts(x, routing_weights, gate, up, down, on_input)``.

    Raises:
        TypeError: If ``apply_router_weight_on_input`` is not a bool.
    """

    arguments = _arguments(node)
    on_input = arguments["apply_router_weight_on_input"]
    if not isinstance(on_input, bool):
        raise TypeError(
            f"apply_router_weight_on_input must be a bool, got {on_input!r}"
        )
    names = ("x", "routing_weights", "gate_proj", "up_proj", "down_proj")
    return MoeExpertsOp(
        operands=tuple(_value(env, arguments[name]) for name in names),
        apply_router_weight_on_input=on_input,
        name_hint=name_hint,
    )


def _convert_gqa_qk(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aloi::gqa_qk(q, k_cache, scale)`` into a ``GqaQKOp``."""

    arguments = _arguments(node)
    return GqaQKOp(
        operands=(_value(env, arguments["q"]), _value(env, arguments["k_cache"])),
        scale=_float("scale", arguments["scale"]),
        name_hint=name_hint,
    )


def _convert_add(
    node: Node, env: dict[Node, Value], name_hint: str | None
) -> Operation:
    """Convert ``aten.add.Tensor(self, other, alpha=1)`` into an ``AddOp``.

    Raises:
        NotImplementedError: If alpha is not 1; ``a + alpha * b`` has no IR op.
    """

    arguments = _arguments(node)
    if arguments["alpha"] != 1:
        raise NotImplementedError(f"add with alpha={arguments['alpha']!r}")
    return AddOp(
        operands=(_value(env, arguments["self"]), _value(env, arguments["other"])),
        name_hint=name_hint,
    )


# Supporting a new op means one converter and one entry here. The aloi::*
# custom ops come from the model; the aten ops are the plain PyTorch ops a
# model writes as F.silu(x), a * b, a + b, x.clone() and x.view(...).
# Several aten activations map to one ActivationOp with different kinds.
_CONVERTERS: dict[OpOverload, _Converter] = {
    torch.ops.aloi.linear.default: _convert_linear,
    torch.ops.aloi.rms_norm.default: _convert_rms_norm,
    torch.ops.aloi.split_heads.default: _convert_split_heads,
    torch.ops.aloi.merge_heads.default: _tensor_op(MergeHeadsOp, "x"),
    torch.ops.aloi.rope.default: _convert_rope,
    torch.ops.aloi.kv_update.default: _tensor_op(
        KvUpdateOp, "cache", "new", "positions"
    ),
    torch.ops.aloi.gqa_qk.default: _convert_gqa_qk,
    torch.ops.aloi.softmax.default: _tensor_op(SoftmaxOp, "x"),
    torch.ops.aloi.gqa_sv.default: _tensor_op(GqaSVOp, "probs", "v_cache"),
    torch.ops.aloi.attn_temperature.default: _convert_attn_temperature,
    torch.ops.aloi.topk_sigmoid.default: _convert_topk_sigmoid,
    torch.ops.aloi.moe_experts.default: _convert_moe_experts,
    torch.ops.aten.silu.default: _activation(ActivationKind.SILU),
    torch.ops.aten.relu.default: _activation(ActivationKind.RELU),
    torch.ops.aten.sigmoid.default: _activation(ActivationKind.SIGMOID),
    torch.ops.aten.tanh.default: _activation(ActivationKind.TANH),
    torch.ops.aten.gelu.default: _convert_gelu,
    torch.ops.aten.mul.Tensor: _tensor_op(MulOp, "self", "other"),
    torch.ops.aten.add.Tensor: _convert_add,
    torch.ops.aten.clone.default: _convert_clone,
    torch.ops.aten.view.default: _convert_reshape,
    torch.ops.aten.reshape.default: _convert_reshape,
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
    hints = _name_hints(
        program.graph.nodes, program.graph_signature.inputs_to_parameters
    )
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
            op = converter(node, env, hints[node])

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
