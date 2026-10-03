"""Deterministic text form of ALOI IR.

The printer only knows the generic IR classes (``Value``, ``Operation``,
``Function``, ``Module``), so the same code prints every dialect: ``aloi.*``
today and ``pim.*``/``aim.*`` later. It never imports a concrete op.

The text is used for stage dumps such as ``01_semantic.aloi`` and for golden
tests, so identical IR must always print identically. Every name and ordering
therefore comes from list order in the IR, never from ``id()`` or set order.

Example::

    func @block(%x: tensor<1x1x8192xfp16>, %w: tensor<1024x8192xfp16>) -> (tensor<1x1x1024xfp16>) {
      %k = aloi.linear(%x, %w) : tensor<1x1x1024xfp16>
      return %k
    }
"""

from __future__ import annotations

import json
from enum import Enum

from .base import Function, Module, Operation, Value
from .types import TensorRole, TensorType

__all__ = ["format_type", "print_function", "print_module"]

_INDENT = "  "


def format_type(tensor_type: TensorType) -> str:
    """Format a tensor type as ``tensor<1x1x8192xfp16>``.

    Annotations are appended only when they differ from their defaults, in a
    fixed order: role, axes, global shape. An unannotated M0 dump therefore
    stays short, while later stages show the full semantic information, e.g.
    ``tensor<1024x8192xfp16, role=k_weight, axes=[out_features, in_features]>``.

    Args:
        tensor_type: Type to format.

    Returns:
        The type's text form.
    """

    # The dtype is joined like one more dimension. A rank-0 tensor has no
    # dimensions and prints as tensor<fp16>.
    shape_and_dtype = "x".join(
        [*(str(dim) for dim in tensor_type.shape), tensor_type.dtype.value]
    )

    annotations: list[str] = []
    if tensor_type.role is not TensorRole.ACTIVATION:
        annotations.append(f"role={tensor_type.role.value}")
    if tensor_type.axes:
        annotations.append(f"axes=[{', '.join(tensor_type.axes)}]")
    if tensor_type.global_shape is not None:
        global_shape = "x".join(str(dim) for dim in tensor_type.global_shape)
        annotations.append(f"global_shape={global_shape}")

    return f"tensor<{', '.join([shape_and_dtype, *annotations])}>"


def print_function(function: Function) -> str:
    """Print one function, without a trailing newline.

    Args:
        function: Function to print.

    Returns:
        The function's text form.

    Raises:
        ValueError: If a value is used before it is defined, or defined twice.
            The printer does not invent names for such values, so a broken
            function cannot hide behind readable output.
    """

    # Names restart for every function, like local variables.
    namer = _Namer()

    arguments = ", ".join(
        f"{namer.define(argument, argument.name)}: {format_type(argument.type)}"
        for argument in function.operands
    )
    body = [_INDENT + _format_operation(op, namer) for op in function.operations]
    returns = ", ".join(namer.reference(value) for value in function.results)
    return_types = ", ".join(format_type(value.type) for value in function.results)

    lines = [
        f"func @{function.name}({arguments}) -> ({return_types}) {{",
        *body,
        # rstrip() keeps a function without returns as "return", not "return ".
        f"{_INDENT}return {returns}".rstrip(),
        "}",
    ]
    return "\n".join(lines)


def print_module(module: Module) -> str:
    """Print a module header followed by its functions.

    Sections are separated by one blank line and the text ends with a newline,
    so it can be written to a dump file unchanged.

    Args:
        module: Module to print.

    Returns:
        The module's text form.
    """

    sections = [f"module @{module.name}", *map(print_function, module.functions)]
    return "\n\n".join(sections) + "\n"


def _format_operation(op: Operation, namer: _Namer) -> str:
    """Format one op as ``%r = name(%a, %b) {attr = v} : type``.

    Args:
        op: Operation to format.
        namer: Names of the values defined so far in this function.

    Returns:
        The op's line without indentation.
    """

    # Operands are looked up before results are defined: an op can never use
    # its own result, and numbering then follows reading order.
    operands = ", ".join(namer.reference(value) for value in op.operands)
    text = f"{op.NAME}({operands})"

    # attributes() follows field declaration order, which is deterministic.
    attributes = op.attributes()
    if attributes:
        formatted = ", ".join(
            f"{name} = {_format_attribute(value)}" for name, value in attributes.items()
        )
        text += f" {{{formatted}}}"

    if not op.results:
        return text

    # A result's own name wins; otherwise the op's name_hint names it, so
    # LinearOp(..., name_hint="q") prints as %q.
    results = ", ".join(
        namer.define(result, result.name if result.name is not None else op.name_hint)
        for result in op.results
    )
    result_types = [format_type(result.type) for result in op.results]
    if len(result_types) == 1:
        type_text = result_types[0]
    else:
        type_text = f"({', '.join(result_types)})"
    return f"{results} = {text} : {type_text}"


def _format_attribute(value: object) -> str:
    """Format one attribute value deterministically.

    Only value types with a stable text form are accepted. A generic ``repr``
    could include a memory address, which would make dumps differ between runs.

    Args:
        value: Attribute value taken from ``Operation.attributes()``.

    Returns:
        The value's text form.

    Raises:
        TypeError: If the value has no deterministic text form yet.
    """

    # Order matters: bool is a subclass of int, and StrEnum/IntEnum members are
    # also str/int instances, so the more specific checks come first.
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, Enum):
        return str(value.value)
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        # repr() is the shortest text that reads back as the same float,
        # for example 1e-05.
        return repr(value)
    if isinstance(value, str):
        # json.dumps adds double quotes and escapes quotes or newlines inside.
        return json.dumps(value)
    if isinstance(value, (tuple, list)):
        return "[" + ", ".join(_format_attribute(item) for item in value) + "]"
    raise TypeError(
        f"cannot print attribute of type {type(value).__name__}; "
        "add a deterministic format for it in printer._format_attribute"
    )


class _Namer:
    """Assign unique ``%`` names to the values of one function.

    Names come from hints when available. Unnamed values are numbered 0, 1,
    2, ... in definition order. Every candidate goes through the same
    uniqueness check, so two values hinted ``x`` print as ``%x`` and ``%x_1``.
    """

    def __init__(self) -> None:
        """Start with no names defined."""

        # Value hashes by identity, so same-typed values are separate keys.
        self._names: dict[Value, str] = {}
        self._used: set[str] = set()
        self._next_number = 0

    def define(self, value: Value, hint: str | None) -> str:
        """Name a value at its definition.

        Args:
            value: Function argument or op result being defined.
            hint: Preferred name, or None to use the next free number.

        Returns:
            The value's name, including the leading ``%``.

        Raises:
            ValueError: If the value was already defined in this function.
        """

        if value in self._names:
            raise ValueError(f"value %{self._names[value]} is defined twice")

        if hint is None:
            # Skip numbers that an earlier hint already claimed, for example
            # a value explicitly named "0".
            while str(self._next_number) in self._used:
                self._next_number += 1
            name = str(self._next_number)
            self._next_number += 1
        else:
            name = hint
            suffix = 1
            while name in self._used:
                name = f"{hint}_{suffix}"
                suffix += 1

        self._used.add(name)
        self._names[value] = name
        return f"%{name}"

    def reference(self, value: Value) -> str:
        """Return the name of a value that has already been defined.

        Args:
            value: Value used as an operand or return value.

        Returns:
            The value's name, including the leading ``%``.

        Raises:
            ValueError: If the value has not been defined in this function.
        """

        if value not in self._names:
            label = value.name if value.name is not None else "<unnamed>"
            raise ValueError(
                f"value {label} of type {format_type(value.type)} is used before "
                "it is defined in this function"
            )
        return f"%{self._names[value]}"
