"""Small, dependency-free SSA IR"""

from __future__ import annotations

from dataclasses import  dataclass, field, fields, replace
from typing import ClassVar, Self, Iterable
from abc import ABC, abstractmethod

from .types import TensorType

@dataclass(frozen = True, eq = False, )
class Value:

    type: TensorType
    name: str | None = None


@dataclass(frozen = True, eq = False, slots = True)
class Operation(ABC):

    NAME: ClassVar[str]
    NUM_OPERANDS: ClassVar[int]

    name_hint: str | None = field(default = None, kw_only = True)
    operands: tuple[Value, ...]
    results: tuple[Value, ...] = field(init = False)

    def __post_init__(self) -> None:
        # Convert before anything else: a generator has no len(), and a list
        # could still be changed by the caller after construction.
        object.__setattr__(self, "operands", tuple(self.operands))

        if len(self.operands) != self.NUM_OPERANDS:
            raise ValueError(
                f"{self.NAME} expects {self.NUM_OPERANDS} operands, "
                f"got {len(self.operands)}"
            )

        result_types = self.infer_result_types()
        results = tuple(Value(type = t) for t in result_types)
        object.__setattr__(self, "results", results)

    # Infer the output type entirely by the operation
    @abstractmethod
    def infer_result_types(self) -> tuple[TensorType, ...]: 
        ...

    # Getting attributes for printing
    def attributes(self) -> dict[str, object]:
        return {
            f.name: getattr(self, f.name)
            for f in fields(self)
            if f.name not in {"name_hint", "operands", "results"}
        }


    def with_operands(self, new_operands: Iterable[Value]) -> Self:
        return replace(self, operands = tuple(new_operands))

@dataclass(frozen = True, eq = False, slots = True)
class Function:
    """One straight-line SSA function"""

    name: str
    operands: tuple[Value, ...]
    operations: tuple[Operation, ...]
    results: tuple[Value, ...]

    def __post_init__(self) -> None:
        operands = tuple(self.operands)
        object.__setattr__(self, "operands", operands)
        
        operations = tuple(self.operations)
        object.__setattr__(self, "operations", operations)

        results = tuple(self.results)
        object.__setattr__(self, "results", results)

    def defining_op(self, value: Value) -> Operation | None:
        for op in self.operations:
            if value in op.results:
                return op

        return None

    def users(self, value: Value) -> tuple[Operation, ...]:
        return tuple(
            op
            for op in self.operations
            if value in op.operands
        )

    def verify(self) -> None:
        # Add arguments one at a time: set(self.operands) would silently merge
        # an argument that is listed twice.
        available: set[Value] = set()
        for argument in self.operands:
            if argument in available:
                raise ValueError(
                    f"function {self.name} lists argument %{argument.name} twice"
                )
            available.add(argument)

        for op in self.operations:
            # NAME says what kind of op failed; name_hint, when present, says which one.
            label = op.NAME if op.name_hint is None else f"{op.NAME} ({op.name_hint})"
            missing = [operand.name for operand in op.operands if operand not in available]
            if missing:
                raise ValueError(f"{label} uses unavailable values: {missing}")
            for result in op.results:
                if result in available:
                    raise ValueError(f"{label} defines duplicate SSA value %{result.name}")
                available.add(result)
        missing_outputs = [value.name for value in self.results if value not in available]
        if missing_outputs:
            raise ValueError(f"function returns unavailable values: {missing_outputs}")

@dataclass(frozen = True)
class Module:
    """A named collection of functions."""

    name: str
    functions: tuple[Function, ... ]

    def __post_init__(self) -> None:
        object.__setattr__(self, "functions", tuple(self.functions))

    def get_function(self, name: str = "main") -> Function:
        for func in self.functions:
            if func.name == name:
                return func
        raise KeyError(f"function not found: {name}")

    def verify(self) -> None:
        # get_function() looks functions up by name, so a repeated name would
        # make the lookup silently return only the first one.
        names: set[str] = set()
        for func in self.functions:
            if func.name in names:
                raise ValueError(f"module @{self.name} defines function @{func.name} twice")
            names.add(func.name)

        for func in self.functions:
            try:
                func.verify()
            except ValueError as e:
                raise ValueError(f"function @{func.name}: {e}") from e


def rebuild_function(func: Function, new_arguments: Iterable[Value]) -> Function:
    """Rebuild ``func`` on top of new argument values.

    Argument i of ``func`` is replaced by ``new_arguments[i]``. Every op is
    then reconstructed with ``with_operands``, which runs type inference
    again. A change to an argument's type (a TP shard's smaller shape, new
    role or axes annotations) therefore flows through the whole function.

    Args:
        func: Function to rebuild; it is not modified.
        new_arguments: One replacement value per argument of ``func``.

    Returns:
        The rebuilt function, sharing no op or result with ``func``.

    Raises:
        ValueError: If the number of new arguments differs from ``func``'s,
            or an op rejects the new operand types.
    """

    new_operands = tuple(new_arguments)
    if len(new_operands) != len(func.operands):
        raise ValueError(
            f"function {func.name} has {len(func.operands)} arguments, "
            f"got {len(new_operands)} replacements"
        )

    # Maps every old value (argument or op result) to its replacement.
    new_map: dict[Value, Value] = dict(zip(func.operands, new_operands, strict=True))

    new_ops: list[Operation] = []
    for op in func.operations:
        new_op = op.with_operands(
            new_operands = tuple(
                new_map[operand] 
                for operand 
                in op.operands),
        )

        new_ops.append(new_op)

        for old_result, new_result in zip(
            op.results,
            new_op.results,
            strict=True,
        ):
            new_map[old_result] = new_result

    return Function(
        name = func.name,
        operands = new_operands,
        operations = tuple(new_ops),
        # The returns are chosen by the original function, not by the ops:
        # look up the copy of each returned value instead of returning every
        # op result.
        results = tuple(new_map[value] for value in func.results),
    )


def clone_function(func: Function) -> Function:
    """Copy ``func`` with fresh values; types and names stay the same.

    Args:
        func: Function to copy; it is not modified.

    Returns:
        A structurally identical function sharing no value with ``func``.
    """

    return rebuild_function(
        func,
        (Value(type = operand.type, name = operand.name) for operand in func.operands),
    )
