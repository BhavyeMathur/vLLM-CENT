"""Infrastructure for implementing IR passes"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Callable
from abc import ABC, abstractmethod

from .base import Function, Module, clone_function

class Pass(ABC):
    """Abstract class for IR passes"""

    NAME: ClassVar[str]

    @abstractmethod
    def run(self, module: Module) -> Module:
        ...


class FunctionPass(Pass):
    """A pass that rewrites every function independently.

    Most passes (annotation, sharding propagation, collective insertion) look
    at one function at a time. Subclasses implement run_on_function; this
    class rebuilds the module around the rewritten functions.
    """

    @abstractmethod
    def run_on_function(self, function: Function) -> Function:
        ...

    def run(self, module: Module) -> Module:
        return Module(
            name = module.name,
            functions = tuple(self.run_on_function(func) for func in module.functions),
        )


class ClonePass(FunctionPass):
    """Copy every function with fresh values; the IR itself is unchanged."""

    NAME = "clone"

    def run_on_function(self, function: Function) -> Function:
        return clone_function(function)


@dataclass(frozen = True)
class PassManager():
    passes: tuple[Pass, ...]
    verify_each: bool = True
    dump: Callable[[str, Module], None] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "passes", tuple(self.passes))

    def run(self, module: Module) -> Module:
        # Verify the input first, so broken frontend output is reported as
        # such instead of being blamed on the first pass.
        if self.verify_each:
            try:
                module.verify()
            except ValueError as e:
                raise ValueError(f"IR invalid before first pass: {e}") from e

        for p in self.passes:
            new_module = p.run(module)

            # The most common pass bug is a missing return, which yields None
            # and would otherwise crash obscurely inside the next pass.
            if not isinstance(new_module, Module):
                raise TypeError(
                    f"pass {p.NAME} returned {type(new_module).__name__}, "
                    "expected Module"
                )

            if self.verify_each:
                try:
                    new_module.verify()
                except ValueError as e:
                    raise ValueError(f"IR invalid after pass {p.NAME}: {e}") from e

            # Dump only after verification: the printer itself rejects broken
            # IR, and a dump file should always hold valid IR.
            if self.dump is not None:
                self.dump(p.NAME, new_module)

            module = new_module

        return module
