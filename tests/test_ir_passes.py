"""Tests for the pass infrastructure: Pass, PassManager, FunctionPass, ClonePass."""

from __future__ import annotations

import importlib
import unittest
from dataclasses import dataclass

from vllm_cent.ir.base import Function, Module, Operation, Value
from vllm_cent.ir.passes import Pass, PassManager
from vllm_cent.ir.printer import print_module
from vllm_cent.ir.types import DType, TensorType

VECTOR = TensorType(shape=(1, 4), dtype=DType.FP16)


@dataclass(frozen=True, eq=False)
class _Identity(Operation):
    """One operand; the result has the operand's type."""

    NAME = "test.identity"
    NUM_OPERANDS = 1

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Pass the operand type through unchanged."""

        return (self.operands[0].type,)


def _function(name: str = "main") -> Function:
    """Build the valid chain ``x -> first -> second``.

    Args:
        name: Function name.

    Returns:
        A function returning ``second``'s result.
    """

    x = Value(type=VECTOR, name="x")
    first = _Identity(operands=(x,))
    second = _Identity(operands=(first.results[0],))
    return Function(
        name=name,
        operands=(x,),
        operations=(first, second),
        results=(second.results[0],),
    )


def _module() -> Module:
    """Build a valid module with two functions.

    Returns:
        Module ``toy`` containing ``main`` and ``helper``.
    """

    return Module(name="toy", functions=(_function("main"), _function("helper")))


def _broken_module(module: Module) -> Module:
    """Drop every op but keep the returns, so returns become undefined.

    Args:
        module: Valid module to break.

    Returns:
        A module that fails verification.
    """

    return Module(
        name=module.name,
        functions=tuple(
            Function(
                name=function.name,
                operands=function.operands,
                operations=(),
                results=function.results,
            )
            for function in module.functions
        ),
    )


# ---------------------------------------------------------------------------
# Test-only passes
# ---------------------------------------------------------------------------


class _RecordingPass(Pass):
    """Append NAME to a shared log and return the module unchanged."""

    NAME = "record"

    def __init__(self, log: list[str]) -> None:
        """Remember the log shared by several passes.

        Args:
            log: List that receives this pass's NAME when it runs.
        """

        self.log = log

    def run(self, module: Module) -> Module:
        """Record the call; returning the input unchanged is a valid pure pass."""

        self.log.append(self.NAME)
        return module


class _PassA(_RecordingPass):
    """First recording pass."""

    NAME = "a"


class _PassB(_RecordingPass):
    """Second recording pass."""

    NAME = "b"


class _BrokenPass(Pass):
    """Return structurally invalid IR."""

    NAME = "broken"

    def run(self, module: Module) -> Module:
        """Drop all ops while keeping the returns."""

        return _broken_module(module)


class _ForgetsReturnPass(Pass):
    """The most common pass bug: no return statement."""

    NAME = "forgets_return"

    def run(self, module: Module) -> Module:
        """Return None instead of a module."""

        return None  # type: ignore[return-value]


class PassTest(unittest.TestCase):
    """Pass is an abstract interface."""

    def test_subclass_without_run_cannot_be_instantiated(self) -> None:
        """Forgetting run() fails when the pass is created."""

        class Incomplete(Pass):
            NAME = "incomplete"

        with self.assertRaises(TypeError):
            Incomplete()  # type: ignore[abstract]


class PassManagerTest(unittest.TestCase):
    """Ordering, verification, error context, and dumping."""

    def test_runs_passes_in_order(self) -> None:
        """Passes run in list order on the previous pass's output."""

        log: list[str] = []
        module = _module()
        result = PassManager(passes=(_PassA(log), _PassB(log))).run(module)
        self.assertEqual(log, ["a", "b"])
        self.assertIs(result, module)

    def test_empty_pipeline_returns_input(self) -> None:
        """With no passes the (verified) input comes back unchanged."""

        module = _module()
        self.assertIs(PassManager(passes=()).run(module), module)

    def test_passes_are_stored_as_tuple(self) -> None:
        """A list of passes is copied into an immutable tuple."""

        manager = PassManager(passes=[_PassA([])])  # type: ignore[arg-type]
        self.assertIsInstance(manager.passes, tuple)

    def test_invalid_input_fails_before_first_pass(self) -> None:
        """Broken frontend output is reported before any pass runs."""

        log: list[str] = []
        manager = PassManager(passes=(_PassA(log),))
        with self.assertRaisesRegex(ValueError, "before first pass"):
            manager.run(_broken_module(_module()))
        self.assertEqual(log, [])

    def test_error_names_the_pass_that_broke_the_ir(self) -> None:
        """The message says which pass produced invalid IR."""

        manager = PassManager(passes=(_PassA([]), _BrokenPass()))
        with self.assertRaisesRegex(ValueError, "after pass broken"):
            manager.run(_module())

    def test_error_keeps_original_cause(self) -> None:
        """``raise ... from e`` keeps the verifier's own error as the cause."""

        manager = PassManager(passes=(_BrokenPass(),))
        with self.assertRaises(ValueError) as caught:
            manager.run(_module())
        self.assertIsInstance(caught.exception.__cause__, ValueError)

    def test_error_names_the_broken_function(self) -> None:
        """Module.verify adds the function name to the verifier's message."""

        manager = PassManager(passes=(_BrokenPass(),))
        with self.assertRaisesRegex(ValueError, "@main"):
            manager.run(_module())

    def test_rejects_pass_that_returns_no_module(self) -> None:
        """A pass that forgets to return is named in a TypeError."""

        manager = PassManager(passes=(_ForgetsReturnPass(),))
        with self.assertRaisesRegex(TypeError, "forgets_return"):
            manager.run(_module())

    def test_verify_each_false_skips_all_verification(self) -> None:
        """Turning verification off skips both input and per-pass checks."""

        broken = _broken_module(_module())
        manager = PassManager(passes=(_BrokenPass(),), verify_each=False)
        result = manager.run(broken)
        self.assertEqual(result.functions[0].operations, ())

    def test_dump_receives_each_output_in_order(self) -> None:
        """dump(name, module) is called once per pass, after it runs."""

        calls: list[tuple[str, Module]] = []
        module = _module()
        manager = PassManager(
            passes=(_PassA([]), _PassB([])),
            dump=lambda name, output: calls.append((name, output)),
        )
        manager.run(module)
        self.assertEqual(calls, [("a", module), ("b", module)])

    def test_invalid_output_is_not_dumped(self) -> None:
        """Verification happens before dumping, so broken IR is never dumped."""

        calls: list[str] = []
        manager = PassManager(
            passes=(_PassA([]), _BrokenPass()),
            dump=lambda name, output: calls.append(name),
        )
        with self.assertRaises(ValueError):
            manager.run(_module())
        self.assertEqual(calls, ["a"])


def _import_pass_class(name: str) -> type[Pass]:
    """Import a pass class that may not be implemented yet.

    Importing inside setUpClass, instead of at module level, keeps a missing
    class from breaking every other test in this file.

    Args:
        name: Class name in ``vllm_cent.ir.passes``.

    Returns:
        The class.
    """

    return getattr(importlib.import_module("vllm_cent.ir.passes"), name)


class FunctionPassTest(unittest.TestCase):
    """FunctionPass applies run_on_function to every function."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load FunctionPass."""

        cls.function_pass = _import_pass_class("FunctionPass")

    def test_maps_every_function_and_keeps_module_name(self) -> None:
        """Each function is processed once, in order, under the same module."""

        seen: list[str] = []

        class Rename(self.function_pass):  # type: ignore[name-defined, misc]
            NAME = "rename"

            def run_on_function(self, function: Function) -> Function:
                seen.append(function.name)
                return Function(
                    name=function.name + "_renamed",
                    operands=function.operands,
                    operations=function.operations,
                    results=function.results,
                )

        result = Rename().run(_module())
        self.assertEqual(seen, ["main", "helper"])
        self.assertEqual(result.name, "toy")
        self.assertEqual(
            [function.name for function in result.functions],
            ["main_renamed", "helper_renamed"],
        )


class ClonePassTest(unittest.TestCase):
    """ClonePass copies every function with fresh values."""

    @classmethod
    def setUpClass(cls) -> None:
        """Load ClonePass."""

        cls.clone_pass = _import_pass_class("ClonePass")

    def test_prints_identically(self) -> None:
        """Same structure and names, so the text form is unchanged."""

        module = _module()
        result = self.clone_pass().run(module)
        self.assertEqual(print_module(result), print_module(module))

    def test_shares_no_values_with_input(self) -> None:
        """Every argument and op result in the output is a new object."""

        def value_ids(module: Module) -> set[int]:
            """Collect the identities of all defined values."""

            ids: set[int] = set()
            for function in module.functions:
                ids.update(id(value) for value in function.operands)
                for op in function.operations:
                    ids.update(id(value) for value in op.results)
            return ids

        module = _module()
        result = self.clone_pass().run(module)
        self.assertEqual(value_ids(module) & value_ids(result), set())

    def test_leaves_input_unchanged(self) -> None:
        """The input module keeps its own function objects."""

        module = _module()
        functions = module.functions
        self.clone_pass().run(module)
        self.assertIs(module.functions, functions)

    def test_runs_under_pass_manager(self) -> None:
        """The cloned module passes verification inside a pipeline."""

        calls: list[str] = []
        manager = PassManager(
            passes=(self.clone_pass(),),
            dump=lambda name, output: calls.append(name),
        )
        manager.run(_module())
        self.assertEqual(calls, [self.clone_pass.NAME])


if __name__ == "__main__":
    unittest.main()
