"""Tests for tensor parallelism: ParallelPlan and ApplyTensorParallel."""

from __future__ import annotations

import unittest

from vllm_cent.ir.annotate import AnnotateRolesAndAxes, TensorAnnotation
from vllm_cent.ir.base import Function, Module, Value
from vllm_cent.ir.passes import PassManager
from vllm_cent.ir.printer import print_module
from vllm_cent.ir.semantic import LinearOp
from vllm_cent.ir.types import (
    DType,
    Replicate,
    Shard,
    TensorRole,
    TensorType,
    packed_axis,
)
from vllm_cent.parallel import ApplyTensorParallel, ParallelPlan

# A tiny attention projection pair: hidden 16 = 4 query heads x head_dim 4.
# The attention between q_proj and o_proj is left out; it works per head, so
# in the real block o_proj's input has the same placement as q_proj's output.
HIDDEN = 16
HEADS = 4
Q_PROJ_OUT = packed_axis("q_head", "head_dim")


def _type(*shape: int) -> TensorType:
    """Create an unannotated fp16 type, as the importer produces.

    Args:
        *shape: Tensor dimensions.

    Returns:
        The tensor type.
    """

    return TensorType(shape=shape, dtype=DType.FP16)


def _q_o_module() -> Module:
    """Build ``o_proj(q_proj(x, wq), wo)`` without annotations.

    Returns:
        Module ``toy`` with one function ``main``.
    """

    x = Value(type=_type(1, HIDDEN), name="x")
    wq = Value(type=_type(HIDDEN, HIDDEN), name="wq")
    wo = Value(type=_type(HIDDEN, HIDDEN), name="wo")
    q_proj = LinearOp(operands=(x, wq), name_hint="q_proj")
    o_proj = LinearOp(operands=(q_proj.results[0], wo), name_hint="o_proj")
    function = Function(
        name="main",
        operands=(x, wq, wo),
        operations=(q_proj, o_proj),
        results=(o_proj.results[0],),
    )
    return Module(name="toy", functions=(function,))


def _q_o_annotations() -> dict[str, TensorAnnotation]:
    """Annotate the q/o module like the Llama block.

    Wq maps hidden to the packed heads (heads on dim 0); Wo maps the heads
    back to hidden (heads on dim 1).

    Returns:
        A new annotation mapping.
    """

    return {
        "x": TensorAnnotation(TensorRole.ACTIVATION, ("batch", "hidden")),
        "wq": TensorAnnotation(TensorRole.WEIGHT, (Q_PROJ_OUT, "hidden")),
        "wo": TensorAnnotation(TensorRole.WEIGHT, ("hidden", Q_PROJ_OUT)),
    }


def _annotated() -> Module:
    """Return the q/o module after AnnotateRolesAndAxes only."""

    return PassManager((AnnotateRolesAndAxes(_q_o_annotations()),)).run(_q_o_module())


def _sharded(plan: ParallelPlan) -> Module:
    """Annotate the q/o module and apply ``plan`` through a verifying manager.

    Args:
        plan: Plan to apply.

    Returns:
        The sharded module.
    """

    return PassManager(
        (AnnotateRolesAndAxes(_q_o_annotations()), ApplyTensorParallel(plan))
    ).run(_q_o_module())


def _head_plan(tp: int = 2) -> ParallelPlan:
    """Return the Megatron-style plan: split the query heads.

    Args:
        tp: Number of ranks.

    Returns:
        The plan.
    """

    plan = ParallelPlan(tp=tp)
    plan.split("q_head", size=HEADS)
    return plan


def _single_projection(out_size: int) -> Module:
    """Build an annotated ``proj(x, w)`` whose output axis is the plain ``out``.

    Args:
        out_size: Length of the ``out`` dimension.

    Returns:
        The module, already annotated.
    """

    x = Value(type=_type(1, HIDDEN), name="x")
    w = Value(type=_type(out_size, HIDDEN), name="w")
    proj = LinearOp(operands=(x, w), name_hint="proj")
    function = Function(
        name="main", operands=(x, w), operations=(proj,), results=(proj.results[0],)
    )
    annotations = {
        "x": TensorAnnotation(TensorRole.ACTIVATION, ("batch", "hidden")),
        "w": TensorAnnotation(TensorRole.WEIGHT, ("out", "hidden")),
    }
    return PassManager((AnnotateRolesAndAxes(annotations),)).run(
        Module(name="toy", functions=(function,))
    )


class ParallelPlanTest(unittest.TestCase):
    """The builder records rules and exceptions and rejects conflicts early."""

    def test_records_rules_sizes_and_exceptions(self) -> None:
        """Rules keep their order; exceptions map names to placements."""

        plan = ParallelPlan(tp=8)
        plan.split("q_head", size=64)
        plan.split("intermediate")
        plan.replicate("k_proj.weight", "v_proj.weight")
        plan.split("hidden", only=("lm_head.weight",))

        self.assertEqual(plan.tp, 8)
        self.assertEqual(plan.split_axes, ("q_head", "intermediate"))
        self.assertEqual(plan.axis_size("q_head"), 64)
        self.assertIsNone(plan.axis_size("intermediate"))
        self.assertEqual(
            dict(plan.exceptions),
            {
                "k_proj.weight": Replicate(),
                "v_proj.weight": Replicate(),
                "lm_head.weight": Shard("hidden"),
            },
        )

    def test_exceptions_are_read_only(self) -> None:
        """Only split() and replicate() may add exceptions."""

        plan = ParallelPlan(tp=2)
        with self.assertRaises(TypeError):
            plan.exceptions["wq"] = Replicate()  # type: ignore[index]

    def test_only_accepts_a_generator(self) -> None:
        """only= is read once, so a one-shot iterable works."""

        plan = ParallelPlan(tp=2)
        plan.split("hidden", only=(name for name in ("wq", "wo")))
        self.assertEqual(set(plan.exceptions), {"wq", "wo"})

    def test_rejects_invalid_calls(self) -> None:
        """Every conflict is reported by the call that causes it."""

        def zero_ranks(_: ParallelPlan) -> None:
            """Build a plan with no ranks."""

            ParallelPlan(tp=0)

        def uneven_heads(_: ParallelPlan) -> None:
            """8 KV heads over 16 ranks would give each half a head."""

            ParallelPlan(tp=16).split("kv_head", size=8)

        def conflicting_size(plan: ParallelPlan) -> None:
            """One axis cannot have two sizes."""

            plan.split("q_head", size=64)
            plan.split("q_head", size=32, only=("wq",))

        def split_twice(plan: ParallelPlan) -> None:
            """The same rule written twice."""

            plan.split("q_head")
            plan.split("q_head")

        def redundant_only(plan: ParallelPlan) -> None:
            """An exception that repeats a rule."""

            plan.split("hidden")
            plan.split("hidden", only=("wq",))

        def empty_only(plan: ParallelPlan) -> None:
            """An exception that names no tensor."""

            plan.split("hidden", only=())

        def empty_replicate(plan: ParallelPlan) -> None:
            """replicate() with no names."""

            plan.replicate()

        def two_exceptions(plan: ParallelPlan) -> None:
            """Two exceptions for one tensor would silently conflict."""

            plan.replicate("wq")
            plan.split("hidden", only=("wq",))

        for build in (
            zero_ranks,
            uneven_heads,
            conflicting_size,
            split_twice,
            redundant_only,
            empty_only,
            empty_replicate,
            two_exceptions,
        ):
            with self.subTest(case=build.__name__), self.assertRaises(ValueError):
                build(ParallelPlan(tp=8))


class ApplyTensorParallelTest(unittest.TestCase):
    """Arguments are sharded, placements derived, all_reduces inserted."""

    def test_q_proj_o_proj_golden(self) -> None:
        """One rule splits Wq by rows and Wo by columns; o_proj is reduced.

        With 4 heads over 2 ranks, each rank keeps 2 heads x 4 = 8 of the 16
        q_head*head_dim entries: Wq is 8x16 (dim 0 split), Wo is 16x8 (dim 1
        split). q_proj's output is this rank's 8 entries. o_proj sums over
        only those 8, so its 1x16 result is partial, and an all_reduce
        completes it. No exception was needed for Wo.
        """

        expected = (
            "module @toy\n"
            "\n"
            "func @main("
            "%x: tensor<1x16xfp16, axes=[batch, hidden]>, "
            "%wq: tensor<8x16xfp16, role=weight, axes=[q_head*head_dim, hidden], "
            "global_shape=16x16, placement=shard(q_head)>, "
            "%wo: tensor<16x8xfp16, role=weight, axes=[hidden, q_head*head_dim], "
            "global_shape=16x16, placement=shard(q_head)>) "
            "-> (tensor<1x16xfp16, axes=[batch, hidden]>) {\n"
            "  %q_proj = aloi.linear(%x, %wq) : tensor<1x8xfp16, "
            "axes=[batch, q_head*head_dim], global_shape=1x16, "
            "placement=shard(q_head)>\n"
            "  %o_proj = aloi.linear(%q_proj, %wo) : tensor<1x16xfp16, "
            "axes=[batch, hidden], placement=partial>\n"
            "  %o_proj_1 = aloi.all_reduce(%o_proj) : tensor<1x16xfp16, "
            "axes=[batch, hidden]>\n"
            "  return %o_proj_1\n"
            "}\n"
        )
        self.assertEqual(print_module(_sharded(_head_plan())), expected)

    def test_replicating_both_weights_needs_no_communication(self) -> None:
        """Exceptions override the rule; the IR equals the unsharded one."""

        plan = _head_plan()
        plan.replicate("wq", "wo")
        self.assertEqual(print_module(_sharded(plan)), print_module(_annotated()))

    def test_inconsistent_exceptions_are_rejected(self) -> None:
        """An exception cannot produce a program the ops do not support."""

        replicate_wo_only = _head_plan()
        replicate_wo_only.replicate("wo")
        # x whole, but Wq split along its input (hidden): x would first need
        # a local slice, which no op provides yet.
        split_wq_input = ParallelPlan(tp=2)
        split_wq_input.split("hidden", only=("wq",))
        for name, plan in {
            "split heads into whole Wo": replicate_wo_only,
            "whole x into split Wq input": split_wq_input,
        }.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, "unsupported placements"):
                    _sharded(plan)

    def test_plain_axis_needs_no_size(self) -> None:
        """A plain axis's size is its dimension length: 8 rows over 4 ranks."""

        plan = ParallelPlan(tp=4)
        plan.split("out")
        module = PassManager((ApplyTensorParallel(plan),)).run(_single_projection(8))
        (w,) = (arg for arg in module.get_function("main").operands if arg.name == "w")
        self.assertEqual(w.type.shape, (2, HIDDEN))
        self.assertEqual(
            module.get_function("main").results[0].type.placement, Shard("out")
        )

    def test_plan_errors_found_in_the_ir(self) -> None:
        """Errors that need the IR are reported with the tensor or axis."""

        no_size = ParallelPlan(tp=2)
        no_size.split("q_head")
        typo_axis = ParallelPlan(tp=2)
        typo_axis.split("q_haed", size=HEADS)
        typo_name = _head_plan()
        typo_name.replicate("wk")
        # 16 entries cannot be 6 whole heads.
        partial_heads = ParallelPlan(tp=2)
        partial_heads.split("q_head", size=6)
        axis_not_on_tensor = ParallelPlan(tp=2)
        axis_not_on_tensor.split("kv_head", size=2, only=("wq",))
        # Wq's dims are q_head*head_dim and hidden: both would be split.
        two_dims = _head_plan()
        two_dims.split("hidden")

        cases = {
            "packed axis without size": (no_size, r"plan\.split\('q_head', size=\.\.\.\)"),
            "misspelled axis": (typo_axis, r"split axes match no tensor: \['q_haed'\]"),
            "misspelled tensor": (typo_name, r"exceptions name unknown tensors: \['wk'\]"),
            "dimension not whole heads": (partial_heads, "not a whole number of q_head"),
            "axis missing on tensor": (axis_not_on_tensor, "%wq: axis 'kv_head'"),
            "two split dimensions": (two_dims, "only one dimension can be split"),
        }
        for name, (plan, message) in cases.items():
            with self.subTest(case=name):
                with self.assertRaisesRegex(ValueError, message):
                    _sharded(plan)

    def test_plain_axis_must_divide(self) -> None:
        """6 rows cannot be split over 4 ranks."""

        plan = ParallelPlan(tp=4)
        plan.split("out")
        with self.assertRaisesRegex(ValueError, "cannot be split over tp=4"):
            PassManager((ApplyTensorParallel(plan),)).run(_single_projection(6))

    def test_requires_annotated_ir(self) -> None:
        """Rules name axes, so the axes must exist."""

        with self.assertRaisesRegex(ValueError, "run AnnotateRolesAndAxes first"):
            PassManager((ApplyTensorParallel(_head_plan()),)).run(_q_o_module())

    def test_single_rank_changes_nothing(self) -> None:
        """With tp=1 one rank owns everything."""

        self.assertEqual(
            print_module(_sharded(_head_plan(tp=1))), print_module(_annotated())
        )

    def test_pass_keeps_its_copy_of_the_plan(self) -> None:
        """Editing the plan after creating the pass does not change the pass."""

        plan = _head_plan()
        apply = ApplyTensorParallel(plan)
        plan.replicate("wq", "wo")
        sharded = PassManager((apply,)).run(_annotated())
        self.assertIn("aloi.all_reduce", print_module(sharded))

    def test_input_module_is_unchanged(self) -> None:
        """The pass is pure."""

        module = _annotated()
        before = print_module(module)
        PassManager((ApplyTensorParallel(_head_plan()),)).run(module)
        self.assertEqual(print_module(module), before)


if __name__ == "__main__":
    unittest.main()
