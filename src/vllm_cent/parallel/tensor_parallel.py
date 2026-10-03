"""Apply a ParallelPlan: shard arguments, derive placements, add all_reduces.

The output is SPMD IR: one function that every rank runs on its own data.
Shapes become local shapes; a sharded tensor also records its global_shape
and placement. Three steps happen in one rebuild of the function:

1. Seed: each argument gets a placement from the plan, and its shape shrinks
   along the split dimension.
2. Propagate: rebuilding re-runs every op's type inference, and each op
   derives its result placement from its operands' (see LinearOp).
3. Insert collectives: an op whose result is a partial sum is followed by an
   all_reduce right away.

The steps share one rebuild because a partial sum must never reach another
op: if the IR could hold one between passes, every op would have to accept
an operand it cannot use.
"""

from __future__ import annotations

import copy
from dataclasses import replace

from ..ir.base import Function, Module, Operation, Value, rebuild_function
from ..ir.passes import FunctionPass
from ..ir.semantic import AllReduceOp
from ..ir.types import Partial, Replicate, Shard, outermost_axis
from .plan import ParallelPlan

__all__ = ["ApplyTensorParallel"]


class ApplyTensorParallel(FunctionPass):
    """Turn annotated Semantic IR into the IR one tensor-parallel rank runs.

    The input must already carry axes (run AnnotateRolesAndAxes first): the
    plan's rules name axes, not dimensions.
    """

    NAME = "apply-tensor-parallel"

    def __init__(self, plan: ParallelPlan) -> None:
        """Create the pass.

        Args:
            plan: Ownership rules. The plan is copied, so later split() or
                replicate() calls do not change this pass.
        """

        self._plan = copy.deepcopy(plan)

    def run(self, module: Module) -> Module:
        """Check the plan against the whole module, then shard each function.

        These checks look at all functions at once: a name or axis unused by
        one function may belong to another.

        Args:
            module: Annotated module; it is not modified.

        Returns:
            The sharded module.

        Raises:
            ValueError: If an argument has no axes, an exception names no
                argument, or a rule's axis appears on no argument.
        """

        arguments = [
            argument for function in module.functions for argument in function.operands
        ]

        # Check annotations first. Without axes, every rule would look unused
        # and the error would point at the plan instead of the missing pass.
        unannotated = [
            argument.name
            for argument in arguments
            if len(argument.type.axes) != argument.type.rank
        ]
        if unannotated:
            raise ValueError(
                f"arguments {unannotated} have no axes; "
                "run AnnotateRolesAndAxes first"
            )

        # A misspelled tensor name would otherwise leave its exception unused.
        names = {argument.name for argument in arguments}
        unknown = sorted(set(self._plan.exceptions) - names)
        if unknown:
            raise ValueError(f"exceptions name unknown tensors: {unknown}")

        # A misspelled axis would otherwise split nothing.
        argument_axes = {
            outermost_axis(axis_name)
            for argument in arguments
            for axis_name in argument.type.axes
        }
        unused = [axis for axis in self._plan.split_axes if axis not in argument_axes]
        if unused:
            raise ValueError(f"split axes match no tensor: {unused}")

        return super().run(module)

    def run_on_function(self, function: Function) -> Function:
        """Shard one function's arguments and rebuild it.

        Args:
            function: Annotated function; it is not modified.

        Returns:
            The function one rank runs.

        Raises:
            ValueError: If an argument cannot be split as planned, or an op
                rejects its operands' placements.
        """

        new_arguments = [self._shard_argument(argument) for argument in function.operands]
        return rebuild_function(function, new_arguments, rewrite=_insert_all_reduce)

    def _placement_for(self, argument: Value) -> Replicate | Shard:
        """Decide which part of ``argument`` each rank owns.

        Args:
            argument: Annotated argument.

        Returns:
            The exception for this argument if there is one, otherwise the
            placement the rules give.

        Raises:
            ValueError: If the rules split more than one of its dimensions;
                one tensor-parallel group can split only one.
        """

        if argument.name is not None:
            exception = self._plan.exceptions.get(argument.name)
            if exception is not None:
                return exception

        split_axes = [
            outermost_axis(axis_name)
            for axis_name in argument.type.axes
            if outermost_axis(axis_name) in self._plan.split_axes
        ]
        if not split_axes:
            return Replicate()
        if len(split_axes) > 1:
            raise ValueError(
                f"%{argument.name}: rules split several of its axes {split_axes}; "
                "only one dimension can be split"
            )
        return Shard(split_axes[0])

    def _shard_argument(self, argument: Value) -> Value:
        """Return the value one rank holds for ``argument``.

        Args:
            argument: Annotated argument.

        Returns:
            A new value. A sharded one has the local shape, the original
            shape as global_shape, and a Shard placement.

        Raises:
            ValueError: If the planned axis is not on the argument or the
                split dimension does not divide evenly.
        """

        argument_type = argument.type
        placement = self._placement_for(argument)
        # With one rank, every rank already owns everything.
        if isinstance(placement, Replicate) or self._plan.tp == 1:
            return Value(type=argument_type, name=argument.name)

        try:
            dim = argument_type.dim_of(placement.axis)
        except ValueError as error:
            raise ValueError(f"%{argument.name}: {error}") from error
        self._check_divisible(argument, dim, placement.axis)

        local_shape = list(argument_type.shape)
        local_shape[dim] //= self._plan.tp
        # replace() re-runs TensorType's checks on the new combination of
        # shape, global_shape and placement.
        sharded_type = replace(
            argument_type,
            shape=tuple(local_shape),
            global_shape=argument_type.shape,
            placement=placement,
        )
        return Value(type=sharded_type, name=argument.name)

    def _check_divisible(self, argument: Value, dim: int, axis: str) -> None:
        """Check that splitting ``axis`` gives every rank a whole share.

        For a plain dimension the axis size is its length. For a packed one
        such as q_head*head_dim of length 8192, the plan supplies the head
        count (64): the length must hold whole heads (8192 / 64 = 128 each),
        and the heads must split evenly over tp (64 / 8 = 8 per rank).

        Args:
            argument: Argument being split.
            dim: Dimension that ``axis`` names.
            axis: Axis being split.

        Raises:
            ValueError: If a packed axis has no size in the plan, the
                dimension is not a whole number of that axis, or the axis
                size is not a multiple of tp.
        """

        dim_name = argument.type.axes[dim]
        length = argument.type.shape[dim]

        if dim_name == axis:
            axis_size = length
        else:
            size = self._plan.axis_size(axis)
            if size is None:
                raise ValueError(
                    f"%{argument.name}: {axis!r} is packed in {dim_name!r}; "
                    f"give its size with plan.split({axis!r}, size=...)"
                )
            if length % size != 0:
                raise ValueError(
                    f"%{argument.name}: {dim_name} of length {length} is not a "
                    f"whole number of {axis} (size {size})"
                )
            axis_size = size

        if axis_size % self._plan.tp != 0:
            raise ValueError(
                f"%{argument.name}: {axis} of size {axis_size} cannot be split "
                f"over tp={self._plan.tp}"
            )


def _insert_all_reduce(op: Operation) -> tuple[Operation, ...]:
    """Follow an op that produces a partial sum with an all_reduce.

    Used as rebuild_function's rewrite: the all_reduce is emitted last, so
    every later user of the op's result reads the completed sum.

    Args:
        op: A rebuilt op.

    Returns:
        ``(op,)``, or ``(op, all_reduce)`` when op's result is partial.
    """

    if not any(isinstance(result.type.placement, Partial) for result in op.results):
        return (op,)
    # Every op that can produce a partial sum today has exactly one result;
    # unpacking fails loudly if a multi-result one ever appears.
    (partial,) = op.results
    # Reusing the producer's hint keeps the reduced value next to it in
    # dumps: the printer names them %o_proj and %o_proj_1.
    return (op, AllReduceOp(operands=(partial,), name_hint=op.name_hint))
