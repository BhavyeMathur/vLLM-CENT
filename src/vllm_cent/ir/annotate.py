"""Turn imported IR into Semantic IR by attaching roles and axes.

The importer only knows shapes and dtypes. What each dimension means
(``hidden``, ``q_head*head_dim``, ...) and which tensors are resident weights
is model knowledge, so the model supplies it for the function arguments. The
pass attaches those annotations and rebuilds the function; each op's
``infer_result_types`` then derives its result axes from its operands. The
pass itself knows no concrete op and never guesses from names.

Later stages read the axes, not the names: tensor parallelism shards along a
named axis wherever that axis sits in each tensor.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from .base import Function, Module, Value, rebuild_function
from .passes import FunctionPass
from .types import TensorRole

__all__ = ["AnnotateRolesAndAxes", "TensorAnnotation"]


@dataclass(frozen=True, slots=True)
class TensorAnnotation:
    """Semantic information for one function argument.

    Attributes:
        role: What kind of tensor the argument is.
        axes: One name per dimension, outermost first, e.g.
            ``("batch", "seq_len", "hidden")``.
    """

    role: TensorRole
    axes: tuple[str, ...]

    def __post_init__(self) -> None:
        """Store ``axes`` as a tuple even when a list is passed.

        The dataclass is frozen, so ``object.__setattr__`` is the only way to
        replace the field during initialization.
        """

        object.__setattr__(self, "axes", tuple(self.axes))


class AnnotateRolesAndAxes(FunctionPass):
    """Attach model-supplied annotations to arguments and propagate them.

    Every argument must be annotated, and every annotation must name an
    argument, so a renamed parameter or a typo fails loudly instead of
    leaving a tensor without axes.
    """

    NAME = "annotate-roles-and-axes"

    def __init__(self, annotations: Mapping[str, TensorAnnotation]) -> None:
        """Create the pass.

        Args:
            annotations: Annotation for each argument, keyed by argument name
                (a parameter FQN such as ``q_proj.weight``, or an input name
                such as ``x``). The mapping is copied.
        """

        self.annotations: dict[str, TensorAnnotation] = dict(annotations)

    def run(self, module: Module) -> Module:
        """Annotate every function of ``module``.

        Unused annotations are checked here rather than per function: a key
        unused by one function may belong to another.

        Args:
            module: Module to annotate; it is not modified.

        Returns:
            The annotated module.

        Raises:
            ValueError: If an annotation names no argument of any function.
        """

        argument_names = {
            argument.name
            for function in module.functions
            for argument in function.operands
        }
        unused = sorted(set(self.annotations) - argument_names)
        if unused:
            raise ValueError(f"annotations for unknown arguments: {unused}")
        return super().run(module)

    def run_on_function(self, function: Function) -> Function:
        """Annotate one function's arguments and re-infer its op results.

        Args:
            function: Function to annotate; it is not modified.

        Returns:
            The annotated function.

        Raises:
            ValueError: If an argument has no annotation, an annotation's axes
                do not match its argument's rank, an op rejects the annotated
                operands, or an op result ends up without axes.
        """

        # Pair each argument with its annotation, collecting every missing one
        # so a single error lists them all. An unnamed argument cannot be
        # looked up, so it counts as missing.
        pairs: list[tuple[Value, TensorAnnotation]] = []
        missing: list[str | None] = []
        for argument in function.operands:
            annotation = (
                None if argument.name is None else self.annotations.get(argument.name)
            )
            if annotation is None:
                missing.append(argument.name)
            else:
                pairs.append((argument, annotation))
        if missing:
            raise ValueError(
                f"function @{function.name}: no annotation for arguments {missing}"
            )

        new_arguments: list[Value] = []
        for argument, annotation in pairs:
            try:
                # replace() re-runs TensorType's checks, which reject an axes
                # tuple whose length differs from the argument's rank.
                annotated_type = replace(
                    argument.type, role=annotation.role, axes=annotation.axes
                )
            except ValueError as error:
                raise ValueError(f"argument %{argument.name}: {error}") from error
            new_arguments.append(Value(type=annotated_type, name=argument.name))

        # Rebuilding runs every op's type inference on the annotated
        # arguments, which propagates the axes through the function.
        annotated = rebuild_function(function, new_arguments)

        # An op that has not learned to propagate axes returns a result
        # without them. Report the op now; TP could not shard its result.
        for op in annotated.operations:
            for result in op.results:
                if len(result.type.axes) != result.type.rank:
                    label = op.NAME if op.name_hint is None else f"{op.NAME} ({op.name_hint})"
                    raise ValueError(f"{label} does not propagate axes to its result")

        return annotated
