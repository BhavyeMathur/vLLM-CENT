"""Connect logical tensor values to a compiled program's physical buffers.

Compilation fixes instructions and storage addresses once. This module checks
new input values for each run, pads them to fill every occupied slot, and
removes padding when results are returned. The underlying runtime only sees
named sequences of scalars and their physical regions.
"""

from dataclasses import dataclass

from ..runtime import CentExecutable, CentNamedScalars
from ..simulator import CentExecutionRequest, execute_functionally
from .ir import Value
from .target import TargetProgram, emit_program

__all__ = ["CompiledArtifact", "compile_target"]

_BINDING_VERSION = 1


@dataclass(frozen=True, slots=True, kw_only=True)
class CompiledArtifact:
    """Package a compiled program with the tensor information needed to run it.

    The ``bind()`` method packs inputs into their assigned buffers, and
    ``execute()`` runs the simulator and removes output padding. New inputs
    reuse the same instructions.

    Execution uses Python-float reference math to check where values flow. It
    does not reproduce exact F32 or BF16 rounding on hardware. Persistent model
    weights and changing key/value-cache addresses are not described here yet.

    Attributes:
        version: Format version of these logical binding rules, currently one.
        inputs: Required tensor names, shapes, and element types in input order.
        outputs: Returned tensor names, shapes, and element types in output order.
        target_ir: Named commands and their mapping, retained for inspection and
            verification of the compiled result.
        executable: Concrete CENT instructions plus a manifest saying where to
            place input scalars and read output scalars.

    Examples:
        For a three-element input stored in two two-lane slots, ``bind()`` turns
        ``(a, b, c)`` into ``(a, b, c, 0)``. The extra zero fills the unused lane;
        ``execute()`` returns only the logical output elements.
    """

    version: int
    inputs: tuple[Value, ...]
    outputs: tuple[Value, ...]
    target_ir: TargetProgram
    executable: CentExecutable

    def __post_init__(self) -> None:
        """Check that shapes, bindings, and instructions describe the same calculation.

        Rebuilding from Target IR also checks objects constructed manually by callers.
        No tensor data is loaded during this check.

        Raises:
            ValueError: If the version is unsupported, logical inputs/outputs differ
                from Target IR, or instructions and physical bindings do not match it.
            NotImplementedError: If verifying the source requires unsupported operations.
        """
        if self.version != _BINDING_VERSION:
            raise ValueError("unsupported logical binding version")
        # Verify Target IR before reading its shapes: a manually assembled
        # object may have an invalid graph as well as incorrect instructions.
        expected_executable = emit_program(self.target_ir)
        rank = self.target_ir.request.graph.ranks[0]
        values = {v.name: v for v in rank.inputs}
        values.update((op.output.name, op.output) for op in rank.operations)
        if self.inputs != rank.inputs or self.outputs != tuple(
            values[name] for name in rank.outputs
        ):
            raise ValueError("logical binding contract does not match Target IR")
        if self.executable != expected_executable:
            raise ValueError("executable does not match Target IR")

    def bind(self, inputs: tuple[CentNamedScalars, ...]) -> CentExecutionRequest:
        """Check new input tensors and add the zeros needed to fill their buffers.

        The caller supplies only tensor elements, not padding. No simulated memory
        is changed here; the returned request tells the executor what to load.

        Args:
            inputs: Named input values for this run, in any name order. Flatten each
                tensor row by row, with its last dimension varying fastest.

        Returns:
            Execution request containing the existing program and input scalars in
            manifest order, with enough trailing zeros to fill every occupied slot.

        Raises:
            ValueError: If input names repeat, are missing or unexpected, or a supplied
                tensor has a different number of elements from its declared shape.
        """
        # A dictionary allows arbitrary input order. The length check also
        # catches duplicate names that a dictionary would otherwise hide.
        supplied = {item.name: item.values for item in inputs}
        if len(supplied) != len(inputs) or set(supplied) != {
            item.name for item in self.inputs
        }:
            raise ValueError("input names must match the logical contract exactly")
        packed: list[CentNamedScalars] = []
        for value, binding in zip(
            self.inputs, self.executable.manifest.inputs, strict=True
        ):
            scalars = supplied[value.name]
            if len(scalars) != value.type.numel:
                raise ValueError(f"input {value.name} has incorrect scalar count")
            # The logical count excludes padding, while the manifest includes
            # every lane touched by instructions. Explicit zeros bridge that gap.
            padding = binding.region.scalar_count - len(scalars)
            packed.append(
                CentNamedScalars(
                    name=value.name,
                    values=scalars + (0.0,) * padding,
                )
            )
        return CentExecutionRequest(executable=self.executable, inputs=tuple(packed))

    def execute(
        self, inputs: tuple[CentNamedScalars, ...]
    ) -> tuple[CentNamedScalars, ...]:
        """Run the program with new inputs and return tensor elements without padding.

        Args:
            inputs: Named flattened tensor values, with the same convention as ``bind()``.

        Returns:
            Named results in the artifact's output order. Each sequence contains only
            the elements of the logical tensor, not unused physical lanes.

        Raises:
            ValueError: If input names or element counts do not match the artifact.
            CentSimulationError: If the executor rejects an instruction, reads data
                that was never written, or fails to compute or commit an instruction.
        """
        result = execute_functionally(self.bind(inputs))
        # This layout places padding only after the logical tensor. Slice by
        # its known element count, never by whether result values happen to be zero.
        return tuple(
            CentNamedScalars(name=value.name, values=raw.values[: value.type.numel])
            for value, raw in zip(self.outputs, result.outputs, strict=True)
        )


def compile_target(program: TargetProgram) -> CompiledArtifact:
    """Compile named target commands and attach their logical input/output shapes.

    Args:
        program: Target commands with the mapping and source graph needed to
            verify them and recover tensor shapes.

    Returns:
        Reusable ``CompiledArtifact`` for the supported single-device calculation.

    Raises:
        ValueError: If the target commands, mapping, or emitted executable are invalid.
        NotImplementedError: If the source needs an unsupported device operation.
    """
    executable = emit_program(program)
    rank = program.request.graph.ranks[0]
    values = {value.name: value for value in rank.inputs}
    values.update((op.output.name, op.output) for op in rank.operations)
    return CompiledArtifact(
        version=_BINDING_VERSION,
        inputs=rank.inputs,
        outputs=tuple(values[name] for name in rank.outputs),
        target_ir=program,
        executable=executable,
    )
