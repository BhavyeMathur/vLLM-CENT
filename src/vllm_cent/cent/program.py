"""Store an ordered CENT program and the hardware it targets."""

from dataclasses import dataclass

from .hardware import CentHardwareSpec
from .instructions import CentInstruction, validate_instruction

__all__ = ["CentProgram"]


# TODO(runtime): Instructions do not tell a loader what each address contains.
# We need tensor locations, initial data, inputs, and output locations. When the
# runtime materializes a logical vector, it must use the vector layout to write
# every occupied lane and explicitly zero each partition's padding lanes.


@dataclass(frozen=True, slots=True, kw_only=True)
class CentProgram:
    """Store one device's ordered commands and check their storage bounds.

    Creating this object validates instruction shapes and addresses against
    ``hardware``. It does not load input values or establish that each operation
    has executable numerical semantics. A runtime executable adds input and
    output bindings separately.

    Attributes:
        hardware: Device on which the instructions will run.
        instructions: Commands in execution order. An empty tuple is valid when
            input/output bindings alone implement an identity or reshape.
    """

    hardware: CentHardwareSpec
    instructions: tuple[CentInstruction, ...]

    def __post_init__(self) -> None:
        """Check that the program contains valid instructions for its device.

        Raises:
            TypeError: If an instruction has no supported validation rules.
            ValueError: If an instruction is incompatible with the hardware.
        """

        # Instruction constructors check rules shared by every target. Validate
        # again here for limits that depend on this program's hardware. This also
        # protects callers that create CentProgram without using the builder.
        for instruction in self.instructions:
            validate_instruction(instruction, self.hardware)
