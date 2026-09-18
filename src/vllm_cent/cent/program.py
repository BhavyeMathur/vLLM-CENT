"""Store an ordered CENT program and the hardware it targets."""

from dataclasses import dataclass

from .hardware import CentHardwareSpec
from .instructions import CentInstruction, validate_instruction

__all__ = ["CentProgram"]


# A CentProgram deliberately contains instructions only. The runtime package's
# CentExecutable pairs it with named physical inputs and outputs. Future logical
# vector bindings must also write every occupied lane and explicitly zero each
# partition's padding lanes; raw bindings do not infer that higher-level layout.


@dataclass(frozen=True, slots=True, kw_only=True)
class CentProgram:
    """An immutable sequence of instructions for one CENT device.

    Attributes:
        hardware: Device on which the instructions will run.
        instructions: Commands in the order in which they should run. The tuple
            must contain at least one command.
    """

    hardware: CentHardwareSpec
    instructions: tuple[CentInstruction, ...]

    def __post_init__(self) -> None:
        """Check that the program contains valid instructions for its device.

        Raises:
            ValueError: If the program is empty or an instruction is
                incompatible with the hardware.
        """

        # An empty program usually means the compiler forgot to emit work.
        if not self.instructions:
            raise ValueError("a CENT program cannot be empty")

        # Instruction constructors check rules shared by every target. Validate
        # again here for limits that depend on this program's hardware. This also
        # protects callers that create CentProgram without using the builder.
        for instruction in self.instructions:
            validate_instruction(instruction, self.hardware)
