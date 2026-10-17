"""Define tensor names, tensor types, and the fields shared by graph operations.

This module does not choose mathematical operations or hardware instructions.
Each concrete operation owns its parameters and checks its own input and output
shapes. The graph verifier only connects names and calls that check.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from math import prod

__all__ = ["Operation", "ScalarType", "TensorType", "Value"]


class ScalarType(Enum):
    """Identify how each tensor element is represented in the source graph.

    Members:
        F32: IEEE 32-bit floating point. This records the source type; the
            functional simulator uses Python floats rather than exact F32 rounding.
    """

    F32 = "f32"


@dataclass(frozen=True, slots=True)
class TensorType:
    """Record a tensor's dimensions and the type of its elements.

    This object contains no values and reserves no memory.

    Attributes:
        shape: Sizes of the dimensions, in order. An empty tuple describes one
            scalar. When values are flattened, the last dimension varies fastest.
        dtype: Number format used by the source graph, currently ``ScalarType.F32`` only.

    Examples:
        ``shape=(2, 3)`` describes two rows of three values.
    """

    shape: tuple[int, ...]
    dtype: ScalarType = ScalarType.F32

    def __post_init__(self) -> None:
        """Require every dimension to contain elements and use the supported type.

        Raises:
            ValueError: If any dimension is zero or negative, or ``dtype`` is not ``ScalarType.F32``.
        """
        if any(dimension <= 0 for dimension in self.shape):
            raise ValueError("Tensor dimensions must be positive")
        if self.dtype is not ScalarType.F32:
            raise ValueError("Only F32 tensor types are supported")

    @property
    def numel(self) -> int:
        """Count the tensor's elements.

        Returns:
            Product of the dimensions. An empty shape gives one because it describes
            a scalar, not an empty tensor.

        Examples:
            ``shape=(2, 3)`` gives six elements.
        """
        return prod(self.shape)


@dataclass(frozen=True, slots=True)
class Value:
    """Give a tensor a name so operations can refer to it.

    A name is defined only once in a graph; later operations cannot redefine it.
    Compiler terminology calls this static single assignment (SSA).

    Attributes:
        name: Tensor name, unique within this graph.
        type: Tensor dimensions and element format. The actual data arrives later.

    Examples:
        For ``z = x + y``, ``x`` and ``y`` name the inputs and ``z`` names a new result.
    """

    name: str
    type: TensorType


@dataclass(frozen=True, slots=True, kw_only=True)
class Operation(ABC):
    """Hold the graph connections common to a single-result calculation.

    A concrete operation adds the parameters needed for its own calculation
    and implements ``verify()``. The base does not list operation kinds, assume
    a fixed number of inputs, or know any operation-specific parameters.
    It is abstract so callers cannot create an operation with no meaning.

    Attributes:
        name: Unique operation name used in schedules and diagnostics.
        inputs: Names of tensors read by this operation, in operand order.
            Each name refers to a graph input or an earlier operation's result.
        output: Name and type of the new tensor this operation produces.
        block: Optional source model-block label for diagnostics. This does not
            change execution order or require a particular model family.
    """

    name: str
    inputs: tuple[str, ...]
    output: Value
    block: str = ""

    @abstractmethod
    def verify(self, input_types: tuple[TensorType, ...]) -> None:
        """Check the mathematical input/output rules of this operation.

        The graph verifier resolves input names before calling this method.
        Implementations check their required number of inputs, shapes, and
        element formats without executing arithmetic or choosing hardware.

        Args:
            input_types: Types of the tensors named in ``inputs``, in the same order.

        Raises:
            ValueError: If inputs or the declared output do not match the
                concrete operation's mathematical rules.
        """
        ...
