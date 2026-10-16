"""Describe tensor arithmetic, copies, and changes of shape.

Each operation checks its own mathematical shapes. These checks do not promise
that the device backend can execute the operation; legalization decides that.
"""

from dataclasses import dataclass
from typing import final

from .base import Operation, TensorType

__all__ = [
    "AddOperation",
    "CopyOperation",
    "LinearOperation",
    "MultiplyOperation",
    "ReshapeOperation",
]


def _binary_inputs(
    operation: Operation, input_types: tuple[TensorType, ...]
) -> tuple[TensorType, TensorType]:
    """Check the shared operand-count and element-type rules of binary math.

    Args:
        operation: Calculation whose output type and operand names are checked.
        input_types: Resolved types in the same order as ``operation.inputs``.

    Returns:
        Left and right types, with element formats matching the declared result.

    Raises:
        ValueError: If there are not two names and two types, or the element
            formats differ. Individual operations check the shape relationship.
    """
    if len(operation.inputs) != 2 or len(input_types) != 2:
        raise ValueError(f"{type(operation).__name__} requires two operands")
    left, right = input_types
    if not left.dtype == right.dtype == operation.output.type.dtype:
        raise ValueError("Operand and result dtypes must agree")
    return left, right


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class AddOperation(Operation):
    """Add matching elements of two tensors with the same shape.

    This operation has no extra parameters and does not broadcast smaller inputs.

    Attributes:
        name: Unique name of this addition in the graph.
        inputs: Left and right tensor names, in that order.
        output: New result name and type; its shape must match both inputs.
        block: Optional source model-block label for diagnostics.

    Examples:
        For ``x=(1, 2)`` and ``y=(3, 4)``, the requested result is ``(4, 6)``.
    """

    def verify(self, input_types: tuple[TensorType, ...]) -> None:
        """Require two inputs and an output with identical shapes and formats.

        Args:
            input_types: Left and right tensor types, resolved by the graph.

        Raises:
            ValueError: If operand count, element formats, or shapes disagree.
        """
        left, right = _binary_inputs(self, input_types)
        if not left.shape == right.shape == self.output.type.shape:
            raise ValueError("Invalid operand/output shape for addition")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class LinearOperation(Operation):
    """Multiply an input by a transposed weight matrix, without adding bias.

    Each input row is multiplied by every weight row to produce an output row.
    Device lowering uses explicit accumulator and packing primitives under the
    opt-in reference contract; this does not establish a physical instruction ABI.

    Attributes:
        name: Unique name of this matrix multiplication in the graph.
        inputs: Input tensor name followed by the weight-matrix name.
        output: New result name and type. Its last dimension is the number of
            weight rows; any leading input dimensions remain unchanged.
        block: Optional source model-block label for diagnostics.

    Examples:
        With ``x`` shaped ``(2, 4)`` and ``weights`` shaped ``(3, 4)``,
        ``x @ weights.T`` has shape ``(2, 3)``. Each input row produces three
        output values.
    """

    def verify(self, input_types: tuple[TensorType, ...]) -> None:
        """Check the shapes for ``x @ weights.T`` and matching element formats.

        Args:
            input_types: Input tensor and weight-matrix types, in that order.

        Raises:
            ValueError: If operand count, formats, dimensions, or result shape
                do not describe the requested matrix multiplication.
        """
        source, weights = _binary_inputs(self, input_types)
        # Weight rows count outputs; weight columns must match input width.
        # Keep leading dimensions so a batch of rows remains a batch of results.
        if not (
            source.shape
            and len(weights.shape) == 2
            and source.shape[-1] == weights.shape[-1]
            and self.output.type.shape == (*source.shape[:-1], weights.shape[0])
        ):
            raise ValueError("Invalid operand/output shape for linear")


def _unary_input(
    operation: Operation, input_types: tuple[TensorType, ...]
) -> TensorType:
    """Resolve one input and require its element format to match the result.

    Args:
        operation: Calculation declaring the input name and output type.
        input_types: Types resolved from those input names by graph verification.

    Returns:
        The sole input type, ready for the operation's shape check.

    Raises:
        ValueError: If there is not exactly one input name and type, or the
            input and output element formats differ.
    """
    if len(operation.inputs) != 1 or len(input_types) != 1:
        raise ValueError(f"{type(operation).__name__} requires one operand")
    source = input_types[0]
    if source.dtype != operation.output.type.dtype:
        raise ValueError("Operand and result dtypes must agree")
    return source


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class MultiplyOperation(Operation):
    """Multiply corresponding elements without broadcasting smaller inputs.

    Both inputs and the result have the same shape. Listing the same input
    twice requests its elementwise square; no separate square operation is needed.

    Attributes:
        name: Unique name of this multiplication in the graph.
        inputs: Left and right tensor names, in that order.
        output: New result name and type, matching both input shapes and formats.
        block: Optional source model-block label for diagnostics.

    Examples:
        For ``x=(2, 3)`` and ``inputs=("x", "x")``, the result is ``(4, 9)``.
    """

    def verify(self, input_types: tuple[TensorType, ...]) -> None:
        """Require two inputs and an output with matching shapes and formats.

        Args:
            input_types: Left and right tensor types in operand order.

        Raises:
            ValueError: If the operand count, shapes, or element formats disagree.
        """
        left, right = _binary_inputs(self, input_types)
        if not left.shape == right.shape == self.output.type.shape:
            raise ValueError("Invalid operand/output shape for multiplication")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class CopyOperation(Operation):
    """Produce a new tensor containing the same elements as the input.

    This describes values rather than a mandatory physical transfer. A later
    storage plan can reuse memory when the old value is no longer needed.

    Attributes:
        name: Unique name of this copy in the graph.
        inputs: One source tensor name.
        output: New result name, with the source shape and element format.
        block: Optional source model-block label for diagnostics.
    """

    def verify(self, input_types: tuple[TensorType, ...]) -> None:
        """Check that the copy preserves the complete input type.

        Args:
            input_types: Type of the sole source tensor.

        Raises:
            ValueError: If the input count, shape, or element format differs.
        """
        source = _unary_input(self, input_types)
        if source.shape != self.output.type.shape:
            raise ValueError("Invalid operand/output shape for copy")


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class ReshapeOperation(Operation):
    """Change tensor dimensions while preserving the flattened element order.

    Tensors here are contiguous, with the last dimension varying fastest.
    Reshape does not transpose elements or describe arbitrary strided views.
    The output type supplies the new shape; no separate shape attribute is needed.

    Attributes:
        name: Unique name of this reshape in the graph.
        inputs: One source tensor name.
        output: Result name and new shape, retaining element count and format.
        block: Optional source model-block label for diagnostics.

    Examples:
        A ``(2, 3)`` input can become ``(6,)`` or ``(3, 2)``. All three shapes
        refer to the same six elements in the same flattened order.
    """

    def verify(self, input_types: tuple[TensorType, ...]) -> None:
        """Require one input and preserve its element count and format.

        Args:
            input_types: Type of the sole source tensor.

        Raises:
            ValueError: If the input count, element count, or format differs.
        """
        source = _unary_input(self, input_types)
        if source.numel != self.output.type.numel:
            raise ValueError("Reshape must preserve the number of elements")
