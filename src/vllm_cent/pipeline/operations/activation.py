"""Describe pointwise nonlinear functions and last-dimension softmax."""

from dataclasses import dataclass

from ...activations import ActivationKind
from .base import Operation, TensorType
from .tensor import _unary_input

__all__ = ["ActivationKind", "ActivationOperation", "SoftmaxOperation"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ActivationOperation(Operation):
    """Apply one selected scalar function independently to every tensor element.

    Exact and tanh-approximate GELU are separate choices and remain distinct
    through lowering. Execution uses the explicit local reference contract;
    no hardware activation identifier or lookup-table accuracy is implied.

    Attributes:
        name: Unique graph operation name.
        inputs: One source tensor name.
        output: Result name with the same shape and scalar type.
        block: Optional model-block label for diagnostics.
        function: Scalar formula to apply, including the chosen GELU approximation.
    """

    function: ActivationKind

    def verify(self, input_types: tuple[TensorType, ...]) -> None:
        """Require one operand and an unchanged result type.

        Args:
            input_types: Type of the sole input tensor.

        Raises:
            ValueError: If operand count, shape, or scalar format differs.
        """
        source = _unary_input(self, input_types)
        if source.shape != self.output.type.shape:
            raise ValueError("Activation must preserve the input shape")


@dataclass(frozen=True, slots=True, kw_only=True)
class SoftmaxOperation(Operation):
    """Convert each last-dimension row of logits into normalized probabilities.

    Subtract the largest finite value before exponentiation. Negative infinity
    masks an element. Padding never contributes to the denominator. The reference
    contract rejects NaN, positive infinity, and rows without a finite element.

    Attributes:
        name: Unique graph operation name.
        inputs: One tensor of logits, optionally already containing mask values.
        output: Probability tensor with the same shape and scalar type.
        block: Optional model-block label for diagnostics.

    Examples:
        An input shape of ``(2, 3)`` normalizes two independent rows of three
        values. It does not normalize all six values together.
    """

    def verify(self, input_types: tuple[TensorType, ...]) -> None:
        """Require an unchanged non-scalar tensor type.

        Args:
            input_types: Type of the sole logits tensor.

        Raises:
            ValueError: If operand count, format, or shape is invalid.
        """
        source = _unary_input(self, input_types)
        if not source.shape or source.shape != self.output.type.shape:
            raise ValueError(
                "Softmax requires an unchanged shape with a last dimension"
            )
