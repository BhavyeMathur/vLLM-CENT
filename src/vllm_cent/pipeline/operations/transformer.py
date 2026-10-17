"""Describe transformer normalization without adding its fields to generic IR."""

from dataclasses import dataclass
from math import isfinite
from typing import final

from .base import Operation, TensorType
from .tensor import _binary_inputs

__all__ = ["RmsNormOperation"]


@final
@dataclass(frozen=True, slots=True, kw_only=True)
class RmsNormOperation(Operation):
    """Normalize each vector by its root-mean-square value, then apply scale.

    The calculation is ``x * scale / sqrt(mean(x**2, axis=-1) + epsilon)``, with
    the denominator applied to every element along the last dimension.
    ``epsilon`` is required here because it changes this operation's result.
    Numerical execution requires the explicit reference contract for reduction,
    scalar math, and broadcasting. Those rules are not a physical CENT calling
    convention or a promise of bit-exact hardware arithmetic.

    Attributes:
        name: Unique name of this normalization in the graph.
        inputs: Input tensor name followed by the learned scale-vector name.
        output: New result name and type, with the same shape as the input.
        block: Optional source model-block label for diagnostics.
        epsilon: Finite number greater than zero added under the square root
            so an all-zero vector does not cause division by zero.
    """

    epsilon: float

    def __post_init__(self) -> None:
        """Require the normalization constant before this operation enters a graph.

        Raises:
            ValueError: If ``epsilon`` is non-finite, zero, or negative.
        """
        if not isfinite(self.epsilon) or self.epsilon <= 0:
            raise ValueError("RMSNorm requires finite positive epsilon")

    def verify(self, input_types: tuple[TensorType, ...]) -> None:
        """Check input width, learned scale length, and the unchanged result shape.

        Args:
            input_types: Input tensor and learned scale-vector types, in that order.

        Raises:
            ValueError: If operand count, element formats, scale length, or
                result shape do not describe RMS normalization.
        """
        source, scale = _binary_inputs(self, input_types)
        # A single learned scale value is used at each position of every vector.
        # For input shape (2, 4), scale is (4,) and the result is still (2, 4).
        if not (
            source.shape
            and scale.shape == (source.shape[-1],)
            and self.output.type.shape == source.shape
        ):
            raise ValueError("Invalid operand/output shape for RMSNorm")
