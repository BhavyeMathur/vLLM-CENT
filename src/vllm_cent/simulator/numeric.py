"""Numeric policies used by functional instruction kernels."""

from dataclasses import dataclass
from typing import Protocol

__all__ = ["CentNumericSemantics", "ReferenceMathSemantics"]


class CentNumericSemantics(Protocol):
    """Define scalar conversion and arithmetic for one simulator profile."""

    def store(self, value: float) -> float:
        """Convert a host value to the profile's stored representation.

        Args:
            value: Host scalar being written to simulated state.

        Returns:
            Scalar represented according to this profile.
        """

        ...

    def add(self, left: float, right: float) -> float:
        """Add two stored values using the profile's arithmetic.

        Args:
            left: Left operand.
            right: Right operand.

        Returns:
            Stored representation of the sum.
        """

        ...

    def multiply(self, left: float, right: float) -> float:
        """Multiply two stored values using the profile's arithmetic.

        Args:
            left: Left operand.
            right: Right operand.

        Returns:
            Stored representation of the product.
        """

        ...


@dataclass(frozen=True, slots=True)
class ReferenceMathSemantics:
    """Use Python float operations to validate compiler-level dataflow.

    This profile intentionally makes no claim about CENT BF16 rounding or
    accelerator special-value behavior.
    """

    def store(self, value: float) -> float:
        """Store one value as a Python float without hardware quantization.

        Args:
            value: Host scalar being written to simulated state.

        Returns:
            Python float preserving signed zero and supported special values.
        """

        return float(value)

    def add(self, left: float, right: float) -> float:
        """Add two values with Python float arithmetic.

        Args:
            left: Left operand.
            right: Right operand.

        Returns:
            Python float sum.
        """

        return self.store(left + right)

    def multiply(self, left: float, right: float) -> float:
        """Multiply two values with Python float arithmetic.

        Args:
            left: Left operand.
            right: Right operand.

        Returns:
            Python float product.
        """

        return self.store(left * right)
