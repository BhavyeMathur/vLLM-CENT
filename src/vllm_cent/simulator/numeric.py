"""Choose how host numbers are stored and how simulated arithmetic behaves."""

from dataclasses import dataclass
from typing import Protocol

__all__ = ["CentNumericSemantics", "ReferenceMathSemantics"]


class CentNumericSemantics(Protocol):
    """Define conversion, addition, and multiplication for one numeric policy.

    Host writes call ``store``. Arithmetic methods return values already in
    the stored representation; copying or committing those values must not
    apply conversion a second time.
    """

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
            left: First stored number used in the calculation.
            right: Second stored number used in the calculation.

        Returns:
            Stored representation of the sum.
        """

        ...

    def multiply(self, left: float, right: float) -> float:
        """Multiply two stored values using the profile's arithmetic.

        Args:
            left: First stored number used in the calculation.
            right: Second stored number used in the calculation.

        Returns:
            Stored representation of the product.
        """

        ...


@dataclass(frozen=True, slots=True)
class ReferenceMathSemantics:
    """Use Python ``float`` values to check how compiled instructions move and combine values.

    This is a reference computation, not a model of CENT rounding. BF16
    (bfloat16) is a 16-bit floating-point format; its rounding and the
    accelerator's treatment of values such as infinity are not modeled here.
    """

    def store(self, value: float) -> float:
        """Store one value as a Python ``float`` without hardware quantization.

        Args:
            value: Host scalar being written to simulated state.

        Returns:
            Python ``float`` preserving signed zero and supported special values.
        """

        return float(value)

    def add(self, left: float, right: float) -> float:
        """Add two values with Python ``float`` arithmetic.

        Args:
            left: First stored number used in the calculation.
            right: Second stored number used in the calculation.

        Returns:
            Python ``float`` sum.
        """

        return self.store(left + right)

    def multiply(self, left: float, right: float) -> float:
        """Multiply two values with Python ``float`` arithmetic.

        Args:
            left: First stored number used in the calculation.
            right: Second stored number used in the calculation.

        Returns:
            Python ``float`` product.
        """

        return self.store(left * right)
