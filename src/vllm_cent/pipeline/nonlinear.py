"""Keep nonlinear function choices visible until reference instruction emission."""

from dataclasses import dataclass

from ..activations import ActivationKind

__all__ = ["ActivationCommand", "SoftmaxCommand"]


@dataclass(frozen=True, slots=True, kw_only=True)
class ActivationCommand:
    """Apply a scalar function to a logical prefix, then clear its padding.

    Attributes:
        source: Shared Buffer object containing input values.
        destination: Shared Buffer object receiving the result; may equal source.
        value_count: Number of logical scalar elements, excluding padded lanes.
        function: Exact scalar formula, including the selected GELU approximation.
        operation: Source operation name retained for diagnostics.
    """

    source: str
    destination: str
    value_count: int
    function: ActivationKind
    operation: str


@dataclass(frozen=True, slots=True, kw_only=True)
class SoftmaxCommand:
    """Normalize contiguous logical rows without including padded lanes.

    Attributes:
        source: Shared Buffer object containing row-major logits.
        destination: Result object; exact input reuse is safe at last use.
        row_count: Number of independent probability distributions.
        row_width: Logical scalar elements in each distribution.
        operation: Source operation name retained for diagnostics.
    """

    source: str
    destination: str
    row_count: int
    row_width: int
    operation: str
