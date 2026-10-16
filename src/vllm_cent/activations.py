"""Name activation formulas independently of a device or execution backend."""

from enum import Enum

__all__ = ["ActivationKind"]


class ActivationKind(str, Enum):
    """Select one parameter-free scalar activation formula.

    GELU means Gaussian error linear unit. Its exact error-function formula and
    common hyperbolic-tangent approximation are separate choices, so a lowering
    pass cannot silently exchange them. SiLU means sigmoid linear unit.

    Members:
        RELU: Return zero for negative inputs and the input otherwise.
        SIGMOID: Return ``1 / (1 + exp(-x))``.
        TANH: Return the hyperbolic tangent of the input.
        SILU: Multiply the input by its sigmoid.
        GELU_ERF: Multiply the input by the standard normal cumulative probability.
        GELU_TANH: Use the GELU approximation with coefficient ``0.044715``.
    """

    RELU = "relu"
    SIGMOID = "sigmoid"
    TANH = "tanh"
    SILU = "silu"
    GELU_ERF = "gelu_erf"
    GELU_TANH = "gelu_tanh"
