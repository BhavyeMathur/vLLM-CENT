"""Semantic IR Implementation"""

from __future__ import annotations

from dataclasses import dataclass

from .types import TensorType
from .base import Value, Operation

@dataclass(frozen = True, eq = False, slots=True)
class LinearOp(Operation):
    """
    LinearOp(
        operands=(x, weight),
    )

    y = x @ weight.T
    """

    NAME = "aloi.linear"
    NUM_OPERANDS = 2

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Matrix multiplication on inner-most dimensions of in_feature with weight"""

        if (self.x.type.dtype != self.weight.type.dtype):
            raise ValueError("Operands should have the same datatype")

        if (len(self.x.type.shape) < 1):
            raise ValueError("x would have dimensions of [*, in_features]")
        
        if (len(self.weight.type.shape) != 2):
            raise ValueError("Weight should have 2 dimensions [out_features, in_features]")

        if (self.x.type.shape[-1] != self.weight.type.shape[-1]):
            raise ValueError("Weight should have shape [out_features, in_features]")

        return (
            TensorType(
                shape = self.x.type.shape[:-1] + (self.out_features,),
                dtype = self.operands[0].type.dtype,
            ),
        )

    @property
    def x(self) -> Value:
        return self.operands[0]

    @property
    def weight(self) -> Value:
        return self.operands[1]

    @property
    def in_features(self) -> int:
        return self.weight.type.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.type.shape[0]

@dataclass(frozen = True, eq = False, slots=True)
class RMSNormOp(Operation):
    NAME = "aloi.rms_norm"
    NUM_OPERANDS = 2

    eps: float | None = None

    @property
    def x(self) -> Value:
        return self.operands[0]

    @property
    def weight(self) -> Value:
        return self.operands[1]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """RMSNorm does not alter the operand type."""

        if (self.eps is not None) and self.eps <= 0:
            raise ValueError(
                f"{self.NAME}: eps must be positive, "
                f"got {self.eps}"
            )

        # Check ranks before indexing shape[-1] below, so a scalar x gives a
        # clear ValueError instead of an IndexError.
        if len(self.x.type.shape) < 1:
            raise ValueError(
                f"{self.NAME}: x needs a last dimension to normalize, "
                f"got shape {self.x.type.shape}"
            )

        if len(self.weight.type.shape) != 1:
            raise ValueError(
                f"{self.NAME}: weight must be 1D, "
                f"got {self.weight.type.shape}"
            )

        if self.x.type.dtype != self.weight.type.dtype:
            raise ValueError(
                f"{self.NAME}: x and weight must share one dtype, "
                f"got {self.x.type.dtype} and {self.weight.type.dtype}"
            )

        if self.x.type.shape[-1] != self.weight.type.shape[0]:
            raise ValueError(
                f"{self.NAME}: weight size must match "
                f"x.shape[-1], got {self.weight.type.shape[0]} "
                f"and {self.x.type.shape[-1]}"
            )

        return (self.x.type,)

    