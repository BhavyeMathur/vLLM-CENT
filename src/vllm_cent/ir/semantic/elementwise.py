"""Ops that compute each result element from the same element of the operands.

Because no element needs another, a split operand stays split the same
way. They reject partial sums: an activation or a product of the parts is
not the activation or product of the whole, so the ranks must add their
parts first.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ...activations import ActivationKind
from ..base import Operation, Value
from ..types import DType, Partial, TensorRole, TensorType

__all__ = [
    "ActivationOp",
    "AddOp",
    "DivOp",
    "ExpOp",
    "MulOp",
    "SubOp",
]

# Dtypes without a fractional part: exp and true division do not apply.
_INTEGER_DTYPES = frozenset({DType.INT64, DType.INT32, DType.INT8})


@dataclass(frozen=True, eq=False, slots=True)
class ActivationOp(Operation):
    """A parameter-free activation applied to every element.

    ``function`` picks the formula: SiLU, ReLU, sigmoid, tanh, or one of the
    two GELU formulas. The exact (erf) GELU and its tanh approximation are
    different choices; no pass may exchange them.

    ActivationOp(operands=(x,), function=ActivationKind.SILU)

    Attributes:
        function: Which formula to apply.
    """

    NAME = "aloi.activation"
    NUM_OPERANDS = 1

    function: ActivationKind

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result has x's shape, dtype, axes and placement.

        A split x stays split the same way: each element's result depends
        only on that element, so no rank needs another rank's data.

        Raises:
            ValueError: If x is a partial sum. None of the formulas is
                linear, so f(a) + f(b) != f(a + b): the parts must be summed
                first.
        """

        if isinstance(self.x.type.placement, Partial):
            raise ValueError(f"{self.NAME}: x is a partial sum; all_reduce it first")
        return (replace(self.x.type, role=TensorRole.ACTIVATION),)


@dataclass(frozen=True, eq=False, slots=True)
class ExpOp(Operation):
    """Natural exponential ``e ** x``, element by element.

    ExpOp(operands=(x,))
    """

    NAME = "aloi.exp"
    NUM_OPERANDS = 1

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result has x's shape, dtype, axes and placement.

        A split x stays split the same way: each element's result depends
        only on that element.

        Raises:
            ValueError: If x has an integer dtype (e ** x is not an integer),
                or x is a partial sum: exp(a) + exp(b) != exp(a + b), so the
                parts must be summed first.
        """

        if self.x.type.dtype in _INTEGER_DTYPES:
            raise ValueError(
                f"{self.NAME}: x must be floating point, got {self.x.type.dtype}"
            )
        if isinstance(self.x.type.placement, Partial):
            raise ValueError(f"{self.NAME}: x is a partial sum; all_reduce it first")
        return (replace(self.x.type, role=TensorRole.ACTIVATION),)


@dataclass(frozen=True, eq=False, slots=True)
class _ElementwiseBinaryOp(Operation):
    """Shared rules of ops that combine two same-shaped tensors elementwise.

    There is no broadcasting: both operands have the same shape, dtype, axes
    and placement, and so does the result. Subclasses only name the op.
    """

    NUM_OPERANDS = 2

    @property
    def lhs(self) -> Value:
        return self.operands[0]

    @property
    def rhs(self) -> Value:
        return self.operands[1]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Check that the operands line up element for element.

        Raises:
            ValueError: If shapes, dtypes, annotated axes or placements
                differ, or the operands are partial sums.
        """

        lhs = self.lhs.type
        rhs = self.rhs.type
        if lhs.shape != rhs.shape:
            raise ValueError(
                f"{self.NAME}: operand shapes differ, {lhs.shape} and {rhs.shape}"
            )
        if lhs.dtype != rhs.dtype:
            raise ValueError(
                f"{self.NAME}: operand dtypes differ, {lhs.dtype} and {rhs.dtype}"
            )
        if lhs.axes and rhs.axes and lhs.axes != rhs.axes:
            raise ValueError(
                f"{self.NAME}: operand axes differ, {lhs.axes} and {rhs.axes}"
            )

        # Equal placements mean every rank holds matching pieces of both
        # operands. A partial sum is rejected because the result is used
        # right away; an all_reduce always follows the op that made it.
        if lhs.placement != rhs.placement or isinstance(lhs.placement, Partial):
            raise ValueError(
                f"{self.NAME}: unsupported placements "
                f"lhs={lhs.placement}, rhs={rhs.placement}"
            )

        axes = lhs.axes if lhs.axes and rhs.axes else ()
        return (
            TensorType(
                shape=lhs.shape,
                dtype=lhs.dtype,
                axes=axes,
                global_shape=lhs.global_shape,
                placement=lhs.placement,
            ),
        )


@dataclass(frozen=True, eq=False, slots=True)
class AddOp(_ElementwiseBinaryOp):
    """Elementwise sum, e.g. a residual connection.

    AddOp(operands=(lhs, rhs))
    """

    NAME = "aloi.add"


@dataclass(frozen=True, eq=False, slots=True)
class SubOp(_ElementwiseBinaryOp):
    """Elementwise difference ``lhs - rhs``, e.g. subtracting each row's maximum.

    SubOp(operands=(lhs, rhs))
    """

    NAME = "aloi.sub"


@dataclass(frozen=True, eq=False, slots=True)
class MulOp(_ElementwiseBinaryOp):
    """Elementwise product, e.g. the SwiGLU gate times the up projection.

    MulOp(operands=(lhs, rhs))
    """

    NAME = "aloi.mul"


@dataclass(frozen=True, eq=False, slots=True)
class DivOp(_ElementwiseBinaryOp):
    """Elementwise quotient ``lhs / rhs``, e.g. dividing by a softmax row sum.

    This is true division of floating-point values. How a target computes it,
    e.g. a reciprocal followed by a multiplication, is a lowering choice.

    DivOp(operands=(lhs, rhs))
    """

    NAME = "aloi.div"

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Check the operands like the other binary ops, and reject integers.

        Raises:
            ValueError: As for the other binary ops, or if the operands are
                integers (integer division would need a rounding mode).
        """

        if self.lhs.type.dtype in _INTEGER_DTYPES:
            raise ValueError(
                f"{self.NAME}: operands must be floating point, "
                f"got {self.lhs.type.dtype}"
            )
        return super().infer_result_types()
