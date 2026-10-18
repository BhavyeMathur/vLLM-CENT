"""Ops that normalize along the last dimension: RMSNorm and softmax.

Each result row depends on the whole last-dimension row of the operand,
so that dimension must not be split across ranks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import inf

from ..base import Operation, Value
from ..types import Partial, Replicate, Shard, TensorRole, TensorType

__all__ = [
    "RMSNormOp",
    "SoftmaxOp",
]


@dataclass(frozen = True, eq = False, slots=True)
class RMSNormOp(Operation):
    """RMS normalization over the last dimension, with an optional scale.

    ``y = x / sqrt(mean(x**2) + eps)``, then multiplied element-wise by
    ``weight`` when one is given. Without a weight this is the unscaled form
    of ``nn.RMSNorm(elementwise_affine=False)``, which Llama 4 applies to
    queries and keys (its "L2Norm").

    RMSNormOp(operands=(x, weight), eps=1e-5)
    RMSNormOp(operands=(x,), eps=1e-5)

    Attributes:
        eps: Constant added to the mean square; None means the machine
            epsilon of x's dtype, as in ``torch.nn.RMSNorm``.
    """

    NAME = "aloi.rms_norm"
    # x, and optionally the learned per-feature scale.
    NUM_OPERANDS = (1, 2)

    eps: float | None = None

    @property
    def x(self) -> Value:
        return self.operands[0]

    @property
    def weight(self) -> Value | None:
        return self.operands[1] if len(self.operands) == 2 else None

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """RMSNorm keeps x's shape, dtype and axes; the result is an activation."""

        # nan fails every comparison, so test for the valid range rather than
        # for eps <= 0; this rejects nan and inf as well as non-positive eps.
        if (self.eps is not None) and not (0 < self.eps < inf):
            raise ValueError(
                f"{self.NAME}: eps must be positive and finite, "
                f"got {self.eps}"
            )

        # Check ranks before indexing shape[-1] below, so a scalar x gives a
        # clear ValueError instead of an IndexError.
        if len(self.x.type.shape) < 1:
            raise ValueError(
                f"{self.NAME}: x needs a last dimension to normalize, "
                f"got shape {self.x.type.shape}"
            )

        weight = self.weight
        if weight is not None and len(weight.type.shape) != 1:
            raise ValueError(
                f"{self.NAME}: weight must be 1D, "
                f"got {weight.type.shape}"
            )

        if weight is not None and self.x.type.dtype != weight.type.dtype:
            raise ValueError(
                f"{self.NAME}: x and weight must share one dtype, "
                f"got {self.x.type.dtype} and {weight.type.dtype}"
            )

        # Placement checks come before the size check, for the same reason as
        # in LinearOp: a sharded x would also fail the size check, with a less
        # helpful message.
        #
        # Every rank scales its rows by the whole weight vector.
        if weight is not None and not isinstance(weight.type.placement, Replicate):
            raise ValueError(
                f"{self.NAME}: weight must be replicated, "
                f"got {weight.type.placement}"
            )
        match self.x.type.placement:
            case Partial():
                # The mean square of a partial sum is not the mean square of
                # the sum, so the ranks must add their parts first.
                raise ValueError(
                    f"{self.NAME}: x is a partial sum; all_reduce it first"
                )
            case Shard(axis) if self.x.type.dim_of(axis) == self.x.type.rank - 1:
                # The mean square runs over the whole last dimension, which no
                # single rank would hold.
                raise ValueError(
                    f"{self.NAME}: x's last dimension is split along {axis!r}; "
                    "normalizing needs the whole dimension on every rank"
                )

        if weight is not None and self.x.type.shape[-1] != weight.type.shape[0]:
            raise ValueError(
                f"{self.NAME}: weight size must match "
                f"x.shape[-1], got {weight.type.shape[0]} "
                f"and {self.x.type.shape[-1]}"
            )

        # The weight scales each element of x's last dimension, so once both
        # are annotated it must be indexed by that same axis.
        x_axes = self.x.type.axes
        weight_axes = () if weight is None else weight.type.axes
        if x_axes and weight_axes and weight_axes != (x_axes[-1],):
            raise ValueError(
                f"{self.NAME}: weight axes {weight_axes} do not match "
                f"x's last axis {x_axes[-1]!r}"
            )

        # Copy x's type but reset the role: whatever x is, the normalized
        # tensor is computed during the forward pass, i.e. an activation.
        # The copy keeps x's placement and global_shape: a row split along
        # batch or seq_len stays split the same way.
        return (replace(self.x.type, role = TensorRole.ACTIVATION),)


@dataclass(frozen = True, eq = False, slots=True)
class SoftmaxOp(Operation):
    """Softmax over the last dimension.

    SoftmaxOp(operands=(x,))
    """

    NAME = "aloi.softmax"
    NUM_OPERANDS = 1

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result has x's shape, dtype, axes and placement.

        Raises:
            ValueError: If x is a scalar, a partial sum, or split along its
                last dimension. The maximum and the sum run over the whole
                last dimension, so every rank needs all of it.
        """

        if self.x.type.rank < 1:
            raise ValueError(f"{self.NAME}: x needs a dimension to normalize over")
        if isinstance(self.x.type.placement, Partial):
            raise ValueError(f"{self.NAME}: x is a partial sum; all_reduce it first")
        if self.x.type.sharded_dim == self.x.type.rank - 1:
            raise ValueError(
                f"{self.NAME}: x's last dimension is split across ranks; "
                "softmax needs the whole dimension on every rank"
            )
        return (replace(self.x.type, role = TensorRole.ACTIVATION),)
