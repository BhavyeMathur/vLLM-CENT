"""Semantic IR Implementation"""

from __future__ import annotations

from dataclasses import dataclass, replace

from .types import Partial, Placement, Replicate, Shard, TensorRole, TensorType
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

        # Check placements before shapes: when only one operand is sharded,
        # the local shapes disagree too, and "unsupported placements" says
        # what is actually wrong.
        placement = self._result_placement()

        if (self.x.type.shape[-1] != self.weight.type.shape[-1]):
            raise ValueError("Weight should have shape [out_features, in_features]")

        # A sharded result remembers its whole size. Its leading dimensions
        # come from x and its last one is the weight's whole out_features.
        # A partial result already has its full shape, so it needs none.
        global_shape = None
        if isinstance(placement, Shard):
            global_shape = (
                *self.x.type.full_shape[:-1],
                self.weight.type.full_shape[0],
            )

        return (
            TensorType(
                shape = self.x.type.shape[:-1] + (self.out_features,),
                dtype = self.operands[0].type.dtype,
                axes = self._result_axes(),
                global_shape = global_shape,
                placement = placement,
            ),
        )

    def _result_placement(self) -> Placement:
        """Derive which part of the result each tensor-parallel rank holds.

        Only the combinations that need no communication before the matmul
        are accepted:

        - x and weight replicated: every rank computes the whole result.
        - x replicated, weight split along its out_features dimension
          (column parallel, e.g. q_proj): each rank computes its own slice of
          the output features.
        - x split along its last dimension and weight split along its
          in_features dimension on the same axis (row parallel, e.g. o_proj):
          each rank sums over only its slice of the contraction, so it holds a
          partial sum that an all_reduce must complete.

        Returns:
            The result's placement.

        Raises:
            ValueError: For any other combination, including a partial
                operand.
        """

        x = self.x.type
        weight = self.weight.type
        # Match both placements at once. Shard(axis) checks the class and
        # binds its axis field (dataclasses generate __match_args__); the
        # "if" guard then checks which dimension that axis splits.
        match x.placement, weight.placement:
            case Replicate(), Replicate():
                return Replicate()
            case Replicate(), Shard(axis) if weight.dim_of(axis) == 0:
                return Shard(axis)
            case Shard(x_axis), Shard(weight_axis) if (
                x_axis == weight_axis
                and x.dim_of(x_axis) == x.rank - 1
                and weight.dim_of(weight_axis) == 1
            ):
                return Partial()
        raise ValueError(
            f"{self.NAME}: unsupported placements "
            f"x={x.placement}, weight={weight.placement}"
        )

    def _result_axes(self) -> tuple[str, ...]:
        """Propagate axis names: x's leading axes, then weight's output axis.

        The last axis of x is summed over against the weight's in_features
        axis, so the two must name the same thing. Equal sizes are not enough:
        hidden and q_head*head_dim are both 8192 in Llama2-70B, but
        multiplying one by the other is a model bug.

        Returns:
            The result's axes, or () while either operand is unannotated.
        """

        x_axes = self.x.type.axes
        weight_axes = self.weight.type.axes
        if not x_axes or not weight_axes:
            return ()

        out_axis, in_axis = weight_axes
        if in_axis != x_axes[-1]:
            raise ValueError(
                f"{self.NAME}: x's last axis {x_axes[-1]!r} does not match "
                f"weight's in_features axis {in_axis!r}"
            )
        return (*x_axes[:-1], out_axis)

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
        """RMSNorm keeps x's shape, dtype and axes; the result is an activation."""

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

        # Placement checks come before the size check, for the same reason as
        # in LinearOp: a sharded x would also fail the size check, with a less
        # helpful message.
        #
        # Every rank scales its rows by the whole weight vector.
        if not isinstance(self.weight.type.placement, Replicate):
            raise ValueError(
                f"{self.NAME}: weight must be replicated, "
                f"got {self.weight.type.placement}"
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

        if self.x.type.shape[-1] != self.weight.type.shape[0]:
            raise ValueError(
                f"{self.NAME}: weight size must match "
                f"x.shape[-1], got {self.weight.type.shape[0]} "
                f"and {self.x.type.shape[-1]}"
            )

        # The weight scales each element of x's last dimension, so once both
        # are annotated it must be indexed by that same axis.
        x_axes = self.x.type.axes
        weight_axes = self.weight.type.axes
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
class AllReduceOp(Operation):
    """Sum a partial tensor over all tensor-parallel ranks.

    Every rank contributes its partial sum and receives the total, so the
    result is replicated. Shape, dtype, role and axes are unchanged.

    AllReduceOp(operands=(x,))
    """

    NAME = "aloi.all_reduce"
    NUM_OPERANDS = 1

    @property
    def x(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result is x's type with a replicated placement.

        Raises:
            ValueError: If x is not a partial sum. Summing a replicated
                tensor over tp ranks would multiply it by tp, and summing a
                sharded one would add unrelated slices.
        """

        if not isinstance(self.x.type.placement, Partial):
            raise ValueError(
                f"{self.NAME}: operand must be a partial sum, "
                f"got {self.x.type.placement}"
            )
        return (replace(self.x.type, placement = Replicate()),)

    