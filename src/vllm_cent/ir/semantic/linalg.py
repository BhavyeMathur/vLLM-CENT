"""Linear-algebra ops: contractions over a named axis.

A contraction sums products over one axis. When that axis is split
across ranks, each rank holds a partial sum of the result.
"""

from __future__ import annotations

from dataclasses import dataclass

from ..base import Operation, Value
from ..types import Partial, Placement, Replicate, Shard, TensorType

__all__ = [
    "LinearOp",
]


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
