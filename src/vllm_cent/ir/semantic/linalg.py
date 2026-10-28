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
    """Linear transform of the last dimension, with an optional bias.

    ``y = x @ weight.T``, plus ``bias`` added to every output row when given.
    The weight uses the ``nn.Linear`` layout ``[out_features, in_features]``,
    and the bias has ``out_features`` elements.

    LinearOp(operands=(x, weight))
    LinearOp(operands=(x, weight, bias))
    """

    NAME = "aloi.linear"
    # x and weight, and optionally a bias.
    NUM_OPERANDS = (2, 3)

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """Matrix multiplication on inner-most dimensions of in_feature with weight"""

        if (self.x.type.dtype != self.weight.type.dtype):
            raise ValueError("Operands should have the same datatype")

        if (len(self.x.type.shape) < 1):
            raise ValueError("x would have dimensions of [*, in_features]")
        
        if (len(self.weight.type.shape) != 2):
            raise ValueError("Weight should have 2 dimensions [out_features, in_features]")

        bias = self.bias
        if bias is not None and bias.type.dtype != self.x.type.dtype:
            raise ValueError("Bias should have the same datatype as x")
        if bias is not None and len(bias.type.shape) != 1:
            raise ValueError("Bias should have 1 dimension [out_features]")

        # Check placements before shapes: when only one operand is sharded,
        # the local shapes disagree too, and "unsupported placements" says
        # what is actually wrong.
        placement = self._result_placement()
        self._check_bias_placement(placement)

        if (self.x.type.shape[-1] != self.weight.type.shape[-1]):
            raise ValueError("Weight should have shape [out_features, in_features]")

        if bias is not None and bias.type.shape[0] != self.out_features:
            raise ValueError(
                f"{self.NAME}: bias has {bias.type.shape[0]} elements, "
                f"expected out_features {self.out_features}"
            )

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

    def _check_bias_placement(self, result_placement: Placement) -> None:
        """Check that each rank holds the part of the bias its result needs.

        The bias is added to each rank's local result, so it must be split
        exactly like that result: whole for a whole result, and split along
        the same axis for a column-parallel result.

        Args:
            result_placement: The placement derived from x and the weight.

        Raises:
            ValueError: If the result is a partial sum (row parallel): every
                rank would add the whole bias, so the all_reduce would add it
                tp times. The bias must be added after the all_reduce instead.
                Also if the bias is split differently from the result.
        """

        # TODO(row-parallel bias): A model with a biased row-parallel layer
        # (e.g. an o_proj with bias) needs the bias added after the
        # all_reduce. The TP pass could split such a linear into a bias-free
        # linear and an add; it does not yet, so the combination is rejected.
        bias = self.bias
        if bias is None:
            return
        if isinstance(result_placement, Partial):
            raise ValueError(
                f"{self.NAME}: the result is a partial sum, so adding the bias "
                "on every rank would add it tp times; add it after the all_reduce"
            )
        if bias.type.placement != result_placement:
            raise ValueError(
                f"{self.NAME}: bias placement {bias.type.placement} must match "
                f"the result's {result_placement}"
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
        # The bias is indexed by the output features, so once both are
        # annotated it must name the weight's out_features axis.
        bias_axes = () if self.bias is None else self.bias.type.axes
        if weight_axes and bias_axes and bias_axes != (weight_axes[0],):
            raise ValueError(
                f"{self.NAME}: bias axes {bias_axes} must be the weight's "
                f"out_features axis ({weight_axes[0]!r},)"
            )
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
    def bias(self) -> Value | None:
        return self.operands[2] if len(self.operands) == 3 else None

    @property
    def in_features(self) -> int:
        return self.weight.type.shape[1]

    @property
    def out_features(self) -> int:
        return self.weight.type.shape[0]
