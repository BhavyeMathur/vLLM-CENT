"""Mixture-of-experts ops: choosing experts per token and running them.

A mixture-of-experts (MoE) layer has several feed-forward networks, the
experts. A router scores every expert for every token; only the best
``top_k`` experts run for that token, weighted by their scores.

These ops describe the computation densely: routing weights are a full
``[*tokens, experts]`` tensor that is zero for unchosen experts, and the
expert op is defined as a sum over all experts. A zero weight removes an
expert's contribution exactly, so the result equals running only the chosen
experts. Which experts run is data-dependent; deciding how to skip the zero
ones is left to lowering.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from ...base import Operation, Value
from ...types import Partial, Placement, Replicate, Shard, TensorRole, TensorType

__all__ = [
    "MoeExpertsOp",
    "TopKSigmoidOp",
]


@dataclass(frozen=True, eq=False, slots=True)
class TopKSigmoidOp(Operation):
    """Keep each token's ``top_k`` largest router logits, squashed by a sigmoid.

    The result has the logits' shape: ``sigmoid(logit)`` for the ``top_k``
    largest logits of each token (last dimension), and 0 for every other
    expert. This is Llama 4's routing: scores are not normalized to sum to 1.

    TopKSigmoidOp(operands=(router_logits,), top_k=1)

    Attributes:
        top_k: Experts chosen per token.

    Examples:
        Logits ``(2.0, -1.0, 0.5)`` with ``top_k=1`` give
        ``(sigmoid(2.0), 0, 0) = (0.881, 0, 0)``.
    """

    NAME = "aloi.topk_sigmoid"
    NUM_OPERANDS = 1

    top_k: int

    @property
    def logits(self) -> Value:
        return self.operands[0]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result has the logits' shape, dtype, axes and placement.

        Raises:
            ValueError: If the logits have no expert dimension, top_k is not
                between 1 and the number of experts, the logits are a partial
                sum, or the experts are split across ranks: choosing the
                largest needs every expert's logit.
        """

        logits = self.logits.type
        if logits.rank < 1:
            raise ValueError(f"{self.NAME}: logits need an expert dimension")
        experts = logits.full_shape[-1]
        if not 1 <= self.top_k <= experts:
            raise ValueError(
                f"{self.NAME}: top_k must be between 1 and {experts}, got {self.top_k}"
            )
        if isinstance(logits.placement, Partial):
            raise ValueError(f"{self.NAME}: logits are a partial sum; all_reduce first")
        if logits.sharded_dim == logits.rank - 1:
            raise ValueError(
                f"{self.NAME}: the experts are split across ranks; choosing "
                "the top_k needs every expert's logit"
            )
        return (replace(logits, role=TensorRole.ACTIVATION),)


@dataclass(frozen=True, eq=False, slots=True)
class MoeExpertsOp(Operation):
    """Sum the SwiGLU experts' outputs, weighted by the routing weights.

    For tokens ``x [*tokens, hidden]`` and weights ``w [*tokens, experts]``,
    expert ``e`` computes ``down[e] @ (silu(gate[e] @ u) * (up[e] @ u))`` of
    its input ``u``. With ``apply_router_weight_on_input``, ``u = w[e] * x``
    and outputs are summed as they are (Llama 4). Otherwise ``u = x`` and each
    output is multiplied by ``w[e]`` before the sum (Mixtral). Weights use the
    ``nn.Linear`` layout, one matrix per expert.

    MoeExpertsOp(
        operands=(x, routing_weights, gate_proj, up_proj, down_proj),
        apply_router_weight_on_input=True,
    )

    Operands:
        x: ``[*tokens, hidden]``.
        routing_weights: ``[*tokens, experts]``; 0 for unchosen experts.
        gate_proj: ``[experts, intermediate, hidden]``.
        up_proj: ``[experts, intermediate, hidden]``.
        down_proj: ``[experts, hidden, intermediate]``.

    Attributes:
        apply_router_weight_on_input: Scale each expert's input (True) or its
            output (False) by the routing weight.
    """

    NAME = "aloi.moe_experts"
    NUM_OPERANDS = 5

    apply_router_weight_on_input: bool

    @property
    def x(self) -> Value:
        return self.operands[0]

    @property
    def routing_weights(self) -> Value:
        return self.operands[1]

    @property
    def gate_proj(self) -> Value:
        return self.operands[2]

    @property
    def up_proj(self) -> Value:
        return self.operands[3]

    @property
    def down_proj(self) -> Value:
        return self.operands[4]

    def infer_result_types(self) -> tuple[TensorType, ...]:
        """The result has x's shape, dtype and axes.

        Raises:
            ValueError: If dtypes differ, the shapes or axes do not describe
                the layout above, or the placements are not one of the
                supported combinations (see ``_result_placement``).
        """

        x = self.x.type
        weights = self.routing_weights.type
        gate, up, down = (v.type for v in self.operands[2:])
        if any(t.dtype != x.dtype for t in (weights, gate, up, down)):
            raise ValueError(f"{self.NAME}: all operands need one dtype")
        if x.rank < 1 or weights.rank != x.rank or any(
            t.rank != 3 for t in (gate, up, down)
        ):
            raise ValueError(
                f"{self.NAME}: expected x [*, hidden], routing_weights "
                "[*, experts] and three [experts, *, *] weights"
            )
        placement = self._result_placement()
        experts, intermediate, hidden = gate.shape
        if not (
            up.shape == gate.shape
            and down.shape == (experts, hidden, intermediate)
            and x.shape[-1] == hidden
            and weights.shape == (*x.shape[:-1], experts)
        ):
            raise ValueError(
                f"{self.NAME}: x {x.shape}, routing_weights {weights.shape}, "
                f"gate {gate.shape}, up {up.shape}, down {down.shape} do not fit"
            )
        if x.axes and weights.axes and gate.axes and up.axes and down.axes:
            expert_axis, intermediate_axis, hidden_axis = gate.axes
            if not (
                up.axes == gate.axes
                and down.axes == (expert_axis, hidden_axis, intermediate_axis)
                and x.axes[-1] == hidden_axis
                and weights.axes == (*x.axes[:-1], expert_axis)
            ):
                raise ValueError(
                    f"{self.NAME}: axes x {x.axes}, routing_weights "
                    f"{weights.axes}, gate {gate.axes}, up {up.axes}, "
                    f"down {down.axes} do not fit"
                )
        return (
            TensorType(
                shape=x.shape, dtype=x.dtype, axes=x.axes, placement=placement
            ),
        )

    def _result_placement(self) -> Placement:
        """Derive which part of the result each tensor-parallel rank holds.

        - Everything replicated: every rank computes the whole result.
        - Tokens and routing weights replicated, and the experts' intermediate
          axis split (gate/up rows, down columns): each rank runs its slice of
          every expert's hidden layer, a partial sum that an all_reduce must
          complete, like a row-parallel ``down_proj``.

        Returns:
            The result's placement.

        Raises:
            ValueError: For any other combination.
        """

        # TODO(expert parallelism): Splitting the expert axis gives each rank
        # whole experts, which also yields a partial sum. It needs a rank-local
        # slice of the replicated routing weights; the IR has no such op yet.
        x = self.x.type
        weights = self.routing_weights.type
        gate, up, down = (v.type for v in self.operands[2:])
        if isinstance(x.placement, Replicate) and isinstance(
            weights.placement, Replicate
        ):
            match gate.placement, up.placement, down.placement:
                case Replicate(), Replicate(), Replicate():
                    return Replicate()
                case Shard(a), Shard(b), Shard(c) if (
                    a == b == c
                    and gate.dim_of(a) == 1
                    and up.dim_of(a) == 1
                    and down.dim_of(a) == 2
                ):
                    return Partial()
        raise ValueError(
            f"{self.NAME}: unsupported placements x {x.placement}, routing_weights "
            f"{weights.placement}, gate {gate.placement}, up {up.placement}, "
            f"down {down.placement}; only the experts' intermediate axis may be split"
        )
