"""Build Llama tensor calculations without choosing hardware or storage.

These graph builders use the staged compiler's semantic operations. They do
not call the older structural instruction lowerers, load weights, or execute
data. A caller chooses a target and compiles the returned graph separately.
"""

from ...activations import ActivationKind
from ...pipeline.ir import SemanticGraph, TensorType, Value, verify_semantic
from ...pipeline.operations import (
    ActivationOperation,
    AddOperation,
    LinearOperation,
    MultiplyOperation,
    RmsNormOperation,
)
from .spec import LlamaModelSpec

__all__ = ["build_feed_forward_graph"]


def build_feed_forward_graph(
    model: LlamaModelSpec,
    *,
    epsilon: float,
    batch_shape: tuple[int, ...] = (),
    block: str = "feed_forward",
) -> SemanticGraph:
    """Build normalized, SiLU-gated projections followed by a residual addition.

    The caller supplies five tensors: ``x``, ``norm_weight``, ``gate_weight``,
    ``up_weight``, and ``down_weight``. Projection matrices store one output's
    weights per row. The returned ``output`` has the same shape as ``x``.
    Leading input dimensions describe independent vectors sharing the weights.

    Args:
        model: Llama dimensions; this subgraph uses ``hidden_size`` and
            ``intermediate_size``. Attention-head counts do not affect it.
        epsilon: Positive finite constant added to the RMS mean square.
        batch_shape: Leading dimensions of ``x``. An empty tuple requests one
            vector; ``(2, 3)`` requests six vectors arranged in two groups.
        block: Source label attached to every operation for diagnostics.

    Returns:
        Verified graph for ``x + down(SiLU(gate(RMSNorm(x))) * up(RMSNorm(x)))``.
        It declares shapes and dependencies but contains no tensor data, device
        addresses, or compilation target. Compile it with the explicit reference
        contract to run it in the current functional simulator.

    Raises:
        ValueError: If a batch dimension is empty or negative, or ``epsilon`` is
            non-finite or not positive.

    Examples:
        For hidden width three and intermediate width five, input ``x`` is
        ``(3,)``, ``norm_weight`` is ``(3,)``, both gate/up matrices are
        ``(5, 3)``, and ``down_weight`` is ``(3, 5)``. The result is ``(3,)``.
    """
    hidden = model.hidden_size
    intermediate = model.intermediate_size
    input_type = TensorType((*batch_shape, hidden))
    expanded_type = TensorType((*batch_shape, intermediate))
    graph = SemanticGraph(
        inputs=(
            Value("x", input_type),
            Value("norm_weight", TensorType((hidden,))),
            Value("gate_weight", TensorType((intermediate, hidden))),
            Value("up_weight", TensorType((intermediate, hidden))),
            Value("down_weight", TensorType((hidden, intermediate))),
        ),
        operations=(
            RmsNormOperation(
                name="normalize",
                inputs=("x", "norm_weight"),
                output=Value("normalized", input_type),
                epsilon=epsilon,
                block=block,
            ),
            # Both projections read normalized. Its storage must survive the
            # first projection, and x must survive until the final residual.
            LinearOperation(
                name="gate_projection",
                inputs=("normalized", "gate_weight"),
                output=Value("gate", expanded_type),
                block=block,
            ),
            LinearOperation(
                name="up_projection",
                inputs=("normalized", "up_weight"),
                output=Value("up", expanded_type),
                block=block,
            ),
            ActivationOperation(
                name="activate_gate",
                inputs=("gate",),
                output=Value("activated_gate", expanded_type),
                function=ActivationKind.SILU,
                block=block,
            ),
            MultiplyOperation(
                name="gate_product",
                inputs=("activated_gate", "up"),
                output=Value("gated", expanded_type),
                block=block,
            ),
            LinearOperation(
                name="down_projection",
                inputs=("gated", "down_weight"),
                output=Value("projected", input_type),
                block=block,
            ),
            AddOperation(
                name="residual",
                inputs=("x", "projected"),
                output=Value("output", input_type),
                block=block,
            ),
        ),
        outputs=("output",),
    )
    verify_semantic(graph)
    return graph
