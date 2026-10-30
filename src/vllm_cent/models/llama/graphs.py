"""Build Llama tensor calculations without choosing hardware or storage.

These graph builders write ``vllm_cent.ir`` functions by hand, with the same
operations the PyTorch importer produces. They do not call the older
structural instruction lowerers, load weights, or execute data. A caller
chooses a target and compiles the returned function separately.
"""

from math import isfinite

from ...activations import ActivationKind
from ...ir import (
    ActivationOp,
    AddOp,
    DType,
    Function,
    LinearOp,
    MulOp,
    RMSNormOp,
    TensorType,
    Value,
)
from .spec import LlamaModelSpec

__all__ = ["build_feed_forward_graph"]


def build_feed_forward_graph(
    model: LlamaModelSpec,
    *,
    epsilon: float,
    batch_shape: tuple[int, ...] = (),
    block: str = "feed_forward",
) -> Function:
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
        block: Model-block label. Every result name starts with it, as a
            module path does: ``feed_forward.gate``, ...,
            ``feed_forward.output``. An empty label leaves the plain names.

    Returns:
        Verified function for ``x + down(SiLU(gate(RMSNorm(x))) * up(RMSNorm(x)))``.
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
    if any(dim <= 0 for dim in batch_shape):
        raise ValueError("Batch dimensions must be positive")
    if not isfinite(epsilon) or epsilon <= 0:
        raise ValueError("RMSNorm epsilon must be positive and finite")
    hidden = model.hidden_size
    intermediate = model.intermediate_size


    def label(name: str) -> str:
        """Prefix a result name with the block label, as module paths do.

        Args:
            name: Result name inside the block, such as ``gate``.

        Returns:
            ``block.name``, or ``name`` when the block label is empty.
        """
        return f"{block}.{name}" if block else name

    x = _fp32_argument("x", (*batch_shape, hidden))
    norm_weight = _fp32_argument("norm_weight", (hidden,))
    gate_weight = _fp32_argument("gate_weight", (intermediate, hidden))
    up_weight = _fp32_argument("up_weight", (intermediate, hidden))
    down_weight = _fp32_argument("down_weight", (hidden, intermediate))

    normalize = RMSNormOp(
        operands=(x, norm_weight), eps=epsilon, name_hint=label("normalized")
    )
    (normalized,) = normalize.results
    # Both projections read normalized. Its storage must survive the first
    # projection, and x must survive until the final residual.
    gate_projection = LinearOp(
        operands=(normalized, gate_weight), name_hint=label("gate")
    )
    up_projection = LinearOp(operands=(normalized, up_weight), name_hint=label("up"))
    activate_gate = ActivationOp(
        operands=gate_projection.results,
        function=ActivationKind.SILU,
        name_hint=label("activated_gate"),
    )
    gate_product = MulOp(
        operands=(*activate_gate.results, *up_projection.results),
        name_hint=label("gated"),
    )
    down_projection = LinearOp(
        operands=(*gate_product.results, down_weight), name_hint=label("projected")
    )
    residual = AddOp(
        operands=(x, *down_projection.results), name_hint=label("output")
    )
    function = Function(
        name="feed_forward",
        operands=(x, norm_weight, gate_weight, up_weight, down_weight),
        operations=(
            normalize,
            gate_projection,
            up_projection,
            activate_gate,
            gate_product,
            down_projection,
            residual,
        ),
        results=residual.results,
    )
    function.verify()
    return function


def _fp32_argument(name: str, shape: tuple[int, ...]) -> Value:
    """Create a named fp32 function argument.

    Args:
        name: Argument name; the compiled program's input of the same name.
        shape: Tensor dimensions.

    Returns:
        A new value with no semantic annotations.
    """
    return Value(type=TensorType(shape=shape, dtype=DType.FP32), name=name)
