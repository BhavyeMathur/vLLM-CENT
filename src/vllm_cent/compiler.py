"""Model-dispatching public compiler API."""

from .cent import CentProgram
from .models.llama import LlamaModelSpec, compile_llama_transformer_block
from .request import CompileRequest

__all__ = ["compile_transformer_block"]

# TODO(frontend API): Define how model compilers are registered.
#
# An isinstance check is enough while Llama is the only supported family. A
# second family will need a public way to register its compiler and report what
# model features it supports.


def compile_transformer_block(request: CompileRequest) -> CentProgram:
    """Compile one supported model block into CENT commands.

    Args:
        request: Model, hardware, placement, and decode-step information.

    Returns:
        An ordered CENT instruction program that passes structural validation.
        TODOs in the model compiler mark dataflow that is not executable yet.

    Raises:
        TypeError: If no compiler frontend supports the model description.
        ValueError: If the selected frontend rejects dimensions or capacity.
    """

    # The request's model type chooses the matching compiler. isinstance also
    # allows a Llama specification to add fields without losing Llama behavior.
    if isinstance(request.model, LlamaModelSpec):
        return compile_llama_transformer_block(request)
    raise TypeError(f"unsupported model specification: {type(request.model).__name__}")
