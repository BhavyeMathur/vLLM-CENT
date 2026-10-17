"""Common interface for source-model descriptions."""

__all__ = ["ModelSpec"]


class ModelSpec:
    """Identify which model family a block specification describes.

    The compiler uses the subclass, such as ``LlamaModelSpec``, to choose the
    code that translates that model family into CENT instructions.
    """
