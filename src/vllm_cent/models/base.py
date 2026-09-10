"""Common interface for source-model descriptions."""

__all__ = ["ModelSpec"]


class ModelSpec:
    """Base class for model architecture descriptions.

    The compiler uses the subclass, such as ``LlamaModelSpec``, to choose the
    code that translates that model family into CENT instructions.
    """
