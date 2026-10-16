"""Expose model descriptions used to select and configure a block compiler."""

from .base import ModelSpec
from .llama import LlamaModelSpec

__all__ = ["LlamaModelSpec", "ModelSpec"]
