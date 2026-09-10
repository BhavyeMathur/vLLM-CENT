"""Source-model interfaces and supported model families."""

from .base import ModelSpec
from .llama import LlamaModelSpec

__all__ = ["LlamaModelSpec", "ModelSpec"]
