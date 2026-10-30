"""Llama decoder layer (Llama 2 / 3 / 3.1) for the ALOI compiler.

Model files: ``llama2_70b.json`` and ``llama3_1_70b.json`` (Hugging Face
config.json), ``model.py`` and ``annotations.py`` (code), ``registry.py``
(registration). Run specs live in ``runs/``.
"""

from .annotations import axis_sizes, semantic_annotations
from .config import LlamaConfig
from .model import (
    LlamaAttention,
    LlamaDecoderLayer,
    LlamaMLP,
    build_meta_decoder_layer,
    example_inputs,
)

__all__ = [
    "LlamaAttention",
    "LlamaConfig",
    "LlamaDecoderLayer",
    "LlamaMLP",
    "axis_sizes",
    "build_meta_decoder_layer",
    "example_inputs",
    "semantic_annotations",
]
