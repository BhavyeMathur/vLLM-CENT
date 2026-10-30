"""Llama 4 text decoder layer (Scout: 16 experts, top-1) for the ALOI compiler.

Model files: ``llama4_scout.json`` (the Hugging Face config.json of
Llama-4-Scout-17B-16E-Instruct, without its ``vision_config``: only the text
model is compiled), ``model.py`` and ``annotations.py`` (code),
``registry.py`` (registration). Run specs live in ``runs/``: layer 0 is a
RoPE layer, layer 3 a NoPE layer.
"""

from .annotations import axis_sizes, semantic_annotations
from .config import Llama4Config
from .model import (
    Llama4Attention,
    Llama4DecoderLayer,
    Llama4MLP,
    Llama4MoE,
    build_meta_decoder_layer,
    example_inputs,
)

__all__ = [
    "Llama4Attention",
    "Llama4Config",
    "Llama4DecoderLayer",
    "Llama4MLP",
    "Llama4MoE",
    "axis_sizes",
    "build_meta_decoder_layer",
    "example_inputs",
    "semantic_annotations",
]
