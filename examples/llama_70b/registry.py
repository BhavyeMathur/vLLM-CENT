"""Register the Llama family with the ALOI compiler.

The compiler imports this file when it loads a config.json from this
directory, then picks the family by the config's ``model_type``.
"""

from __future__ import annotations

from vllm_cent.pipeline import register_model

from .annotations import axis_sizes, semantic_annotations
from .config import LlamaConfig
from .model import build_meta_decoder_layer, example_inputs

__all__: list[str] = []

register_model(
    "llama",
    parse_config=LlamaConfig.from_hf,
    build_model=build_meta_decoder_layer,
    example_inputs=lambda config, batch, context_len: example_inputs(
        config, context_len=context_len, batch=batch
    ),
    annotations=lambda config: semantic_annotations(),
    axis_sizes=axis_sizes,
)
