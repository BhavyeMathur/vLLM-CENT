"""ALOI for CENT: compile PyTorch models into CENT kernels and AiM traces.

- ``frontend``: torch.export graphs to Semantic IR.
- ``ir``: the Semantic, Device and Tile IR, their printer and passes.
- ``parallel``, ``legalize``: tensor parallelism; lowering to a device's ops.
- ``device``: device specs (geometry, supported ops), read from YAML.
- ``tile``: physical layouts, the Tile DSL, kernel verification, expansion
  into CENT instructions, and AiM trace generation.
- ``cent``: CENT's hardware description and typed instructions.
- ``driver``: ``aloi-compile``, from a model and a run spec to every stage.

Importing this package loads no model and no torch.
"""
