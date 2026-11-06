"""CENT kernel mappings for Llama 70B, written in the Tile DSL.

Each module maps Device IR ops onto Tile DSL kernels that follow CENT's own
mapping (``../ALOI/cent_simulation``): ``gemv`` holds the GEMV schedules,
``llama3_1_70b`` maps one decoder layer of the evaluated configuration and
writes its AiM trace.
"""
