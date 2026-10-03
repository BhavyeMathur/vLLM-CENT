"""PyTorch frontend: export a model and import it as ALOI IR.

This is the only package that imports torch.
"""

from .importer import import_exported_program
from .torch_export import export_model

__all__ = ["export_model", "import_exported_program"]
