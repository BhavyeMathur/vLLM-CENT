"""IR passes: rewrites of a Module into a new Module.

The Pass interface and the PassManager that runs a sequence of passes are in
``vllm_cent.ir.pass_manager``. The passes here are, in pipeline order:

- ``AnnotateRolesAndAxes``: attach the model's roles and axes (Semantic IR).
- ``ApplyTensorParallel``: split tensors across ranks; insert all-reduces.
- ``DecomposeOps``: replace composite ops by the ops that define them.
- ``FuseAttention``: an adaptor for the CENT pipeline's fused attention.
- ``ClonePass``: copy the IR unchanged, for testing.
"""

from .annotate import AnnotateRolesAndAxes, TensorAnnotation
from .clone import ClonePass
from .decompose import DecomposeOps
from .fuse_attention import FuseAttention
from .tensor_parallel import ApplyTensorParallel

__all__ = [
    "AnnotateRolesAndAxes",
    "ApplyTensorParallel",
    "ClonePass",
    "DecomposeOps",
    "FuseAttention",
    "TensorAnnotation",
]
