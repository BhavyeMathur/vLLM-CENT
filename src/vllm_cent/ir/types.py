"""Types shared by all ALOI IR levels"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, IntEnum, auto
from typing import assert_never

class DType(StrEnum):
    INT64 = "int64"
    INT32 = "int32"
    FP32 = "fp32"
    BF16 = "bf16"
    FP16 = "fp16"
    INT8 = "int8"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    FP4 = "fp4"

    @property
    def bits(self) -> int:
        match self:
            case DType.INT64:
                return 64
            case DType.FP32 | DType.INT32:
                return 32
            case DType.FP16 | DType.BF16:
                return 16
            case DType.INT8 | DType.FP8_E4M3 | DType.FP8_E5M2:
                return 8
            case DType.FP4:
                return 4
        
        assert_never(self)

    @classmethod
    def normalize(cls, dtype: object) -> DType:
        """Return stable ALOI spelling for a dtype-like object

        Accepts a ``DType``, its value (``"fp16"``), a ``torch.dtype``, or the
        torch name with or without the ``torch.`` prefix (``"torch.float16"``,
        ``"half"``). Matching goes through ``str()``, so this module never
        imports torch.
        """

        text = str(dtype).lower().replace("torch.", "")
        # Keys are torch dtype names (str(torch.float16) == "torch.float16")
        # and torch's short aliases (torch.half is torch.float16). The torch
        # names int64, int32 and int8 already equal ALOI values, so they need
        # no entry.
        #
        # Look-alike torch dtypes are deliberately missing, so they raise:
        # - float8_e4m3fnuz / float8_e5m2fnuz use a different encoding (no
        #   negative zero or infinity, different exponent bias).
        # - float4_e2m1fn_x2 packs two fp4 values into one element, so its
        #   shape counts pairs, not fp4 values; it is not DType.FP4.
        aliases = {
            "long": "int64",
            "int": "int32",
            "float32": "fp32",
            "float": "fp32",
            "float16": "fp16",
            "half": "fp16",
            "bfloat16": "bf16",
            "float8_e4m3fn": "fp8_e4m3",
            "float8_e5m2": "fp8_e5m2",
        }

        text = aliases.get(text, text)
        try:
            return cls(text)
        except ValueError as e:
            raise ValueError(f"Unsupported dtype: {dtype!r}") from e

class TensorRole(StrEnum):
    ACTIVATION = auto()
    Q_WEIGHT = auto()
    K_WEIGHT = auto()
    V_WEIGHT = auto()
    QUERY = auto()
    KEY = auto()
    VALUE = auto()
    K_CACHE = auto()
    V_CACHE = auto()
    O_WEIGHT = auto()

@dataclass(frozen = True)
class TensorType:
    """Tensor shape + semantics + physical annotations

    ''shape'' is the current tensor shape after sharding
    ''global shape'' is the shape of the whole tensor before sharding, set only after parallel plan is applied
    """

    shape: tuple[int, ...]
    dtype: DType
    role: TensorRole = TensorRole.ACTIVATION
    axes: tuple[str, ...] = ()
    global_shape: tuple[int, ...] | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "shape", tuple(int(dim) for dim in self.shape))
        object.__setattr__(self, "dtype", DType.normalize(self.dtype))
        object.__setattr__(self, "axes", tuple(self.axes))

        if any(dim < 0 for dim in self.shape):
            raise ValueError(f"tensor dimensions must be non-negative: {self.shape}")
        if self.axes and len(self.axes) != len(self.shape):
            raise ValueError(f"{len(self.axes)} axes do not describe rank-{len(self.shape)} tensor")
        if self.global_shape is not None and len(self.global_shape) != len(self.shape):
            raise ValueError("global and local tensor ranks differ")

    @property
    def rank(self) -> int:
        return len(self.shape)
