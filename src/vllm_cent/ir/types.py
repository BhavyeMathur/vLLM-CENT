"""Types shared by all ALOI IR levels"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum, auto
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
    """What kind of tensor a value is, independent of the model.

    The role says how a tensor lives on the device, not which tensor it is.
    Which tensor it is (query vs. key, q_proj vs. o_proj) is told by its axes:
    tensor parallelism shards along a named axis, wherever that axis sits.

    Members:
        ACTIVATION: Computed during the forward pass and consumed by later
            ops. Every op result is an activation; so is the model input.
        WEIGHT: A parameter loaded once and kept resident on the device.
    """

    ACTIVATION = auto()
    WEIGHT = auto()


# Joins the names of semantic axes that share one flat dimension.
_PACKED_AXIS_SEPARATOR = "*"


def packed_axis(*axes: str) -> str:
    """Name one flat dimension that packs several semantic axes.

    The first axis is the outermost. For example, q_proj's output dimension
    holds 64 heads of 128 elements, head after head, so its axis is
    ``packed_axis("q_head", "head_dim") == "q_head*head_dim"``. Sharding along
    ``q_head`` then splits that dimension into whole heads.

    Args:
        *axes: Axis names, outermost first.

    Returns:
        The packed axis name.

    Raises:
        ValueError: If fewer than two axes are given, or an axis is empty or
            already packed.
    """

    if len(axes) < 2:
        raise ValueError(f"a packed axis combines at least two axes, got {axes}")
    for axis in axes:
        if not axis or _PACKED_AXIS_SEPARATOR in axis:
            raise ValueError(f"cannot pack axis {axis!r}")
    return _PACKED_AXIS_SEPARATOR.join(axes)


@dataclass(frozen = True)
class TensorType:
    """Tensor shape + semantics + physical annotations

    ''shape'' is the current tensor shape after sharding
    ''global shape'' is the shape of the whole tensor before sharding, set only after parallel plan is applied

    ''axes'' names the meaning of each dimension, e.g. ("batch", "seq_len",
    "hidden"). It is empty until AnnotateRolesAndAxes runs; after that every
    dimension has a name. A dimension that packs several axes uses a name
    made by packed_axis(), e.g. "q_head*head_dim".
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
