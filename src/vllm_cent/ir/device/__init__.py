"""Device IR: where each computation runs.

Device IR ops implement semantic ops on one execution domain (PIM, PNM), move
tensors between domains (``device.transfer``), or communicate between devices
(``cxl.*``). Transformer-only ops are in ``transformer``. They reuse the semantic ops' type rules and add the domain rules
in ``domain``. Which channel, bank or buffer holds a tensor is decided later,
by mapping.
"""

from .cxl import CXLAllReduceOp
from .data import (
    DeviceBroadcastOp,
    DeviceCopyOp,
    DeviceMergeHeadsOp,
    DeviceReshapeOp,
    DeviceSplitHeadsOp,
    DeviceTransferOp,
    ViewOp,
)
from .domain import DeviceOp
from .pim import PIMActivationOp, PIMEwmulOp, PIMGemvOp
from .pnm import (
    PNMActivationOp,
    PNMAddOp,
    PNMDivOp,
    PNMExpOp,
    PNMMaxOp,
    PNMMulOp,
    PNMRMSNormOp,
    PNMSoftmaxOp,
    PNMSubOp,
    PNMSumOp,
)
from .transformer import PIMCacheWriteOp, PIMQKOp, PIMSVOp, PNMRopeOp

__all__ = [
    "CXLAllReduceOp",
    "DeviceBroadcastOp",
    "DeviceCopyOp",
    "DeviceMergeHeadsOp",
    "DeviceOp",
    "DeviceReshapeOp",
    "DeviceSplitHeadsOp",
    "DeviceTransferOp",
    "PIMActivationOp",
    "PIMCacheWriteOp",
    "PIMEwmulOp",
    "PIMGemvOp",
    "PIMQKOp",
    "PIMSVOp",
    "PNMActivationOp",
    "PNMAddOp",
    "PNMDivOp",
    "PNMExpOp",
    "PNMMaxOp",
    "PNMMulOp",
    "PNMRMSNormOp",
    "PNMRopeOp",
    "PNMSoftmaxOp",
    "PNMSubOp",
    "PNMSumOp",
    "ViewOp",
]
