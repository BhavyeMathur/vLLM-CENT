"""Lower Semantic IR to the Device IR ops a target device supports.

Which device ops a target runs comes from its ``DeviceSpec``; this package
chooses, for each semantic op, a supported device op or a decomposition.
"""

from .lower import LowerToDevice

__all__ = ["LowerToDevice"]
