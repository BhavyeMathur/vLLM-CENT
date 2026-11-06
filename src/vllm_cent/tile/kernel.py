"""CENT Tile IR kernel: a Tile IR function paired with its parameter layouts."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from ..ir.base import Function, Value
from .layout import PhysicalLayout

__all__ = ["CentKernel"]


@dataclass(frozen=True, slots=True)
class CentKernel:
    """A CENT Tile IR function and the physical layouts of its storage parameters.

    Attributes:
        function: The Tile IR function defining the kernel's loop nest and ops.
        layouts: Maps each function argument (in ``pnm.sb`` or ``pim.dram``) to
            its resolved ``PhysicalLayout``.
    """

    function: Function
    layouts: Mapping[Value, PhysicalLayout]

    def __post_init__(self) -> None:
        """Copy ``layouts`` into a read-only mapping."""
        object.__setattr__(self, "layouts", MappingProxyType(dict(self.layouts)))
