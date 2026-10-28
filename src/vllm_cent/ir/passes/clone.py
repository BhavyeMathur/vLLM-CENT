"""A pass that copies the IR, for testing the pass infrastructure."""

from __future__ import annotations

from ..base import Function, clone_function
from ..pass_manager import FunctionPass

__all__ = ["ClonePass"]


class ClonePass(FunctionPass):
    """Copy every function with fresh values; the IR itself is unchanged."""

    NAME = "clone"

    def run_on_function(self, function: Function) -> Function:
        return clone_function(function)
