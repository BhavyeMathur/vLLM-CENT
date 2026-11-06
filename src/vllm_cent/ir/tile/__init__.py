"""Tile IR: structured loops over tiles, below Device IR."""

from .cent import CentTensorType, CentTileOp
from .tile import INDEX, TileForOp

__all__ = ["INDEX", "CentTensorType", "CentTileOp", "TileForOp"]
