"""racerts.embedder of legacy racerts."""

from .embedder import BaseEmbedder, BoundsMatrixEmbedder, CmapEmbedder

embedders = {"dm": BoundsMatrixEmbedder, "cmap": CmapEmbedder, "base": BaseEmbedder}

__all__ = ["BaseEmbedder", "BoundsMatrixEmbedder", "CmapEmbedder", "embedders"]
