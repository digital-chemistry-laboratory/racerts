from .embedder import BaseEmbedder, BoundsMatrixEmbedder, CmapEmbedder

embedders = {"dm": BoundsMatrixEmbedder, "cmap": CmapEmbedder, "base": BaseEmbedder}
