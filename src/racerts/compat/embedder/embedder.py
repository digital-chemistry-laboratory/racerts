"""
racerts.embedder.embedder of legacy racerts: the embedders with the legacy method
embed_TS, as subclasses of those of racerts.embed.
"""

import racerts.embed
from racerts.task import FrozenSet

from .utils import get_bounds_matrix, print_bounds_matrix_errors

__all__ = [
    "BaseEmbedder",
    "BoundsMatrixEmbedder",
    "CmapEmbedder",
    "get_bounds_matrix",
    "print_bounds_matrix_errors",
]


class BaseEmbedder(racerts.embed.BaseEmbedder):
    """
    A legacy racerts embedder: subclasses implement embed_TS. The stages call embed,
    which runs embed_TS, so legacy embedders and legacy subclasses of the built-in ones
    work there too.
    """

    # For subclasses whose own __init__ does not set them: verbose, which embed passes
    # on, and the settings that are new since legacy racerts.
    verbose = False
    etkdg = False
    chirality_fallback = True
    sequential_seeds = False

    def embed(self, mol, reference, frozen, n):
        return self.embed_TS(
            mol_ts=reference,
            mol=mol,
            reacting_atoms=list(frozen.core),
            frozen_atoms=list(frozen.hard),
            n=n,
            verbose=self.verbose,
        )

    def embed_TS(self, mol_ts, mol, reacting_atoms, frozen_atoms, n=10, verbose=False):
        """Embed n conformers into mol, the frozen atoms at the positions of mol_ts."""
        raise NotImplementedError(f"{type(self).__name__} does not implement embed_TS.")


class _LegacyDefaults(racerts.embed.DistanceGeometryEmbedder):
    """
    The signature and the settings of legacy racerts, whatever those of racerts.embed:
    remove_all_conformers is stored and unused, as it was; of the other keywords only
    num_threads is read.
    """

    def __init__(
        self,
        verbose=False,
        randomSeed=12,
        pruneRmsThresh=-1,
        remove_all_conformers=True,
        ETversion=2,
        useRandomCoords=True,
        etkdg=False,
        chirality_fallback=True,
        sequential_seeds=False,
        **kwargs,
    ):
        super().__init__(
            verbose=verbose,
            randomSeed=randomSeed,
            pruneRmsThresh=pruneRmsThresh,
            ETversion=ETversion,
            useRandomCoords=useRandomCoords,
            etkdg=etkdg,
            chirality_fallback=chirality_fallback,
            sequential_seeds=sequential_seeds,
            num_threads=kwargs.get("num_threads", 1),
        )
        self.remove_all_conformers = remove_all_conformers


class BoundsMatrixEmbedder(
    BaseEmbedder, _LegacyDefaults, racerts.embed.BoundsMatrixEmbedder
):
    def embed_TS(self, mol_ts, mol, reacting_atoms, frozen_atoms, n=10, verbose=False):
        frozen = FrozenSet(hard=tuple(frozen_atoms), core=tuple(reacting_atoms))
        return racerts.embed.BoundsMatrixEmbedder.embed(self, mol, mol_ts, frozen, n)


class CmapEmbedder(BaseEmbedder, _LegacyDefaults, racerts.embed.CmapEmbedder):
    def embed_TS(self, mol_ts, mol, reacting_atoms, frozen_atoms, n=10, verbose=False):
        frozen = FrozenSet(hard=tuple(frozen_atoms), core=tuple(reacting_atoms))
        return racerts.embed.CmapEmbedder.embed(self, mol, mol_ts, frozen, n)
