"""Distance-geometry embedders: coordinate map (CmapEmbedder) or bounds matrix."""

import logging
from typing import Optional

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import EmbedMultipleConfs  # type: ignore
from rdkit.Chem.rdDistGeom import EmbedFailureCauses

from racerts.task import FrozenSet

from .base import BaseEmbedder
from .bounds import bounds_matrix, fixed_distance_pairs, log_inconsistent_bounds

logger = logging.getLogger(__name__)


class DistanceGeometryEmbedder(BaseEmbedder):
    """
    Embedding with RDKit's EmbedMultipleConfs. Plain distance geometry by default;
    etkdg=True uses ETKDGv3 (with the same settings otherwise).

    With chirality_fallback (legacy racerts, for TS embedding, where the fixed atoms can
    contradict a chiral tag), the first min(n, 3) conformers check for chirality
    problems: if most of them fail on chirality, n conformers are embedded without
    chiral tags (or without enforcing chirality), with a warning. Otherwise the other
    n - 3 are added. The ground-state defaults turn it off: stereocentres of the input
    are then never given up.
    """

    def __init__(
        self,
        verbose: bool = False,
        randomSeed: int = 12,
        pruneRmsThresh: Optional[float] = -1,
        remove_all_conformers: bool = True,
        ETversion: int = 2,
        useRandomCoords: bool = True,
        etkdg: bool = False,
        chirality_fallback: bool = True,
        **kwargs,
    ):
        self.verbose = verbose
        self.randomSeed = randomSeed
        self.pruneRmsThresh = pruneRmsThresh
        self.remove_all_conformers = remove_all_conformers
        self.ETversion = ETversion
        self.useRandomCoords = useRandomCoords
        self.etkdg = etkdg
        self.chirality_fallback = chirality_fallback
        self.num_threads = kwargs.get("num_threads", 1)

    def _configure(
        self,
        params,
        mol: Chem.Mol,
        reference: Optional[Chem.Mol],
        frozen: FrozenSet,
    ) -> None:
        """Set what keeps the frozen atoms in place (coordinate map or bounds)."""
        raise NotImplementedError

    def embed(
        self, mol: Chem.Mol, reference: Optional[Chem.Mol], frozen: FrozenSet, n: int
    ):
        if not isinstance(mol, Chem.rdchem.Mol):
            raise TypeError("Embedding: input for embedding is not a molecule!")

        params = AllChem.ETKDGv3() if self.etkdg else AllChem.EmbedParameters()
        params.verbose = self.verbose
        params.ETversion = self.ETversion
        params.useMacrocycleTorsions = True
        params.useMacrocycle14config = True
        params.useSmallRingTorsions = True
        params.embedFragmentsSeparately = False
        params.clearConfs = False
        self._configure(params, mol, reference, frozen)
        params.trackFailures = True
        params.pruneRmsThresh = self.pruneRmsThresh
        params.randomSeed = self.randomSeed
        params.numThreads = self.num_threads
        logger.debug(
            "Random coordinates are %sbeing used.",
            "" if self.useRandomCoords else "not ",
        )
        params.useRandomCoords = self.useRandomCoords

        chiral_check = min(n, 3)
        result = EmbedMultipleConfs(mol, chiral_check, params)
        error_counts = params.GetFailureCounts()
        fallback = None
        if self.chirality_fallback:
            fallback = chirality_fallback(
                len(result), error_counts, chiral_check, params.maxIterations
            )
        if fallback == "strip_tags":
            logger.warning(
                "Most of the first %d conformers failed on chirality; embedding "
                "without chiral tags (stereocentres may be inverted).",
                chiral_check,
            )
            for atom in mol.GetAtoms():
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            result = EmbedMultipleConfs(mol, n, params)
        elif fallback == "no_enforce":
            logger.warning(
                "Most of the first %d conformers failed on chirality; embedding "
                "without enforcing chirality (stereocentres may be inverted).",
                chiral_check,
            )
            params.enforceChirality = False
            result = EmbedMultipleConfs(mol, n, params)
        elif chiral_check < n:
            rest_confs = n - chiral_check
            params.clearConfs = False
            result = EmbedMultipleConfs(mol, rest_confs, params)

        if mol.GetNumConformers() == 0:
            if self.useRandomCoords:
                logger.warning(
                    "Problem while generating the conformers... try a different "
                    "approach"
                )
            else:
                logger.warning(
                    "Problem while generating the conformers... Random coordinates will "
                    "be used."
                )
                params.useRandomCoords = True
                result = EmbedMultipleConfs(mol, n, params)

        error_counts = params.GetFailureCounts()
        logger.debug("Embedding failure counts: %s", list(error_counts))

        return result, error_counts


def chirality_fallback(
    n_embedded: int, failure_counts, chiral_check: int, max_iterations: int
) -> Optional[str]:
    """
    The legacy racerts rule after the first chiral_check conformers: if at most half of
    them were embedded, "strip_tags" when the first minimization failed often, or
    "no_enforce" when the final chirality checks failed often; otherwise None.
    """
    if n_embedded >= chiral_check / 2:
        return None
    limit = max_iterations * chiral_check / 2
    if failure_counts[EmbedFailureCauses.FIRST_MINIMIZATION] > limit:
        return "strip_tags"
    if (
        failure_counts[EmbedFailureCauses.FINAL_CHIRAL_BOUNDS] > limit
        or failure_counts[EmbedFailureCauses.FINAL_CENTER_IN_VOLUME] > limit
    ):
        return "no_enforce"
    return None


class CmapEmbedder(DistanceGeometryEmbedder):
    """The frozen atoms are placed at the reference positions (coordinate map)."""

    def _configure(self, params, mol, reference, frozen):
        if reference is None:
            return
        cmap = {
            frozen.hard[i]: reference.GetConformer().GetAtomPosition(frozen.hard[i])
            for i in range(len(frozen.hard))
        }
        params.SetCoordMap(cmap)  # type: ignore


class BoundsMatrixEmbedder(DistanceGeometryEmbedder):
    """
    The distances between the core (reacting) atoms and all frozen atoms are fixed at
    the reference geometry in the bounds matrix.
    """

    def _configure(self, params, mol, reference, frozen):
        bounds = bounds_matrix(
            mol, reference=reference, pairs=fixed_distance_pairs(frozen)
        )
        log_inconsistent_bounds(bounds)
        params.SetBoundsMat(bounds)
