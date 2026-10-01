"""Distance-geometry embedders: coordinate map (CmapEmbedder) or bounds matrix."""

import logging
from typing import Optional, Sequence, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import EmbedMultipleConfs  # type: ignore
from rdkit.Chem.rdDistGeom import EmbedFailureCauses

from racerts.system.stereo import StereoCheck, reference_tags
from racerts.task import FrozenSet

from .base import BaseEmbedder
from .bounds import bounds_matrix, fixed_distance_pairs, log_inconsistent_bounds

logger = logging.getLogger(__name__)


class DistanceGeometryEmbedder(BaseEmbedder):
    """
    Embedding with RDKit's EmbedMultipleConfs. Plain distance geometry by default;
    etkdg=True uses ETKDGv3 (with the same settings otherwise).

    The first min(n, 3) conformers check for chirality problems (the fixed atoms of a
    TS can contradict a chiral tag); then the other n - 3 are added. If most of the
    first ones fail on chirality, chirality_fallback decides:
    - True or "legacy" (legacy racerts): n conformers are embedded without any chiral
      tags, or without enforcing chirality, with a warning; free stereocentres can
      then come out inverted.
    - "frozen_first": where legacy racerts drops all chiral tags (the first
      minimization fails), only the tags of the frozen atoms are dropped first (the
      reference fixes their configuration) and the check is repeated; the legacy
      fallback follows only if it still fails. After any fallback, the graph keeps
      its chiral tags, and conformers whose specified stereo (outside the core atoms,
      e.g. the reacting atoms) is inverted are removed, with a warning.
    - False: no fallback (the ground-state default: stereocentres of the input are
      never given up).

    With sequential_seeds, conformer i is embedded with the seed start + i
    (enableSequentialRandomSeeds) across all calls of one embedding, where start is
    derived from randomSeed (stream_start), so that the streams of different seeds do
    not overlap. Legacy racerts restarts the seed for the second call, so its first 3
    conformers are embedded twice.
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
        chirality_fallback: Union[bool, str] = True,
        sequential_seeds: bool = False,
        **kwargs,
    ):
        if chirality_fallback not in CHIRALITY_FALLBACKS:
            raise ValueError(
                f"chirality_fallback must be one of {CHIRALITY_FALLBACKS}, not "
                f"{chirality_fallback!r}."
            )
        self.verbose = verbose
        self.randomSeed = randomSeed
        self.pruneRmsThresh = pruneRmsThresh
        self.remove_all_conformers = remove_all_conformers
        self.ETversion = ETversion
        self.useRandomCoords = useRandomCoords
        self.etkdg = etkdg
        self.chirality_fallback = chirality_fallback
        self.sequential_seeds = sequential_seeds
        self.num_threads = kwargs.get("num_threads", 1)

    def _configure(
        self,
        params,
        mol: Chem.Mol,
        reference: Optional[Chem.Mol],
        frozen: FrozenSet,
        restraints: Sequence = (),
    ) -> None:
        """
        Set what keeps the frozen atoms in place (coordinate map or bounds) and the
        windows of the restraints.
        """
        raise NotImplementedError

    def embed(
        self,
        mol: Chem.Mol,
        reference: Optional[Chem.Mol],
        frozen: FrozenSet,
        n: int,
        restraints: Sequence = (),
    ):
        """
        Add n conformers to mol with the frozen atoms at the reference positions and
        the distances of the restraints (DistanceRestraint) in their windows.
        """
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
        self._configure(params, mol, reference, frozen, restraints)
        if restraints and self.etkdg:
            # ETKDG's torsion terms override about 60 % of bounds windows otherwise
            # (verified on RDKit 2023.09-2026.03); plain DG keeps most of them.
            params.boundsMatForceScaling = 100.0
        params.trackFailures = True
        params.pruneRmsThresh = self.pruneRmsThresh
        params.randomSeed = self.randomSeed
        params.numThreads = self.num_threads
        logger.debug(
            "Random coordinates are %sbeing used.",
            "" if self.useRandomCoords else "not ",
        )
        params.useRandomCoords = self.useRandomCoords

        requested = 0  # conformers requested so far, the offset of sequential seeds
        original = Chem.Mol(mol)  # the graph with its chiral tags

        def embed_more(count):
            nonlocal requested
            if self.sequential_seeds:
                params.enableSequentialRandomSeeds = True
                if self.randomSeed >= 0:
                    params.randomSeed = stream_start(self.randomSeed) + requested
            requested += count
            return EmbedMultipleConfs(mol, count, params)

        chiral_check = min(n, 3)
        result = embed_more(chiral_check)
        fallback = self._chirality_problem(result, params, chiral_check)
        # After a fallback, remove the conformers with inverted stereo at the end.
        guard = fallback is not None and self.chirality_fallback == "frozen_first"
        if fallback == "strip_tags" and self.chirality_fallback == "frozen_first":
            tagged = [
                i
                for i in (*frozen.hard, *frozen.soft)
                if mol.GetAtomWithIdx(i).GetChiralTag()
                != Chem.ChiralType.CHI_UNSPECIFIED
            ]
            if tagged:
                logger.warning(
                    "Most of the first %d conformers failed on chirality; embedding "
                    "without the chiral tags of the frozen atoms %s (the reference "
                    "fixes their configuration).",
                    chiral_check,
                    tagged,
                )
                for i in tagged:
                    mol.GetAtomWithIdx(i).SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                mol.RemoveAllConformers()
                result = embed_more(chiral_check)
                fallback = self._chirality_problem(result, params, chiral_check)
        if fallback == "strip_tags":
            logger.warning(
                "Most of the first %d conformers failed on chirality; embedding "
                "without chiral tags (stereocentres may be inverted).",
                chiral_check,
            )
            for atom in mol.GetAtoms():
                atom.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
            result = embed_more(n)
        elif fallback == "no_enforce":
            logger.warning(
                "Most of the first %d conformers failed on chirality; embedding "
                "without enforcing chirality (stereocentres may be inverted).",
                chiral_check,
            )
            params.enforceChirality = False
            result = embed_more(n)
        elif chiral_check < n:
            rest_confs = n - chiral_check
            params.clearConfs = False
            result = embed_more(rest_confs)

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
                result = embed_more(n)

        if guard:
            self._remove_inverted(mol, original, reference, frozen)
        error_counts = params.GetFailureCounts()
        logger.debug("Embedding failure counts: %s", list(error_counts))

        return result, error_counts

    @staticmethod
    def _remove_inverted(
        mol: Chem.Mol, original: Chem.Mol, reference: Chem.Mol, frozen: FrozenSet
    ) -> None:
        """
        Restore the chiral tags of original, except that the frozen atoms get the
        configuration of the reference (where its geometry defines one), and remove
        the conformers whose stereo differs from these tags.
        """
        for atom, before in zip(mol.GetAtoms(), original.GetAtoms()):
            atom.SetChiralTag(before.GetChiralTag())
        tagged = [
            i
            for i in (*frozen.hard, *frozen.soft)
            if original.GetAtomWithIdx(i).GetChiralTag()
            != Chem.ChiralType.CHI_UNSPECIFIED
        ]
        if tagged and reference is not None:
            from_reference = reference_tags(mol, reference, tagged)
            changed = [
                i
                for i in tagged
                if from_reference[i] != original.GetAtomWithIdx(i).GetChiralTag()
            ]
            if changed:
                logger.warning(
                    "The chiral tags of the frozen atoms %s contradict the reference "
                    "geometry; their configuration is taken from the reference.",
                    changed,
                )
            for i in changed:
                mol.GetAtomWithIdx(i).SetChiralTag(from_reference[i])
        check = StereoCheck(mol, exempt=frozen.core)
        reasons = {c.GetId(): check.mismatch(c) for c in mol.GetConformers()}
        inverted = [conf_id for conf_id, reason in reasons.items() if reason]
        if inverted and len(inverted) == len(reasons):
            # e.g. a frozen stereocentre whose one free substituent sets its
            # configuration: legacy racerts returns only the other stereoisomer
            logger.warning(
                "All %d conformers have inverted stereo after the chirality fallback "
                "(e.g. %s) and are removed: the frozen atoms and the chiral tags could "
                "not be embedded together. chirality_fallback='legacy' keeps them, "
                "with the inverted stereo.",
                len(inverted),
                reasons[inverted[0]],
            )
        elif inverted:
            logger.warning(
                "Removing %d of %d conformers whose stereo is inverted after the "
                "chirality fallback: %s",
                len(inverted),
                mol.GetNumConformers(),
                inverted,
            )
        for conf_id in inverted:
            mol.RemoveConformer(conf_id)

    def _chirality_problem(self, result, params, chiral_check: int) -> Optional[str]:
        """The legacy fallback that the first conformers call for (None: none)."""
        if not self.chirality_fallback:
            return None
        return chirality_fallback(
            len(result), params.GetFailureCounts(), chiral_check, params.maxIterations
        )


CHIRALITY_FALLBACKS = (True, False, "legacy", "frozen_first")

# Room for this many conformers after the start of a seed stream (RDKit seeds are
# 31-bit integers).
_STREAM_LENGTH = 2**24


def stream_start(seed: int) -> int:
    """
    The RDKit seed of the first conformer of a sequential-seed embedding with the
    user seed seed: spread over the seed range, so that seeds 1, 2, ... do not give
    shifted copies of the same stream (as seed + i would).
    """
    return int(np.random.default_rng(seed).integers(0, 2**31 - 1 - _STREAM_LENGTH))


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
    """
    The frozen atoms are placed at the reference positions (coordinate map). With
    restraints, a bounds matrix holds their windows and the distances between the
    frozen atoms; without, there is none (as in legacy racerts).
    """

    def _configure(self, params, mol, reference, frozen, restraints=()):
        if restraints:
            hard = frozen.hard if reference is not None else ()
            pairs = [(a, b) for k, a in enumerate(hard) for b in hard[k + 1 :]]
            params.SetBoundsMat(
                bounds_matrix(mol, reference, pairs=pairs, windows=restraints)
            )
        if reference is None:
            return
        conf = reference.GetConformer()
        # Soft atoms start at the reference too; refinement then lets them move.
        placed = [*frozen.hard, *frozen.soft]
        cmap = {i: conf.GetAtomPosition(i) for i in placed}
        params.SetCoordMap(cmap)  # type: ignore


class BoundsMatrixEmbedder(DistanceGeometryEmbedder):
    """
    The distances between the core (reacting) atoms and all frozen atoms are fixed at
    the reference geometry in the bounds matrix.
    """

    def _configure(self, params, mol, reference, frozen, restraints=()):
        if frozen.soft:
            raise ValueError(
                "Soft atoms are placed by a coordinate map: use the CmapEmbedder "
                "(embed mode 'cmap')."
            )
        bounds = bounds_matrix(
            mol,
            reference=reference,
            pairs=fixed_distance_pairs(frozen),
            windows=restraints,
        )
        log_inconsistent_bounds(bounds)
        params.SetBoundsMat(bounds)
