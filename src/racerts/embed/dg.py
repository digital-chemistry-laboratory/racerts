"""Distance-geometry embedders: coordinate map (CmapEmbedder) or bounds matrix."""

import hashlib
import logging
from typing import List, Optional, Sequence, Union

from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.AllChem import EmbedMultipleConfs  # type: ignore
from rdkit.Chem.rdDistGeom import EmbedFailureCauses

from racerts.system.stereo import (
    StereoCheck,
    reference_fixed,
    reference_tags,
    stereo_anchors,
)
from racerts.task import FrozenSet

from .base import BaseEmbedder
from .bounds import (
    check_reference_bounds,
    fixed_distance_pairs,
    hard_pairs,
    log_inconsistent_bounds,
    smoothed_bounds,
    widened_bounds,
    widened_message,
)

logger = logging.getLogger(__name__)

CHIRALITY_FALLBACK_MODES = ("legacy", "frozen_first")  # the choices of the config
CHIRALITY_FALLBACKS = (True, False, *CHIRALITY_FALLBACK_MODES)  # True: "legacy"


class DistanceGeometryEmbedder(BaseEmbedder):
    """
    Embedding with RDKit's EmbedMultipleConfs. Plain distance geometry by default;
    etkdg=True uses ETKDGv3 (with the same settings otherwise, e.g. the torsion
    preferences of small rings and macrocycles).

    The first min(n, 3) conformers check for chirality problems (the fixed atoms of a
    TS can contradict a chiral tag); then the other n - 3 are added. If most of the
    first ones fail on chirality, chirality_fallback decides:
    - True or "legacy" (legacy racerts): n conformers are embedded without any chiral
      tags, or without enforcing chirality, with a warning; free stereocentres can
      then come out inverted.
    - "frozen_first": where legacy racerts drops all chiral tags (the first
      minimization fails), only the tags of the frozen atoms whose configuration the
      reference fixes (see reference_fixed) are dropped first and the check is
      repeated; the legacy fallback follows only if it still fails. After any
      fallback, the graph keeps its chiral tags, and conformers whose specified stereo
      is inverted are removed, with a warning (not checked: core atoms that are free
      or fixed by the reference, e.g. the reacting atoms). In this mode the
      coordinate-map embedder always places the free substituent that alone sets the
      configuration of a frozen stereocentre (racerts.system.stereo.stereo_anchors).
    - False: no fallback (the ground-state default: stereocentres of the input are
      never given up).

    With sequential_seeds (the default), conformer i is embedded with the seed
    start + i (enableSequentialRandomSeeds) across all calls of one embedding, where
    start is derived from randomSeed (stream_start), so that the streams of different
    seeds do not overlap. False reproduces legacy racerts, which restarts the seed for
    the second call, so that its first 3 conformers are embedded twice.

    reference_bounds decides what happens to distance bounds of the graph that exclude
    a distance of the reference geometry (a TS core far from the graph's equilibrium
    geometry, e.g. a metal over a ring bond; see bounds.widened_bounds):
    - "fallback": they are widened to the reference only when triangle smoothing
      fails without (RDKit embeds nothing then), with a warning;
    - "never": as legacy racerts, no conformers then;
    - "always": they are widened whenever they exclude the reference.
    The embedder's attribute widened lists the widened pairs of its last embedding.
    """

    def __init__(
        self,
        verbose: bool = False,
        randomSeed: int = 12,
        pruneRmsThresh: Optional[float] = -1,
        ETversion: int = 2,
        useRandomCoords: bool = True,
        etkdg: bool = False,
        chirality_fallback: Union[bool, str] = True,
        sequential_seeds: bool = True,
        reference_bounds: str = "fallback",
        num_threads: int = 1,
    ):
        check_reference_bounds(reference_bounds)
        if chirality_fallback not in CHIRALITY_FALLBACKS:
            raise ValueError(
                f"chirality_fallback must be one of {CHIRALITY_FALLBACKS}, not "
                f"{chirality_fallback!r}."
            )
        self.verbose = verbose
        self.randomSeed = randomSeed
        self.pruneRmsThresh = pruneRmsThresh
        self.ETversion = ETversion
        self.useRandomCoords = useRandomCoords
        self.etkdg = etkdg
        self.chirality_fallback = chirality_fallback
        self.sequential_seeds = sequential_seeds
        self.reference_bounds = reference_bounds
        self.widened: List = []
        self.num_threads = num_threads

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

    def _setup_fallback(self, params, mol, reference, frozen, restraints) -> bool:
        """
        Called when RDKit embedded nothing without trying (it could not set up its
        bounds): whether params now hold bounds to try again with.
        """
        return False

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
        if self.randomSeed == 0 and not self.sequential_seeds and n > 1:
            logger.warning(
                "With randomSeed 0 all conformers are identical (RDKit seeds conformer "
                "i with (i + 1) * seed): use another seed, or sequential seeds."
            )

        params = AllChem.ETKDGv3() if self.etkdg else AllChem.EmbedParameters()
        params.verbose = self.verbose
        params.ETversion = self.ETversion
        params.useMacrocycleTorsions = True
        params.useMacrocycle14config = True
        params.useSmallRingTorsions = True
        params.embedFragmentsSeparately = False
        params.clearConfs = False
        self.widened = []
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
        if (
            not len(result)
            and not any(params.GetFailureCounts())
            and self._setup_fallback(params, mol, reference, frozen, restraints)
        ):
            requested = 0
            result = embed_more(chiral_check)
        fallback = self._chirality_problem(result, params, chiral_check)
        # After a fallback, remove the conformers with inverted stereo at the end.
        guard = fallback is not None and self.chirality_fallback == "frozen_first"
        if fallback == "strip_tags" and self.chirality_fallback == "frozen_first":
            tagged = [
                i
                for i in reference_fixed(mol, frozen)
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
        Restore the chiral tags of original, except that the frozen atoms whose
        configuration the reference fixes get the tag that its geometry gives them, and
        remove the conformers whose stereo differs from these tags.
        """
        for atom, before in zip(mol.GetAtoms(), original.GetAtoms()):
            atom.SetChiralTag(before.GetChiralTag())
        fixed = reference_fixed(mol, frozen)
        tagged = [
            i
            for i in fixed
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
        # Core atoms held with two or more free neighbours can invert: they are checked.
        held = {*frozen.hard, *frozen.soft}
        exempt = [i for i in frozen.core if i not in held or i in fixed]
        check = StereoCheck(mol, exempt=exempt)
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
        return needed_fallback(
            len(result), params.GetFailureCounts(), chiral_check, params.maxIterations
        )


# Room for this many conformers after the start of a seed stream (RDKit seeds are
# 31-bit integers).
_STREAM_LENGTH = 2**24


def stream_start(seed: int) -> int:
    """
    The RDKit seed of the first conformer of a sequential-seed embedding with the
    user seed seed: spread over the seed range, so that seeds 1, 2, ... do not give
    shifted copies of the same stream (as seed + i would). It is a hash of the seed
    (SHA-256), so that a seed gives the same conformers with every version of the
    libraries.
    """
    digest = hashlib.sha256(str(int(seed)).encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**31 - 1 - _STREAM_LENGTH)


def needed_fallback(
    n_embedded: int, failure_counts, chiral_check: int, max_iterations: int
) -> Optional[str]:
    """
    The legacy racerts rule after the first chiral_check conformers: if fewer than half
    of them were embedded, "strip_tags" when the first minimization failed more than
    max_iterations * chiral_check / 2 times, or "no_enforce" when the final chirality
    checks did; otherwise None. RDKit's maxIterations is 0 unless set, so one such
    failure is enough.
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
            pairs = hard_pairs(frozen, reference)
            bounds, self.widened = smoothed_bounds(
                mol,
                reference,
                pairs=pairs,
                windows=restraints,
                reference_bounds=self.reference_bounds,
            )
            params.SetBoundsMat(bounds)
        if reference is None:
            return
        placed = self._placed(mol, frozen)
        if not restraints and self.reference_bounds == "always":
            self._widen(params, mol, reference, placed, fallback=False)
        conf = reference.GetConformer()
        cmap = {i: conf.GetAtomPosition(i) for i in placed}
        params.SetCoordMap(cmap)  # type: ignore

    def _placed(self, mol, frozen) -> List[int]:
        # Soft atoms start at the reference too; refinement then lets them move.
        placed = [*frozen.hard, *frozen.soft]
        if self.chirality_fallback == "frozen_first":
            placed += stereo_anchors(mol, frozen)
        return placed

    def _setup_fallback(self, params, mol, reference, frozen, restraints) -> bool:
        if restraints or reference is None or self.reference_bounds != "fallback":
            return False  # windows: bounds_matrix falls back itself
        return self._widen(
            params, mol, reference, self._placed(mol, frozen), fallback=True
        )

    def _widen(self, params, mol, reference, placed, fallback) -> bool:
        """
        Bounds widened to the reference, with the distances between the placed atoms
        fixed as the coordinate map fixes them; whether any bound was widened.
        """
        pairs = [(a, b) for k, a in enumerate(placed) for b in placed[k + 1 :]]
        bounds, self.widened = widened_bounds(mol, reference, pairs)
        if not self.widened:
            return False
        logger.log(
            logging.WARNING if fallback else logging.INFO,
            "%s.",
            widened_message(mol, self.widened, fallback),
        )
        params.SetBoundsMat(bounds)
        return True


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
        bounds, self.widened = smoothed_bounds(
            mol,
            reference=reference,
            pairs=fixed_distance_pairs(frozen),
            windows=restraints,
            reference_bounds=self.reference_bounds,
        )
        log_inconsistent_bounds(bounds)
        params.SetBoundsMat(bounds)
