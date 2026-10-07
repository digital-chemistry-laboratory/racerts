"""Distance-geometry embedders: coordinate map (CmapEmbedder) or bounds matrix."""

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
    trans_in_small_rings,
)
from racerts.task import FrozenSet
from racerts.utils import seeds

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
# With restraints, ETKDG weights the distance bounds this much against its torsion
# terms (RDKit's boundsMatForceScaling; 1.0 without restraints).
RESTRAINT_BOUNDS_WEIGHT = 100.0


class DistanceGeometryEmbedder(BaseEmbedder):
    """
    Embedding with RDKit's EmbedMultipleConfs: plain distance geometry by default,
    ETKDGv3 with etkdg=True (the same settings otherwise).

    chirality_fallback: the first min(n, 3) conformers show whether the fixed atoms
    contradict a chiral tag. If most of them fail on chirality:
    - True or "legacy": all n are embedded without chiral tags, or without enforcing
      chirality, with a warning; free stereocentres can then come out inverted.
    - "frozen_first": only the tags of the frozen atoms whose configuration the
      reference fixes (reference_fixed) are dropped first; the legacy fallback follows
      only if that fails too. The graph keeps its tags, and conformers whose specified
      stereo is inverted are removed with a warning (not checked: core atoms). The
      coordinate-map embedder then also places the free substituent that alone sets
      the configuration of a frozen stereocentre (system.stereo.stereo_anchors).
    - False: no fallback (the ground-state default).

    sequential_seeds (default): conformer i gets the seed start + i across all calls
    of one embedding, with start derived from randomSeed (utils.seeds.derive), so the
    streams of different seeds do not overlap. False is legacy racerts: the seed
    restarts for the second call, and the first 3 conformers are embedded twice.

    reference_bounds: what happens to bounds of the graph that exclude a distance of
    the reference geometry (e.g. a metal over a ring bond; bounds.widened_bounds):
    - "fallback": widened to the reference only when RDKit embeds nothing without,
      with a warning;
    - "always": widened whenever they exclude the reference;
    - "never": as legacy racerts, no conformers then.
    The attribute widened lists the widened pairs of the last embedding.

    max_attempts: RDKit's attempts per conformer (maxIterations; default ten per
    atom). A conformer that does not embed within them is left out.
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
        max_attempts: Optional[int] = None,
    ):
        check_reference_bounds(reference_bounds)
        if max_attempts is not None and max_attempts < 1:
            raise ValueError("max_attempts must be positive, or None for RDKit's.")
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
        self.max_attempts = max_attempts
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
        if trans_in_small_rings(mol):
            # ETKDG's torsion preferences hold the double bonds of small rings cis
            params.useExpTorsionAnglePrefs = False
        self.widened = []
        self._configure(params, mol, reference, frozen, restraints)
        if restraints and self.etkdg:
            # ETKDG's torsion terms override about 60 % of bounds windows otherwise
            # (verified on RDKit 2023.09-2026.03); plain DG keeps most of them. The
            # small-ring terms are not scaled down from RDKit 2026.09 on: a window
            # that needs a boat embeds in 1 % of the attempts with them (66 % before).
            params.boundsMatForceScaling = RESTRAINT_BOUNDS_WEIGHT
            params.useSmallRingTorsions = False
        params.trackFailures = True
        if getattr(self, "max_attempts", None) is not None:
            params.maxIterations = int(self.max_attempts)
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
                    params.randomSeed = seeds.derive(self.randomSeed) + requested
            requested += count
            try:
                return EmbedMultipleConfs(mol, count, params)
            except RuntimeError as error:
                if not (self.etkdg and params.useSmallRingTorsions):
                    raise
                # e.g. cyclopentane rings: "bad direction in linearSearch"
                logger.warning(
                    "RDKit failed with its small-ring torsion terms (%s): embedding "
                    "without the small-ring torsion terms.",
                    " ".join(str(error).split()[:6]),
                )
                params.useSmallRingTorsions = False
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
