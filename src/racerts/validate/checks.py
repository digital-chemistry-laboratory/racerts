"""
Geometric validators: connectivity and stereo of the graph, the frozen core, clashes and
restraints; gate() combines them after a refinement.
"""

import logging
from typing import Dict, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from racerts.geometry import superpose
from racerts.pipeline import ConformerEnsemble
from racerts.restraints.model import applying, position_restraints
from racerts.system.stereo import StereoCheck

from .base import Validate

logger = logging.getLogger(__name__)


class Connectivity:
    """
    Whether each conformer still is the molecule of the graph: the bonds perceived
    from its geometry (RDKit's DetermineConnectivity) are those of the graph, and the
    stereocentres and double bonds that the graph specifies have the same
    configuration (unspecified stereo is a wildcard). The check goes by atom index,
    since the conformers share the graph. Close contacts that the graph does not
    bond, e.g. of ions or coordinated metals, count as extra bonds.

    Args:
        exempt: Atoms whose bonds among each other and whose stereo are not checked;
            default: the core atoms of the task (FrozenSet.core): the reacting atoms
            of a TransitionState, whose bonds form or break; the hard atoms of
            Constrained, which the reference fixes; none for GroundState.
        stereo: Check the stereo.
        bonds: Check the bonds. Without them the validator is named "stereo".
    """

    name = "connectivity"

    def __init__(
        self,
        exempt: Optional[Sequence[int]] = None,
        stereo: bool = True,
        bonds: bool = True,
    ):
        if not (bonds or stereo):
            raise ValueError("Connectivity needs bonds or stereo to check.")
        self.exempt = None if exempt is None else tuple(exempt)
        self.stereo = stereo
        self.bonds = bonds
        if not bonds:
            self.name = "stereo"

    def validate(self, ctx, ensemble: ConformerEnsemble) -> Dict[int, str]:
        exempt = self.exempt
        if exempt is None:
            exempt = ctx.frozen.core if ctx is not None else ()
        exempt = set(exempt)
        graph = ensemble.mol
        expected = _bonds(graph, exempt)
        stereo = StereoCheck(graph, exempt) if self.stereo else None
        bondless = None
        if self.bonds:
            bondless = Chem.RWMol(Chem.Mol(graph, True))  # without conformers
            for bond in list(bondless.GetBonds()):
                bondless.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
            bondless = bondless.GetMol()
            reference = ctx.reference if ctx is not None else None
            if reference is not None:
                wrong = self._bond_mismatch(
                    bondless, reference.GetConformer(), expected, exempt
                )
                if wrong is not None:
                    logger.warning(
                        "The graph does not fit the reference geometry (%s), so no "
                        "conformer can pass: check the SMILES and its atom mapping.",
                        wrong,
                    )
        reasons = {}
        for conf in graph.GetConformers():
            reason = None
            if bondless is not None:
                reason = self._bond_mismatch(bondless, conf, expected, exempt)
            if reason is None and stereo is not None:
                reason = stereo.mismatch(conf)
            if reason is not None:
                reasons[conf.GetId()] = reason
        return reasons

    def _bond_mismatch(self, bondless, conf, expected, exempt) -> Optional[str]:
        perceived = Chem.Mol(bondless)
        perceived.AddConformer(Chem.Conformer(conf), assignId=True)
        rdDetermineBonds.DetermineConnectivity(perceived, useHueckel=False)
        found = _bonds(perceived, exempt)
        if found == expected:
            return None
        parts = []
        if expected - found:
            parts.append(f"bonds {sorted(expected - found)} missing")
        if found - expected:
            parts.append(f"bonds {sorted(found - expected)} extra")
        return ", ".join(parts)

    def __repr__(self) -> str:
        return (
            f"Connectivity(exempt={self.exempt}, bonds={self.bonds}, "
            f"stereo={self.stereo})"
        )


def _bonds(mol, exempt) -> set:
    return {
        tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
        for b in mol.GetBonds()
        if not (b.GetBeginAtomIdx() in exempt and b.GetEndAtomIdx() in exempt)
    }


class FrozenCore:
    """
    Whether the frozen (hard) atoms of the task are still at the reference geometry:
    after the best superposition on them, no frozen atom may be further than
    tolerance (A) from its reference position.
    """

    name = "frozen_core"

    def __init__(self, tolerance: float = 1e-3):
        self.tolerance = tolerance

    def validate(self, ctx, ensemble: ConformerEnsemble) -> Dict[int, str]:
        atoms = list(ctx.frozen.hard)
        if not atoms or ctx.reference is None:
            return {}
        reasons = {}
        for reference, conf_ids in ctx.by_reference(ensemble):
            target = reference.GetConformer().GetPositions()[atoms]
            for conf_id in conf_ids:
                positions = ensemble.mol.GetConformer(conf_id).GetPositions()[atoms]
                moved = _max_deviation(positions, target)
                if moved > self.tolerance:
                    reasons[conf_id] = f"frozen atoms moved by up to {moved:.3f} A"
        return reasons


class ReactionCore:
    """
    Whether a conformer is still a TS of the reaction of the reference: no distance
    between two core atoms of the task (for a TS the reacting atoms) may differ from the
    reference by more than tolerance (A). Meant after a free saddle search: an imaginary
    mode and the connectivity outside the reacting atoms do not tell another saddle of
    the same atoms (e.g. the proton already on its acceptor) from the TS.
    """

    name = "reaction_core"

    def __init__(self, tolerance: float = 0.5):
        self.tolerance = tolerance

    def validate(self, ctx, ensemble: ConformerEnsemble) -> Dict[int, str]:
        atoms = list(ctx.frozen.core)
        if len(atoms) < 2 or ctx.reference is None:
            return {}
        pairs = [(a, b) for k, a in enumerate(atoms) for b in atoms[k + 1 :]]
        first, second = (list(x) for x in zip(*pairs))
        reasons = {}
        for reference, conf_ids in ctx.by_reference(ensemble):
            x = reference.GetConformer().GetPositions()
            expected = np.linalg.norm(x[first] - x[second], axis=1)
            for conf_id in conf_ids:
                y = ensemble.mol.GetConformer(conf_id).GetPositions()
                change = np.abs(np.linalg.norm(y[first] - y[second], axis=1) - expected)
                worst = int(np.argmax(change))
                if change[worst] > self.tolerance:
                    a, b = pairs[worst]
                    reasons[conf_id] = (
                        f"reacting-atom distances changed by up to "
                        f"{change[worst]:.2f} A ({a}-{b})"
                    )
        return reasons


def _max_deviation(positions: np.ndarray, reference: np.ndarray) -> float:
    """Largest atom deviation after the best superposition (Kabsch, rotations only)."""
    q = reference - reference.mean(axis=0)
    return float(np.linalg.norm(superpose(positions, reference) - q, axis=1).max())


class IdentityFilter(Validate):
    """
    Drops conformers that are no longer the molecule of the graph (see Connectivity);
    raises if none is left.
    """

    name = "identity_filter"

    def __init__(self, **settings):
        super().__init__(Connectivity(**settings))


MIN_FACE_HEIGHT = 0.3  # A: a partner closer to the plane has no defined face
MIN_PLANE_SINE = 0.17  # neighbours within 10 degrees of a line span no plane


class AttackFace:
    """
    Whether each forming bond approaches from the same face as in the reference: for
    every active bond (i, j), the side of j relative to the plane of i's other
    neighbours (three: their plane; two: their plane with i, unless nearly collinear),
    and the same from j's side. A side counts only where j lies at least min_height
    from the plane in the reference, and a conformer is flagged only where j lies at
    least min_height on the other side: a partner near the plane (e.g. in the plane of
    a ring) has no defined face. Atoms with one other neighbour, or four and more,
    have no face either.

    Loosened active-bond windows let distance geometry put the partner on the wrong
    face. The default pipeline runs the filter after refinement: raw embedded
    geometries are too rough for it.

    Args:
        pairs: The bonds; default: the active bonds of the task (TransitionState).
        min_height: See above (A).
    """

    name = "attack_face"

    def __init__(
        self,
        pairs: Optional[Sequence[Sequence[int]]] = None,
        min_height: float = MIN_FACE_HEIGHT,
    ):
        self.pairs = None if pairs is None else [tuple(p) for p in pairs]
        self.min_height = min_height

    def validate(self, ctx, ensemble: ConformerEnsemble) -> Dict[int, str]:
        pairs = self.pairs
        if pairs is None:
            pairs = (
                ctx.task.active_pairs(ctx.mol)
                if hasattr(ctx.task, "active_pairs")
                else []
            )
        if not pairs or ctx.reference is None:
            return {}
        mol = ensemble.mol
        sides = [(i, j) for a, b in pairs for i, j in ((a, b), (b, a))]
        neighbors = {
            (i, j): sorted(
                n.GetIdx()
                for n in mol.GetAtomWithIdx(i).GetNeighbors()
                if n.GetIdx() != j
            )
            for i, j in sides
        }
        reasons = {}
        for reference, conf_ids in ctx.by_reference(ensemble):
            reference_positions = reference.GetConformer().GetPositions()
            expected = {}
            for side in sides:
                height = _height(reference_positions, *side, neighbors[side])
                if height is not None and abs(height) >= self.min_height:
                    expected[side] = np.sign(height)
            for conf_id in conf_ids:
                positions = mol.GetConformer(conf_id).GetPositions()
                flipped = []
                for (i, j), sign in expected.items():
                    height = _height(positions, i, j, neighbors[(i, j)])
                    if height is not None and sign * height <= -self.min_height:
                        flipped.append(f"{j} on {i}")
                if flipped:
                    reasons[conf_id] = (
                        f"attack from the other face ({', '.join(flipped)})"
                    )
        return reasons


def _height(
    positions: np.ndarray, i: int, j: int, neighbors: Sequence[int]
) -> Optional[float]:
    """
    The signed distance (A) of atom j from the plane of i's neighbours (two: their
    plane with i), measured from i; None without a plane.
    """
    if len(neighbors) == 3:
        a, b, c = (positions[k] for k in neighbors)
        normal = np.cross(b - a, c - a)
        scale = np.linalg.norm(b - a) * np.linalg.norm(c - a)
    elif len(neighbors) == 2:
        a, b = (positions[k] - positions[i] for k in neighbors)
        normal = np.cross(a, b)
        scale = np.linalg.norm(a) * np.linalg.norm(b)
    else:  # one neighbour, or four and more: no single face
        return None
    length = np.linalg.norm(normal)
    if length < MIN_PLANE_SINE * scale:  # (nearly) collinear: no plane
        return None
    return float(np.dot(normal / length, positions[j] - positions[i]))


CLASH_FACTOR = 0.7  # heavy atoms closer than this times their vdW sum clash
# Heavy atoms closer than this times their vdW sum overlap: below every lower bound that
# RDKit's distance geometry sets for the pairs Clash checks (0.7 for most pairs four
# bonds apart, 0.555 for some). An embedded conformer can sit at such a bound, which
# refinement relaxes, but not below it.
OVERLAP_FACTOR = 0.5
REFERENCE_MARGIN = 0.2  # A: pairs that close in the reference clash only this closer


def clash_limits(mol, held=(), reference=None, factor: float = CLASH_FACTOR):
    """
    The atom pairs to check for clashes and their distance limits (A): heavy atoms more
    than three bonds apart (or in different fragments), not both held (e.g. the hard
    and core atoms of a task, which keep the reference geometry), clash when closer than
    factor times their vdW sum. A pair closer than that in the reference geometry
    (positions; e.g. a coordination that the graph lacks) clashes only if it comes
    more than REFERENCE_MARGIN closer.

    Returns:
        Arrays (first atoms, second atoms, limits) of the pairs to check.
    """
    table = Chem.GetPeriodicTable()
    numbers = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
    heavy = np.array([z > 1 for z in numbers])
    radii = np.array([table.GetRvdw(z) for z in numbers])
    check = np.triu(Chem.GetDistanceMatrix(mol) > 3, k=1)
    check &= heavy[:, None] & heavy[None, :]
    held = sorted(set(held))
    if held:
        check[np.ix_(held, held)] = False
    first, second = np.nonzero(check)
    limits = factor * (radii[first] + radii[second])
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        seed = np.linalg.norm(reference[first] - reference[second], axis=1)
        limits = np.where(seed < limits, seed - REFERENCE_MARGIN, limits)
    return first, second, limits


def first_clash(positions, pairs) -> Optional[str]:
    """The first clash in positions among pairs (from clash_limits), or None."""
    first, second, limits = pairs
    if not len(first):
        return None
    positions = np.asarray(positions, dtype=float)
    distances = np.linalg.norm(positions[first] - positions[second], axis=1)
    hits = np.nonzero(distances < limits)[0]
    if not len(hits):
        return None
    k = hits[0]
    return (
        f"atoms {first[k]} and {second[k]} at {distances[k]:.2f} A "
        f"(limit {limits[k]:.2f} A)"
    )


class Clash:
    """
    Whether heavy atoms clash: closer than factor times their vdW sum while more than
    three bonds apart or in different fragments (see clash_limits). Pairs of hard and
    core atoms of the task keep the reference geometry (e.g. a forming bond) and are not
    checked; pairs that close in the reference only count if they come closer.
    """

    name = "clash"

    def __init__(self, factor: float = CLASH_FACTOR):
        self.factor = factor

    def validate(self, ctx, ensemble: ConformerEnsemble) -> Dict[int, str]:
        held = (*ctx.frozen.hard, *ctx.frozen.core) if ctx is not None else ()
        groups = (
            ctx.by_reference(ensemble)
            if ctx is not None
            else [(None, ensemble.conf_ids)]
        )
        reasons = {}
        for reference, conf_ids in groups:
            seed = None
            if reference is not None:
                seed = reference.GetConformer().GetPositions()
            pairs = clash_limits(ensemble.mol, held, seed, self.factor)
            for conf_id in conf_ids:
                positions = ensemble.mol.GetConformer(conf_id).GetPositions()
                reason = first_clash(positions, pairs)
                if reason is not None:
                    reasons[conf_id] = reason
        return reasons

    def __repr__(self) -> str:
        return f"Clash(factor={self.factor})"


class RestraintViolation:
    """
    Whether the restraints that refinement holds are kept: no distance window of the
    context (stage "refine" or "both"; of the optional ones only those of the
    conformer's embedding batch, see restraints.applying) and no soft atom (held near
    its reference position) may be violated by more than tolerance (A). Meant for
    contacts that broke, not for the small excess that flat-bottom terms allow.
    """

    name = "restraints"

    def __init__(self, tolerance: float = 0.5):
        self.tolerance = tolerance

    def validate(self, ctx, ensemble: ConformerEnsemble) -> Dict[int, str]:
        if ctx is None:
            return {}
        windows = list(ctx.restraints.for_stage("refine"))
        reasons = {}
        for reference, conf_ids in ctx.by_reference(ensemble):
            soft = []
            if ctx.frozen.soft and reference is not None:
                soft = position_restraints(reference, ctx.frozen.soft)
            for conf_id in conf_ids:
                held = applying(windows, ensemble.provenance(conf_id)) + soft
                if not held:
                    continue
                positions = ensemble.mol.GetConformer(conf_id).GetPositions()
                worst = max(held, key=lambda r: r.violation(positions))
                excess = worst.violation(positions)
                if excess > self.tolerance:
                    reasons[conf_id] = f"{worst.label} violated by {excess:.2f} A"
        return reasons

    def __repr__(self) -> str:
        return f"RestraintViolation(tolerance={self.tolerance})"


GATE_FROZEN_TOLERANCE = 0.1  # A: see gate


def gate(
    clash_factor: float = CLASH_FACTOR,
    tolerance: float = 0.5,
    frozen_tolerance: float = GATE_FROZEN_TOLERANCE,
    warn_above: float = 0.3,
) -> Validate:
    """
    The validity gate after a refinement: a Validate stage that drops the conformers
    whose frozen atoms moved (FrozenCore), whose bonds or stereo changed
    (Connectivity), with clashes (Clash) or with a broken restraint
    (RestraintViolation). If more than warn_above of the conformers fail, a warning
    says so: that points to a wrong charge, restraint or hypothesis rather than to
    single bad conformers.

    frozen_tolerance: MMFF/UFF hold the frozen atoms with stiff springs, which the
    minimizer does not always pull fully onto the reference: they stay up to about
    0.09 A away (the same in every conformer), so the gate allows 0.1 A. Broken anchors
    move them much further.
    """
    return Validate(
        FrozenCore(frozen_tolerance),
        Connectivity(),
        Clash(clash_factor),
        RestraintViolation(tolerance),
        on_fail="drop",
        warn_above=warn_above,
    )
