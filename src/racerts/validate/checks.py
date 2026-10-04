"""Geometric validators: connectivity and stereo of the graph, the frozen core."""

from typing import Dict, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from racerts.geometry import superpose
from racerts.pipeline import ConformerEnsemble
from racerts.system.stereo import StereoCheck

from .base import Validate


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
