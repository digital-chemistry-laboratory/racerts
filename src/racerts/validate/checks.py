"""Geometric validators: connectivity and stereo of the graph, the frozen core."""

from typing import Dict, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

from racerts.pipeline import ConformerEnsemble
from racerts.prune.rmsd import superpose
from racerts.system.stereo import StereoCheck

from .base import Validate


class Connectivity:
    """
    Whether each conformer still is the molecule of the graph: the bonds perceived
    from its geometry (RDKit's DetermineConnectivity, by covalent radii) are those of
    the graph, and the stereocentres and double bonds that the graph specifies have the
    same configuration (unspecified stereo is a wildcard). This is catmlp's identity
    check (geometry_matches_smiles), by atom index since the conformers share the
    graph.

    Args:
        exempt: Atoms whose bonds among each other and whose stereo are not checked;
            default: the core atoms of the task (FrozenSet.core): the reacting atoms
            of a TransitionState, whose bonds form or break; the hard atoms of
            Constrained, which the reference fixes; none for GroundState.
        bonds: Check the bonds.
        stereo: Check the stereo.
        cov_factor: Scale of the covalent radii for perceived bonds (RDKit default).
    """

    name = "connectivity"

    def __init__(
        self,
        exempt: Optional[Sequence[int]] = None,
        stereo: bool = True,
        cov_factor: float = 1.3,
        bonds: bool = True,
    ):
        if not (bonds or stereo):
            raise ValueError("Connectivity needs bonds or stereo to check.")
        self.exempt = None if exempt is None else tuple(exempt)
        self.stereo = stereo
        self.cov_factor = cov_factor
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
        rdDetermineBonds.DetermineConnectivity(
            perceived, useHueckel=False, covFactor=self.cov_factor
        )
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


def _max_deviation(positions: np.ndarray, reference: np.ndarray) -> float:
    """Largest atom deviation after the best superposition (Kabsch, rotations only)."""
    q = reference - reference.mean(axis=0)
    return float(np.linalg.norm(superpose(positions, reference) - q, axis=1).max())


class IdentityFilter(Validate):
    """
    Drops conformers that are no longer the molecule of the graph (see Connectivity);
    raises if none is left. catmlp's filter_conformers_by_connectivity.
    """

    name = "identity_filter"

    def __init__(self, **settings):
        super().__init__(Connectivity(**settings))
