"""Rigid attachment: poses of a swapped fragment from its conformers and rotations."""

import logging

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from racerts.pipeline import ConformerEnsemble
from racerts.system.swap import SwapResult, rotation_between

logger = logging.getLogger(__name__)

CLASH_FACTOR = 0.7  # heavy atoms closer than this times their vdW sum clash


def rigid_attach(
    result: SwapResult,
    n_fragment_conformers: int = 3,
    n_rotations: int = 12,
    seed: int = 0xF00D,
    clash_factor: float = CLASH_FACTOR,
) -> ConformerEnsemble:
    """
    Poses of the fragment of a single-attachment swap on every reference conformer:
    n_fragment_conformers conformers of the fragment (ETKDG), each placed along the
    removed bond as apply_swap places it and turned about the attachment bond in
    n_rotations steps. Poses whose heavy atoms clash with the kept heavy atoms
    (closer than clash_factor times the vdW sum, beyond three bonds) are left out.

    Returns an ensemble of result.mol with the provenance route "rigid", "reference"
    (the reference conformer id) and "pose" (fragment conformer, rotation step).
    """
    if not result.placed:
        raise ValueError(
            "Rigid attachment needs a single attachment that replaces a bond."
        )
    if n_fragment_conformers < 1 or n_rotations < 1:
        raise ValueError("n_fragment_conformers and n_rotations must be positive.")
    fragment = Chem.Mol(result.fragment)
    dummy = next(a.GetIdx() for a in fragment.GetAtoms() if a.GetAtomicNum() == 0)
    root = fragment.GetAtomWithIdx(dummy).GetNeighbors()[0].GetIdx()
    kept, root_new = result.attachments[0]
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    params.pruneRmsThresh = 0.1
    with rdBase.BlockLogs():  # UFF typer messages for the dummy
        ids = list(AllChem.EmbedMultipleConfs(fragment, n_fragment_conformers, params))
    if not ids:
        raise ValueError("Could not embed the fragment.")

    mol = result.mol
    new_atoms = sorted(result.fragment_map.values())
    kept_heavy = [
        i for i in result.conserved if mol.GetAtomWithIdx(i).GetAtomicNum() > 1
    ]
    new_heavy = [i for i in new_atoms if mol.GetAtomWithIdx(i).GetAtomicNum() > 1]
    topological = Chem.GetDistanceMatrix(mol)
    table = Chem.GetPeriodicTable()
    pairs = [(i, j) for i in new_heavy for j in kept_heavy if topological[i, j] > 3]
    limits = np.array(
        [
            clash_factor
            * (
                table.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum())
                + table.GetRvdw(mol.GetAtomWithIdx(j).GetAtomicNum())
            )
            for i, j in pairs
        ]
    )
    frag_idx = np.array(list(result.fragment_map))
    new_idx = np.array([result.fragment_map[j] for j in frag_idx])

    out = Chem.Mol(mol)
    out.RemoveAllConformers()
    ensemble = ConformerEnsemble(out)
    clashes = 0
    for ref in mol.GetConformers():
        positions = ref.GetPositions()
        anchor = positions[kept]
        axis = positions[root_new] - anchor
        length = np.linalg.norm(axis)
        axis /= length
        for frag_id in ids:
            xyz = fragment.GetConformer(frag_id).GetPositions()
            placed = (xyz - xyz[root]) @ rotation_between(
                xyz[root] - xyz[dummy], axis
            ).T + (anchor + length * axis)
            for step in range(n_rotations):
                turn = _axis_rotation(axis, 2 * np.pi * step / n_rotations)
                pose = positions.copy()
                pose[new_idx] = (placed[frag_idx] - anchor) @ turn.T + anchor
                if pairs:
                    first, second = zip(*pairs)
                    d = np.linalg.norm(pose[list(first)] - pose[list(second)], axis=1)
                    if (d < limits).any():
                        clashes += 1
                        continue
                conf = Chem.Conformer(mol.GetNumAtoms())
                for k, p in enumerate(pose):
                    conf.SetAtomPosition(k, p.tolist())
                conf_id = out.AddConformer(conf, assignId=True)
                ensemble.add_provenance(
                    conf_id,
                    route="rigid",
                    reference=ref.GetId(),
                    pose=[int(frag_id), step],
                )
    logger.info(
        "Rigid attachment: %d poses, %d clashing ones left out.", len(ensemble), clashes
    )
    return ensemble


def _axis_rotation(axis: np.ndarray, angle: float) -> np.ndarray:
    """Rotation by angle about the unit vector axis (Rodrigues)."""
    x, y, z = axis
    skew = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * skew + (1 - np.cos(angle)) * skew @ skew
