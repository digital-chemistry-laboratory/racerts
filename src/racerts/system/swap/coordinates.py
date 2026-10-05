"""
The coordinates of the result: the kept atoms where they are, a fragment with one
attachment grafted along the removed bond.
"""

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

from racerts.utils import seeds

from .model import SwapError


def _coordinates(
    mol, result, fragment, attachments, frag_to_new, ref_to_new, placed, seed
):
    """Every reference conformer, with the fragment grafted if placed."""
    xyz = None
    if placed:
        embedded = Chem.Mol(fragment)
        with rdBase.BlockLogs():  # UFF typer messages for the dummies
            start = seeds.derive(seed, "swap fragment")
            if AllChem.EmbedMolecule(embedded, randomSeed=start) < 0:
                raise SwapError("Could not embed the fragment.")
        xyz = embedded.GetConformer().GetPositions()
        a = attachments[0]
    for conf in mol.GetConformers():
        positions = conf.GetPositions()
        new_positions = np.zeros((result.GetNumAtoms(), 3))
        for i, k in ref_to_new.items():
            new_positions[k] = positions[i]
        if placed:
            anchor = positions[a.kept]
            direction = positions[a.partner] - anchor
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm < 1e-8:
                raise SwapError("The attachment has coincident or invalid coordinates.")
            direction /= norm
            length = _bond_length(mol, fragment, a, norm)
            rotation = rotation_between(xyz[a.root] - xyz[a.dummy], direction)
            grafted = (xyz - xyz[a.root]) @ rotation.T + anchor + length * direction
            for j, k in frag_to_new.items():
                new_positions[k] = grafted[j]
        else:  # roots that replace an atom: at its position (same element) or along
            for b in attachments:  # its bond at the sum of the covalent radii
                if b.partner is None:
                    continue
                new_positions[frag_to_new[b.root]] = _root_position(
                    mol, fragment, b, positions
                )
        new_conf = Chem.Conformer(result.GetNumAtoms())
        for k, p in enumerate(new_positions):
            new_conf.SetAtomPosition(k, p.tolist())
        new_conf.SetId(conf.GetId())
        new_conf.Set3D(conf.Is3D())
        result.AddConformer(new_conf, assignId=False)


def _bond_length(mol, fragment, attachment, removed_length: float) -> float:
    """
    The length of the new bond: the removed one scaled by covalent radii, so that a
    bond stretched in the reference (e.g. a leaving group of a TS) stays stretched;
    for a dummy that leaves (a template site), the sum of the covalent radii.
    """
    table = Chem.GetPeriodicTable()
    kept = table.GetRcovalent(mol.GetAtomWithIdx(attachment.kept).GetAtomicNum())
    root = table.GetRcovalent(fragment.GetAtomWithIdx(attachment.root).GetAtomicNum())
    partner = mol.GetAtomWithIdx(attachment.partner).GetAtomicNum()
    if partner == 0:
        return kept + root
    return removed_length * (kept + root) / (kept + table.GetRcovalent(partner))


def _root_position(mol, fragment, attachment, positions) -> np.ndarray:
    anchor = positions[attachment.kept]
    direction = positions[attachment.partner] - anchor
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < 1e-8:
        raise SwapError("The attachment has coincident or invalid coordinates.")
    return anchor + direction / norm * _bond_length(mol, fragment, attachment, norm)


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The proper rotation taking the direction source onto target."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cosine = float(np.clip(source @ target, -1, 1))
    if cosine < -1 + 1e-12:  # antiparallel: a half turn about a perpendicular axis
        axis = np.cross(source, np.eye(3)[np.argmin(np.abs(source))])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    cross = np.cross(source, target)
    skew = np.array(
        [[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]]
    )
    return np.eye(3) + skew + skew @ skew / (1 + cosine)
