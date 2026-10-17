"""The torsions that Recombine passes on: which bonds, their angles, and a structure
with some of them turned to other values."""

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Sequence, Tuple

import numpy as np
from rdkit import Chem

from racerts.exploit.moves import _side, rotate


@dataclass(frozen=True)
class Torsion:
    """
    The torsion about the bond atoms[1]-atoms[2], measured as the dihedral of the
    four atoms; moving: the atoms that turn with atoms[2] about the bond.
    """

    atoms: Tuple[int, int, int, int]
    moving: Tuple[int, ...]


def torsions(mol: Chem.Mol, held: Iterable[int] = ()) -> List[Torsion]:
    """
    A torsion for every single bond outside rings between two heavy atoms that each
    carry another heavy atom (so not of methyl or hydroxyl groups), and not next to a
    linear atom. A side of the bond without held atoms turns (the smaller one if both
    qualify); the two atoms of the bond stay where they are, so they may be held. A
    bond whose turn would move held atoms on either side is no torsion.

    Unlike the moves of Exploit (torsion_moves) these include the planar bonds of
    amides and esters: a torsion is only ever set to the value that another conformer
    has, so no isomer is made that the ensemble does not have. A symmetric rotor (a
    tert-butyl or phenyl group) counts too; its equivalent states give duplicates.
    """
    held = set(held)
    linear = Chem.HybridizationType.SP
    found = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.SINGLE or bond.IsInRing():
            continue
        ends = bond.GetBeginAtom(), bond.GetEndAtom()
        if linear in (ends[0].GetHybridization(), ends[1].GetHybridization()):
            continue
        a, b = (atom.GetIdx() for atom in ends)
        others = [
            sorted(
                n.GetIdx()
                for n in atom.GetNeighbors()
                if n.GetAtomicNum() > 1 and n.GetIdx() not in (a, b)
            )
            for atom in ends
        ]
        if 1 in (ends[0].GetAtomicNum(), ends[1].GetAtomicNum()) or not all(others):
            continue
        side_a, side_b = _side(mol, b, a), _side(mol, a, b)
        turn_a, turn_b = side_a - {a}, side_b - {b}  # a and b stay on the axis
        if turn_a & held and turn_b & held:
            continue
        if turn_b & held or (not turn_a & held and len(side_a) < len(side_b)):
            atoms = (others[1][0], b, a, others[0][0])
            moving = turn_a
        else:
            atoms = (others[0][0], a, b, others[1][0])
            moving = turn_b
        found.append(Torsion(atoms, tuple(sorted(moving))))
    return found


def torsion_angles(positions, torsions: Sequence[Torsion]) -> np.ndarray:
    """
    The dihedral angles (degrees, -180 to 180) of the torsions in positions: (atoms,
    3) for one structure, or (structures, atoms, 3) for a row of angles for each.
    """
    positions = np.asarray(positions, dtype=float)
    index = np.array([t.atoms for t in torsions], dtype=int).reshape(-1, 4)
    a, b, c, d = (positions[..., index[:, k], :] for k in range(4))
    axis = c - b
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    v = (a - b) - ((a - b) * axis).sum(axis=-1, keepdims=True) * axis
    w = (d - c) - ((d - c) * axis).sum(axis=-1, keepdims=True) * axis
    sine = (np.cross(axis, v) * w).sum(axis=-1)
    return np.degrees(np.arctan2(sine, (v * w).sum(axis=-1)))


def set_torsions(
    positions, torsions: Sequence[Torsion], values: Mapping[int, float]
) -> np.ndarray:
    """
    A copy of positions in which torsion k has the angle values[k] (degrees) for
    every k in values: its moving side is turned rigidly about the bond. All bond
    lengths, angles and the other torsions stay, and so do the atoms on the side
    that does not turn.
    """
    turned = np.array(positions, dtype=float)
    now = torsion_angles(turned, torsions)
    for k, value in values.items():
        _, fixed, end, _ = torsions[k].atoms
        rotate(
            turned,
            list(torsions[k].moving),
            turned[end].copy(),
            turned[end] - turned[fixed],
            float(value - now[k]),
        )
    return turned
