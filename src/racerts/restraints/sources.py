"""Where restraints come from: the user and the seed geometry."""

from numbers import Real
from typing import Iterable, List, Sequence, Tuple

import numpy as np
from rdkit import Chem

from racerts.system.spec import split_fragments

# Hydrogen bonds D-H...A: donor atoms of the hydrogen and acceptor elements.
HBOND_DONORS = ("N", "O")
HBOND_ACCEPTORS = ("N", "O", "F", "S", "Cl", "Br", "I")

Triplet = Tuple[int, int, float]


def reference_positions(mol: Chem.Mol, what: str) -> np.ndarray:
    if mol.GetNumConformers() == 0:
        raise ValueError(f"{what} need a reference geometry (a conformer of mol).")
    return mol.GetConformer().GetPositions()


def check_pair(mol: Chem.Mol, pair: Sequence, what: str) -> Tuple[int, int]:
    """The pair as two ints, checked against mol."""
    if isinstance(pair, (Real, str)) or len(pair) != 2:
        raise ValueError(f"{what} {pair!r} is not a pair of atoms.")
    first, second = pair
    for atom in (first, second):
        if isinstance(atom, bool) or not isinstance(atom, (int, np.integer)):
            raise TypeError(f"{what} {tuple(pair)}: atom indices must be integers.")
    if first == second:
        raise ValueError(f"{what} {tuple(pair)} needs two different atoms.")
    n = mol.GetNumAtoms()
    if not (0 <= first < n and 0 <= second < n):
        raise ValueError(f"{what} {tuple(pair)} is outside the {n}-atom molecule.")
    return int(first), int(second)


def check_triplets(mol: Chem.Mol, triplets: Iterable[Sequence]) -> List[Triplet]:
    """(atom, atom, target distance) triplets, checked."""
    if isinstance(triplets, Real) or (
        isinstance(triplets, (list, tuple))
        and len(triplets) == 3
        and isinstance(triplets[2], Real)
        and not isinstance(triplets[0], (list, tuple))
    ):
        raise ValueError(
            "Constraints must be a list of (atom, atom, distance) triplets, e.g. "
            "[(2, 7, 2.2)]."
        )
    checked = []
    for triplet in triplets:
        if isinstance(triplet, Real) or len(triplet) != 3:
            raise ValueError(f"{triplet!r} is not an (atom, atom, distance) triplet.")
        first, second = check_pair(mol, triplet[:2], "Constraint")
        target = triplet[2]
        if isinstance(target, bool) or not isinstance(target, Real) or target <= 0:
            raise ValueError(f"Constraint {tuple(triplet)} needs a positive distance.")
        checked.append((first, second, float(target)))
    return checked


def hydrogen_bonds(
    mol: Chem.Mol,
    max_h_acceptor: float = 2.5,
    max_donor_acceptor: float = 3.5,
    min_angle: float = 120.0,
) -> List[Triplet]:
    """
    Hydrogen bonds D-H...A of the reference geometry (the conformer of mol) at their
    distances: two triplets each, H...A and D...A, which keep the contact and its
    direction. Donors are N-H and O-H; acceptors N, O, F, S and the heavier halogens.
    """
    positions = reference_positions(mol, "Hydrogen bonds of the seed")
    acceptors = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in HBOND_ACCEPTORS]
    triplets = []
    for hydrogen in mol.GetAtoms():
        neighbors = hydrogen.GetNeighbors()
        if (
            hydrogen.GetAtomicNum() != 1
            or len(neighbors) != 1
            or neighbors[0].GetSymbol() not in HBOND_DONORS
        ):
            continue
        h, d = hydrogen.GetIdx(), neighbors[0].GetIdx()
        for a in acceptors:
            h_a = positions[a] - positions[h]
            h_d = positions[d] - positions[h]
            d_ha = float(np.linalg.norm(h_a))
            d_da = float(np.linalg.norm(positions[a] - positions[d]))
            cos_angle = np.dot(h_a, h_d) / (d_ha * np.linalg.norm(h_d))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            if (
                d_ha <= max_h_acceptor
                and d_da <= max_donor_acceptor
                and angle >= min_angle
            ):
                triplets += [(h, a, d_ha), (d, a, d_da)]
    return triplets


def contacts(mol: Chem.Mol, pairs: Iterable[Sequence[int]]) -> List[Triplet]:
    """
    Declared non-covalent contacts (a, b) at their distances in the reference
    geometry, with a and the neighbours of b, and b and the neighbours of a, which
    keep the orientation of the contact.
    """
    pairs = list(pairs)
    if not pairs:
        return []
    positions = reference_positions(mol, "Contacts")
    topological = Chem.GetDistanceMatrix(mol)
    triplets = {}
    for pair in pairs:
        a, b = check_pair(mol, pair, "Contact")
        if topological[a, b] < 3:
            raise ValueError(
                f"Atoms {a} and {b} are bonded or share a neighbour; contacts are for "
                "non-covalent interactions."
            )
        partners = [(a, b)]
        partners += [(a, n.GetIdx()) for n in mol.GetAtomWithIdx(b).GetNeighbors()]
        partners += [(b, n.GetIdx()) for n in mol.GetAtomWithIdx(a).GetNeighbors()]
        for i, j in partners:
            key = (min(i, j), max(i, j))
            triplets[key] = (*key, float(np.linalg.norm(positions[i] - positions[j])))
    return list(triplets.values())


def fragment_contacts(
    mol: Chem.Mol, core_atoms: Sequence[int]
) -> List[Tuple[int, int]]:
    """
    Contacts of the reference geometry that attach every fragment without core atoms
    (e.g. solvent molecules or counterions) to the core: fragment by fragment, the
    closest first (relative to vdW radii), each by its closest contact to the core or
    to a fragment attached before, so that a solvent cluster stays at the core.
    """
    core, free = split_fragments(mol, core_atoms)
    if not free:
        return []
    positions = reference_positions(mol, "Fragment contacts")
    table = Chem.GetPeriodicTable()
    radius = np.array([table.GetRvdw(atom.GetAtomicNum()) for atom in mol.GetAtoms()])
    scaled = np.linalg.norm(positions[:, None] - positions[None], axis=-1) / (
        radius[:, None] + radius[None]
    )
    links = []
    while free:
        closest = None  # (scaled distance, free atom, attached atom, fragment)
        for fragment in free:
            block = scaled[np.ix_(fragment, core)]
            i, j = np.unravel_index(block.argmin(), block.shape)
            if closest is None or block[i, j] < closest[0]:
                closest = (block[i, j], fragment[i], core[j], fragment)
        _, a, b, fragment = closest
        links.append((min(a, b), max(a, b)))
        core = core + fragment
        free.remove(fragment)
    return links
