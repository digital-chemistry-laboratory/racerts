"""Where restraints come from: the user, the reference geometry, fragment links."""

from itertools import combinations
from numbers import Real
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem

from racerts.system.spec import split_fragments
from racerts.utils import seeds
from racerts.utils.checks import is_integer

# Hydrogen bonds D-H...A: donor atoms of the hydrogen and acceptor elements.
HBOND_DONORS = ("N", "O")
HBOND_ACCEPTORS = ("N", "O", "F", "S", "Cl", "Br", "I")
# The windows of fragment links, in units of the sum of the vdW radii.
LINK_LOWER_FACTORS = (1.0, 0.8)
LINK_UPPER_FACTOR = 1.3
LINK_SEED = 0xF00D  # breaks ties between equally exposed atoms of fragment links
MAX_HINTS = 8

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
        if not is_integer(atom):
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
    positions = reference_positions(mol, "Hydrogen bonds of the reference")
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


def link_window(
    mol: Chem.Mol, pair: Tuple[int, int], lower_factor: float
) -> Tuple[float, float]:
    """The contact window of a fragment link: [lower_factor, 1.3] x the vdW sum."""
    table = Chem.GetPeriodicTable()
    vdw = sum(table.GetRvdw(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in pair)
    return lower_factor * vdw, LINK_UPPER_FACTOR * vdw


def fallback_links(
    mol: Chem.Mol, active_atoms: Optional[Sequence[int]] = None, seed: int = LINK_SEED
) -> List[Tuple[int, int]]:
    """
    Candidate links between fragments: first between fragments of opposite
    charge (their least buried atoms of the fragment's charge sign), then between
    representatives of every pair of fragments (active atoms first, else heavy atoms,
    the least buried; ties broken by a seeded random choice).
    """
    fragments = Chem.GetMolFrags(mol, asMols=False)
    if len(fragments) < 2:
        return []
    active = set(active_atoms or [])
    rng = seeds.Stream(seed, "fragment links")
    distances = Chem.GetDistanceMatrix(mol)
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]

    def burial(idx):
        return sum(1 for other in heavy if other != idx and distances[idx][other] <= 2)

    def least_buried(candidates):
        least = min(burial(i) for i in candidates)
        return rng.choice([i for i in candidates if burial(i) == least])

    charges = [
        sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in fragment)
        for fragment in fragments
    ]
    charged = {}
    for k, (fragment, charge) in enumerate(zip(fragments, charges)):
        if charge:
            same_sign = [
                i
                for i in fragment
                if mol.GetAtomWithIdx(i).GetFormalCharge() * charge > 0
            ]
            charged[k] = least_buried(same_sign)
    links = [
        (charged[a], charged[b])
        for a, b in combinations(range(len(fragments)), 2)
        if charges[a] * charges[b] < 0
    ]
    representatives = [
        least_buried(
            [i for i in fragment if i in active]
            or [i for i in fragment if i in heavy]
            or list(fragment)
        )
        for fragment in fragments
    ]
    links += [pair for pair in combinations(representatives, 2) if pair not in links]
    return links


def carrier_links(
    mol: Chem.Mol, candidates: Iterable[Sequence[int]]
) -> List[Tuple[int, int]]:
    """
    Links that join all fragments, taken from the candidates in order: a spanning
    tree over the fragments. Raises if the candidates do not join them all.
    """
    fragments = Chem.GetMolFrags(mol, asMols=False)
    if len(fragments) < 2:
        return []
    fragment_of = {i: k for k, fragment in enumerate(fragments) for i in fragment}
    parent = list(range(len(fragments)))

    def find(k):
        while parent[k] != k:
            parent[k] = parent[parent[k]]
            k = parent[k]
        return k

    links = []
    for pair in candidates:
        a, b = sorted(check_pair(mol, pair, "Fragment link"))
        root_a, root_b = find(fragment_of[a]), find(fragment_of[b])
        if root_a == root_b:
            continue
        parent[root_b] = root_a
        links.append((a, b))
    if len({find(k) for k in range(len(fragments))}) != 1:
        raise ValueError("The fragment links do not join every fragment.")
    return links


# Acceptors that graph hints leave out: amide/thioamide and carbamate
# N, aniline N, pyrrole-type aromatic N, and the alkoxy O of esters and carbamates.
_POOR_ACCEPTORS = [
    Chem.MolFromSmarts(smarts)
    for smarts in ("[NX3][CX3]=[O,S]", "[NX3]a", "[nX3]", "[OX2]([#6])[CX3]=[O,S]")
]
HINT_WINDOW = (1.7, 2.3)  # A, H...acceptor


def graph_hints(
    mol: Chem.Mol,
    max_hints: int = MAX_HINTS,
    window: Tuple[float, float] = HINT_WINDOW,
    charged: bool = False,
    min_ring: int = 6,
) -> List[Tuple[int, int, float, float]]:
    """
    Candidate hydrogen bonds from the graph alone: N-H or O-H
    donors, N, O or F acceptors without the poor ones (amide and aniline N, aromatic
    N-H type N, ester alkoxy O), pairs that close a pseudo-ring of at least min_ring
    atoms (or lie in different fragments), without charged partners unless charged
    (MMFF's Coulomb term already pulls those together). Ranked by how close the
    pseudo-ring is to 7 atoms; at most max_hints, as (H, acceptor, lower, upper).
    """
    poor = {
        match[0]
        for pattern in _POOR_ACCEPTORS
        for match in mol.GetSubstructMatches(pattern)
    }
    acceptors = [
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetSymbol() in ("N", "O", "F")
        and a.GetIdx() not in poor
        and a.GetFormalCharge() <= 0
        and a.GetTotalDegree() < 4
        and (charged or a.GetFormalCharge() == 0)
    ]
    topological = Chem.GetDistanceMatrix(mol)
    candidates = []
    for hydrogen in mol.GetAtoms():
        neighbors = hydrogen.GetNeighbors()
        if hydrogen.GetAtomicNum() != 1 or len(neighbors) != 1:
            continue
        donor = neighbors[0]
        if donor.GetSymbol() not in HBOND_DONORS:
            continue
        if not charged and donor.GetFormalCharge() != 0:
            continue
        h = hydrogen.GetIdx()
        for a in acceptors:
            if a == donor.GetIdx():
                continue
            d = topological[h, a]
            ring = np.inf if d > 1e6 else d + 1  # different fragments: no ring
            if ring < min_ring:
                continue
            candidates.append((0 if ring == np.inf else abs(ring - 7), h, a))
    candidates.sort()
    return [(h, a, *window) for _, h, a in candidates[:max_hints]]
