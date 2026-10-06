"""
The symmetry-aware RMSD of two conformers of one graph, without listing every
equivalent atom mapping via the minimum over the automorphisms of the graph and
over the proper rotations. Listing the automorphisms explodes with hydrogens.

SymmetricRMSD lists them if there are at most max_maps (the exact result). Otherwise it
strips terminal atoms from the graph round by round (the hydrogens, then e.g. the
methyl carbons that became terminal) until what is left, the skeleton, has at most
max_maps automorphisms, and lists only those. The stripped atoms hang on the skeleton
as trees; sibling trees of one shape on one parent (the H of a methyl, the F of a CF3,
the methyls of a tert-butyl) may be permuted freely, so the automorphisms are the
skeleton maps times the sibling permutations, and the second factor is never listed.
Per pair of conformers:

1. for every skeleton map, a lower bound of the deviation of all its continuations:
   Kabsch on the atoms whose image the map fixes and on the centroids of the sets of
   atoms that the sibling permutations mix;
2. in the order of that bound, while it is below the best value so far: the best
   sibling permutations by assignment per family for the rotation of the bound (all
   permutations of up to four siblings, the Hungarian method above), then Kabsch with
   the resulting atom map, and again until the map stays; the same from two more
   starting rotations.

Every value is the Kabsch RMSD of an explicit automorphism, so it is never below the
true minimum. It can be above it only if step 2 ends in a local minimum, which was
seen for pairs of different conformers (5 of 3060, by at most 0.006 A) and for no
duplicate.

bound_descriptors gives two vectors per conformer whose distance between two
conformers is never above their symmetry-minimised RMSD: pairs further apart need no
RMSD at all.
"""

from __future__ import annotations

import itertools
import logging
import math
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
from rdkit import Chem
from scipy.optimize import linear_sum_assignment

from racerts.geometry import symmetrize_terminal_atoms

logger = logging.getLogger(__name__)

POLAR_PARENTS = (7, 8, 15, 16)  # "polar" hydrogens: bonded to N, O, P or S
_INDEX = "_symmetric_rmsd_index"
_FORCED = 1000  # isotope offset of the forced automorphism test
_BRUTE = 4  # families up to this size: all permutations; above: Hungarian method
_CHUNK = 20000  # maps per batch of the plain enumeration


def matching_graph(
    mol: Chem.Mol,
    atoms: Union[str, Sequence[int]] = "heavy",
    polar_parents=POLAR_PARENTS,
) -> Tuple[Chem.Mol, np.ndarray, np.ndarray]:
    """
    The graph whose automorphisms define the RMSD over atoms, as (graph, index, weight):
    index[k] is the atom of mol that is atom k of the graph, weight[k] whether it counts
    in the RMSD.

    atoms: "heavy" (what Chem.RemoveHs keeps: racerts' include_hs=False), "all"
    (include_hs=True), "polar" (heavy plus the hydrogens on N, O, P, S), or atom indices
    (their hydrogens stay explicit, all other hydrogens become implicit, other atoms
    that are not listed stay in the graph with weight False).
    """
    hydrogens = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() == 1]
    selection = None
    if isinstance(atoms, str):
        if atoms == "all":
            keep = set(hydrogens)
        elif atoms in ("heavy", "none"):
            keep = set()
        elif atoms == "polar":
            keep = {
                h
                for h in hydrogens
                if any(
                    nbr.GetAtomicNum() in polar_parents
                    for nbr in mol.GetAtomWithIdx(h).GetNeighbors()
                )
            }
        else:
            raise ValueError(
                "atoms must be 'heavy', 'none', 'polar', 'all' or atom indices."
            )
    else:
        selection = sorted({int(i) for i in atoms})
        if not selection or selection[0] < 0 or selection[-1] >= mol.GetNumAtoms():
            raise ValueError("atoms must be a nonempty set of atom indices of mol.")
        keep = set(selection) & set(hydrogens)
    graph = Chem.Mol(mol)
    graph.RemoveAllConformers()
    for atom in graph.GetAtoms():
        atom.SetIntProp(_INDEX, atom.GetIdx())
    if len(keep) < len(hydrogens):
        # Chem.RemoveHs keeps hydrogens with an atom-map number if told so: the marker.
        numbers = {h: graph.GetAtomWithIdx(h).GetAtomMapNum() for h in keep}
        for h in hydrogens:
            graph.GetAtomWithIdx(h).SetAtomMapNum(1 if h in keep else 0)
        params = Chem.RemoveHsParameters()
        params.removeMapped = False
        try:
            graph = Chem.RemoveHs(graph, params, sanitize=True)
        except Exception:
            graph = Chem.RemoveHs(graph, params, sanitize=False)
        for atom in graph.GetAtoms():
            if atom.GetIntProp(_INDEX) in numbers:
                atom.SetAtomMapNum(numbers[atom.GetIntProp(_INDEX)])
    index = np.array(
        [atom.GetIntProp(_INDEX) for atom in graph.GetAtoms()], dtype=np.intp
    )
    if selection is None:
        weight = np.ones(len(index), dtype=bool)
    else:
        weight = np.isin(index, selection)
        if int(weight.sum()) != len(selection):
            raise ValueError("Some of the atoms are not in the graph.")
    return Chem.Mol(symmetrize_terminal_atoms(graph)), index, weight


def _matches(graph: Chem.Mol, max_matches: int) -> np.ndarray:
    """The automorphisms of graph, (n_maps, n_atoms), the identity first."""
    n = graph.GetNumAtoms()
    found = graph.GetSubstructMatches(
        graph,
        maxMatches=max_matches,
        uniquify=False,
        useChirality=True,
        useQueryQueryMatches=False,
    )
    maps = np.array(found, dtype=np.intp).reshape(-1, n)
    identity = np.arange(n)
    return np.vstack([identity, maps[(maps != identity).any(axis=1)]])


def _msd(a: np.ndarray, b: np.ndarray, n: int, align: bool = True) -> np.ndarray:
    """Mean squared deviation of a (n, 3) from each b[k] (k, n, 3) (Kabsch, proper)."""
    if not align:
        return ((b - a) ** 2).sum(axis=(-2, -1)) / n
    h = np.matmul(np.swapaxes(b, -1, -2), a)
    s = np.linalg.svd(h, compute_uv=False)
    sign = np.where(np.linalg.det(h) < 0, -1.0, 1.0)
    trace = s[..., 0] + s[..., 1] + sign * s[..., 2]
    squares = (a * a).sum() + (b * b).sum(axis=(-2, -1))
    return np.maximum(squares - 2.0 * trace, 0.0) / n


def _centred(x: np.ndarray, weight: np.ndarray) -> np.ndarray:
    """x minus the centroid of the weighted atoms, the weightless rows zero."""
    if weight.all():
        return x - x.mean(axis=0)
    out = x - x[weight].mean(axis=0)
    out[~weight] = 0.0
    return out


def _rotation(h: np.ndarray) -> Tuple[np.ndarray, float]:
    """For h = b^T a: the proper rotation r with b @ r closest to a, and trace(r^T h)."""
    u, s, vt = np.linalg.svd(h)
    r = u @ vt
    a, b, c = r.tolist()  # the determinant of an orthogonal matrix, without numpy calls
    det = (
        a[0] * (b[1] * c[2] - b[2] * c[1])
        - a[1] * (b[0] * c[2] - b[2] * c[0])
        + a[2] * (b[0] * c[1] - b[1] * c[0])
    )
    if det < 0:
        u[:, 2] = -u[:, 2]
        return u @ vt, float(s[0] + s[1] - s[2])
    return r, float(s[0] + s[1] + s[2])


def _explicit(a: np.ndarray, b: np.ndarray, n: int, align: bool) -> float:
    """The RMSD of paired coordinates, by rotating b explicitly (exact near zero)."""
    if align:
        b = b @ _rotation(b.T @ a)[0]
    delta = b - a
    return float(np.sqrt((delta * delta).sum() / n))


# ------------------------------------------------------ lower bounds that need no map
def symmetry_classes(
    graph: Chem.Mol, weight: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    A partition of the atoms that no automorphism crosses (classes are unions of
    orbits): colour refinement from element, charge, isotope, radical electrons and
    weight over the bonds. Bond types are used only if no bond is of the wildcard type
    that symmetrize_terminal_atoms creates. Coarser than the orbits at worst, which
    only weakens bound_descriptors.
    """
    n = graph.GetNumAtoms()
    weight = np.ones(n, dtype=bool) if weight is None else weight
    types = {
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx()): int(b.GetBondType())
        for b in graph.GetBonds()
    }
    if any(t == int(Chem.BondType.UNSPECIFIED) for t in types.values()):
        types = {k: 0 for k in types}
    nbrs: List[List[Tuple[int, int]]] = [[] for _ in range(n)]
    for (i, j), t in types.items():
        nbrs[i].append((j, t))
        nbrs[j].append((i, t))
    keys = [
        (
            a.GetAtomicNum(),
            a.GetFormalCharge(),
            a.GetIsotope(),
            a.GetNumRadicalElectrons(),
            bool(weight[a.GetIdx()]),
        )
        for a in graph.GetAtoms()
    ]
    colour = _dense(keys)
    while True:
        refined = _dense(
            [
                (colour[i], tuple(sorted((t, colour[j]) for j, t in nbrs[i])))
                for i in range(n)
            ]
        )
        if len(set(refined)) == len(set(colour)):
            return np.array(refined)
        colour = refined


def _dense(keys) -> List[int]:
    ids = {key: k for k, key in enumerate(sorted(set(keys)))}
    return [ids[key] for key in keys]


def bound_descriptors(
    positions: np.ndarray, index: np.ndarray, weight: np.ndarray, classes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Per conformer (positions: (..., n_atoms, 3)), two vectors whose Euclidean distance
    between two conformers is at most their symmetry-minimised RMSD (any rotation, any
    automorphism), each already divided by sqrt(n):

      the three singular values of the centred coordinates (Mirsky's inequality), and
      the distances of the atoms from the centroid, sorted inside each symmetry class
      (classes: symmetry_classes(graph, weight)).
    """
    x = np.asarray(positions, dtype=float)[..., index[weight], :]
    x = x - x.mean(axis=-2, keepdims=True)
    scale = 1.0 / math.sqrt(x.shape[-2])
    radii = np.sqrt((x * x).sum(axis=-1))
    cls = np.asarray(classes)[weight]
    profile = np.empty_like(radii)
    for label in np.unique(cls):
        members = np.flatnonzero(cls == label)
        profile[..., members] = np.sort(radii[..., members], axis=-1)
    return np.linalg.svd(x, compute_uv=False) * scale, profile * scale


# ------------------------------------------------------------------ the factorisation
class _Block(NamedTuple):
    """
    The family pairs of one depth and one size k. A node pair (u, v) is an atom u of a
    and a candidate image v in b. pairs[f, p, i] is the node pair (child i of the family
    in a, child perms[p][i] of the family in b), for the brute force over p; child[f]
    the k x k node pairs for the Hungarian method; parent[f] the node pair the two
    families hang on (None: on atoms whose image is fixed).
    """

    k: int
    pairs: Optional[np.ndarray]
    child: np.ndarray
    parent: Optional[np.ndarray]
    count: np.ndarray


class Prepared(NamedTuple):
    """A conformer as SymmetricRMSD.prepare returns it."""

    x: np.ndarray  # (n_graph, 3) centred on the weighted atoms, weightless rows zero
    points: np.ndarray  # (n_points, 3) the points of the lower bound
    points_t: np.ndarray  # (3, n_points) the same, transposed and contiguous
    norm: float  # sum of x^2
    norm_points: float
    align: bool


class SymmetricRMSD:
    """
    The symmetry-minimised RMSD of conformers of mol over atoms (see the module text).

    Args:
        mol: The molecule (its graph; a conformer is needed to choose a level unless
            reference is given or the graph has at most max_maps automorphisms).
        atoms: "heavy", "polar", "all" or atom indices (matching_graph).
        max_maps: The most maps to enumerate without stripping (further). With at most
            max_maps automorphisms in total nothing is stripped and the result is the
            plain enumeration (exact).
        hard_max_maps: If no level gets below max_maps, the skeleton maps of the
            deepest usable level are cut here (truncated is set; the result stays an
            upper bound of the RMSD).
        reference: (n_conf, n_atoms, 3) or (n_atoms, 3) positions to judge whether the
            fixed points of a level determine a rotation (default: the conformers of
            mol, at most 10).
        min_spread: A level is refused if the root-mean-square extent of its fixed
            points along their second principal axis is below this (A).

    Attributes: index, weight (matching_graph), n (atoms in the RMSD), level (rounds
    of stripping; 0: plain enumeration), maps (the skeleton maps, continued onto all
    graph atoms), order (the number of automorphisms represented, an int), truncated,
    stats (counts: pairs, pairs that needed the bounds of step 1, pairs that needed
    step 2 in within, skeleton maps evaluated in step 2, assignments).
    """

    max_rounds = 8
    starts = 3

    def __init__(
        self,
        mol: Chem.Mol,
        atoms: Union[str, Sequence[int]] = "heavy",
        max_maps: int = 100,
        hard_max_maps: int = 10000,
        reference=None,
        min_spread: float = 0.2,
        max_pairs: int = 200000,
        polar_parents=POLAR_PARENTS,
    ):
        if max_maps < 1 or hard_max_maps < max_maps:
            raise ValueError("Need 1 <= max_maps <= hard_max_maps.")
        self.graph, self.index, self.weight = matching_graph(mol, atoms, polar_parents)
        self.n = int(self.weight.sum())
        self._all = bool(self.weight.all())
        n = self.graph.GetNumAtoms()
        self._nbrs: List[List[int]] = [[] for _ in range(n)]
        self._bond = {}
        for bond in self.graph.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            self._nbrs[i].append(j)
            self._nbrs[j].append(i)
            self._bond[(i, j)] = self._bond[(j, i)] = int(bond.GetBondType())
        self._inv = [
            (
                a.GetAtomicNum(),
                a.GetFormalCharge(),
                a.GetIsotope(),
                a.GetNumRadicalElectrons(),
                bool(self.weight[a.GetIdx()]),
            )
            for a in self.graph.GetAtoms()
        ]
        maps = self._skeleton_maps(0, max_maps + 1)
        layout = self._layout(0, max_pairs)
        level = 0
        self.n_pinned = self.n_generators = 0
        self.max_level = 0
        if len(maps) > max_maps:
            reference = self._reference(mol, reference)
            pinned = self._pinned()
            while True:  # strip, and let RDKit confirm every sibling transposition
                self._strip(pinned)
                self._shapes()
                rejected = self._rejected_transpositions()
                if not len(rejected):
                    break
                pinned[rejected] = True
            self.n_pinned = int(pinned.sum())
            self.max_level = int(self._level.max())
            for deeper in range(1, self.max_level + 1):
                candidate = self._layout(deeper, max_pairs)
                if candidate is None or self._spread(candidate, reference) < min_spread:
                    break
                level, layout = deeper, candidate
                maps = self._skeleton_maps(level, max_maps + 1)
                if len(maps) <= max_maps:
                    break
        self.truncated = False
        if len(maps) > max_maps:
            maps = self._skeleton_maps(level, hard_max_maps + 1)
            self.truncated = len(maps) > hard_max_maps
            maps = maps[:hard_max_maps]
            if self.truncated:
                logger.warning(
                    "More than %d skeleton maps at the deepest usable level (%d): the "
                    "list is cut, and the RMSD can miss equivalent atom mappings (it "
                    "stays an upper bound).",
                    hard_max_maps,
                    level,
                )
        self.level = level
        self.maps = maps
        self.stats = dict.fromkeys(
            ("pairs", "bounds", "evaluated", "evaluations", "assignments"), 0
        )
        self._install(layout)

    # ---------------------------------------------------------------- set-up: the graph
    def _reference(self, mol, reference) -> Optional[np.ndarray]:
        if reference is None:
            confs = list(mol.GetConformers())[:10]
            if not confs:
                return None
            reference = np.array([c.GetPositions() for c in confs])
        reference = np.asarray(reference, dtype=float)
        if reference.ndim == 2:
            reference = reference[None]
        return np.array([_centred(x[self.index], self.weight) for x in reference])

    def _pinned(self) -> np.ndarray:
        """Atoms that are never stripped: those whose permutation RDKit's chirality and
        double-bond stereo tests can refuse (stereo atoms and their neighbours), and
        dummy atoms. _rejected_transpositions catches whatever this misses."""
        pinned = np.zeros(self.graph.GetNumAtoms(), dtype=bool)
        centres = [
            a.GetIdx()
            for a in self.graph.GetAtoms()
            if a.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
            or a.GetAtomicNum() == 0
        ]
        for bond in self.graph.GetBonds():
            if bond.GetStereo() != Chem.BondStereo.STEREONONE:
                centres += [bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()]
        for i in centres:
            pinned[i] = True
            pinned[self._nbrs[i]] = True
        return pinned

    def _strip(self, pinned: np.ndarray) -> None:
        """
        level[i]: the round in which atom i is stripped (0: never), parent[i]: the atom
        it hung on (-1: none, the atom was alone). A round strips the unpinned atoms
        with one remaining neighbour (not the two ends of a last bond) or none (not if
        nothing would be left).
        """
        n = len(pinned)
        level = np.zeros(n, dtype=int)
        parent = np.full(n, -1, dtype=int)
        alive = np.ones(n, dtype=bool)
        degree = np.array([len(x) for x in self._nbrs])
        current = 0
        while True:
            current += 1
            chosen = []
            for i in np.flatnonzero(alive & ~pinned & (degree <= 1)):
                if degree[i] == 0:
                    chosen.append((int(i), -1))
                    continue
                p = next(j for j in self._nbrs[i] if alive[j])
                if degree[p] == 1 and not pinned[p]:
                    continue  # the last bond of a tree: both ends stay
                chosen.append((int(i), p))
            if not chosen or len(chosen) == int(alive.sum()):
                break
            for i, p in chosen:
                level[i], parent[i], alive[i] = current, p, False
                if p >= 0:
                    degree[p] -= 1
        self._level, self._parent = level, parent

    def _shapes(self) -> None:
        """
        The shape of the tree that each strippable atom carries: its own invariants,
        the bond to its parent and the shapes of its children. Families: the children
        of one shape on one parent (or without parent), in the order of their indices.
        """
        shape = np.full(len(self._level), -1, dtype=int)
        self._families = {}  # parent (or -1) -> {shape: [children]}
        ids = {}
        for i in sorted(
            np.flatnonzero(self._level > 0), key=lambda i: (self._level[i], i)
        ):
            i = int(i)
            p = int(self._parent[i])
            kids = self._families.get(i, {})
            key = (
                self._inv[i],
                self._bond[(i, p)] if p >= 0 else -1,
                tuple((s, len(kids[s])) for s in sorted(kids)),
            )
            shape[i] = ids.setdefault(key, len(ids))
            self._families.setdefault(p, {}).setdefault(int(shape[i]), []).append(i)
        self._shape = shape
        self._subtree_cache = {}

    def _subtree(self, i: int) -> List[int]:
        """The atoms of the tree on i, in an order in which trees of one shape agree."""
        if i not in self._subtree_cache:
            out = [i]
            kids = self._families.get(i, {})
            for s in sorted(kids):
                for child in kids[s]:
                    out += self._subtree(child)
            self._subtree_cache[i] = out
        return self._subtree_cache[i]

    def _rejected_transpositions(self) -> np.ndarray:
        """
        Every transposition of two neighbouring siblings of a family (with their trees)
        must be an automorphism for RDKit: all permutations of the family follow. The
        test forces the map with unique isotopes on both sides. Returns the atoms of
        the families that fail.
        """
        query = Chem.Mol(self.graph)
        for atom in query.GetAtoms():
            atom.SetIsotope(_FORCED + atom.GetIdx())
        target = Chem.Mol(query)
        rejected = []
        self.n_generators = 0
        for by_shape in self._families.values():
            for family in by_shape.values():
                for u, v in zip(family, family[1:]):
                    self.n_generators += 1
                    su, sv = self._subtree(u), self._subtree(v)
                    moved = dict(
                        zip(su + sv, sv + su)
                    )  # atom i of a with moved[i] of b
                    for i, j in moved.items():
                        target.GetAtomWithIdx(j).SetIsotope(_FORCED + i)
                    ok = target.HasSubstructMatch(
                        query, useChirality=True, useQueryQueryMatches=False
                    )
                    for j in moved:
                        target.GetAtomWithIdx(j).SetIsotope(_FORCED + j)
                    if not ok:
                        rejected += family
                        break
        return np.array(sorted(set(rejected)), dtype=int)

    # ------------------------------------------------------------ set-up: one level
    def _layout(self, level: int, max_pairs: int):
        """
        The tables of a level, or None if it needs more than max_pairs node pairs.

        fixed: the atoms whose image a skeleton map determines (the skeleton and the
        stripped atoms that have no sibling of their shape all the way up). The other
        stripped atoms are 'free'. A node pair (u, v) is a free atom u of a and a
        candidate image v of it in b when the skeleton map is the identity; family
        pairs collect the node pairs of two families that can be matched.
        """
        n = self.graph.GetNumAtoms()
        if level == 0:
            stripped = np.zeros(n, dtype=bool)
            families = {}
        else:
            stripped = (self._level > 0) & (self._level <= level)
            families = {
                p: {s: f for s, f in by_shape.items() if stripped[f[0]]}
                for p, by_shape in self._families.items()
            }
            families = {p: by_shape for p, by_shape in families.items() if by_shape}
        fixed = ~stripped
        for i in sorted(np.flatnonzero(stripped), key=lambda i: -self._level[i]):
            p = int(self._parent[i])
            alone = len(families[p][int(self._shape[i])]) == 1
            fixed[i] = alone and (p < 0 or fixed[p])
        pu, pv = [], []
        blocks = {}

        def add(fa, fb, parent_pair, depth):
            k = len(fa)
            child = np.empty((k, k), dtype=np.intp)
            for i, u in enumerate(fa):
                for j, v in enumerate(fb):
                    q = len(pu)
                    if q >= max_pairs:
                        raise OverflowError
                    pu.append(u)
                    pv.append(v)
                    child[i, j] = q
                    kids_u, kids_v = families.get(u, {}), families.get(v, {})
                    for s in sorted(kids_u):
                        add(kids_u[s], kids_v[s], q, depth + 1)
            blocks.setdefault((depth, k), []).append((child, parent_pair))

        try:
            for p in sorted(families):
                if p >= 0 and not fixed[p]:
                    continue
                for s in sorted(families[p]):
                    if len(families[p][s]) > 1:
                        add(families[p][s], families[p][s], -1, 1)
        except OverflowError:
            return None
        # the sets that the sibling permutations mix: the free atoms by their path of
        # shapes from the fixed atom (or from nothing) they hang on
        group_of = {}
        groups: List[List[int]] = []
        for i in sorted(
            np.flatnonzero(stripped & ~fixed), key=lambda i: -self._level[i]
        ):
            i = int(i)
            p = int(self._parent[i])
            above = ("fixed", p) if p < 0 or fixed[p] else ("free", group_of[p])
            key = (above, int(self._shape[i]))
            if key not in group_of:
                group_of[key] = len(groups)
                groups.append([])
            group_of[i] = group_of[key]
            groups[group_of[i]].append(i)
        order = 1
        for by_shape in families.values():
            for family in by_shape.values():
                order *= math.factorial(len(family))
        return {
            "fixed": np.flatnonzero(fixed),
            "pu": np.array(pu, dtype=np.intp),
            "pv": np.array(pv, dtype=np.intp),
            "blocks": blocks,
            "groups": groups,
            "order": order,
            "n_families": sum(len(b) for b in families.values()),
            "n_stripped": int(stripped.sum()),
        }

    def _points_of(self, x: np.ndarray, layout) -> np.ndarray:
        """The points of the lower bound for centred coordinates x (..., n, 3)."""
        fixed = [i for i in layout["fixed"] if self.weight[i]]
        parts = [x[..., fixed, :]]
        for members in layout["groups"]:
            members = [i for i in members if self.weight[i]]
            if members:
                parts.append(
                    x[..., members, :].sum(axis=-2, keepdims=True)
                    / math.sqrt(len(members))
                )
        return np.concatenate(parts, axis=-2)

    def _spread(self, layout, reference) -> float:
        """The smallest extent (A, per atom of the RMSD) of the points of the bound
        along their second principal axis, over the reference conformers."""
        if reference is None:
            raise ValueError(
                "More automorphisms than max_maps: choosing a level needs positions "
                "(a conformer of mol, or reference)."
            )
        points = self._points_of(reference, layout)
        if points.shape[-2] < 2:
            return 0.0
        s = np.linalg.svd(points, compute_uv=False)
        return float(s[..., 1].min() / math.sqrt(self.n))

    def _skeleton_maps(self, level: int, limit: int) -> np.ndarray:
        """
        The skeleton maps of a level, each continued onto the stripped atoms in the
        fixed way (child k of a family onto child k of the image family): the
        automorphisms of the graph in which every stripped atom carries its shape and
        its place in its family as a label. Enumerated by RDKit, at most limit. Level
        0: all automorphisms.
        """
        labelled = Chem.Mol(self.graph)
        place = {}
        if level:
            for by_shape in self._families.values():
                for family in by_shape.values():
                    for k, i in enumerate(family):
                        place[i] = k
        ids = {}
        for atom in labelled.GetAtoms():
            i = atom.GetIdx()
            if level and 0 < self._level[i] <= level:
                key = ("stripped", int(self._shape[i]), place[i])
            else:
                key = ("skeleton", atom.GetIsotope(), bool(self.weight[i]))
            atom.SetIsotope(1 + ids.setdefault(key, len(ids)))
        return _matches(labelled, limit)

    def _install(self, layout) -> None:
        """The arrays of the chosen level."""
        n = self.graph.GetNumAtoms()
        self.order = len(self.maps) * layout["order"]
        self.n_stripped = layout["n_stripped"]
        self.n_families = layout["n_families"]
        self._fixed = layout["fixed"]
        self._pu, self._pv = layout["pu"], layout["pv"]
        self.n_pairs = len(self._pu)
        self._free = self.n_pairs > 0
        self._blocks: List[_Block] = []
        for depth, k in sorted(layout["blocks"], reverse=True):  # deepest first
            entries = layout["blocks"][(depth, k)]
            child = np.array([c for c, _ in entries], dtype=np.intp).reshape(-1, k, k)
            parent = np.array([p for _, p in entries], dtype=np.intp)
            pairs = None
            if k <= _BRUTE:
                perms = np.array(list(itertools.permutations(range(k))), dtype=np.intp)
                pairs = child[:, np.arange(k), perms]  # (families, k!, k)
            self._blocks.append(
                _Block(
                    k,
                    pairs,
                    child,
                    None if depth == 1 else parent,
                    np.arange(len(child)),
                )
            )
        # the points of the bound: the weighted fixed atoms, then the weighted groups
        point_of = np.full(n, -1, dtype=np.intp)
        fixed = np.array([i for i in self._fixed if self.weight[i]], dtype=np.intp)
        point_of[fixed] = np.arange(len(fixed))
        members = [[i for i in g if self.weight[i]] for g in layout["groups"]]
        members = [g for g in members if g]
        for k, g in enumerate(members):
            point_of[g] = len(fixed) + k
        self._point_atoms = fixed
        self._group_atoms = np.array([i for g in members for i in g], dtype=np.intp)
        self._group_start = np.cumsum([0] + [len(g) for g in members[:-1]]).astype(
            np.intp
        )
        self._group_scale = np.array([1.0 / math.sqrt(len(g)) for g in members])[
            :, None
        ]
        first = np.concatenate([fixed, [g[0] for g in members]]).astype(np.intp)
        self._point_maps = point_of[self.maps[:, first]]
        if (self._point_maps < 0).any():
            raise RuntimeError("A skeleton map does not respect the point sets.")
        self.n_points = len(first)
        self._points_are_atoms = len(fixed) == n  # nothing free, nothing weightless

    # --------------------------------------------------------------------- per pair
    def prepare(self, positions, align: bool = True) -> Prepared:
        """A conformer ((n_atoms, 3) positions of the molecule) for rmsd and within."""
        if isinstance(positions, Prepared):
            if positions.align != align:
                raise ValueError("The conformer was prepared for the other align.")
            return positions
        x = np.asarray(positions, dtype=float)
        if x.ndim != 2 or x.shape[1] != 3 or len(x) <= self.index.max():
            raise ValueError("positions must be (n_atoms, 3) of the molecule.")
        x = x[self.index]
        if align:
            x = _centred(x, self.weight)
        elif not self._all:
            x = x * self.weight[:, None]
        norm = float((x * x).sum())
        if self._points_are_atoms:
            return Prepared(x, x, np.ascontiguousarray(x.T), norm, norm, align)
        points = x[self._point_atoms]
        if len(self._group_atoms):
            sums = np.add.reduceat(x[self._group_atoms], self._group_start, axis=0)
            points = np.concatenate([points, sums * self._group_scale])
        transposed = np.ascontiguousarray(points.T)
        return Prepared(
            x, points, transposed, norm, float((points * points).sum()), align
        )

    def _bounds(
        self, pa: Prepared, pb: Prepared
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Per skeleton map: a lower bound of the summed squared deviation of all its
        continuations (the exact value if no atom is free), and b^T a of the points."""
        self.stats["bounds"] += 1
        if len(self.maps) <= _CHUNK:
            return self._bounds_of(pa, pb, self._point_maps)
        parts = [
            self._bounds_of(pa, pb, self._point_maps[s : s + _CHUNK])
            for s in range(0, len(self.maps), _CHUNK)
        ]
        h = None if parts[0][1] is None else np.concatenate([p[1] for p in parts])
        return np.concatenate([p[0] for p in parts]), h

    @staticmethod
    def _bounds_of(pa, pb, point_maps):
        """One gather, one matrix product and one batched SVD for all maps:
        h[k] = (b in the order of map k)^T a."""
        q = np.take(pb.points_t, point_maps, axis=1)  # (3, maps, points)
        if not pa.align:
            delta = q - pa.points_t[:, None, :]
            return np.einsum("ike,ike->k", delta, delta), None
        k = len(point_maps)
        h = (q.reshape(3 * k, -1) @ pa.points).reshape(3, k, 3).transpose(1, 0, 2)
        s = np.linalg.svd(h, compute_uv=False)
        sign = np.where(np.linalg.det(h) < 0, -1.0, 1.0)  # proper rotations only
        trace = s[:, 0] + s[:, 1] + sign * s[:, 2]
        return np.maximum(pa.norm_points + pb.norm_points - 2.0 * trace, 0.0), h

    def _assign(self, xa: np.ndarray, y: np.ndarray) -> np.ndarray:
        """The node pairs of the best sibling permutations for a and the rotated b:
        the exact minimum over the sibling permutations of the summed squared
        deviation in this frame."""
        self.stats["assignments"] += 1
        delta = np.take(xa, self._pu, axis=0)
        delta -= np.take(y, self._pv, axis=0)
        cost = np.einsum("ij,ij->i", delta, delta)
        chosen = []
        for block in self._blocks:  # deepest first
            if block.k == 1:
                picked = block.child[:, :, 0]
                best = cost[picked[:, 0]]
            elif block.pairs is not None:
                sums = cost[block.pairs].sum(axis=2)
                choice = sums.argmin(axis=1)
                best = sums[block.count, choice]
                picked = block.pairs[block.count, choice]
            else:
                m = cost[block.child]
                choice = np.array([linear_sum_assignment(c)[1] for c in m])
                picked = np.take_along_axis(block.child, choice[:, :, None], axis=2)[
                    :, :, 0
                ]
                best = cost[picked].sum(axis=1)
            chosen.append(picked)
            if block.parent is not None:
                np.add.at(cost, block.parent, best)
        selected = np.zeros(len(cost), dtype=bool)
        for block, picked in zip(reversed(self._blocks), reversed(chosen)):
            if block.parent is not None:
                picked = picked[selected[block.parent]]
            selected[picked.ravel()] = True
        return np.flatnonzero(selected)

    def _evaluate(self, pa: Prepared, pb: Prepared, k: int, h, limit: float):
        """
        (summed squared deviation, node pairs) for skeleton map k: sibling permutations
        and rotation in turn until the permutations repeat, from up to three rotations
        (of the bound, h; of all atoms paired in the fixed way; of the fixed atoms).
        Stops as soon as a value is within limit.
        """
        self.stats["evaluations"] += 1
        xa = pa.x
        xb = pb.x[self.maps[k]] if k else pb.x
        fixed, pu, pv = self._fixed, self._pu, self._pv
        if not pa.align:  # one frame: one assignment is exact
            pairs = self._assign(xa, xb)
            d1 = xa[fixed] - xb[fixed]
            d2 = xa[pu[pairs]] - xb[pv[pairs]]
            return float((d1 * d1).sum() + (d2 * d2).sum()), pairs
        h_fixed = xb[fixed].T @ xa[fixed]
        best, best_pairs = np.inf, None
        seen = set()
        for start in range(self.starts):
            if start == 0:
                rot = _rotation(h)[0]
            elif start == 1:
                rot = _rotation(xb.T @ xa)[0]
            else:
                rot = _rotation(h_fixed)[0]
            for _ in range(self.max_rounds):
                pairs = self._assign(xa, xb @ rot)
                key = pairs.tobytes()
                if key in seen:
                    break
                seen.add(key)
                rot, trace = _rotation(
                    h_fixed
                    + np.take(xb, pv[pairs], axis=0).T @ np.take(xa, pu[pairs], axis=0)
                )
                ssd = max(pa.norm + pb.norm - 2.0 * trace, 0.0)
                if ssd < best:
                    best, best_pairs = ssd, pairs
                if best <= limit:
                    return best, best_pairs
        return best, best_pairs

    def _full_map(self, k: int, pairs: Optional[np.ndarray]) -> np.ndarray:
        tau = np.arange(len(self.index))
        if pairs is not None:
            tau[self._pu[pairs]] = self._pv[pairs]
        return self.maps[k][tau]

    def rmsd(
        self, positions_a, positions_b, align: bool = True, return_map: bool = False
    ):
        """
        The RMSD of the best automorphism found: exact if nothing is stripped (level
        0), else an upper bound of the symmetry-minimised RMSD (see the module text).
        With return_map also the map over the graph atoms (index, weight): atom
        index[i] of a is paired with atom index[map[i]] of b.
        """
        pa, pb = self.prepare(positions_a, align), self.prepare(positions_b, align)
        self.stats["pairs"] += 1
        bounds, h = self._bounds(pa, pb)
        if not self._free:
            k = int(np.argmin(bounds)) if len(bounds) > 1 else 0
            full = self.maps[k]
        else:
            best, k, pairs = np.inf, 0, None
            for j in np.argsort(bounds, kind="stable"):
                if bounds[j] > best + 1e-9:
                    break
                ssd, found = self._evaluate(
                    pa, pb, int(j), None if h is None else h[j], -1.0
                )
                if ssd < best:
                    best, k, pairs = ssd, int(j), found
            full = self._full_map(k, pairs)
        value = _explicit(pa.x, pb.x[full], self.n, align)
        return (value, full) if return_map else value

    def within(
        self, positions_a, positions_b, threshold: float, align: bool = True
    ) -> bool:
        """
        Whether rmsd(positions_a, positions_b) <= threshold, with early exits: the
        identity map first, then only the skeleton maps whose lower bound allows it.
        True is never wrong; False is wrong only where rmsd is above the true minimum.
        """
        pa, pb = self.prepare(positions_a, align), self.prepare(positions_b, align)
        self.stats["pairs"] += 1
        limit = threshold * threshold * self.n * (1.0 + 1e-12)
        if align:
            identity = pa.norm + pb.norm - 2.0 * _rotation(pb.x.T @ pa.x)[1]
        else:
            delta = pa.x - pb.x
            identity = float((delta * delta).sum())
        if identity <= limit:
            return True
        if not self._free and len(self.maps) == 1:
            return False
        bounds, h = self._bounds(pa, pb)
        if not self._free:
            return bool(bounds.min() <= limit)
        possible = np.flatnonzero(bounds <= limit)
        self.stats["evaluated"] += bool(len(possible))
        for j in possible[np.argsort(bounds[possible], kind="stable")]:
            ssd, _ = self._evaluate(pa, pb, int(j), None if h is None else h[j], limit)
            if ssd <= limit:
                return True
        return False

    def lower_bound(self, positions_a, positions_b, align: bool = True) -> float:
        """A rigorous lower bound of the symmetry-minimised RMSD (step 1 alone), unless
        the skeleton maps were truncated."""
        pa, pb = self.prepare(positions_a, align), self.prepare(positions_b, align)
        return float(np.sqrt(self._bounds(pa, pb)[0].min() / self.n))

    def describe(self) -> str:
        return (
            f"{self.n} atoms, level {self.level}/{self.max_level}: {len(self.maps)} maps"
            f"{' (truncated)' if self.truncated else ''}, {self.n_stripped} stripped atoms "
            f"in {self.n_families} families, {self.n_pairs} node pairs, "
            f"{self.order:.4g} automorphisms"
        )
