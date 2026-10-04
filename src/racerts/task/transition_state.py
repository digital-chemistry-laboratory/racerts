"""Transition states: the reacting atoms and their neighbours are kept fixed."""

import logging
from numbers import Real
from typing import Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
from rdkit import Chem

from .base import FrozenSet, check_atom_indices

logger = logging.getLogger(__name__)


Window = Union[None, float, Tuple[float, float]]

# Active bonds by default: reacting pairs not bonded in the graph and closer than this
# factor times the sum of their covalent radii (the forming bonds of the TS).
FORMING_BOND_FACTOR = 1.6
# 1,3-pairs at this angle (degrees) or wider are not forming bonds by default.
MIN_OPEN_ANGLE = 80.0
MIN_WINDOW_FACTOR = 0.9  # of the covalent bond length: lower window starts are warned


class TransitionState:
    """
    A TS conformer search, as in legacy racerts.

    The reacting atoms (atoms that change connectivity during the reaction) and all
    their bonded neighbours are kept at the TS geometry, unless frozen_atoms are given.
    In the bounds-matrix embedder, the distances between the reacting atoms and the
    frozen atoms are fixed.

    Active-bond windows (active_window): the lengths of the active bonds are sampled
    in a window instead of kept at the seed. Then only the neighbours of the reacting
    atoms that are not reacting are held at the seed geometry; the reacting atoms are
    placed by distance windows: [lo, hi] for the active bonds, the seed distance +/-
    neighbor_window to their bonded neighbours (both in embedding and in MMFF/UFF
    refinement, with force constant window_force_constant).

    Args:
        active_bonds: The pairs whose length is sampled; default: the bonds that form
            or break (bond_changes, from from_endpoints), else the reacting pairs that
            the graph does not bond and that are closer than 1.6 times the sum of
            their covalent radii in the seed, except 1,3-pairs at an angle of 80
            degrees or more (e.g. ring atoms; a narrow angle is a three-membered TS).
        active_window: None (legacy: the seed length); a number d (the seed length +/-
            d); or (lo, hi) in A, for every active bond.
        neighbor_window: +/- A for the distances of the reacting atoms to their bonded
            neighbours, in window mode.
        stratify: 0: the embedding places the lengths in the window (not evenly:
            distance geometry puts most at its lower edge); k >= 2: k target lengths
            evenly spaced in the window, one per embedding batch. Either way,
            refinement holds each conformer at its target (+/- 0.02 A, with
            target_force_constant; a flat-bottom window would let the force field push
            all conformers to one edge). The provenance records the targets
            ("active_bond_targets") and the lengths ("active_bond_lengths").
        stereo_filter: In window mode, drop conformers whose reacting atoms are
            attacked from the other face than in the seed (see AttackFace).
    """

    needs_reference = True

    def __init__(
        self,
        reacting_atoms: Sequence[int],
        frozen_atoms: Optional[Sequence[int]] = None,
        active_bonds: Optional[Sequence[Sequence[int]]] = None,
        active_window: Window = None,
        neighbor_window: float = 0.10,
        stratify: int = 0,
        stereo_filter: bool = True,
        window_force_constant: float = 10000.0,
        target_force_constant: float = 10000.0,
    ):
        self.reacting_atoms = [int(i) for i in reacting_atoms]
        self.user_frozen_atoms = (
            [] if frozen_atoms is None else [int(i) for i in frozen_atoms]
        )
        # The bonds that form or break, if known (from_endpoints).
        self.bond_changes: Optional[List[Tuple[int, int]]] = None
        self.active_bonds = (
            None
            if active_bonds is None
            else [tuple(sorted(map(int, b))) for b in active_bonds]
        )
        if active_window is not None:
            if isinstance(active_window, Real):
                if active_window <= 0:
                    raise ValueError("active_window must be positive.")
            elif len(active_window) != 2 or not 0 < active_window[0] < active_window[1]:
                raise ValueError(
                    "active_window must be a number or (lo, hi) with 0 < lo < hi."
                )
        if stratify < 0 or (stratify and active_window is None):
            raise ValueError(
                "stratify needs an active_window and must not be negative."
            )
        if stratify == 1:
            raise ValueError(
                "stratify: use 0 (anywhere in the window) or at least 2 targets."
            )
        if active_window is not None and self.user_frozen_atoms:
            raise ValueError("Give either frozen_atoms or an active_window.")
        self.active_window = active_window
        self.neighbor_window = neighbor_window
        self.stratify = int(stratify)
        self.stereo_filter = stereo_filter
        self.window_force_constant = window_force_constant
        self.target_force_constant = target_force_constant

    @property
    def windowed(self) -> bool:
        return self.active_window is not None

    def frozen_atoms(self, mol: Chem.Mol) -> FrozenSet:
        check_atom_indices(mol, self.reacting_atoms, "reacting atoms")
        if self.windowed:
            reacting = set(self.reacting_atoms)
            held = reacting_neighbourhood(mol, self.reacting_atoms)
            hard = tuple(i for i in held if i not in reacting)
            # core: the reacting atoms, as in legacy mode (fragments, stereo, the
            # bounds-matrix pairs, which the windows then replace)
            return FrozenSet(hard=hard, core=tuple(self.reacting_atoms))
        frozen = get_frozen_atoms(mol, self.reacting_atoms, self.user_frozen_atoms)
        return FrozenSet(hard=tuple(frozen), core=tuple(self.reacting_atoms))

    def active_pairs(self, mol: Chem.Mol) -> List[Tuple[int, int]]:
        """The active bonds (see active_bonds)."""
        if self.active_bonds is not None:
            check_atom_indices(
                mol, [i for pair in self.active_bonds for i in pair], "active bonds"
            )
            outside = [
                p for p in self.active_bonds if not set(p) <= set(self.reacting_atoms)
            ]
            if outside or any(a == b for a, b in self.active_bonds):
                raise ValueError(
                    f"Active bonds must join two different reacting atoms, not {outside or self.active_bonds}."
                )
            return list(self.active_bonds)
        if self.bond_changes:
            return list(self.bond_changes)
        positions = mol.GetConformer().GetPositions()
        table = Chem.GetPeriodicTable()
        reacting = sorted(self.reacting_atoms)
        pairs = []
        neighbors = {
            i: {n.GetIdx() for n in mol.GetAtomWithIdx(i).GetNeighbors()}
            for i in reacting
        }
        for k, a in enumerate(reacting):
            for b in reacting[k + 1 :]:
                # Bonded pairs are not forming bonds, nor 1,3-pairs at an open angle
                # (e.g. ring atoms 2.4 A apart at 120 degrees), whose distance the angle
                # sets; a 1,3-pair at a narrow angle closes a three-membered TS (a
                # reductive elimination, a 1,2-shift, an epoxide).
                if b in neighbors[a] or any(
                    _angle(positions, a, n, b) >= MIN_OPEN_ANGLE
                    for n in neighbors[a] & neighbors[b]
                ):
                    continue
                radii = sum(
                    table.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum())
                    for i in (a, b)
                )
                if (
                    np.linalg.norm(positions[a] - positions[b])
                    < FORMING_BOND_FACTOR * radii
                ):
                    pairs.append((a, b))
        return pairs

    def active_windows(
        self, mol: Chem.Mol
    ) -> Dict[Tuple[int, int], Tuple[float, float]]:
        """The window of each active bond (in window mode)."""
        positions = mol.GetConformer().GetPositions()
        windows = {}
        for a, b in self.active_pairs(mol):
            if isinstance(self.active_window, Real):
                d = float(np.linalg.norm(positions[a] - positions[b]))
                windows[(a, b)] = (
                    max(0.0, d - self.active_window),
                    d + self.active_window,
                )
            else:
                windows[(a, b)] = tuple(map(float, self.active_window))
        return windows

    def targets(self, mol: Chem.Mol) -> List[Dict[Tuple[int, int], float]]:
        """The target lengths of each batch (stratify > 0): k evenly spaced per bond."""
        windows = self.active_windows(mol)
        return [
            {
                pair: lo + (hi - lo) * i / (self.stratify - 1)
                for pair, (lo, hi) in windows.items()
            }
            for i in range(self.stratify)
        ]

    def restraints(self, mol: Chem.Mol):
        """
        In window mode, the windows of the active bonds and of the reacting atoms to
        their neighbours (a RestraintSet); else none.
        """
        from racerts.restraints import DistanceRestraint, RestraintSet

        if not self.windowed:
            return RestraintSet()
        windows = self.active_windows(mol)
        if not windows:
            raise ValueError(
                "No active bonds: give active_bonds (the pairs whose length is sampled)."
            )
        positions = mol.GetConformer().GetPositions()
        _warn_low_windows(mol, windows)
        k = self.window_force_constant
        restraints = RestraintSet(
            DistanceRestraint(a, b, lo, hi, force_constant=k, source="active")
            for (a, b), (lo, hi) in windows.items()
        )
        taken = set(windows)
        for r in self.reacting_atoms:
            for neighbor in mol.GetAtomWithIdx(r).GetNeighbors():
                pair = tuple(sorted((r, neighbor.GetIdx())))
                if pair in taken:
                    continue
                restraints.add(
                    DistanceRestraint.around(
                        *pair,
                        _length(positions, pair),
                        self.neighbor_window,
                        force_constant=k,
                        source="neighbor",
                    )
                )
                taken.add(pair)
        # The other distances of the reacting atoms within the core, e.g. O...O of a
        # proton transfer at 2.5 A, contradict RDKit's default (vdW) bounds, which the
        # coordinate map overrides in legacy mode. In window mode they get an
        # embedding-only window around the seed distance, as wide as the active bonds
        # and the neighbour windows let them change.
        spread = 2 * self.neighbor_window + max(
            max(abs(lo - _length(positions, pair)), abs(hi - _length(positions, pair)))
            for pair, (lo, hi) in windows.items()
        )
        for a in self.reacting_atoms:
            for b in sorted(reacting_neighbourhood(mol, self.reacting_atoms)):
                pair = tuple(sorted((a, b)))
                if a == b or pair in taken:
                    continue
                restraints.add(
                    DistanceRestraint.around(
                        *pair,
                        _length(positions, pair),
                        spread,
                        stage="embed",
                        source="core",
                    )  # fmt: skip
                )
                taken.add(pair)
        return restraints

    def active_lengths(
        self, mol: Chem.Mol, conf_id: int, pairs: Sequence[Tuple[int, int]]
    ) -> Dict[str, float]:
        """The lengths of the active bonds pairs (see active_pairs of the reference)
        in a conformer ("a-b": A)."""
        positions = mol.GetConformer(conf_id).GetPositions()
        return {f"{a}-{b}": round(_length(positions, (a, b)), 4) for a, b in pairs}

    @classmethod
    def from_endpoints(
        cls,
        reactant: Chem.Mol,
        product: Chem.Mol,
        frozen_atoms: Optional[Sequence[int]] = None,
        **settings,
    ) -> "TransitionState":
        """
        The TS between two atom-aligned endpoints (atom i is the same atom in both):
        the reacting atoms are the atoms of the bonds that form or break. A bond
        whose order changes but that stays is not counted. The bonds are kept as
        bond_changes, the default active bonds of a window (settings: the further
        arguments of TransitionState, e.g. active_window).
        """
        changes = formed_or_broken_bonds(reactant, product)
        if not changes:
            raise ValueError(
                "The endpoints have the same bonds: no bond forms or breaks."
            )
        reacting = sorted({atom for bond in changes for atom in bond})
        task = cls(reacting, frozen_atoms, **settings)
        task.bond_changes = changes
        return task

    def remap(self, index_map: Mapping[int, int]) -> "TransitionState":
        """
        The same task for a molecule whose atom i is index_map[i] (e.g. after a swap);
        every atom of the task must be in index_map.
        """

        def mapped(atoms, what):
            missing = [i for i in atoms if i not in index_map]
            if missing:
                raise ValueError(f"The {what} {missing} are not in the new molecule.")
            return [int(index_map[i]) for i in atoms]

        def pairs(bonds, what):
            if bonds is None:
                return None
            return [tuple(mapped(bond, what)) for bond in bonds]

        task = TransitionState(
            mapped(self.reacting_atoms, "reacting atoms"),
            mapped(self.user_frozen_atoms, "frozen atoms") or None,
            active_bonds=pairs(self.active_bonds, "active bond atoms"),
            active_window=self.active_window,
            neighbor_window=self.neighbor_window,
            stratify=self.stratify,
            stereo_filter=self.stereo_filter,
            window_force_constant=self.window_force_constant,
            target_force_constant=self.target_force_constant,
        )
        task.bond_changes = pairs(self.bond_changes, "bond change atoms")
        if task.bond_changes is not None:
            task.bond_changes = [tuple(sorted(b)) for b in task.bond_changes]
        return task

    def __repr__(self) -> str:
        window = "" if not self.windowed else f", active_window={self.active_window}"
        return f"TransitionState(reacting_atoms={self.reacting_atoms}{window})"


def _warn_low_windows(mol, windows) -> None:
    """Warn about windows that reach below 0.9 times the covalent bond length
    (distance geometry and the force field would place the atoms there)."""
    table = Chem.GetPeriodicTable()
    for (a, b), (lo, _) in windows.items():
        bond = sum(
            table.GetRcovalent(mol.GetAtomWithIdx(i).GetAtomicNum()) for i in (a, b)
        )
        if lo < MIN_WINDOW_FACTOR * bond:
            logger.warning(
                "The window of active bond %d-%d starts at %.2f A, below %.1f times its "
                "covalent length (%.2f A).",
                a,
                b,
                lo,
                MIN_WINDOW_FACTOR,
                bond,
            )


def _angle(positions, a, center, b) -> float:
    u, v = positions[a] - positions[center], positions[b] - positions[center]
    cosine = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v))
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _length(positions, pair) -> float:
    return float(np.linalg.norm(positions[pair[0]] - positions[pair[1]]))


def formed_or_broken_bonds(
    reactant: Chem.Mol, product: Chem.Mol
) -> List[Tuple[int, int]]:
    """
    The bonds (i, j), i < j, present in one of two atom-aligned molecules but not in
    the other, sorted.
    """
    if reactant.GetNumAtoms() != product.GetNumAtoms() or any(
        a.GetAtomicNum() != b.GetAtomicNum()
        for a, b in zip(reactant.GetAtoms(), product.GetAtoms())
    ):
        raise ValueError(
            "The endpoints must have the same atoms in the same order "
            f"({reactant.GetNumAtoms()} and {product.GetNumAtoms()} atoms)."
        )

    def bonds(mol):
        return {
            tuple(sorted((b.GetBeginAtomIdx(), b.GetEndAtomIdx())))
            for b in mol.GetBonds()
        }

    return sorted(bonds(reactant) ^ bonds(product))


def reacting_neighbourhood(mol: Chem.Mol, reacting_atoms: Sequence[int]) -> List[int]:
    """
    The reacting atoms and their bonded neighbours, in the order of legacy racerts (the
    neighbours of each reacting atom, then the atom), which enters the force-field sums.
    """
    atoms: List[int] = []
    for atom_idx in reacting_atoms:
        for neighbor in mol.GetAtomWithIdx(atom_idx).GetNeighbors():
            if neighbor.GetIdx() not in atoms:
                atoms.append(neighbor.GetIdx())
        if atom_idx not in atoms:
            atoms.append(atom_idx)
    return atoms


def get_frozen_atoms(
    mol_ts: Chem.Mol, reacting_atoms: List, frozen_atoms: List = [], verbose=False
):
    """
    The atoms to keep fixed, as in legacy racerts: frozen_atoms if given, else the
    reacting atoms (atoms that change connectivity in the reaction) and their bonded
    neighbours. verbose is not used; the choice is logged at INFO level.
    """
    frozen_atoms_new = reacting_neighbourhood(mol_ts, reacting_atoms)
    if frozen_atoms is None or len(frozen_atoms) == 0:
        logger.info(
            "No frozen atoms are given by the user. The following frozen atoms are "
            "considered: %s",
            frozen_atoms_new,
        )
        frozen_atoms = frozen_atoms_new
    else:
        logger.info(
            "Following frozen atoms are given by the user: %s. Detected frozen atoms "
            "(not further used) would have been: %s",
            frozen_atoms,
            frozen_atoms_new,
        )

    return frozen_atoms
