"""Transition states: the reacting atoms and their neighbours are kept fixed."""

import logging
from typing import List, Optional, Sequence, Tuple

from rdkit import Chem

from .base import FrozenSet, check_atom_indices

logger = logging.getLogger(__name__)


class TransitionState:
    """
    A TS conformer search, as in legacy racerts.

    The reacting atoms (atoms that change connectivity during the reaction) and all
    their bonded neighbours are kept at the TS geometry, unless frozen_atoms are given.
    In the bounds-matrix embedder, the distances between the reacting atoms and the
    frozen atoms are fixed.
    """

    needs_reference = True

    def __init__(
        self,
        reacting_atoms: Sequence[int],
        frozen_atoms: Optional[Sequence[int]] = None,
    ):
        self.reacting_atoms = list(reacting_atoms)
        self.user_frozen_atoms = list(frozen_atoms) if frozen_atoms else []
        # The bonds that form or break, if known (from_endpoints).
        self.bond_changes: Optional[List[Tuple[int, int]]] = None

    def frozen_atoms(self, mol: Chem.Mol) -> FrozenSet:
        check_atom_indices(mol, self.reacting_atoms, "reacting atoms")
        frozen = get_frozen_atoms(mol, self.reacting_atoms, self.user_frozen_atoms)
        return FrozenSet(hard=tuple(frozen), core=tuple(self.reacting_atoms))

    @classmethod
    def from_endpoints(
        cls,
        reactant: Chem.Mol,
        product: Chem.Mol,
        frozen_atoms: Optional[Sequence[int]] = None,
    ) -> "TransitionState":
        """
        The TS between two atom-aligned endpoints (atom i is the same atom in both):
        the reacting atoms are the atoms of the bonds that form or break. A bond
        whose order changes but that stays is not counted. The bonds are kept as
        bond_changes.
        """
        changes = formed_or_broken_bonds(reactant, product)
        if not changes:
            raise ValueError(
                "The endpoints have the same bonds: no bond forms or breaks."
            )
        task = cls(sorted({atom for bond in changes for atom in bond}), frozen_atoms)
        task.bond_changes = changes
        return task

    def __repr__(self) -> str:
        return f"TransitionState(reacting_atoms={self.reacting_atoms})"


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


def get_frozen_atoms(
    mol_ts: Chem.Mol, reacting_atoms: List, frozen_atoms: List = [], verbose=False
):
    """
    Retrieve the atoms to be fixed from the molecular graph and the reacting atoms.

    Args:
        mol_ts (Chem.Mol): RDKit mol object.
        reacting_atoms (List): atom indeces of reacting atoms (atoms that change connectivity during reaction)
        (Optional) frozen_atoms (List): atom indeces of atoms to be fixed, if these should not be inferred from the graph.

    Returns:
        List: atom indeces of atoms to be fixed
    """
    frozen_atoms_new = []
    for atom_idx in reacting_atoms:
        for neighbor in mol_ts.GetAtomWithIdx(atom_idx).GetNeighbors():
            id = neighbor.GetIdx()
            if id not in frozen_atoms_new:
                frozen_atoms_new.append(id)
        if atom_idx not in frozen_atoms_new:
            frozen_atoms_new.append(atom_idx)
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
