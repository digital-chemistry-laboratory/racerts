"""Transition states: the reacting atoms and their neighbours are kept fixed."""

import logging
from typing import List, Optional, Sequence

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

    def frozen_atoms(self, mol: Chem.Mol) -> FrozenSet:
        check_atom_indices(mol, self.reacting_atoms, "reacting atoms")
        frozen = get_frozen_atoms(mol, self.reacting_atoms, self.user_frozen_atoms)
        return FrozenSet(hard=tuple(frozen), core=tuple(self.reacting_atoms))

    def __repr__(self) -> str:
        return f"TransitionState(reacting_atoms={self.reacting_atoms})"


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
