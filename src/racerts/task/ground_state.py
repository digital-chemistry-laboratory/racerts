"""Ground states: nothing is kept fixed."""

from typing import Optional

from rdkit import Chem

from racerts.system.graph import radical_multiplicity

from .base import FrozenSet


class GroundState:
    """A conformer search without fixed atoms; no reference geometry is needed."""

    needs_reference = False

    def frozen_atoms(self, mol: Chem.Mol) -> FrozenSet:
        return FrozenSet()

    def multiplicity(self, mol: Chem.Mol) -> Optional[int]:
        """
        The multiplicity where none is given: 1 + the radical electrons of the graph,
        if it has any (e.g. "[CH2]", a triplet carbene); else None, the lowest one.
        """
        found = radical_multiplicity(mol)
        return found if found > 1 else None

    def remap(self, index_map) -> "GroundState":
        return GroundState()

    def __repr__(self) -> str:
        return "GroundState()"
