"""Ground states: nothing is kept fixed."""

from rdkit import Chem

from .base import FrozenSet


class GroundState:
    """A conformer search without fixed atoms; no reference geometry is needed."""

    needs_reference = False

    def frozen_atoms(self, mol: Chem.Mol) -> FrozenSet:
        return FrozenSet()

    def __repr__(self) -> str:
        return "GroundState()"
