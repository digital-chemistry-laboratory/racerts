"""Conformer searches with a chosen set of atoms kept at the reference geometry."""

from typing import Sequence

from rdkit import Chem

from .base import FrozenSet


class Constrained:
    """
    Keeps the hard atoms at the reference geometry, e.g. a binding motif or a core
    taken from another structure. In the bounds-matrix embedder, all distances between
    the hard atoms are fixed.
    """

    needs_reference = True

    def __init__(self, hard: Sequence[int]):
        if not hard:
            raise ValueError("Constrained needs at least one hard atom.")
        if len(set(hard)) != len(hard):
            raise ValueError(f"Repeated atoms in hard: {list(hard)}.")
        self.hard = tuple(hard)

    def frozen_atoms(self, mol: Chem.Mol) -> FrozenSet:
        return FrozenSet(hard=self.hard)

    def __repr__(self) -> str:
        return f"Constrained(hard={list(self.hard)})"
