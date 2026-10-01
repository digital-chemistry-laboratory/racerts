"""Conformer searches with a chosen set of atoms kept at the reference geometry."""

from typing import Mapping, Optional, Sequence

from rdkit import Chem

from .base import FrozenSet


class Constrained:
    """
    Keeps the hard atoms at the reference geometry, e.g. a binding motif or a core
    taken from another structure. In the bounds-matrix embedder, all distances between
    the hard atoms are fixed (between the core atoms and the hard atoms, if a core is
    given). Soft atoms start at the reference coordinates and are held near them by
    position restraints in refinement (see FrozenSet).
    """

    needs_reference = True

    def __init__(
        self,
        hard: Sequence[int] = (),
        soft: Sequence[int] = (),
        core: Optional[Sequence[int]] = None,
    ):
        if not hard and not soft:
            raise ValueError("Constrained needs at least one hard or soft atom.")
        for name, atoms in (("hard", hard), ("soft", soft)):
            if len(set(atoms)) != len(atoms):
                raise ValueError(f"Repeated atoms in {name}: {list(atoms)}.")
        self.hard = tuple(int(i) for i in hard)
        self.soft = tuple(int(i) for i in soft)
        self.core = None if core is None else tuple(int(i) for i in core)
        FrozenSet(hard=self.hard, soft=self.soft)  # checks the overlap

    def frozen_atoms(self, mol: Chem.Mol) -> FrozenSet:
        return FrozenSet(hard=self.hard, core=self.core, soft=self.soft)

    def remap(self, index_map: Mapping[int, int]) -> "Constrained":
        """The same task for a molecule whose atom i is index_map[i] (e.g. a swap)."""
        return Constrained(
            _remap(index_map, self.hard, "hard atoms"),
            _remap(index_map, self.soft, "soft atoms"),
            None if self.core is None else _remap(index_map, self.core, "core atoms"),
        )

    def __repr__(self) -> str:
        parts = [f"hard={list(self.hard)}"]
        if self.soft:
            parts.append(f"soft={list(self.soft)}")
        if self.core is not None:
            parts.append(f"core={list(self.core)}")
        return f"Constrained({', '.join(parts)})"


def _remap(index_map: Mapping[int, int], atoms: Sequence[int], what: str) -> list:
    missing = [i for i in atoms if i not in index_map]
    if missing:
        raise ValueError(f"The {what} {missing} are not in the new molecule.")
    return [index_map[i] for i in atoms]
