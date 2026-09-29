"""What a conformer search keeps fixed: the Task protocol and FrozenSet."""

from dataclasses import dataclass
from typing import Optional, Protocol, Tuple, runtime_checkable

from rdkit import Chem


@dataclass(frozen=True)
class FrozenSet:
    """
    Atoms kept at their reference positions.

    Attributes:
        hard: Atoms fixed at the reference coordinates, in embedding (coordinate map)
            and refinement (fixed anchor points). The order is kept: it enters the
            force-field sums and the alignment.
        core: Atoms whose distances to all hard atoms are fixed in the bounds-matrix
            embedder (for a TS the reacting atoms); default: hard.
    """

    hard: Tuple[int, ...] = ()
    core: Optional[Tuple[int, ...]] = None

    def __post_init__(self):
        if self.core is None:
            object.__setattr__(self, "core", self.hard)

    def __bool__(self) -> bool:
        return bool(self.hard)


@runtime_checkable
class Task(Protocol):
    """
    What is kept fixed. frozen_atoms gets the molecular graph (with the reference
    geometry as its conformer, if the task needs one) and returns the FrozenSet.
    """

    needs_reference: bool

    def frozen_atoms(self, mol: Chem.Mol) -> FrozenSet: ...


def check_atom_indices(mol: Chem.Mol, atoms, what: str) -> None:
    """Raise ValueError if any atom index is not an atom of mol."""
    invalid = sorted(set(atoms) - set(range(mol.GetNumAtoms())))
    if invalid:
        raise ValueError(
            f"Invalid {what}: {invalid} (the molecule has {mol.GetNumAtoms()} atoms)."
        )
