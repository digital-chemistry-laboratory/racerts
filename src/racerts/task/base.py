"""What a conformer search keeps fixed: the Task protocol and FrozenSet."""

from dataclasses import dataclass
from typing import Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

from rdkit import Chem


@dataclass(frozen=True)
class FrozenSet:
    """
    Atoms kept at their reference positions.

    Attributes:
        hard: Atoms fixed at the reference coordinates, in embedding (coordinate map)
            and refinement (fixed anchor points). The order is kept: it enters the
            force-field sums and the alignment.
        core: The atoms that define the task (for a TS the reacting atoms); default:
            hard. The bounds-matrix embedder fixes their distances to all hard atoms;
            the stereo and connectivity checks exempt them; fragments without a core
            atom count as free (count policy "fragments", keep_fragments); and
            ReactionCore compares their distances with the reference.
        soft: Atoms placed at the reference coordinates in embedding (coordinate map,
            like hard atoms) and held near them in MMFF/UFF refinement by a flat-bottom
            position restraint (see racerts.restraints.PositionRestraint), e.g. the
            kept atoms of a swap.
    """

    hard: Tuple[int, ...] = ()
    core: Optional[Tuple[int, ...]] = None
    soft: Tuple[int, ...] = ()

    def __post_init__(self):
        object.__setattr__(self, "hard", tuple(self.hard))
        if self.core is None:
            object.__setattr__(self, "core", self.hard)
        object.__setattr__(self, "core", tuple(self.core))
        object.__setattr__(self, "soft", tuple(self.soft))
        both = sorted(set(self.hard) & set(self.soft))
        if both:
            raise ValueError(f"Atoms {both} are both hard and soft.")

    def __bool__(self) -> bool:
        return bool(self.hard or self.soft)


@runtime_checkable
class Task(Protocol):
    """
    What is kept fixed. frozen_atoms gets the molecular graph (with the reference
    geometry as its conformer, if the task needs one) and returns the FrozenSet.

    A task may also have: remap(index_map), the task for new atom indices (needed by
    racerts.swap); restraints(mol), distance windows of its own; and, for sampled
    active bonds as in TransitionState, windowed, stratify, stereo_filter, targets,
    active_pairs, active_windows, active_lengths and target_force_constant.
    """

    needs_reference: bool

    def frozen_atoms(self, mol: Chem.Mol) -> FrozenSet: ...


def remapped(index_map: Mapping[int, int], atoms: Sequence[int], what: str) -> list:
    """The atoms in the indices of a new molecule (index_map: old -> new)."""
    missing = [i for i in atoms if i not in index_map]
    if missing:
        raise ValueError(f"The {what} {missing} are not in the new molecule.")
    return [int(index_map[i]) for i in atoms]


def check_atom_indices(mol: Chem.Mol, atoms, what: str) -> None:
    """Raise ValueError if any atom index is not an atom of mol."""
    invalid = sorted(set(atoms) - set(range(mol.GetNumAtoms())))
    if invalid:
        raise ValueError(
            f"Invalid {what}: {invalid} (the molecule has {mol.GetNumAtoms()} atoms)."
        )
