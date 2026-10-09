"""Context: what every stage of a pipeline can use."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Optional

from rdkit import Chem

from racerts.system.spec import set_charge_and_multiplicity
from racerts.task import FrozenSet, Task
from racerts.task.base import check_atom_indices


@dataclass
class Context:
    """
    Attributes:
        mol: The molecular graph with its charge and multiplicity as properties; its
            conformer (if any) is the reference geometry.
        task: What is kept fixed.
        frozen: The FrozenSet of the task for mol.
        seed: The random seed; embedders that stages create use it as their RDKit
            seed (-1: random).
    """

    mol: Chem.Mol
    task: Task
    frozen: FrozenSet
    seed: int = 12

    @property
    def reference(self) -> Optional[Chem.Mol]:
        """The molecule with the reference geometry, if there is one."""
        return self.mol if self.mol.GetNumConformers() > 0 else None

    def graph(self) -> Chem.Mol:
        """A copy of the molecular graph without conformers, e.g. to embed into."""
        mol = deepcopy(self.mol)
        mol.RemoveAllConformers()
        return mol

    @classmethod
    def create(
        cls,
        mol: Chem.Mol,
        task: Task,
        seed: int = 12,
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
    ) -> Context:
        """
        A context for a copy of mol: charge and multiplicity are settled and stored as
        properties of the copy (see set_charge_and_multiplicity), then the frozen atoms
        of the task are determined and checked.
        """
        if not isinstance(mol, Chem.Mol):
            raise TypeError(f"Expected an RDKit Mol, not {type(mol).__name__}.")
        if task.needs_reference and mol.GetNumConformers() == 0:
            raise ValueError(
                f"{task!r} needs a reference geometry (a conformer of mol)."
            )
        mol = Chem.Mol(mol)
        set_charge_and_multiplicity(mol, charge, multiplicity)
        frozen = task.frozen_atoms(mol)
        check_atom_indices(mol, frozen.hard + frozen.core, "frozen atoms")
        return cls(mol=mol, task=task, frozen=frozen, seed=seed)
