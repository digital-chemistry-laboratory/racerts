"""Context: what every stage of a pipeline can use."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from functools import cached_property
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
        task: What is kept fixed. Stages ask it for what depends on the kind of
            search (e.g. Embed for the default embedder of the task).
        seed: The random seed; embedders that stages create use it as their RDKit
            seed (-1: random).
    """

    mol: Chem.Mol
    task: Task
    seed: int = 12

    @cached_property
    def frozen(self) -> FrozenSet:
        """
        What the task freezes in mol: task.frozen_atoms(mol).
        """
        frozen = self.task.frozen_atoms(self.mol)
        check_atom_indices(self.mol, frozen.hard + frozen.core, "frozen atoms")
        return frozen

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
        ctx = cls(mol=mol, task=task, seed=seed)
        _ = ctx.frozen  # asked and checked here: a wrong task fails before any stage
        return ctx
