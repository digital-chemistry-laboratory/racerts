"""Context: what every stage of a pipeline can use."""

import logging
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from rdkit import Chem

from racerts.restraints.model import RestraintSet
from racerts.system.spec import set_charge_and_multiplicity
from racerts.task import FrozenSet, Task
from racerts.task.base import check_atom_indices

logger = logging.getLogger(__name__)


@dataclass
class Context:
    """
    Attributes:
        mol: The molecular graph with its charge and multiplicity as properties; its
            conformer (if any) is the reference geometry. With several conformers
            (several references, see Embed(references=...)), the first is the
            reference unless a conformer's provenance names another.
        task: What is kept fixed.
        frozen: The FrozenSet of the task for mol.
        seed: The random seed; embedders that stages create use it as their RDKit
            seed (-1: random).
        restraints: Distance windows for embedding and force-field refinement (see
            racerts.restraints).
    """

    mol: Chem.Mol
    task: Task
    frozen: FrozenSet
    seed: int = 12
    restraints: RestraintSet = field(default_factory=RestraintSet)

    @property
    def reference(self) -> Optional[Chem.Mol]:
        """The molecule with the reference geometry, if there is one."""
        return self.mol if self.mol.GetNumConformers() > 0 else None

    def reference_mol(self, conf_id: int) -> Chem.Mol:
        """The molecule with only its conformer conf_id, as a reference geometry."""
        mol = Chem.Mol(self.mol)
        conf = Chem.Conformer(self.mol.GetConformer(conf_id))
        mol.RemoveAllConformers()
        mol.AddConformer(conf)
        return mol

    def by_reference(self, ensemble) -> List[Tuple[Optional[Chem.Mol], List[int]]]:
        """
        (reference, conformer ids) pairs: the conformers embedded from each reference
        (provenance "reference", see Embed(references=...)), or all conformers with
        the reference.
        """
        conf_ids = ensemble.conf_ids
        if self.mol.GetNumConformers() <= 1:
            return [(self.reference, conf_ids)]
        groups = {}
        for conf_id in conf_ids:
            reference = ensemble.provenance(conf_id).get("reference")
            groups.setdefault(reference, []).append(conf_id)
        if list(groups) == [None]:
            return [(self.reference, conf_ids)]
        return [
            (self.reference if ref is None else self.reference_mol(ref), ids)
            for ref, ids in groups.items()
        ]

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
        restraints: Optional[RestraintSet] = None,
    ) -> "Context":
        """
        A context for a copy of mol: charge and multiplicity are settled and stored as
        properties of the copy (see set_charge_and_multiplicity), then the frozen atoms
        of the task are determined and checked. Restraints between two frozen atoms
        are left out (their distance is fixed), with a warning.
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
        check_atom_indices(mol, frozen.hard + frozen.core + frozen.soft, "frozen atoms")
        restraints = RestraintSet(restraints or ())
        restraints.check_atoms(mol.GetNumAtoms())
        if hasattr(task, "restraints"):  # e.g. active-bond windows of a TS
            restraints = _with_task_windows(mol, task, restraints)
        kept = restraints.without_pairs_within(frozen.hard)
        if len(kept) < len(restraints):
            dropped = sorted({r.pair for r in restraints} - {r.pair for r in kept})
            logger.warning(
                "Restraints %s are ignored: both atoms are frozen at the reference.",
                dropped,
            )
        return cls(mol=mol, task=task, frozen=frozen, seed=seed, restraints=kept)


def _with_task_windows(mol, task, restraints):
    """
    The task's windows (active bonds of a TS) with the other restraints: a user
    restraint on one of their pairs raises; generated ones there are left out, and so
    are hints that do not fit with the windows.
    """
    from racerts.restraints.build import _consistent

    windows = task.restraints(mol)
    if not windows:
        return restraints
    held = {r.pair for r in windows}
    clash = [r for r in restraints if r.pair in held]
    user = [r.pair for r in clash if r.source == "user"]
    if user:
        raise ValueError(
            f"Restraints {user} are on pairs that the active-bond windows of the task "
            "hold; change the window (active_window, active_bonds) instead."
        )
    if clash:
        logger.warning(
            "Restraints %s are left out: the active-bond windows hold these pairs.",
            [r.label for r in clash],
        )
    others = RestraintSet(r for r in restraints if r.pair not in held)
    hints = RestraintSet(r for r in others if r.source == "hint")
    kept = windows.merge(r for r in others if r.source != "hint")
    if hints:
        frozen = task.frozen_atoms(mol)
        for hint in _consistent(mol, frozen, list(kept), hints, each=True):
            kept.add(hint)
    return kept
