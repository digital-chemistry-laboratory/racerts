"""The optimizer interface."""

import logging
from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from rdkit import Chem
from rdkit.Chem.AllChem import AlignMol  # type: ignore

from racerts.restraints.model import PositionRestraint, accepts_restraints

logger = logging.getLogger(__name__)


class BaseOptimizer(ABC):
    """
    Refines the conformers of a molecule in place and stores their energies (kcal/mol)
    as the conformer property "energy". Subclasses implement _refine.
    """

    conf_id_ref = -1  # the conformer of the reference that align_mols uses; -1: first

    def refine(
        self,
        mol: Chem.Mol,
        reference: Optional[Chem.Mol] = None,
        anchors: Sequence[int] = (),
        restraints: Sequence = (),
    ):
        """
        Refine all conformers; the anchors stay at their reference positions.
        restraints (DistanceRestraint and PositionRestraint) go to optimizers whose
        _refine takes them (MMFF, UFF of racerts.refine); others refine without them,
        which is logged.
        """
        if anchors and reference is None:
            raise ValueError("Anchor atoms need a reference geometry.")
        if restraints:
            if accepts_restraints(self._refine):
                return self._refine(mol, reference, anchors, restraints=restraints)
            soft = [r for r in restraints if isinstance(r, PositionRestraint)]
            if soft:  # position restraints: nothing else holds these atoms
                logger.warning(
                    "%s takes no restraints: the %d soft atoms are free in this "
                    "refinement.",
                    type(self).__name__,
                    len(soft),
                )
            if len(soft) < len(restraints):
                # A force field could hold them: a legacy class or subclass without
                # the restraints argument. Other optimizers (ASE) take none by design.
                legacy = hasattr(self, "_add_restraint")
                logger.log(
                    logging.WARNING if legacy else logging.INFO,
                    "%s refines without the %d distance restraints%s.",
                    type(self).__name__,
                    len(restraints) - len(soft),
                    " (the optimizers of racerts.refine hold them)" if legacy else "",
                )
        return self._refine(mol, reference, anchors)

    @abstractmethod
    def _refine(
        self, mol: Chem.Mol, reference: Optional[Chem.Mol], anchors: Sequence[int]
    ):
        """refine, with the arguments checked."""
        raise NotImplementedError

    def align_mols(
        self,
        mol: Chem.Mol,
        reference: Chem.Mol,
        align_indices: Optional[List[int]] = None,
        conf_id: Optional[int] = None,
    ) -> None:
        """
        Align conformer conf_id of mol (all conformers for None) on the atoms
        align_indices to conformer conf_id_ref of the reference.
        """
        if not align_indices:
            return

        atom_map = [(idx, idx) for idx in align_indices]
        conformers = (
            mol.GetConformers() if conf_id is None else [mol.GetConformer(conf_id)]
        )
        for conformer in conformers:
            AlignMol(
                mol,
                reference,
                atomMap=atom_map,
                prbCid=conformer.GetId(),
                refCid=self.conf_id_ref,
            )
