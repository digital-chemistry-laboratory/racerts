"""The optimizer interface."""

from abc import ABC, abstractmethod
from typing import List, Optional, Sequence

from rdkit import Chem
from rdkit.Chem.AllChem import AlignMol  # type: ignore


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
    ):
        """Refine all conformers; the anchors stay at their reference positions."""
        if anchors and reference is None:
            raise ValueError("Anchor atoms need a reference geometry.")
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
