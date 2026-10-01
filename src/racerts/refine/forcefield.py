"""MMFF and UFF refinement with the anchor atoms held at the reference positions."""

import logging
from concurrent.futures import ThreadPoolExecutor
from typing import Optional, Sequence

from rdkit import Chem
from rdkit.Chem.AllChem import (
    MMFFGetMoleculeForceField,  # type: ignore
    MMFFGetMoleculeProperties,  # type: ignore
    UFFGetMoleculeForceField,  # type: ignore
)

from .base import BaseOptimizer

logger = logging.getLogger(__name__)


class ForceFieldOptimizer(BaseOptimizer):
    """
    Minimizes each conformer with an RDKit force field. Each anchor atom is tied to a
    fixed extra point at its reference position by a distance constraint
    (force_constant, kcal/mol/A^2); conformers are aligned on the anchors to the
    reference before and after. Minimize is called up to maxIter times per conformer.

    Args:
        anchor_free_energies: Report the energy of the force field without the anchor
            terms; legacy racerts includes them (0.02-0.18 kcal/mol on test systems).
    """

    def __init__(
        self,
        verbose=False,
        conf_id_ref=-1,
        force_constant=1000000,
        num_threads=1,
        anchor_free_energies: bool = False,
    ):
        self.verbose = verbose
        self.conf_id_ref = conf_id_ref
        self.force_constant = force_constant
        self.num_threads = num_threads
        self.anchor_free_energies = anchor_free_energies
        self.maxIter = 100

    def _setup(self, mol: Chem.Mol):
        """Per-molecule data for _force_field (e.g. MMFF parameters)."""
        return None

    def _force_field(self, mol: Chem.Mol, conf_id: int, setup):
        raise NotImplementedError

    def _refine(
        self, mol: Chem.Mol, reference: Optional[Chem.Mol], anchors: Sequence[int]
    ) -> int:
        """Returns the number of Minimize calls that did not converge."""
        align_indices = list(anchors)
        coordinates_ref = None
        if align_indices:
            coordinates_ref = reference.GetConformer(self.conf_id_ref).GetPositions()
        setup = self._setup(mol)

        def _optimize(conf_id: int) -> int:
            """
            Returns the number of failed minimisation attempts for this conformer.
            """
            self.align_mols(
                mol=mol,
                reference=reference,
                align_indices=align_indices,
                conf_id=conf_id,
            )
            ff = self._force_field(mol, conf_id, setup)

            for idx in align_indices:
                point = coordinates_ref[idx]
                ep_idx = ff.AddExtraPoint(*point, fixed=True) - 1
                ff.AddDistanceConstraint(ep_idx, idx, 0, 0, self.force_constant)

            ff.Initialize()

            local_fail = 0
            for _ in range(self.maxIter):
                if ff.Minimize() == 0:
                    break
                local_fail += 1

            if self.anchor_free_energies and align_indices:
                energy = self._force_field(mol, conf_id, setup).CalcEnergy()
            else:
                energy = ff.CalcEnergy()
            mol.GetConformer(conf_id).SetDoubleProp("energy", energy)

            self.align_mols(
                mol=mol,
                reference=reference,
                align_indices=align_indices,
                conf_id=conf_id,
            )

            return local_fail

        conformer_ids = [c.GetId() for c in mol.GetConformers()]
        if self.num_threads > 1:
            with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                failures = sum(pool.map(_optimize, conformer_ids))
        else:
            failures = sum(_optimize(cid) for cid in conformer_ids)

        logger.info("%s: %d unconverged Minimize calls", type(self).__name__, failures)
        return failures


class UFFOptimizer(ForceFieldOptimizer):
    def _force_field(self, mol, conf_id, setup):
        return UFFGetMoleculeForceField(
            mol, confId=conf_id, ignoreInterfragInteractions=False
        )


class MMFFOptimizer(ForceFieldOptimizer):
    def _setup(self, mol):
        mmffVerbosity = 2 if self.verbose else 0
        ff_props = MMFFGetMoleculeProperties(mol, mmffVerbosity=mmffVerbosity)
        if ff_props is None:
            raise ValueError("MMFF parameters are not available for this molecule")
        return ff_props

    def _force_field(self, mol, conf_id, setup):
        return MMFFGetMoleculeForceField(
            mol,
            setup,
            confId=conf_id,
            ignoreInterfragInteractions=False,
        )
