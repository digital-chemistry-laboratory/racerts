"""
racerts.pruner.pruner of legacy racerts: the pruners of racerts.prune (same API), and
the energy pruner with its extended-Hueckel energies.
"""

import logging
import warnings

from rdkit import Chem
from rdkit.Chem import rdEHTTools

from racerts import prune
from racerts.prune import BasePruner, RMSDPruner
from racerts.utils.units import EV_TO_KCAL_MOL

logger = logging.getLogger(__name__)

__all__ = ["EV_TO_KCAL_MOL", "BasePruner", "EnergyPruner", "RMSDPruner"]


class EnergyPruner(prune.EnergyPruner):
    """
    The energy pruner of legacy racerts. With YAeHMOP_energies (deprecated) the
    energies are first replaced by extended-Hueckel ones (energy_method "EHT"); the
    pipeline ranks by another level with the Rescore stage instead.
    """

    def __init__(self, threshold: float = 20.0, verbose: bool = False, **kwargs):
        super().__init__(threshold, verbose)
        self.YAeHMOP_energies = kwargs.get("YAeHMOP_energies", False)
        if self.YAeHMOP_energies:
            warnings.warn(
                "YAeHMOP_energies is deprecated; rescore the ensemble with an ASE "
                "calculator instead (the Rescore stage).",
                FutureWarning,  # shown by default, unlike DeprecationWarning
                stacklevel=2,
            )

    def set_QM_energies(self, mol, verbose=False):
        """
        Sets quantum mechanical (QM) energies for all conformers of a molecule using RDKit's
        EHT tools. QM energies are set as a property on each conformer (in kcal/mol);
        conformers for which the calculation fails are left without an energy.

        Args:
            mol (RDKit Mol): The molecule whose conformers will have QM energies set.
            verbose (bool, optional): If True, prints failure messages for conformers where
                                    QM energy calculation fails. Defaults to False.
        """
        idx = [conf.GetId() for conf in mol.GetConformers()]

        for id in idx:
            passed, res = rdEHTTools.RunMol(mol, confId=id)

            if passed is True:
                # YAeHMOP works in eV (its H_ii parameters are ionization energies, e.g.
                # -13.6 eV for H 1s; H2 gives a sigma orbital at -17.7 eV).
                e = res.totalEnergy * EV_TO_KCAL_MOL
                mol.GetConformer(id).SetDoubleProp("energy", e)
            else:
                # Don't rank a stale energy from an earlier step against EHT energies.
                mol.GetConformer(id).ClearProp("energy")
                logger.debug("rdEHTTools failed for conformer %d", id)

    def prune(self, mol: Chem.Mol):
        if self.YAeHMOP_energies:
            self.set_QM_energies(mol, self.verbose)
            mol.SetProp("energy_method", "EHT")
        return super().prune(mol)
