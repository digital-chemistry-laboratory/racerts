"""Pruning by energy window."""

import logging
import warnings

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdEHTTools

from racerts.utils.units import EV_TO_KCAL_MOL

from .base import BasePruner, check_threshold, drop_conformers_without_energy

logger = logging.getLogger(__name__)


class EnergyPruner(BasePruner):
    """
    Drops conformers more than threshold (kcal/mol) above the lowest one. With
    YAeHMOP_energies (deprecated) the energies are first replaced by extended-Hueckel
    ones (energy_method "EHT").
    """

    def __init__(self, threshold: float = 20.0, verbose: bool = False, **kwargs):
        check_threshold(threshold, "threshold")
        self.YAeHMOP_energies = kwargs.get("YAeHMOP_energies", False)
        if self.YAeHMOP_energies:
            warnings.warn(
                "YAeHMOP_energies is deprecated; rescore the ensemble with an ASE "
                "calculator instead (the Rescore stage).",
                FutureWarning,  # shown by default, unlike DeprecationWarning
                stacklevel=2,
            )
        self.threshold = threshold
        self.verbose = verbose

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

    def get_minimal_energy(self, mol, verbose=False):
        """
        Finds and returns the minimal energy among all conformers of a molecule. The minimal
        energy is also set as a property on the molecule.

        Args:
            mol (RDKit Mol): The molecule to evaluate.
            verbose (bool, optional): If True, prints the minimal energy found. Defaults to False.

        Returns:
            float: The minimal energy found among all conformers.

        Raises:
            ValueError: If no conformer has an energy.
        """
        energies = [
            conf.GetDoubleProp("energy")
            for conf in mol.GetConformers()
            if conf.HasProp("energy") and np.isfinite(conf.GetDoubleProp("energy"))
        ]
        if not energies:
            raise ValueError(
                "No conformer carries an 'energy' property. Run an optimizer first."
            )
        # Never cache: energies change after every (re-)optimization.
        min_energy = float(np.min(energies))
        mol.SetDoubleProp("minimal_energy", min_energy)

        logger.debug("Minimal energy conformer: %s kcal/mol", min_energy)

        return min_energy

    def prune(self, mol: Chem.Mol):

        if self.YAeHMOP_energies:
            self.set_QM_energies(mol, self.verbose)
            mol.SetProp("energy_method", "EHT")

        min_energy = self.get_minimal_energy(mol, self.verbose)
        drop_conformers_without_energy(mol)
        conformers_to_remove = []

        for conf in mol.GetConformers():
            if conf.GetDoubleProp("energy") - min_energy > self.threshold:
                conformers_to_remove.append(conf.GetId())

        for conf_id in conformers_to_remove:
            mol.RemoveConformer(conf_id)

        return mol
