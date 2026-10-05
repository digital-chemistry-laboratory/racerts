"""Pruning by energy window."""

import logging

import numpy as np
from rdkit import Chem

from .base import BasePruner, check_threshold, drop_conformers_without_energy

logger = logging.getLogger(__name__)


class EnergyPruner(BasePruner):
    """Drops conformers more than threshold (kcal/mol) above the lowest one."""

    def __init__(self, threshold: float = 20.0, verbose: bool = False):
        check_threshold(threshold, "threshold")
        self.threshold = threshold
        self.verbose = verbose

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
        min_energy = self.get_minimal_energy(mol, self.verbose)
        drop_conformers_without_energy(mol)
        conformers_to_remove = []

        for conf in mol.GetConformers():
            if conf.GetDoubleProp("energy") - min_energy > self.threshold:
                conformers_to_remove.append(conf.GetId())

        for conf_id in conformers_to_remove:
            mol.RemoveConformer(conf_id)

        return mol
