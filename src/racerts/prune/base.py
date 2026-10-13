"""The pruner interface."""

import logging
from abc import abstractmethod

from rdkit import Chem

logger = logging.getLogger(__name__)


def drop_conformers_without_energy(mol: Chem.Mol) -> None:
    """Remove conformers without an 'energy' property, e.g. from failed calculations."""
    missing = [
        conf.GetId() for conf in mol.GetConformers() if not conf.HasProp("energy")
    ]
    if missing:
        logger.warning(
            "Dropping %d conformer(s) without an energy: %s", len(missing), missing
        )
    for conf_id in missing:
        mol.RemoveConformer(conf_id)


class BasePruner:
    @abstractmethod
    def __init__(self, threshold: float, verbose: bool = False, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def prune(self, mol: Chem.Mol) -> Chem.Mol:
        raise NotImplementedError
