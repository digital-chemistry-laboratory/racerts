"""The pruner interface."""

import logging
import math
from abc import abstractmethod
from numbers import Real

from rdkit import Chem

logger = logging.getLogger(__name__)


def check_threshold(value, name: str) -> None:
    """Raise unless value is a finite, non-negative number (TypeError, ValueError)."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number, not {value!r}.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and not negative, not {value!r}.")


def drop_conformers_without_energy(mol: Chem.Mol) -> None:
    """
    Remove conformers without a finite 'energy' property, e.g. from failed
    calculations.
    """
    missing = [
        conf.GetId()
        for conf in mol.GetConformers()
        if not conf.HasProp("energy") or not math.isfinite(conf.GetDoubleProp("energy"))
    ]
    if missing:
        logger.warning(
            "Dropping %d conformer(s) without an energy (or with a non-finite one): %s",
            len(missing),
            missing,
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
