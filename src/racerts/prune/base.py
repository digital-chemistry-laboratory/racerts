"""The pruner interface."""

import logging
import math
from abc import abstractmethod
from numbers import Real

from rdkit import Chem

from racerts.utils.checks import is_integer

logger = logging.getLogger(__name__)


def check_threshold(value, name: str) -> None:
    """Raise unless value is a finite, non-negative number (TypeError, ValueError)."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a number, not {value!r}.")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be finite and not negative, not {value!r}.")


def check_n_max(n_max) -> int:
    """n_max as an int; raises unless it is a positive integer."""
    if not is_integer(n_max):
        raise TypeError(f"n_max must be an integer, not {n_max!r}.")
    if n_max < 1:
        raise ValueError("n_max must be positive.")
    return int(n_max)


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
    """Removes conformers of a molecule in place; subclasses implement prune."""

    @abstractmethod
    def __init__(self, threshold: float, verbose: bool = False, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def prune(self, mol: Chem.Mol) -> Chem.Mol:
        raise NotImplementedError
