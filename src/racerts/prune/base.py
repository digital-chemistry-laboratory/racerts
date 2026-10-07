"""The pruner interface."""

import logging
import math
from abc import abstractmethod
from numbers import Real

import numpy as np
from rdkit import Chem

from racerts.symmetry import SymmetricRMSD
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


def check_graph(graph) -> None:
    """Raise unless graph is an RDKit molecule or None (TypeError)."""
    if graph is not None and not isinstance(graph, Chem.Mol):
        raise TypeError(f"graph must be an RDKit molecule, not {graph!r}.")


def symmetry_kernel(mol: Chem.Mol, graph, conf_ids, atoms, **limits) -> SymmetricRMSD:
    """
    The symmetry-aware RMSD of the conformers conf_ids of mol: over the equivalent
    atoms of graph if one is given (a molecule of the same atoms in the same order,
    for conformers that are stored without bonds), else of mol itself.
    """
    if graph is None:
        if mol.GetNumBonds() == 0 and mol.GetNumAtoms() > 2:
            logger.warning(
                "The molecule has no bonds: all atoms of an element count as "
                "equivalent and no hydrogen is left out, so that different structures "
                "can be taken for one and copies of a structure can be missed. For a "
                "molecule stored without its bonds, give the bonded one as graph=."
            )
        return SymmetricRMSD(mol, atoms, **limits)
    if [a.GetAtomicNum() for a in mol.GetAtoms()] != [
        a.GetAtomicNum() for a in graph.GetAtoms()
    ]:
        raise ValueError(
            "graph must have the atoms of the conformers: the same elements in the "
            "same order."
        )
    reference = np.array(
        [mol.GetConformer(int(i)).GetPositions() for i in list(conf_ids)[:10]]
    )
    return SymmetricRMSD(graph, atoms, reference=reference, **limits)


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
