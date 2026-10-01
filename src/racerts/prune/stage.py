"""The pruning stages."""

from numbers import Integral
from typing import Optional

import numpy as np

from racerts.pipeline import ConformerEnsemble

from .base import BasePruner
from .cluster import ClusterPruner
from .energy import EnergyPruner
from .rmsd import RMSDPruner


class _Prune:
    """Prunes the ensemble in place with pruner (default: default_pruner())."""

    name: str
    default_pruner: type

    def __init__(self, pruner: Optional[BasePruner] = None):
        self.pruner = pruner

    def run(self, ctx, ensemble: ConformerEnsemble) -> ConformerEnsemble:
        pruner = self.pruner if self.pruner is not None else self.default_pruner()
        pruner.prune(mol=ensemble.mol)
        return ensemble


class PruneEnergy(_Prune):
    """
    Drops conformers more than the threshold (kcal/mol) above the lowest one.

    Args:
        pruner: Any BasePruner; default EnergyPruner() (threshold 20 kcal/mol).
    """

    name = "prune_energy"
    default_pruner = EnergyPruner


class PruneRMSD(_Prune):
    """
    Drops duplicates (RMSD below the threshold), keeping the lowest in energy.

    Args:
        pruner: Any BasePruner; default RMSDPruner() (threshold 0.125 A).
    """

    name = "prune_rmsd"
    default_pruner = RMSDPruner


class PruneCluster(_Prune):
    """
    Keeps one conformer per cluster (see ClusterPruner).

    Args:
        pruner: Any BasePruner; default ClusterPruner() (Butina at 1.5 A, the
            lowest-energy member of each cluster).
    """

    name = "prune_cluster"
    default_pruner = ClusterPruner


class PruneCount:
    """
    Keeps the n_max conformers of lowest energy, ordered by energy (catmlp
    prune_to_max_conformers); with renumber, their ids become 0, 1, ... in that
    order. Every conformer needs a finite energy.
    """

    name = "prune_count"

    def __init__(self, n_max: int, renumber: bool = False):
        if isinstance(n_max, bool) or not isinstance(n_max, Integral):
            raise TypeError(f"n_max must be an integer, not {n_max!r}.")
        if n_max < 1:
            raise ValueError("n_max must be positive.")
        self.n_max = int(n_max)
        self.renumber = renumber

    def run(self, ctx, ensemble: ConformerEnsemble) -> ConformerEnsemble:
        energies = ensemble.energies()
        if not len(energies):
            raise ValueError("PruneCount got an ensemble without conformers.")
        if not np.isfinite(energies).all():
            missing = [
                i for i, e in zip(ensemble.conf_ids, energies) if not np.isfinite(e)
            ]
            raise ValueError(
                f"Conformers {missing} have no finite energy; refine or rescore first."
            )
        order = np.argsort(energies, kind="stable")[: self.n_max]
        return ensemble.filter(
            [ensemble.conf_ids[i] for i in order], renumber=self.renumber
        )
