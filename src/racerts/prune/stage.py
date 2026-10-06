"""The pruning stages."""

from typing import List, Optional

import numpy as np

from racerts.pipeline import ConformerEnsemble

from .base import BasePruner, check_n_max
from .cluster import ClusterPruner
from .energy import EnergyPruner
from .rmsd import RMSDPruner
from .targets import in_turns, target_groups


class _Prune:
    """
    Prunes the ensemble in place with pruner (default: default_pruner()). For a TS
    with active-bond windows, each target (stratified) or fifth of the window
    (unstratified) is pruned on its own: energies and geometries at different
    constrained lengths are not comparable.
    """

    name: str
    default_pruner: type

    def __init__(self, pruner: Optional[BasePruner] = None):
        if pruner is not None and not callable(getattr(pruner, "prune", None)):
            raise TypeError(
                f"{type(self).__name__} takes a pruner (an object with prune(mol)), "
                f"not {pruner!r}."
            )
        self.pruner = pruner

    def run(self, ctx, ensemble: ConformerEnsemble) -> ConformerEnsemble:
        pruner = self.pruner if self.pruner is not None else self.default_pruner()
        groups = target_groups(ctx, ensemble)
        if len(groups) <= 1:
            pruner.prune(mol=ensemble.mol)
            return ensemble
        kept = set()
        for conf_ids in groups:
            part = ensemble.filter(conf_ids)
            pruner.prune(mol=part.mol)
            kept.update(part.conf_ids)
        for conf_id in ensemble.conf_ids:
            if conf_id not in kept:
                ensemble.mol.RemoveConformer(conf_id)
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
        pruner: Any BasePruner; default RMSDPruner() (threshold 0.125 A, with the
            polar hydrogens).
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
    Keeps the n_max conformers of lowest energy, ordered by energy; with renumber,
    their ids become 0, 1, ... in that order. Every conformer needs a finite energy.

    The targets of a TS with active-bond windows take turns (the lowest of each target,
    then the second lowest, ...; the result is in that order): energies at different
    constrained lengths are not comparable.
    """

    name = "prune_count"

    def __init__(self, n_max: int, renumber: bool = False):
        self.n_max = check_n_max(n_max)
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
        order = in_turns(ctx, ensemble, by_energy)
        return ensemble.filter(order[: self.n_max], renumber=self.renumber)


def by_energy(ensemble: ConformerEnsemble) -> List[int]:
    """The conformer ids by energy; conformers of the same energy keep their order."""
    order = np.argsort(ensemble.energies(), kind="stable")
    return [ensemble.conf_ids[i] for i in order]
