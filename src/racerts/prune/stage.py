"""The pruning stages."""

from typing import Optional

import numpy as np

from racerts.pipeline import ConformerEnsemble

from .base import BasePruner, check_n_max
from .cluster import ClusterPruner
from .energy import EnergyPruner
from .rmsd import RMSDPruner


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
        groups = _target_groups(ctx, ensemble)
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


LENGTH_BINS = 5  # unstratified active-bond windows are pruned per fifth of the window


def _target_groups(ctx, ensemble):
    """
    The conformer ids per target of the active bonds of a windowed TS: per target
    (stratified) or per fifth of the window (unstratified); else one group.
    """
    groups = {}
    for conf_id, key in window_bins(ctx, ensemble).items():
        groups.setdefault(key, []).append(conf_id)
    return list(groups.values()) or [ensemble.conf_ids]


def window_bins(ctx, ensemble) -> dict:
    """
    The window bin of every conformer of a windowed TS, whose energies and geometries
    are comparable only within it: its targets (stratified) or the fifths of the
    windows that its targets fall in (uniform). () for every conformer otherwise.
    """
    if ctx is None or not getattr(ctx.task, "windowed", False):
        return {conf_id: () for conf_id in ensemble.conf_ids}
    windows = {f"{a}-{b}": w for (a, b), w in ctx.task.active_windows(ctx.mol).items()}
    stratified = getattr(ctx.task, "stratify", 0)
    bins = {}
    for conf_id in ensemble.conf_ids:
        targets = ensemble.provenance(conf_id).get("active_bond_targets") or {}
        key = []
        for bond, t in sorted(targets.items()):
            if stratified or bond not in windows:
                key.append((bond, t))
            else:
                lo, hi = windows[bond]
                key.append(
                    (
                        bond,
                        min(int((t - lo) / (hi - lo) * LENGTH_BINS), LENGTH_BINS - 1),
                    )
                )
        bins[conf_id] = tuple(key)
    return bins


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
    Keeps the n_max conformers of lowest energy, ordered by energy; with renumber,
    their ids become 0, 1, ... in that order. Every conformer needs a finite energy.
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
        order = np.argsort(energies, kind="stable")[: self.n_max]
        return ensemble.filter(
            [ensemble.conf_ids[i] for i in order], renumber=self.renumber
        )
