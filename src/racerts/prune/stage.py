"""The pruning stages."""

from typing import Optional

from racerts.pipeline import ConformerEnsemble

from .base import BasePruner
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
