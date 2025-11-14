from .pruner import BasePruner, EnergyPruner, RMSDPruner

pruners = {
    "base": BasePruner,
    "energy": EnergyPruner,
    "rmsd": RMSDPruner,
}
