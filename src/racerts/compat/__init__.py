"""
The legacy racerts API, kept for existing code: ConformerGenerator, the components with
their legacy methods, the registries and the legacy command line. It runs on the
pipeline and gives the same results.

This package has the module layout of legacy racerts (0.1.7), and its modules are also
registered under the legacy names: racerts.embedder is racerts.compat.embedder,
racerts.optimizer.ase is racerts.compat.optimizer.ase, and so on. racerts.utils forwards
the legacy functions to compat.utils, and racerts.cli hands the legacy command line to
compat.cli.
"""

import sys
from importlib import import_module

from .conformer_generator import DEFAULT_CONF_FACTOR, ConformerGenerator
from .embedder import BaseEmbedder, BoundsMatrixEmbedder, CmapEmbedder, embedders
from .mol_getter import (
    BaseMolGetter,
    MolGetterBonds,
    MolGetterConnectivity,
    MolGetterSMILES,
    mol_getters,
)
from .optimizer import (
    ASEOptimizer,
    BaseOptimizer,
    MMFFOptimizer,
    UFFOptimizer,
    optimizers,
)
from .pruner import BasePruner, EnergyPruner, RMSDPruner, pruners
from .utils import atom_idx_input_validation

LEGACY_MODULES = (
    "conformer_generator",
    "embedder",
    "embedder.embedder",
    "embedder.utils",
    "mol_getter",
    "mol_getter.mol_getter",
    "optimizer",
    "optimizer.ase",
    "optimizer.ff_optimizer",
    "optimizer.parallel",
    "pruner",
    "pruner.pruner",
    "visualizer",
)
for _name in LEGACY_MODULES:
    sys.modules[f"racerts.{_name}"] = import_module(f"{__name__}.{_name}")

__all__ = [
    "DEFAULT_CONF_FACTOR",
    "ASEOptimizer",
    "BaseEmbedder",
    "BaseMolGetter",
    "BaseOptimizer",
    "BasePruner",
    "BoundsMatrixEmbedder",
    "CmapEmbedder",
    "ConformerGenerator",
    "EnergyPruner",
    "MMFFOptimizer",
    "MolGetterBonds",
    "MolGetterConnectivity",
    "MolGetterSMILES",
    "RMSDPruner",
    "UFFOptimizer",
    "atom_idx_input_validation",
    "embedders",
    "mol_getters",
    "optimizers",
    "pruners",
]
