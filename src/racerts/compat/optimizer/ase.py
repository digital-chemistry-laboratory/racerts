"""
racerts.optimizer.ase of legacy racerts: ASEOptimizer with the legacy method
tune_ts_conformers, and the names this module had.

ASEOptimizer builds the ASE Atoms with rdkit_conformer_to_ase_atoms of this module,
looked up at call time, so code that replaces it here (as catmlp does) keeps working.
"""

from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from typing import Any, Dict, List, Optional

import racerts.refine
from racerts.io import ase as ase_io
from racerts.io.ase import write_ase_positions_to_rdkit
from racerts.refine.parallel import (
    OptimizationConfig,
    OptimizationTask,
    run_optimization,
    run_optimizations_in_processes,
)
from racerts.system import count_electrons, infer_charge_and_multiplicity
from racerts.utils.units import EV_TO_KCAL_MOL

from .ff_optimizer import BaseOptimizer

__all__ = [
    "ASEOptimizer",
    "BaseOptimizer",
    "EV_TO_KCAL_MOL",
    "Import",
    "OptimizationConfig",
    "OptimizationTask",
    "count_electrons",
    "infer_charge_and_multiplicity",
    "rdkit_conformer_to_ase_atoms",
    "requires_dependency",
    "run_optimization",
    "run_optimizations_in_processes",
    "write_ase_positions_to_rdkit",
]


def rdkit_conformer_to_ase_atoms(mol, conf_id, multiplicity=None, charge=None):
    """racerts.io.ase.rdkit_conformer_to_ase_atoms (looked up at call time)."""
    return ase_io.rdkit_conformer_to_ase_atoms(
        mol, conf_id, multiplicity=multiplicity, charge=charge
    )


class ASEOptimizer(BaseOptimizer, racerts.refine.ASEOptimizer):
    def tune_ts_conformers(self, mol, reference, align_indices=None):
        return racerts.refine.ASEOptimizer._refine(
            self, mol, reference, align_indices or []
        )

    def _to_atoms(self, mol, conf_id, state):
        return rdkit_conformer_to_ase_atoms(mol, conf_id=conf_id, **state)


@dataclass(frozen=True)
class Import:
    """An import for requires_dependency."""

    module: str
    item: Optional[str] = None
    alias: Optional[str] = None


def requires_dependency(imports: List[Import], scope: Dict[str, Any]):
    """Decorator: import the given names into scope before the first call."""

    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            try:
                for imp in imports:
                    symbol_name = imp.alias or imp.item or imp.module.rsplit(".", 1)[-1]
                    if scope.get(symbol_name) is not None:
                        continue
                    module = import_module(imp.module)
                    symbol = getattr(module, imp.item) if imp.item else module
                    scope[symbol_name] = symbol
            except Exception as exc:
                raise ImportError(
                    "ASE is required for ASEOptimizer. Install with "
                    "`pip install racerts[ase]`."
                ) from exc
            return func(*args, **kwargs)

        return _wrapper

    return _decorator
