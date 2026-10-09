"""racerts.optimizer of legacy racerts."""

from .ase import ASEOptimizer
from .ff_optimizer import BaseOptimizer, MMFFOptimizer, UFFOptimizer

optimizers = {
    "mmff": MMFFOptimizer,
    "uff": UFFOptimizer,
    "ase": ASEOptimizer,
    "base": BaseOptimizer,
}

__all__ = [
    "ASEOptimizer",
    "BaseOptimizer",
    "MMFFOptimizer",
    "UFFOptimizer",
    "optimizers",
]
