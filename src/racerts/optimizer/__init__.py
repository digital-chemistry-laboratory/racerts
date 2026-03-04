from .ase import ASEOptimizer
from .ff_optimizer import MMFFOptimizer, UFFOptimizer, BaseOptimizer

optimizers = {
    "mmff": MMFFOptimizer,
    "uff": UFFOptimizer,
    "ase": ASEOptimizer,
    "base": BaseOptimizer,
}
