from .ff_optimizer import MMFFOptimizer, UFFOptimizer, BaseOptimizer

optimizers = {"mmff": MMFFOptimizer, "uff": UFFOptimizer, "base": BaseOptimizer}
