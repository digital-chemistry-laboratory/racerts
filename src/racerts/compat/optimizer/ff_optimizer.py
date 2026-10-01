"""
racerts.optimizer.ff_optimizer of legacy racerts: the force-field optimizers with the
legacy method tune_ts_conformers, as subclasses of those of racerts.refine.
"""

from abc import abstractmethod

import racerts.refine


class BaseOptimizer(racerts.refine.BaseOptimizer):
    """
    A legacy racerts optimizer: subclasses implement tune_ts_conformers. The stages
    call refine, which runs tune_ts_conformers, so legacy optimizers and legacy
    subclasses of the built-in ones work there too.
    """

    # For subclasses whose own __init__ does not set them (the UFF fallback copies
    # them).
    verbose = False
    force_constant = 1e6
    num_threads = 1
    anchor_free_energies = False

    def _refine(self, mol, reference, anchors):
        return self.tune_ts_conformers(
            mol=mol, reference=reference, align_indices=list(anchors)
        )

    @abstractmethod
    def tune_ts_conformers(self, mol, reference, align_indices):
        """Optimize the conformers of mol in place (here, as in legacy racerts: no-op)."""


class MMFFOptimizer(BaseOptimizer, racerts.refine.MMFFOptimizer):
    def tune_ts_conformers(self, mol, reference, align_indices):
        return racerts.refine.MMFFOptimizer._refine(
            self, mol, reference, align_indices or []
        )


class UFFOptimizer(BaseOptimizer, racerts.refine.UFFOptimizer):
    def tune_ts_conformers(self, mol, reference, align_indices):
        return racerts.refine.UFFOptimizer._refine(
            self, mol, reference, align_indices or []
        )
