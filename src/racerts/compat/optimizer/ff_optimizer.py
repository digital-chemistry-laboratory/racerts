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
    converge = False
    anchor_free_energies = False

    def _refine(self, mol, reference, anchors):
        return self.tune_ts_conformers(
            mol=mol, reference=reference, align_indices=list(anchors)
        )

    @abstractmethod
    def tune_ts_conformers(self, mol, reference, align_indices):
        """Optimize the conformers of mol in place (here, as in legacy racerts: no-op)."""


class _LegacyDefaults(racerts.refine.ForceFieldOptimizer):
    """The settings of legacy racerts as defaults, whatever those of racerts.refine."""

    def __init__(
        self,
        verbose=False,
        conf_id_ref=-1,
        force_constant=1000000,
        num_threads=1,
        converge=False,
        anchor_free_energies=False,
        **kwargs,
    ):
        super().__init__(
            verbose=verbose,
            conf_id_ref=conf_id_ref,
            force_constant=force_constant,
            num_threads=num_threads,
            converge=converge,
            anchor_free_energies=anchor_free_energies,
            **kwargs,
        )


class MMFFOptimizer(BaseOptimizer, _LegacyDefaults, racerts.refine.MMFFOptimizer):
    def tune_ts_conformers(self, mol, reference, align_indices):
        return racerts.refine.MMFFOptimizer._refine(
            self, mol, reference, align_indices or []
        )


class UFFOptimizer(BaseOptimizer, _LegacyDefaults, racerts.refine.UFFOptimizer):
    def tune_ts_conformers(self, mol, reference, align_indices):
        return racerts.refine.UFFOptimizer._refine(
            self, mol, reference, align_indices or []
        )
