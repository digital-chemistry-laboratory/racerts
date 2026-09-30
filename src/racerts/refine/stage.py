"""The Refine stage and the MMFF -> UFF fallback."""

import logging
from typing import Callable, Optional, Type

from racerts.pipeline import ConformerEnsemble

from .base import BaseOptimizer
from .forcefield import MMFFOptimizer, UFFOptimizer

logger = logging.getLogger(__name__)

REFINE_BACKENDS = {"mmff": MMFFOptimizer, "uff": UFFOptimizer}


def refine_with_fallback(
    optimizer: BaseOptimizer,
    run: Callable[[BaseOptimizer], object],
    fallback: Optional[Type[UFFOptimizer]] = UFFOptimizer,
    num_threads: Optional[int] = None,
) -> str:
    """
    Call run(optimizer). If an MMFFOptimizer fails (e.g. MMFF has no parameters for
    an element) and there is a fallback (a UFFOptimizer class), log a warning and call
    run with a fallback optimizer of the same settings (num_threads: default that of
    the MMFFOptimizer) instead. Other errors are passed on.

    Returns:
        str: The class name of the optimizer that produced the energies.
    """
    try:
        run(optimizer)
    except Exception as e:
        if fallback is None or not isinstance(optimizer, MMFFOptimizer):
            raise
        logger.warning(
            "%s failed (%s); falling back to UFF. UFF energies are less reliable for "
            "ranking conformers.",
            type(optimizer).__name__,
            e,
        )
        optimizer = fallback(
            verbose=optimizer.verbose,
            conf_id_ref=optimizer.conf_id_ref,
            force_constant=optimizer.force_constant,
            num_threads=optimizer.num_threads if num_threads is None else num_threads,
        )
        run(optimizer)
    return type(optimizer).__name__


class Refine:
    """
    Refines the conformers in place, with the hard frozen atoms of the task as anchors,
    and records the optimizer as the molecule property "energy_method".

    Args:
        optimizer: Any BaseOptimizer; default MMFFOptimizer.
        fallback: Fall back to UFF if MMFF fails.
        anchors: Hold the hard frozen atoms at the reference (default). False refines
            all atoms freely, e.g. for a saddle-point search from TS-like conformers
            (ASEOptimizer with Sella).
    """

    name = "refine"

    def __init__(
        self,
        optimizer: Optional[BaseOptimizer] = None,
        fallback: bool = True,
        anchors: bool = True,
    ):
        self.optimizer = optimizer
        self.fallback = fallback
        self.anchors = anchors

    def run(self, ctx, ensemble: ConformerEnsemble) -> ConformerEnsemble:
        optimizer = self.optimizer if self.optimizer is not None else MMFFOptimizer()
        anchors = ctx.frozen.hard if self.anchors else ()
        energy_method = refine_with_fallback(
            optimizer,
            lambda opt: opt.refine(ensemble.mol, ctx.reference, anchors),
            fallback=UFFOptimizer if self.fallback else None,
        )
        ensemble.mol.SetProp("energy_method", energy_method)
        return ensemble
