"""From TS-like conformers to transition states: the free saddle search, checked."""

from typing import Any, Callable, Optional, Sequence

from racerts.pipeline import Pipeline
from racerts.prune import FamilySelector, PruneCount, PruneRMSD
from racerts.refine import Refine
from racerts.validate import (
    Connectivity,
    Converged,
    ReactionCore,
    ReactionMode,
    Validate,
)

from .staged import POOLS


def saddles(
    search,
    calculator: Any = None,
    pool: Optional[int] = None,
    pool_by: str = "family",
    min_stretch: float = 0.3,
    tolerance: float = 0.5,
    hessian: Optional[Callable] = None,
    checks: Sequence = (),
) -> Pipeline:
    """
    The saddle search as a pipeline, for the TS-like conformers of a TransitionState
    (the result of the default pipeline or of staged):

    1. with pool, at most that many conformers, by structural family or by energy;
    2. the free saddle search, Refine(search, anchors=False): the frozen atoms and the
       windows of the task are released;
    3. the checks: the search converged (Converged); exactly one imaginary mode, which
       moves an active bond (ReactionMode); the distances of the reacting atoms are
       those of the reference within tolerance (ReactionCore); the bonds and the stereo
       of the graph outside the reacting atoms (Connectivity); then the checks given
       with checks. Conformers that fail are dropped, with the reason in the log;
    4. the RMSD pruning: several TS-like conformers can end in one saddle point.

    Args:
        search: An optimizer for the saddle search, e.g. ASEOptimizer(calculator,
            optimizer_cls=Sella, optimizer_kwargs={"order": 1}, fmax=0.01,
            max_steps=300), or a Refine stage of one with anchors=False.
        calculator: Of the finite-difference Hessians (default: that of search).
        pool, pool_by: As of a Level of staged ("family" or "energy").
        min_stretch: Of ReactionMode.
        tolerance: Of ReactionCore (A).
        hessian: Instead of the finite differences (6 N force calls per conformer): a
            function of ASE Atoms that returns the Hessian in eV/A^2, e.g. an
            analytical one of the calculator (see ImaginaryModes).
        checks: Further validators after the built-in ones, e.g. one that follows the
            imaginary mode downhill to both sides (see racerts.validate.Validator).
    """
    if pool_by not in POOLS:
        raise ValueError(f"pool_by must be one of {POOLS}, not {pool_by!r}.")
    if isinstance(search, Refine):
        if search.anchors:
            raise ValueError(
                "The saddle search must be free: Refine(optimizer, anchors=False)."
            )
    else:
        search = Refine(search, fallback=False, anchors=False)
    if calculator is not None and hessian is not None:
        raise ValueError("Give either a calculator or a hessian function, not both.")
    if calculator is None and hessian is None:
        calculator = getattr(search.optimizer, "calculator", None)
        if calculator is None:
            raise ValueError(
                "The search has no calculator for the Hessians of the mode check: "
                "give calculator= or hessian=."
            )
    stages = []
    if pool is not None:
        stages.append(FamilySelector(pool) if pool_by == "family" else PruneCount(pool))
    checks = Validate(
        Converged(),
        ReactionMode(calculator, min_stretch=min_stretch, hessian=hessian),
        ReactionCore(tolerance),
        Connectivity(),
        *checks,
    )
    return Pipeline([*stages, search, checks, PruneRMSD()])
