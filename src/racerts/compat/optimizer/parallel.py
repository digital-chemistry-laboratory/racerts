"""
racerts.optimizer.parallel of legacy racerts: the relaxation of ASE structures is
racerts.refine.parallel, which gives an Outcome per structure; the functions here give
the tuples of legacy racerts.
"""

from __future__ import annotations

from typing import Any

from racerts.refine.parallel import (
    OptimizationConfig,
    OptimizationTask,
    _in_processes,
    optimize_one,
    pin_to_single_thread,
)

__all__ = [
    "OptimizationConfig",
    "OptimizationTask",
    "pin_to_single_thread",
    "run_optimization",
    "run_optimizations_in_processes",
]

# (conf_id, positions, energy, converged, error); positions and energy are None on error.
OptimizationResult = tuple[int, Any, "float | None", bool, "str | None"]


def run_optimization(
    calculator: Any,
    config: OptimizationConfig,
    task: OptimizationTask,
) -> OptimizationResult:
    """Relax one structure and return its id, positions, energy, convergence and error.

    See [optimize_one][racerts.refine.optimize_one], which also returns the number of
    steps and the time.
    """
    return optimize_one(calculator, config, task).as_tuple()


def run_optimizations_in_processes(
    tasks: list[OptimizationTask],
    calculator: Any,
    calculator_is_factory: bool,
    config: OptimizationConfig,
    num_workers: int | None,
) -> list[OptimizationResult]:
    """Relax conformers in spawned, single-thread-pinned worker processes."""
    outcomes = _in_processes(
        tasks, calculator, calculator_is_factory, config, num_workers
    )
    return [outcome.as_tuple() for outcome in outcomes]
