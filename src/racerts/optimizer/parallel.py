from __future__ import annotations

import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ase import Atoms as ASEAtoms


OptimizationTask = tuple[int, "ASEAtoms"]
OptimizationResult = tuple[int, Any, float, bool]


@dataclass(frozen=True)
class OptimizationConfig:
    """Optimizer settings shared by every conformer."""

    optimizer_cls: type
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    fmax: float = 0.05
    max_steps: int = 100


# Per-worker calculator/config, set once in the pool initializer and reused across tasks.
_WORKER_CALC: Any = None
_WORKER_CONFIG: OptimizationConfig | None = None
# Hold the threadpoolctl controller for the worker's lifetime so its limits are not
# restored between tasks.
_PIN: Any = None


def pin_to_single_thread() -> None:
    """Pin the calling process to one native thread across all thread pools."""
    global _PIN
    from threadpoolctl import threadpool_limits

    _PIN = threadpool_limits(limits=1)


def _init_worker(
    calculator_payload: Any,
    is_factory: bool,
    config: OptimizationConfig,
) -> None:
    """Pin a pool worker and initialize its resident calculator and settings."""
    global _WORKER_CALC, _WORKER_CONFIG
    pin_to_single_thread()
    if is_factory:
        calculator = calculator_payload()
        if calculator is None:
            raise ValueError("`calculator` callable returned None.")
        _WORKER_CALC = calculator
    else:
        _WORKER_CALC = calculator_payload
    _WORKER_CONFIG = config


def run_optimization(
    calculator: Any,
    config: OptimizationConfig,
    task: OptimizationTask,
) -> OptimizationResult:
    """Relax one structure and return its id, positions, energy, and convergence."""
    conf_id, atoms = task
    atoms.calc = calculator
    optimizer = config.optimizer_cls(atoms, **config.optimizer_kwargs)
    converged = optimizer.run(fmax=config.fmax, steps=config.max_steps)
    return (
        conf_id,
        atoms.get_positions(),
        float(atoms.get_potential_energy()),
        bool(converged),
    )


def _run_worker_optimization(task: OptimizationTask) -> OptimizationResult:
    """Relax a structure with the current worker's resident calculator."""
    if _WORKER_CONFIG is None:  # pragma: no cover - initializer contract
        raise RuntimeError("Optimization worker was not initialized.")
    return run_optimization(_WORKER_CALC, _WORKER_CONFIG, task)


def _resolve_workers(n_workers: int | None, n_tasks: int) -> int:
    """Clamp a requested worker count to the available CPUs and task count."""
    if hasattr(os, "sched_getaffinity"):
        available = len(os.sched_getaffinity(0))
    else:
        available = os.cpu_count() or 1
    if n_workers is None or n_workers <= 0:
        n_workers = available
    return max(1, min(n_workers, n_tasks))


def _calculator_payload(calculator: Any, is_factory: bool) -> tuple[Any, bool]:
    """Prepare a calculator factory or independent instance for spawned workers."""
    if is_factory:
        return calculator, True
    try:
        return deepcopy(calculator), False
    except Exception as exc:
        raise RuntimeError(
            "Unable to deepcopy the ASE calculator for process-parallel execution. "
            "Pass a picklable `calculator` factory or set `num_workers=1`."
        ) from exc


def run_optimizations_in_processes(
    tasks: list[OptimizationTask],
    calculator: Any,
    calculator_is_factory: bool,
    config: OptimizationConfig,
    num_workers: int | None,
) -> list[OptimizationResult]:
    """Relax conformers in spawned, single-thread-pinned worker processes."""
    payload, is_factory = _calculator_payload(calculator, calculator_is_factory)
    workers = _resolve_workers(num_workers, len(tasks))
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(payload, is_factory, config),
    ) as pool:
        return list(pool.map(_run_worker_optimization, tasks))
