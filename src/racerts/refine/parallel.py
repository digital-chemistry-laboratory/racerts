"""Relaxation of ASE structures, one by one or in spawned worker processes."""

from __future__ import annotations

import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

# (conf_id, ase.Atoms)
OptimizationTask = tuple[int, Any]
# (conf_id, positions, energy, converged, error); positions and energy are None on error.
OptimizationResult = tuple[int, Any, "float | None", bool, "str | None"]


@dataclass(frozen=True)
class OptimizationConfig:
    """
    Optimizer settings shared by every conformer.

    prepare(calculator, reference_atoms), if given, is called once for every
    calculator instance before its first structure, e.g. to warm a calculator that
    perceives a topology from the first geometry it sees (GFN-FF) on the reference.
    For worker processes it must be picklable (a module-level function).
    """

    optimizer_cls: type
    optimizer_kwargs: dict[str, Any] = field(default_factory=dict)
    fmax: float = 0.05
    max_steps: int = 100
    prepare: Optional[Callable[[Any, Any], None]] = None


@dataclass(frozen=True)
class Outcome:
    """The result of one structure; positions and energy (eV) are None on error."""

    conf_id: int
    positions: Any = None
    energy: Optional[float] = None
    converged: bool = False
    error: Optional[str] = None
    n_steps: int = 0
    seconds: float = 0.0

    def as_tuple(self) -> OptimizationResult:
        return self.conf_id, self.positions, self.energy, self.converged, self.error


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


def _new_calculator(payload: Any, is_factory: bool) -> Any:
    if not is_factory:
        return payload
    calculator = payload()
    if calculator is None:
        raise ValueError("`calculator` callable returned None.")
    return calculator


def _prepared(calculator: Any, config: OptimizationConfig, reference: Any) -> Any:
    if config.prepare is not None and reference is not None:
        config.prepare(calculator, reference.copy())
    return calculator


def _init_worker(
    calculator_payload: Any,
    is_factory: bool,
    config: OptimizationConfig,
    reference: Any = None,
) -> None:
    """Pin a pool worker and initialize its resident calculator and settings."""
    global _WORKER_CALC, _WORKER_CONFIG
    pin_to_single_thread()
    calculator = _new_calculator(calculator_payload, is_factory)
    _WORKER_CALC = _prepared(calculator, config, reference)
    _WORKER_CONFIG = config


def optimize_one(
    calculator: Any, config: OptimizationConfig, task: OptimizationTask
) -> Outcome:
    """
    Relax one structure. With max_steps=0, only the energy is computed (a single
    point). A failing calculation (e.g. an SCF that does not converge) is reported as
    an error instead of raising, so that it only costs this conformer and not the
    whole ensemble.
    """
    conf_id, atoms = task
    start = time.perf_counter()
    n_steps = 0
    try:
        atoms.calc = calculator
        converged = True
        # ASE optimizers ask for forces even without steps, and ASE < 3.23 treats
        # steps=0 as no limit.
        if config.max_steps > 0:
            optimizer = config.optimizer_cls(atoms, **config.optimizer_kwargs)
            converged = optimizer.run(fmax=config.fmax, steps=config.max_steps)
            n_steps = int(getattr(optimizer, "nsteps", 0))
        energy = float(atoms.get_potential_energy())
    except Exception as exc:
        return Outcome(
            conf_id,
            error=f"{type(exc).__name__}: {exc}",
            n_steps=n_steps,
            seconds=time.perf_counter() - start,
        )
    return Outcome(
        conf_id,
        positions=atoms.get_positions(),
        energy=energy,
        converged=bool(converged),
        n_steps=n_steps,
        seconds=time.perf_counter() - start,
    )


def run_optimization(
    calculator: Any,
    config: OptimizationConfig,
    task: OptimizationTask,
) -> OptimizationResult:
    """Relax one structure and return its id, positions, energy, convergence and error.

    See optimize_one, which also returns the number of steps and the time.
    """
    return optimize_one(calculator, config, task).as_tuple()


def _run_worker_optimization(task: OptimizationTask) -> Outcome:
    """Relax a structure with the current worker's resident calculator."""
    if _WORKER_CONFIG is None:  # pragma: no cover - initializer contract
        raise RuntimeError("Optimization worker was not initialized.")
    return optimize_one(_WORKER_CALC, _WORKER_CONFIG, task)


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


def optimize_all(
    tasks: list[OptimizationTask],
    calculator: Any,
    calculator_is_factory: bool,
    config: OptimizationConfig,
    num_workers: int | None = 1,
    reference: Any = None,
) -> list[Outcome]:
    """
    Relax all structures: one after the other (a factory makes a new calculator for
    every structure, as in legacy racerts), or, for num_workers None or > 1, in
    spawned single-thread worker processes with one calculator each. reference: the
    Atoms for config.prepare.
    """
    if num_workers is not None and num_workers <= 1:
        shared = None
        if not calculator_is_factory:
            shared = _prepared(calculator, config, reference)
        outcomes = []
        for task in tasks:
            calc = shared
            if calc is None:
                calc = _prepared(_new_calculator(calculator, True), config, reference)
            outcomes.append(optimize_one(calc, config, task))
        return outcomes
    return _in_processes(
        tasks, calculator, calculator_is_factory, config, num_workers, reference
    )


def _in_processes(
    tasks, calculator, calculator_is_factory, config, num_workers, reference=None
) -> list[Outcome]:
    payload, is_factory = _calculator_payload(calculator, calculator_is_factory)
    workers = _resolve_workers(num_workers, len(tasks))
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=ctx,
        initializer=_init_worker,
        initargs=(payload, is_factory, config, reference),
    ) as pool:
        return list(pool.map(_run_worker_optimization, tasks))


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
