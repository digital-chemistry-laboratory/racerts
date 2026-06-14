from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable as ABCCallable
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass, field
from functools import wraps
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Type

from rdkit import Chem
from rdkit.Geometry import Point3D

from .ff_optimizer import BaseOptimizer

if TYPE_CHECKING:
    from ase import Atoms as ASEAtoms

Atoms: Any = None
FixAtoms: Any = None
BFGS: Any = None


@dataclass(frozen=True)
class Import:
    module: str
    item: Optional[str] = None
    alias: Optional[str] = None


def requires_dependency(imports: List[Import], scope: Dict[str, Any]):
    def _decorator(func):
        @wraps(func)
        def _wrapper(*args, **kwargs):
            try:
                for imp in imports:
                    symbol_name = imp.alias or imp.item or imp.module.rsplit(".", 1)[-1]
                    if scope.get(symbol_name) is not None:
                        continue
                    module = import_module(imp.module)
                    symbol = getattr(module, imp.item) if imp.item else module
                    scope[symbol_name] = symbol
            except Exception as exc:  # pragma: no cover - import path dependent
                raise ImportError(
                    "ASE is required for ASEOptimizer. Install with `pip install racerts[ase]`."
                ) from exc
            return func(*args, **kwargs)

        return _wrapper

    return _decorator


EV_TO_KCAL_MOL = 23.06054783061903


def infer_charge_and_multiplicity(mol: Chem.Mol) -> Dict[str, int]:
    """
    Infer the total formal charge and spin multiplicity of an RDKit molecule.

    Args:
        mol (Chem.Mol): The molecule to inspect.

    Returns:
        dict: A dict with charge (sum of per-atom formal charges) and multiplicity
            (2S + 1 = n_radical_electrons + 1).
    """
    charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
    multiplicity = 1 + sum(a.GetNumRadicalElectrons() for a in mol.GetAtoms())
    return {"charge": charge, "multiplicity": multiplicity}


@requires_dependency([Import(module="ase", item="Atoms")], globals())
def rdkit_conformer_to_ase_atoms(
    mol: Chem.Mol,
    conf_id: int,
    multiplicity: Optional[int] = None,
    charge: Optional[int] = None,
) -> ASEAtoms:
    """
    Convert one conformer of an RDKit Mol to an ASE Atoms object.

    Charge and spin multiplicity are stored on atoms.info (charge, spin and multiplicity,
    where spin is the multiplicity expected by UMA models) so that downstream calculators
    treat charged/radical species correctly. When either is None it is inferred from mol
    via infer_charge_and_multiplicity.

    Args:
        mol (Chem.Mol): The molecule with at least one embedded conformer.
        conf_id (int): The conformer to read.
        multiplicity (int): The spin multiplicity. Inferred from mol when None.
        charge (int): The total formal charge. Inferred from mol when None.

    Returns:
        ASEAtoms: The conformer geometry, with charge and spin set on atoms.info.
    """
    if multiplicity is None or charge is None:
        inferred = infer_charge_and_multiplicity(mol)
        if multiplicity is None:
            multiplicity = inferred["multiplicity"]
        if charge is None:
            charge = inferred["charge"]

    conf = mol.GetConformer(conf_id)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    atoms = Atoms(symbols=symbols, positions=conf.GetPositions())
    atoms.info.update(
        {
            "charge": int(charge),
            "spin": int(multiplicity),  # for uma models
            "multiplicity": int(multiplicity),
        }
    )
    return atoms


def write_ase_positions_to_rdkit(atoms: ASEAtoms, mol: Chem.Mol, conf_id: int) -> None:
    conf = mol.GetConformer(conf_id)
    positions = atoms.get_positions()
    for idx, xyz in enumerate(positions):
        conf.SetAtomPosition(
            idx,
            Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])),
        )


# Process-based parallel optimization.
#
# Each conformer is relaxed in its own spawned, single-thread-pinned worker process (many
# ASE calculators are not thread-safe, so processes are used instead of threads). Atoms
# objects carrying only geometry and constraints are shipped keyed by conformer id, and
# results are echoed back with their id so the parent can write the optimized positions
# and energies onto the RDKit Mol.


@dataclass(frozen=True)
class _RunConfig:
    """
    Optimizer settings shared by every conformer (pickled once per worker).
    """

    optimizer_cls: Type
    optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)
    fmax: float = 0.05
    max_steps: int = 100


# Per-worker calculator/config, set once in the pool initializer and reused across tasks.
_WORKER_CALC: Any = None
_WORKER_CONFIG: Optional[_RunConfig] = None
# Holds the threadpoolctl controller so the single-thread limit lives for the worker's
# whole life (and is never restored).
_PIN: Any = None


def pin_to_single_thread() -> None:
    """
    Pin the calling process to a single native thread across all thread pools.

    Caps OpenMP (libgomp) and BLAS/LAPACK (OpenBLAS/MKL) to one thread via threadpoolctl.
    This is required per dedicated worker: xTB/GFN-FF anti-scale with thread count on
    small systems, so N concurrent workers must not each spawn N x (OpenMP + BLAS)
    threads and thrash the node.
    """
    global _PIN
    from threadpoolctl import threadpool_limits

    _PIN = threadpool_limits(limits=1)


def _init_worker(calculator_payload: Any, is_factory: bool, config: _RunConfig) -> None:
    """
    Pool initializer: pin to one thread, then build this worker's calculator.

    The calculator_payload is either a factory (called here) or a deepcopied calculator
    instance (delivered, and unpickled, per worker via initargs), giving each worker its
    own independent calculator.

    Args:
        calculator_payload (Any): A calculator factory or a deepcopied calculator
            instance.
        is_factory (bool): Whether calculator_payload should be called to build the
            calculator.
        config (_RunConfig): The shared optimizer settings.
    """
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


def _optimize_one(
    calc: Any, config: _RunConfig, conf_id: int, atoms: ASEAtoms
) -> Tuple[int, Any, float, bool]:
    """
    Relax one structure and return (conf_id, positions, energy_eV, converged).
    """
    atoms = atoms.copy()  # never mutate the caller's shipped geometry
    atoms.calc = calc
    optimizer = config.optimizer_cls(atoms, **config.optimizer_kwargs)
    converged = optimizer.run(fmax=config.fmax, steps=config.max_steps)
    return (
        conf_id,
        atoms.get_positions(),
        float(atoms.get_potential_energy()),
        bool(converged),
    )


def _optimize_task(task: Tuple[int, ASEAtoms]) -> Tuple[int, Any, float, bool]:
    """
    Picklable pool entry point: relax with this worker's resident calculator.
    """
    conf_id, atoms = task
    return _optimize_one(_WORKER_CALC, _WORKER_CONFIG, conf_id, atoms)


def _resolve_workers(n_workers: Optional[int], n_tasks: int) -> int:
    """
    Clamp the requested worker count to the cores available and tasks on hand.

    A value of None or <= 0 auto-sizes to the number of CPUs this process may run on (the
    SLURM allocation, via sched_getaffinity when present), capped at one worker per task.

    Args:
        n_workers (int): The requested worker count, or None to auto-size.
        n_tasks (int): The number of tasks to distribute.

    Returns:
        int: The resolved worker count.
    """
    if hasattr(os, "sched_getaffinity"):
        available = len(os.sched_getaffinity(0))
    else:
        available = os.cpu_count() or 1
    if n_workers is None or n_workers <= 0:
        n_workers = available
    return max(1, min(n_workers, n_tasks))


class ASEOptimizer(BaseOptimizer):
    @requires_dependency([Import(module="ase.optimize", item="BFGS")], globals())
    def __init__(
        self,
        calculator=None,
        optimizer_cls: Optional[Type] = None,
        optimizer_kwargs: Optional[Dict] = None,
        fmax: float = 0.05,
        max_steps: int = 100,
        verbose: bool = False,
        conf_id_ref: int = -1,
        force_constant: float = 1e6,
        num_threads: int = 1,
        num_workers: Optional[int] = 1,
    ):
        if calculator is None:
            raise ValueError("`calculator` must be provided.")

        self.calculator = calculator
        self._calculator_is_factory = self._is_calculator_factory(calculator)
        if not self._calculator_is_factory and not hasattr(calculator, "get_property"):
            raise ValueError(
                "`calculator` must be an ASE calculator instance or a callable "
                "returning one."
            )
        self.optimizer_cls = optimizer_cls if optimizer_cls is not None else BFGS
        self.optimizer_kwargs = dict(optimizer_kwargs or {})
        self.fmax = fmax
        self.max_steps = max_steps
        self.verbose = verbose
        self.conf_id_ref = conf_id_ref
        self.force_constant = force_constant
        self.num_threads = num_threads
        self.num_workers = num_workers

        if "logfile" not in self.optimizer_kwargs and not self.verbose:
            self.optimizer_kwargs["logfile"] = None

    @staticmethod
    def _is_calculator_factory(calculator) -> bool:
        # Treat classes/callables passed via `calculator=...` as factories.
        return isinstance(calculator, type) or (
            isinstance(calculator, ABCCallable)
            and not hasattr(calculator, "get_property")
        )

    def _use_processes(self) -> bool:
        """
        Whether to run the process pool rather than serially.

        num_workers of None means auto-size to all available cores (resolved by
        _resolve_workers), matching the convention of catmlp's relax_conformers; a
        value greater than 1 requests that many workers; 0 or 1 runs serially.

        Returns:
            bool: True if the process pool should be used.
        """
        return self.num_workers is None or self.num_workers > 1

    def _get_calculator(self):
        if self._calculator_is_factory:
            calculator = self.calculator()
            if calculator is None:
                raise ValueError("`calculator` callable returned None.")
            return calculator

        if self.num_threads > 1:
            try:
                return deepcopy(self.calculator)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to deepcopy the ASE calculator for threaded execution. "
                    "Use a callable `calculator` or set `num_threads=1`."
                ) from exc

        return self.calculator

    def _calculator_payload(self) -> Tuple[Any, bool]:
        """
        Return (payload, is_factory) to ship to worker processes.

        A factory is shipped as-is (each worker calls it); a calculator instance is
        deepcopied so that each worker unpickles its own independent copy across the
        spawn boundary.

        Returns:
            tuple: The calculator payload and a flag for whether it is a factory.
        """
        if self._calculator_is_factory:
            return self.calculator, True
        try:
            return deepcopy(self.calculator), False
        except Exception as exc:
            raise RuntimeError(
                "Unable to deepcopy the ASE calculator for process-parallel execution. "
                "Pass a picklable `calculator` factory or set `num_workers=1`."
            ) from exc

    def _optimize_parallel(
        self,
        mol: Chem.Mol,
        constraints: Optional[List[Any]],
    ) -> int:
        """
        Relax every conformer across a spawned process pool and write results back.

        Geometry (with constraints) is shipped per conformer; the parent writes the
        optimized positions and energy onto mol. This matches the serial optimize
        contract.

        Args:
            mol (Chem.Mol): The molecule whose conformers are optimized in place.
            constraints (list): The ASE constraints applied to every conformer.

        Returns:
            int: The total number of conformers that failed to converge.
        """
        tasks: List[Tuple[int, ASEAtoms]] = []
        for conformer in mol.GetConformers():
            conf_id = conformer.GetId()
            atoms = rdkit_conformer_to_ase_atoms(mol, conf_id=conf_id)
            if constraints:
                atoms.set_constraint(constraints)
            tasks.append((conf_id, atoms))

        if not tasks:
            return 0

        config = _RunConfig(
            optimizer_cls=self.optimizer_cls,
            optimizer_kwargs=dict(self.optimizer_kwargs),
            fmax=self.fmax,
            max_steps=self.max_steps,
        )
        payload, is_factory = self._calculator_payload()
        workers = _resolve_workers(self.num_workers, len(tasks))
        ctx = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers,
            mp_context=ctx,
            initializer=_init_worker,
            initargs=(payload, is_factory, config),
        ) as pool:
            results = list(pool.map(_optimize_task, tasks))

        failures = 0
        for conf_id, positions, energy_ev, converged in results:
            conf = mol.GetConformer(conf_id)
            for idx, xyz in enumerate(positions):
                conf.SetAtomPosition(
                    idx, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2]))
                )
            conf.SetDoubleProp("energy", energy_ev * EV_TO_KCAL_MOL)
            failures += 0 if converged else 1
        return failures

    def optimize(
        self,
        mol: Chem.Mol,
        constraints: Optional[List[Any]] = None,
        conf_id: Optional[int] = None,
    ) -> int:
        if conf_id is None:
            if self._use_processes():
                return self._optimize_parallel(mol=mol, constraints=constraints)
            return sum(
                self.optimize(
                    mol=mol,
                    constraints=constraints,
                    conf_id=conformer.GetId(),
                )
                for conformer in mol.GetConformers()
            )

        atoms = rdkit_conformer_to_ase_atoms(mol, conf_id=conf_id)
        if constraints:
            atoms.set_constraint(constraints)

        atoms.calc = self._get_calculator()
        ase_optimizer = self.optimizer_cls(atoms, **self.optimizer_kwargs)
        converged = ase_optimizer.run(fmax=self.fmax, steps=self.max_steps)

        write_ase_positions_to_rdkit(atoms, mol=mol, conf_id=conf_id)
        energy_kcal_mol = atoms.get_potential_energy() * EV_TO_KCAL_MOL
        mol.GetConformer(conf_id).SetDoubleProp("energy", energy_kcal_mol)

        return 0 if bool(converged) else 1

    @requires_dependency([Import(module="ase.constraints", item="FixAtoms")], globals())
    def tune_ts_conformers(
        self,
        mol: Chem.Mol,
        reference: Chem.Mol,
        align_indices: Optional[List[int]] = None,
    ):
        align_indices = [] if align_indices is None else list(align_indices)

        if self.conf_id_ref == -1:
            self.conf_id_ref = reference.GetConformer().GetId()

        def _optimize(conf_id: int) -> int:
            self.align_mols(
                mol=mol,
                reference=reference,
                align_indices=align_indices,
                conf_id=conf_id,
            )

            constraints = None
            if align_indices:
                constraints = [FixAtoms(indices=align_indices)]

            local_fail = self.optimize(
                mol=mol,
                constraints=constraints,
                conf_id=conf_id,
            )

            self.align_mols(
                mol=mol,
                reference=reference,
                align_indices=align_indices,
                conf_id=conf_id,
            )

            return local_fail

        conformer_ids = [conf.GetId() for conf in mol.GetConformers()]

        if self._use_processes():
            # Process-parallel path overrides the thread path. Alignment is cheap RDKit
            # work and each conformer is independent, so align all in this process, relax
            # them across the pool in one shot, then re-align. This is equivalent to the
            # per-conformer align, optimize, align loop below.
            constraints = [FixAtoms(indices=align_indices)] if align_indices else None
            for conf_id in conformer_ids:
                self.align_mols(
                    mol=mol,
                    reference=reference,
                    align_indices=align_indices,
                    conf_id=conf_id,
                )
            ase_failures = self.optimize(mol=mol, constraints=constraints)
            for conf_id in conformer_ids:
                self.align_mols(
                    mol=mol,
                    reference=reference,
                    align_indices=align_indices,
                    conf_id=conf_id,
                )
        elif self.num_threads > 1:
            with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                ase_failures = sum(pool.map(_optimize, conformer_ids))
        else:
            ase_failures = sum(_optimize(conf_id) for conf_id in conformer_ids)

        if self.verbose:
            print(f"ASE failures: {ase_failures}")
