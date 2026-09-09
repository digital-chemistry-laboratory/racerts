from __future__ import annotations

from collections.abc import Callable as ABCCallable
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
import warnings

from rdkit import Chem
from rdkit.Geometry import Point3D

from .ff_optimizer import BaseOptimizer
from .parallel import (
    OptimizationConfig,
    OptimizationTask,
    run_optimization,
    run_optimizations_in_processes,
)

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
        num_threads: Optional[int] = 1,
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
        self.num_workers = num_workers

        if num_threads != 1:
            warnings.warn("Threads-based parallelism within ASEOptimizer is no longer supported and will be deprecated in future versions. Please set the number of parallel processes with num_workers."
            "For now, the num_workers will be inferred from num_threads, if num_workers is not set.")
            if self.num_workers == 1:
                self.num_workers = num_threads

        if "logfile" not in self.optimizer_kwargs and not self.verbose:
            self.optimizer_kwargs["logfile"] = None

    @staticmethod
    def _is_calculator_factory(calculator) -> bool:
        # Treat classes/callables passed via `calculator=...` as factories.
        return isinstance(calculator, type) or (
            isinstance(calculator, ABCCallable)
            and not hasattr(calculator, "get_property")
        )

    def _get_calculator(self):
        if self._calculator_is_factory:
            calculator = self.calculator()
            if calculator is None:
                raise ValueError("`calculator` callable returned None.")
            return calculator

        return self.calculator

    def optimize(
        self,
        mol: Chem.Mol,
        constraints: Optional[List[Any]] = None,
        conf_id: Optional[int] = None,
    ) -> int:
        """Optimize one conformer or the full ensemble in place.

        Full-ensemble calls use spawned processes when ``num_workers`` is ``None``
        or greater than one; otherwise conformers are optimized serially. A call
        with ``conf_id`` always optimizes that conformer serially.
        """
        conf_ids = (
            [conf_id]
            if conf_id is not None
            else [conformer.GetId() for conformer in mol.GetConformers()]
        )
        tasks: List[OptimizationTask] = []
        for task_conf_id in conf_ids:
            atoms = rdkit_conformer_to_ase_atoms(mol, conf_id=task_conf_id)
            if constraints:
                atoms.set_constraint(constraints)
            tasks.append((task_conf_id, atoms))

        if not tasks:
            return 0

        config = OptimizationConfig(
            optimizer_cls=self.optimizer_cls,
            optimizer_kwargs=dict(self.optimizer_kwargs),
            fmax=self.fmax,
            max_steps=self.max_steps,
        )
        use_processes = conf_id is None and (
            self.num_workers is None or self.num_workers > 1
        )
        if use_processes:
            results = run_optimizations_in_processes(
                tasks=tasks,
                calculator=self.calculator,
                calculator_is_factory=self._calculator_is_factory,
                config=config,
                num_workers=self.num_workers,
            )
        else:
            results = [
                run_optimization(self._get_calculator(), config, task) for task in tasks
            ]

        failures = 0
        for result_conf_id, positions, energy_ev, converged in results:
            conf = mol.GetConformer(result_conf_id)
            for idx, xyz in enumerate(positions):
                conf.SetAtomPosition(
                    idx, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2]))
                )
            conf.SetDoubleProp("energy", energy_ev * EV_TO_KCAL_MOL)
            failures += 0 if converged else 1
        return failures

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

        conformer_ids = [conf.GetId() for conf in mol.GetConformers()]
        for conf_id in conformer_ids:
            self.align_mols(
                mol=mol,
                reference=reference,
                align_indices=align_indices,
                conf_id=conf_id,
            )

        constraints = [FixAtoms(indices=align_indices)] if align_indices else None
        ase_failures = self.optimize(mol=mol, constraints=constraints)

        for conf_id in conformer_ids:
            self.align_mols(
                mol=mol,
                reference=reference,
                align_indices=align_indices,
                conf_id=conf_id,
            )

        if self.verbose:
            print(f"ASE failures: {ase_failures}")
