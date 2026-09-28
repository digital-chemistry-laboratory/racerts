from __future__ import annotations

from collections.abc import Callable as ABCCallable
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Type
import logging
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

logger = logging.getLogger(__name__)

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


def infer_charge_and_multiplicity(
    mol: Chem.Mol, charge: Optional[int] = None, multiplicity: Optional[int] = None
) -> Dict[str, int]:
    """
    Charge and spin multiplicity to use for mol.

    Given values come first, then the "charge" and "multiplicity" properties of mol
    (set by generate_conformers). Otherwise, the charge is the sum of the formal
    charges and the multiplicity the lowest one for the number of electrons (1 or 2).
    Radical electrons of the graph are not used: in a TS graph, they mostly stand for
    bonds that are forming or breaking.
    """
    if charge is None:
        charge = (
            mol.GetIntProp("charge")
            if mol.HasProp("charge")
            else Chem.GetFormalCharge(mol)
        )
    if multiplicity is None:
        multiplicity = (
            mol.GetIntProp("multiplicity")
            if mol.HasProp("multiplicity")
            else 1 + count_electrons(mol, charge) % 2
        )
    return {"charge": int(charge), "multiplicity": int(multiplicity)}


def count_electrons(mol: Chem.Mol, charge: int) -> int:
    """Number of electrons of mol (all atoms explicit) at the given charge."""
    return sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - charge


@requires_dependency([Import(module="ase", item="Atoms")], globals())
def rdkit_conformer_to_ase_atoms(
    mol: Chem.Mol,
    conf_id: int,
    multiplicity: Optional[int] = None,
    charge: Optional[int] = None,
) -> ASEAtoms:
    """
    Convert one conformer of an RDKit Mol to an ASE Atoms object.

    Charge and multiplicity (from infer_charge_and_multiplicity when None) are stored
    in atoms.info as charge, spin (the multiplicity, as read by UMA models) and
    multiplicity, and as initial charges and magnetic moments (read e.g. by tblite).
    """
    if multiplicity is None or charge is None:
        state = infer_charge_and_multiplicity(mol, charge, multiplicity)
        charge, multiplicity = state["charge"], state["multiplicity"]

    conf = mol.GetConformer(conf_id)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    # Codes such as tblite only read the totals; put what the formal charges do not
    # explain, and all unpaired electrons, on the first atom.
    charges = [float(atom.GetFormalCharge()) for atom in mol.GetAtoms()]
    charges[0] += charge - sum(charges)
    magmoms = [0.0] * mol.GetNumAtoms()
    magmoms[0] = float(multiplicity - 1)
    atoms = Atoms(
        symbols=symbols,
        positions=conf.GetPositions(),
        charges=charges,
        magmoms=magmoms,
    )
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
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
    ):
        """
        charge / multiplicity: override the values of the molecule (see
        infer_charge_and_multiplicity), e.g. multiplicity=3 for a triplet.
        """
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
        self.charge = charge
        self.multiplicity = multiplicity

        if num_threads != 1:
            warnings.warn(
                "Threads-based parallelism within ASEOptimizer is no longer supported and will be deprecated in future versions. "
                "Please set the number of parallel processes with num_workers."
                "For now, the num_workers will be inferred from num_threads, if num_workers is not set."
            )

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

        A conformer whose calculation fails (e.g. an SCF that does not converge) is
        left without an energy, so the pruners drop it. If all fail, RuntimeError is
        raised.
        """
        conf_ids = (
            [conf_id]
            if conf_id is not None
            else [conformer.GetId() for conformer in mol.GetConformers()]
        )
        state = infer_charge_and_multiplicity(mol, self.charge, self.multiplicity)
        tasks: List[OptimizationTask] = []
        for task_conf_id in conf_ids:
            atoms = rdkit_conformer_to_ase_atoms(mol, conf_id=task_conf_id, **state)
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
        errors = {}
        for result_conf_id, positions, energy_ev, converged, error in results:
            conf = mol.GetConformer(result_conf_id)
            if error is not None:
                errors[result_conf_id] = error
                conf.ClearProp("energy")  # no stale energy from an earlier step
                failures += 1
                continue
            for idx, xyz in enumerate(positions):
                conf.SetAtomPosition(
                    idx, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2]))
                )
            conf.SetDoubleProp("energy", energy_ev * EV_TO_KCAL_MOL)
            failures += 0 if converged else 1

        if errors:
            first_error = next(iter(errors.values()))
            if len(errors) == len(results):
                raise RuntimeError(
                    f"ASE optimization failed for all {len(results)} conformers: "
                    f"{first_error}"
                )
            logger.warning(
                "ASE optimization failed for %d of %d conformers %s; they are left "
                "without an energy. First error: %s",
                len(errors),
                len(results),
                sorted(errors),
                first_error,
            )
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
