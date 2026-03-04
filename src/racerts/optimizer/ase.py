from __future__ import annotations

from collections.abc import Callable as ABCCallable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from functools import wraps
from importlib import import_module
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Type

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


@requires_dependency([Import(module="ase", item="Atoms")], globals())
def rdkit_conformer_to_ase_atoms(mol: Chem.Mol, conf_id: int) -> ASEAtoms:
    conf = mol.GetConformer(conf_id)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    return Atoms(symbols=symbols, positions=conf.GetPositions())


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
        num_threads: int = 1,
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

        if self.num_threads > 1:
            try:
                return deepcopy(self.calculator)
            except Exception as exc:
                raise RuntimeError(
                    "Unable to deepcopy the ASE calculator for threaded execution. "
                    "Use a callable `calculator` or set `num_threads=1`."
                ) from exc

        return self.calculator

    def optimize(
        self,
        mol: Chem.Mol,
        constraints: Optional[List[Any]] = None,
        conf_id: Optional[int] = None,
    ) -> int:
        if conf_id is None:
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

        if self.num_threads > 1:
            with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                ase_failures = sum(pool.map(_optimize, conformer_ids))
        else:
            ase_failures = sum(_optimize(conf_id) for conf_id in conformer_ids)

        if self.verbose:
            print(f"ASE failures: {ase_failures}")
