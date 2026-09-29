"""Refinement with any ASE calculator (xTB, MLIPs, ...), optionally in processes."""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable as ABCCallable
from typing import Any, Dict, List, Optional, Sequence, Type

from rdkit import Chem

from racerts.io import ase as ase_io
from racerts.system.spec import infer_charge_and_multiplicity
from racerts.utils.optional import require
from racerts.utils.units import EV_TO_KCAL_MOL

from .base import BaseOptimizer
from .parallel import (
    OptimizationConfig,
    OptimizationTask,
    run_optimization,
    run_optimizations_in_processes,
)

logger = logging.getLogger(__name__)


class ASEOptimizer(BaseOptimizer):
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
        if optimizer_cls is None:
            optimizer_cls = require("ase.optimize", "ase").BFGS
        self.optimizer_cls = optimizer_cls
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
            atoms = self._to_atoms(mol, task_conf_id, state)
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
            ase_io.set_positions(conf, positions)
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

    def _to_atoms(self, mol: Chem.Mol, conf_id: int, state: Dict[str, int]):
        """
        The conformer as ASE Atoms with its charge and multiplicity (state). Looked up
        at call time: patching racerts.io.ase.rdkit_conformer_to_ase_atoms, or
        overriding this method, changes the conversion.
        """
        return ase_io.rdkit_conformer_to_ase_atoms(mol, conf_id=conf_id, **state)

    def _refine(
        self, mol: Chem.Mol, reference: Optional[Chem.Mol], anchors: Sequence[int]
    ) -> int:
        """
        Optimize all conformers, with the anchors fixed (FixAtoms) after aligning the
        conformers on them to the reference. Returns the number of conformers that
        failed or did not converge.
        """
        anchors = list(anchors)
        self.align_mols(mol, reference, anchors)
        constraints = None
        if anchors:
            constraints = [require("ase.constraints", "ase").FixAtoms(indices=anchors)]
        failures = self.optimize(mol=mol, constraints=constraints)
        self.align_mols(mol, reference, anchors)
        logger.info("ASE failures: %d", failures)
        return failures
