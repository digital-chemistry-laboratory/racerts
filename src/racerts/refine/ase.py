"""Refinement with any ASE calculator (xTB, MLIPs, ...), optionally in processes."""

from __future__ import annotations

import logging
import warnings
from typing import Any, Callable, Dict, List, Optional, Sequence, Type

from rdkit import Chem

from racerts.io import ase as ase_io
from racerts.pipeline.ensemble import ConformerEnsemble
from racerts.system.spec import infer_charge_and_multiplicity
from racerts.utils.optional import require
from racerts.utils.units import EV_TO_KCAL_MOL

from .base import BaseOptimizer
from .parallel import OptimizationConfig, OptimizationTask, optimize_all

logger = logging.getLogger(__name__)


class ASEOptimizer(BaseOptimizer):
    """
    Refinement with any ASE calculator (xTB, MLIPs, ...), with the anchor atoms fixed.

    Args:
        calculator: An ASE calculator, or a callable that returns one (a factory: one
            calculator per worker process, or per call without workers; the way to
            use calculators that keep state between structures).
        optimizer_cls: Any class with the ASE optimizer interface,
            cls(atoms, **optimizer_kwargs).run(fmax=..., steps=...); default BFGS.
            E.g. Sella, with optimizer_kwargs={"order": 1} for saddle points (then
            refine without anchors: Refine(ASEOptimizer(...), anchors=False)).
        fmax, max_steps: Convergence criterion (eV/A) and step limit; max_steps=0
            gives single points.
        num_workers: Worker processes; 1 (default): none; None: all CPUs.
        charge, multiplicity: Override the values of the molecule (see
            infer_charge_and_multiplicity), e.g. multiplicity=3 for a triplet.
        drop_unconverged: Remove conformers whose optimization did not converge
            within max_steps (default: keep them).
        prepare: prepare(calculator, reference_atoms), called once for every
            calculator instance before its first conformer, with the reference
            geometry (or the first conformer), e.g. to warm GFN-FF on the TS topology.
            Must be picklable for worker processes.

    Every conformer records "converged", "n_steps" and "wall_time" in its provenance.
    """

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
        drop_unconverged: bool = False,
        prepare: Optional[Callable[[Any, Any], None]] = None,
    ):
        if calculator is None:
            raise ValueError("`calculator` must be provided.")

        self.calculator = calculator
        self._calculator_is_factory = ase_io.check_calculator(calculator)
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
        self.drop_unconverged = drop_unconverged
        self.prepare = prepare

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

    def optimize(
        self,
        mol: Chem.Mol,
        constraints: Optional[List[Any]] = None,
        conf_id: Optional[int] = None,
        reference: Optional[Chem.Mol] = None,
    ) -> int:
        """Optimize one conformer or the full ensemble in place.

        Full-ensemble calls use spawned processes when ``num_workers`` is ``None``
        or greater than one; otherwise conformers are optimized serially. A call
        with ``conf_id`` always optimizes that conformer serially. reference: the
        geometry passed to prepare (default: the first conformer).

        A conformer whose calculation fails (e.g. an SCF that does not converge) is
        left without an energy, so the pruners drop it. If all fail, RuntimeError is
        raised. Returns the number of conformers that failed or did not converge.
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
            prepare=self.prepare,
        )
        reference_atoms = None
        if self.prepare is not None:
            if reference is not None:
                ref_id = reference.GetConformer(self.conf_id_ref).GetId()
                reference_atoms = self._to_atoms(reference, ref_id, state)
            else:  # a copy: the first conformer is relaxed in place
                reference_atoms = tasks[0][1].copy()
        use_processes = conf_id is None and (
            self.num_workers is None or self.num_workers > 1
        )
        outcomes = optimize_all(
            tasks,
            self.calculator,
            self._calculator_is_factory,
            config,
            num_workers=self.num_workers if use_processes else 1,
            reference=reference_atoms,
        )

        ensemble = ConformerEnsemble(mol)  # a view, to record the provenance
        failures = 0
        errors = {}
        unconverged = []
        for outcome in outcomes:
            conf = mol.GetConformer(outcome.conf_id)
            ensemble.add_provenance(
                outcome.conf_id,
                converged=outcome.converged,
                n_steps=outcome.n_steps,
                wall_time=round(outcome.seconds, 3),
            )
            if outcome.error is not None:
                errors[outcome.conf_id] = outcome.error
                conf.ClearProp("energy")  # no stale energy from an earlier step
                failures += 1
                continue
            ase_io.set_positions(conf, outcome.positions)
            conf.SetDoubleProp("energy", outcome.energy * EV_TO_KCAL_MOL)
            if not outcome.converged:
                failures += 1
                unconverged.append(outcome.conf_id)

        if errors:
            first_error = next(iter(errors.values()))
            if len(errors) == len(outcomes):
                raise RuntimeError(
                    f"ASE optimization failed for all {len(outcomes)} conformers: "
                    f"{first_error}"
                )
            logger.warning(
                "ASE optimization failed for %d of %d conformers %s; they are left "
                "without an energy. First error: %s",
                len(errors),
                len(outcomes),
                sorted(errors),
                first_error,
            )
        if unconverged and self.drop_unconverged:
            if len(unconverged) + len(errors) == len(outcomes):
                raise RuntimeError(
                    f"No conformer converged within {self.max_steps} steps."
                )
            logger.warning(
                "Dropping %d conformers that did not converge within %d steps: %s",
                len(unconverged),
                self.max_steps,
                unconverged,
            )
            for unconverged_id in unconverged:
                mol.RemoveConformer(unconverged_id)
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
        failures = self.optimize(mol=mol, constraints=constraints, reference=reference)
        self.align_mols(mol, reference, anchors)
        logger.info("ASE failures: %d", failures)
        return failures
