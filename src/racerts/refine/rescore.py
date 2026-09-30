"""The Rescore stage: energies from single points of another method."""

import logging
import math
from functools import partial
from typing import Any, Callable, List, Optional, Sequence

from racerts.io.ase import rdkit_conformer_to_ase_atoms
from racerts.pipeline import ConformerEnsemble
from racerts.system.spec import infer_charge_and_multiplicity
from racerts.utils.optional import require
from racerts.utils.units import EV_TO_KCAL_MOL

from .ase import ASEOptimizer
from .parallel import OptimizationConfig, optimize_all

logger = logging.getLogger(__name__)

ON_FAIL = ("clear", "drop")


class Rescore:
    """
    Replaces the energies of the conformers by single points of an ASE calculator
    (e.g. GFN2-xTB with tblite, or a machine-learned potential), or of a batch
    function; the geometries stay as they are.

    Args:
        calculator: An ASE calculator, or a callable that returns one (see
            ASEOptimizer).
        batch: Instead of a calculator: batch(list of ASE Atoms) -> energies in eV,
            e.g. a model that evaluates many structures at once on a GPU; None or NaN
            for a structure that failed.
        method: The name of the energies ("energy_method"); default: the class name
            of the calculator, or of the batch function.
        on_fail: For conformers whose calculation fails: "clear" leaves them without
            an energy (the pruners then drop them), "drop" removes them.
        num_workers, charge, multiplicity, prepare: As in ASEOptimizer (not used with
            batch, except charge and multiplicity).

    The energy that is replaced, and its method, go into the provenance of each
    conformer ("previous_energy", "previous_energy_method"). If no conformer gets an
    energy, RuntimeError is raised.
    """

    name = "rescore"

    def __init__(
        self,
        calculator: Any = None,
        batch: Optional[Callable[[List[Any]], Sequence[Optional[float]]]] = None,
        method: Optional[str] = None,
        on_fail: str = "clear",
        num_workers: Optional[int] = 1,
        charge: Optional[int] = None,
        multiplicity: Optional[int] = None,
        prepare: Optional[Callable[[Any, Any], None]] = None,
    ):
        if (calculator is None) == (batch is None):
            raise ValueError("Give either a calculator or a batch function.")
        if on_fail not in ON_FAIL:
            raise ValueError(f"on_fail must be one of {ON_FAIL}.")
        if calculator is not None:
            # Checks the calculator as ASEOptimizer does; also the conversion to Atoms.
            self._single_point = ASEOptimizer(
                calculator=calculator,
                max_steps=0,
                num_workers=num_workers,
                charge=charge,
                multiplicity=multiplicity,
                prepare=prepare,
            )
        self.calculator = calculator
        self.batch = batch
        self.method = method or _name(calculator if batch is None else batch)
        self.on_fail = on_fail
        self.charge = charge
        self.multiplicity = multiplicity

    def run(self, ctx, ensemble: ConformerEnsemble) -> ConformerEnsemble:
        mol = ensemble.mol
        conf_ids = ensemble.conf_ids
        if not conf_ids:
            raise ValueError("Rescore got an ensemble without conformers.")
        previous_method = ensemble.energy_method
        for conf_id in conf_ids:
            energy = ensemble.energy(conf_id)
            if energy is not None:
                ensemble.add_provenance(
                    conf_id,
                    previous_energy=energy,
                    previous_energy_method=previous_method,
                )

        energies, errors = self._energies(ctx, mol, conf_ids)
        failed = []
        for conf_id, energy in zip(conf_ids, energies):
            conf = mol.GetConformer(conf_id)
            if energy is None or not math.isfinite(energy):
                conf.ClearProp("energy")
                failed.append(conf_id)
            else:
                conf.SetDoubleProp("energy", energy * EV_TO_KCAL_MOL)
        if len(failed) == len(conf_ids):
            detail = f": {errors[0]}" if errors else ""
            raise RuntimeError(
                f"Rescoring failed for all {len(conf_ids)} conformers{detail}"
            )
        if failed:
            logger.warning(
                "Rescoring failed for %d of %d conformers %s; they are %s.%s",
                len(failed),
                len(conf_ids),
                failed,
                "removed" if self.on_fail == "drop" else "left without an energy",
                f" First error: {errors[0]}" if errors else "",
            )
            if self.on_fail == "drop":
                for conf_id in failed:
                    mol.RemoveConformer(conf_id)
        mol.SetProp("energy_method", self.method)
        return ensemble

    def _energies(self, ctx, mol, conf_ids):
        """Energies in eV (None where the calculation failed) and the errors."""
        state = infer_charge_and_multiplicity(mol, self.charge, self.multiplicity)
        if self.batch is not None:
            atoms = [
                rdkit_conformer_to_ase_atoms(mol, conf_id, **state)
                for conf_id in conf_ids
            ]
            energies = list(self.batch(atoms))
            if len(energies) != len(atoms):
                raise ValueError(
                    f"The batch function returned {len(energies)} energies for "
                    f"{len(atoms)} structures."
                )
            return [None if e is None else float(e) for e in energies], []

        single_point = self._single_point
        tasks = [
            (conf_id, single_point._to_atoms(mol, conf_id, state))
            for conf_id in conf_ids
        ]
        reference = None
        if single_point.prepare is not None:
            reference = tasks[0][1].copy()
            if ctx.reference is not None:
                ref = ctx.reference
                reference = single_point._to_atoms(
                    ref, ref.GetConformer().GetId(), state
                )
        config = OptimizationConfig(
            optimizer_cls=require("ase.optimize", "ase").BFGS,  # not used: 0 steps
            max_steps=0,
            prepare=single_point.prepare,
        )
        outcomes = optimize_all(
            tasks,
            single_point.calculator,
            single_point._calculator_is_factory,
            config,
            num_workers=single_point.num_workers,
            reference=reference,
        )
        errors = [o.error for o in outcomes if o.error is not None]
        return [o.energy for o in outcomes], errors


def _name(calculator) -> str:
    """Class name of a calculator instance, class, function or partial of either."""
    while isinstance(calculator, partial):
        calculator = calculator.func
    if isinstance(calculator, type):
        return calculator.__name__
    if hasattr(calculator, "get_property"):
        return type(calculator).__name__
    return getattr(calculator, "__name__", type(calculator).__name__)
