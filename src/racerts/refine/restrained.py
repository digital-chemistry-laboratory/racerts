"""Restraints for ASE calculators: flat-bottom terms added to any calculator."""

from typing import Iterable, Tuple

import numpy as np
from ase.calculators.calculator import Calculator, all_changes

from racerts.restraints.model import DistanceRestraint, PositionRestraint
from racerts.utils.units import EV_TO_KCAL_MOL


def restraint_terms(positions, restraints) -> Tuple[float, np.ndarray]:
    """
    Energy (kcal/mol) and forces (kcal/(mol A)) of flat-bottom restraints at positions
    (A), in RDKit's convention: E = 1/2 k (d - bound)^2 outside the window of a
    DistanceRestraint, and 1/2 k (d - tolerance)^2 beyond the tolerance of a
    PositionRestraint (d the distance from its point).
    """
    positions = np.asarray(positions, dtype=float)
    energy = 0.0
    forces = np.zeros_like(positions)
    for r in restraints:
        if isinstance(r, PositionRestraint):
            delta = positions[r.atom] - np.asarray(r.point)
            d = float(np.linalg.norm(delta))
            excess = d - r.tolerance
            if excess <= 0:
                continue
            energy += 0.5 * r.force_constant * excess**2
            forces[r.atom] -= r.force_constant * excess * delta / d
            continue
        i, j = r.pair
        delta = positions[i] - positions[j]
        d = float(np.linalg.norm(delta))
        if d > r.upper:
            excess = d - r.upper
        elif d < r.lower:
            excess = d - r.lower  # negative: pushes the atoms apart
        else:
            continue
        energy += 0.5 * r.force_constant * excess**2
        if d > 0:
            gradient = r.force_constant * excess * delta / d  # dE/d(position of i)
            forces[i] -= gradient
            forces[j] += gradient
    return energy, forces


class RestrainedCalculator(Calculator):
    """
    Another ASE calculator plus flat-bottom restraints: distance windows
    (DistanceRestraint) and position restraints (PositionRestraint, e.g. the soft atoms
    of a task), with force constants in kcal/(mol A^2) as for MMFF/UFF.

    results["energy"] and results["forces"] include the restraint terms, so an
    optimizer sees them; results["restraint_energy"] (eV) is their part of the energy,
    which ASEOptimizer subtracts: stored conformer energies are restraint-free.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, calculator, restraints: Iterable, **kwargs):
        super().__init__(**kwargs)
        restraints = tuple(restraints)
        for r in restraints:
            if not isinstance(r, (DistanceRestraint, PositionRestraint)):
                raise TypeError(
                    f"{r!r} is not a restraint that ASE refinement takes "
                    "(DistanceRestraint or PositionRestraint)."
                )
        self.calculator = calculator
        self.restraints = restraints

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy = self.calculator.get_potential_energy(self.atoms)
        forces = self.calculator.get_forces(self.atoms)
        extra, extra_forces = restraint_terms(
            self.atoms.get_positions(), self.restraints
        )
        extra /= EV_TO_KCAL_MOL
        total = energy + extra
        self.results = {
            "energy": total,
            "free_energy": total,
            "forces": forces + extra_forces / EV_TO_KCAL_MOL,
            "restraint_energy": extra,
        }
