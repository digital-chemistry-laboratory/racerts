"""Validation of stationary points by their imaginary vibrational modes."""

from typing import Any, Dict

import numpy as np

from racerts.io.ase import check_calculator, rdkit_conformer_to_ase_atoms
from racerts.pipeline import ConformerEnsemble
from racerts.utils.optional import require


class ImaginaryModes:
    """
    Whether each conformer has the expected number of imaginary modes (1 for a
    transition state, 0 for a minimum), from a finite-difference Hessian of any ASE
    calculator, with translations and rotations projected out. Meaningful only at
    stationary points, e.g. after a saddle-point search (ASEOptimizer with Sella).

    Args:
        calculator: An ASE calculator, or a callable that returns one.
        expected: The number of imaginary modes.
        threshold: Imaginary modes with |frequency| below this (cm^-1) are noise.
        delta: Displacement of the finite differences (A).

    The reason of a failed conformer lists its imaginary frequencies (cm^-1); see
    frequencies() for all of them.
    """

    name = "imaginary_modes"

    def __init__(
        self,
        calculator: Any,
        expected: int = 1,
        threshold: float = 50.0,
        delta: float = 0.005,
    ):
        self.calculator = calculator
        self._calculator_is_factory = check_calculator(calculator)
        self.expected = expected
        self.threshold = threshold
        self.delta = delta

    def validate(self, ctx, ensemble: ConformerEnsemble) -> Dict[int, str]:
        reasons = {}
        for conf_id in ensemble.conf_ids:
            atoms = rdkit_conformer_to_ase_atoms(ensemble.mol, conf_id)
            atoms.calc = self._calculator()
            try:
                frequencies = self.frequencies(atoms)
            except Exception as exc:  # e.g. an SCF failure: the conformer fails
                reasons[conf_id] = f"frequencies failed: {type(exc).__name__}: {exc}"
                continue
            imaginary = frequencies[frequencies < -self.threshold]
            if len(imaginary) != self.expected:
                listed = ", ".join(f"{f:.0f}" for f in imaginary) or "none"
                reasons[conf_id] = (
                    f"{len(imaginary)} imaginary modes (expected {self.expected}; "
                    f"cm^-1: {listed})"
                )
        return reasons

    def _calculator(self):
        return self.calculator() if self._calculator_is_factory else self.calculator

    def frequencies(self, atoms) -> np.ndarray:
        """Vibrational frequencies (cm^-1, imaginary negative), without the 5-6
        external modes."""
        hessian = hessian_by_finite_differences(atoms, self.delta)
        return vibrational_frequencies(hessian, atoms.get_masses(), atoms.positions)


def hessian_by_finite_differences(atoms, delta: float = 0.005) -> np.ndarray:
    """The Hessian (eV/A^2, 3N x 3N) from central differences of the forces."""
    positions = atoms.get_positions().copy()
    n = 3 * len(atoms)
    hessian = np.empty((n, n))
    try:
        for k in range(n):
            atom, axis = divmod(k, 3)
            columns = []
            for sign in (1, -1):
                displaced = positions.copy()
                displaced[atom, axis] += sign * delta
                atoms.set_positions(displaced)
                columns.append(-atoms.get_forces().reshape(-1))
            hessian[:, k] = (columns[0] - columns[1]) / (2 * delta)
    finally:
        atoms.set_positions(positions)
    return 0.5 * (hessian + hessian.T)


def vibrational_frequencies(
    hessian: np.ndarray, masses: np.ndarray, positions: np.ndarray
) -> np.ndarray:
    """
    Frequencies (cm^-1, imaginary ones negative) of a Hessian in eV/A^2, after
    projecting out translations and rotations (in mass-weighted coordinates).
    """
    units = require("ase.units", "ase")
    sqrt_m = np.repeat(np.sqrt(masses), 3)
    weighted = hessian / np.outer(sqrt_m, sqrt_m)
    external = _external_modes(masses, positions)
    projector = np.eye(len(weighted)) - external @ external.T
    eigenvalues = np.linalg.eigvalsh(projector @ weighted @ projector)
    # Drop the external modes: the eigenvalues closest to zero.
    order = np.argsort(np.abs(eigenvalues))
    internal = np.sort(eigenvalues[order[external.shape[1] :]])
    # eV / (A^2 amu) -> s^-2 -> cm^-1
    to_wavenumber = np.sqrt(units._e / (units._amu * 1e-20)) / (
        2 * np.pi * units._c * 100
    )
    return np.sign(internal) * np.sqrt(np.abs(internal)) * to_wavenumber


def _external_modes(masses: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """Orthonormal mass-weighted translation and rotation vectors (3N x 5 or 6)."""
    sqrt_m = np.sqrt(masses)
    center = (masses[:, None] * positions).sum(axis=0) / masses.sum()
    r = positions - center
    vectors = []
    for axis in range(3):
        t = np.zeros_like(positions)
        t[:, axis] = 1.0
        vectors.append((t * sqrt_m[:, None]).reshape(-1))
    for axis in range(3):
        e = np.zeros(3)
        e[axis] = 1.0
        vectors.append((np.cross(e, r) * sqrt_m[:, None]).reshape(-1))
    basis, singular, _ = np.linalg.svd(np.array(vectors).T, full_matrices=False)
    # 5 vectors for a linear molecule (one rotation vanishes).
    rank = int((singular > 1e-6 * singular.max()).sum())
    return basis[:, :rank]
