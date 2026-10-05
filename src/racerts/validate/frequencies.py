"""Validation of stationary points by their imaginary vibrational modes."""

from typing import Any, Dict, Optional, Sequence, Tuple

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

    The reason of a failed conformer lists its imaginary frequencies (cm^-1); modes()
    gives all frequencies and modes of one structure.
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
            try:
                frequencies, modes = self.modes(atoms)
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
                continue
            reason = self._judge(ctx, ensemble, conf_id, atoms, imaginary, modes)
            if reason is not None:
                reasons[conf_id] = reason
        return reasons

    def modes(self, atoms) -> Tuple[np.ndarray, np.ndarray]:
        """The frequencies (cm^-1, imaginary negative, without the 5-6 external modes)
        and the modes of atoms (see vibrational_modes)."""
        calculator = self.calculator
        atoms.calc = calculator() if self._calculator_is_factory else calculator
        hessian = hessian_by_finite_differences(atoms, self.delta)
        return vibrational_modes(hessian, atoms.get_masses(), atoms.get_positions())

    def frequencies(self, atoms) -> np.ndarray:
        """The frequencies of modes(atoms)."""
        return self.modes(atoms)[0]

    def _judge(self, ctx, ensemble, conf_id, atoms, imaginary, modes) -> Optional[str]:
        """The reason why a conformer with the expected number of imaginary modes still
        fails, or None."""
        return None


class ReactionMode(ImaginaryModes):
    """
    Whether each conformer is a transition state of the reaction: exactly one imaginary
    mode, and that mode moves an active bond. A first-order saddle point can be one of
    another motion (a rotor of a loosely bound complex, another reaction step), which
    ImaginaryModes does not tell apart.

    The stretch of a bond is the change of its length along the mode, for a
    displacement of unit length over all atoms: 1.41 for the stretch of two atoms
    alone, about 0 for a mode elsewhere in the molecule.

    Args:
        calculator, threshold, delta: As ImaginaryModes.
        bonds: The atom pairs; default: the active bonds of the task (for a
            TransitionState the bonds that form or break, see its active_bonds).
        min_stretch: The least stretch of one of the bonds.

    Every conformer with one imaginary mode records it in its provenance:
    "imaginary_frequency" (cm^-1) and "mode_stretch" ({"a-b": stretch}).
    """

    name = "reaction_mode"

    def __init__(
        self,
        calculator: Any,
        bonds: Optional[Sequence[Tuple[int, int]]] = None,
        min_stretch: float = 0.3,
        threshold: float = 50.0,
        delta: float = 0.005,
    ):
        super().__init__(calculator, 1, threshold, delta)
        self.bonds = None if bonds is None else [tuple(map(int, b)) for b in bonds]
        self.min_stretch = min_stretch

    def validate(self, ctx, ensemble: ConformerEnsemble) -> Dict[int, str]:
        self._active(ctx)  # raises before the first Hessian
        return super().validate(ctx, ensemble)

    def _active(self, ctx) -> list:
        bonds = self.bonds
        if bonds is None:
            active = getattr(ctx.task, "active_pairs", None)
            bonds = list(active(ctx.mol)) if active is not None else []
        if not bonds:
            raise ValueError(
                "ReactionMode needs the bonds that the reaction changes: a "
                "TransitionState task with active bonds, or bonds=[(a, b), ...]."
            )
        return bonds

    def _judge(self, ctx, ensemble, conf_id, atoms, imaginary, modes) -> Optional[str]:
        positions, mode = atoms.get_positions(), modes[0]  # the lowest: the imaginary
        stretch = {}
        for a, b in self._active(ctx):
            axis = positions[b] - positions[a]
            stretch[f"{a}-{b}"] = float(
                abs((mode[b] - mode[a]) @ axis) / np.linalg.norm(axis)
            )
        ensemble.add_provenance(
            conf_id,
            imaginary_frequency=float(imaginary[0]),
            mode_stretch={k: round(v, 4) for k, v in stretch.items()},
        )
        if max(stretch.values()) >= self.min_stretch:
            return None
        listed = ", ".join(f"{k}: {v:.2f}" for k, v in stretch.items())
        return (
            f"the imaginary mode ({imaginary[0]:.0f} cm^-1) moves no active bond "
            f"(stretch {listed}; at least {self.min_stretch:g})"
        )


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


def vibrational_modes(
    hessian: np.ndarray, masses: np.ndarray, positions: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """
    The frequencies (cm^-1, imaginary ones negative, ascending) of a Hessian in
    eV/A^2, after projecting out translations and rotations (in mass-weighted
    coordinates), and their modes: for each, the Cartesian displacements of the atoms
    (n_atoms x 3), of unit length over all atoms.
    """
    units = require("ase.units", "ase")
    sqrt_m = np.repeat(np.sqrt(masses), 3)
    weighted = hessian / np.outer(sqrt_m, sqrt_m)
    external = _external_modes(masses, positions)
    projector = np.eye(len(weighted)) - external @ external.T
    eigenvalues, vectors = np.linalg.eigh(projector @ weighted @ projector)
    # Drop the external modes: the eigenvalues closest to zero.
    order = np.argsort(np.abs(eigenvalues))[external.shape[1] :]
    order = order[np.argsort(eigenvalues[order])]
    # eV / (A^2 amu) -> s^-2 -> cm^-1
    to_wavenumber = np.sqrt(units._e / (units._amu * 1e-20)) / (
        2 * np.pi * units._c * 100
    )
    values = eigenvalues[order]
    frequencies = np.sign(values) * np.sqrt(np.abs(values)) * to_wavenumber
    cartesian = (vectors[:, order] / sqrt_m[:, None]).T
    cartesian /= np.linalg.norm(cartesian, axis=1)[:, None]
    return frequencies, cartesian.reshape(len(order), -1, 3)


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
