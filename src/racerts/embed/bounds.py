"""Bounds matrices with distances fixed at a reference geometry."""

import logging
from typing import List, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.DistanceGeometry import DoTriangleSmoothing

from racerts.task import FrozenSet

logger = logging.getLogger(__name__)


def tol_function(
    tol: float, a: float = 1.2, b: float = 0.002, verbose: bool = False
) -> float:
    """
    Gradually increases the tolerance value for triangle smoothing.

    Args:
        tol (float): Current tolerance value.
        a (float): Multiplicative factor for tolerance adjustment.
        b (float): Additive factor for tolerance adjustment.
        verbose (bool): Unused; the new tolerance is logged at DEBUG level.

    Returns:
        float: Updated tolerance value.
    """
    tol = a * tol + b
    logger.debug("Triangle smoothing tolerance: %s", tol)
    return tol


def distance_matrix(coordinates, max_elements: int = 1000000) -> np.ndarray:
    """
    All pairwise distances, with the arithmetic of scipy.spatial.distance_matrix
    (used by legacy racerts): above max_elements (n * n * 3) one row at a time, to keep
    the memory small.
    """
    x = np.asarray(coordinates, dtype=np.float64)
    if x.size * len(x) <= max_elements:
        squared = np.sum(
            np.abs(x[np.newaxis, :, :] - x[:, np.newaxis, :]) ** 2, axis=-1
        )
        return squared ** (1.0 / 2)
    result = np.empty((len(x), len(x)))
    for i in range(len(x)):
        result[i, :] = np.sum(np.abs(x - x[i]) ** 2, axis=-1) ** (1.0 / 2)
    return result


def fixed_distance_pairs(frozen: FrozenSet) -> List[Tuple[int, int]]:
    """Pairs whose distance is fixed: each core atom with every other hard atom."""
    return [
        (atom, other) for atom in frozen.core for other in frozen.hard if other != atom
    ]


INCONSISTENT_RESTRAINTS = (
    "The distance restraints are inconsistent with each other or with the frozen "
    "atoms: triangle smoothing would have to move other bounds (e.g. stretch bonds)."
)

# How bounds that exclude a distance of the reference geometry are handled: widened to
# include it when smoothing fails without ("fallback"), always ("always"), or never.
REFERENCE_BOUNDS = ("fallback", "never", "always")

# (i, j, RDKit's lower bound, RDKit's upper bound, reference distance)
Widened = Tuple[int, int, float, float, float]


class SmoothingError(Exception):
    """Triangle smoothing of a bounds matrix fails, even with the tolerance allowed."""


def check_reference_bounds(value: str) -> None:
    if value not in REFERENCE_BOUNDS:
        raise ValueError(
            f"reference_bounds must be one of {REFERENCE_BOUNDS}, not {value!r}."
        )


def widened_bounds(
    mol: Chem.Mol,
    reference: Chem.Mol,
    pairs: Sequence[Tuple[int, int]] = (),
    windows: Sequence = (),
) -> Tuple[np.ndarray, List[Widened]]:
    """
    RDKit's distance bounds of mol (from its graph), widened wherever they exclude a
    distance of the reference geometry, with the pairs at their reference distances
    and the windows (DistanceRestraint) replacing the bounds of their pairs; triangle
    smoothed. The graph's bounds assume equilibrium geometry, which a TS core need not
    have (e.g. a metal over a ring bond, or a bond of the graph that the TS lacks); the
    reference satisfies the widened bounds, so they are consistent.

    Returns:
        The bounds, and the widened pairs (i, j, RDKit's lower and upper bound, the
        reference distance) other than those of pairs and windows.

    Raises:
        InconsistentRestraints: If the windows contradict the other bounds.
    """
    windows = list(windows)
    bounds = AllChem.GetMoleculeBoundsMatrix(  # type: ignore[attr-defined]
        mol, doTriangleSmoothing=False, useMacrocycle14config=True
    )
    d = distance_matrix(reference.GetConformer().GetPositions())
    i, j = np.triu_indices(len(bounds), 1)
    upper, lower, distance = bounds[i, j], bounds[j, i], d[i, j]
    outside = (distance > upper) | (distance < lower)
    bounds[i, j] = np.maximum(upper, distance)  # the upper triangle holds upper bounds
    bounds[j, i] = np.minimum(lower, distance)
    given = set()
    for a, b in pairs:
        bounds[a, b] = bounds[b, a] = d[a, b]
        given.add((min(a, b), max(a, b)))
    for window in windows:
        bounds[window.first, window.second] = window.upper
        bounds[window.second, window.first] = window.lower
        given.add((window.first, window.second))
    # A tiny tolerance for the rounding of (nearly) collinear atoms.
    if not DoTriangleSmoothing(bounds, tol=1e-6):
        if windows:
            raise InconsistentRestraints()
        raise SmoothingError("The bounds widened to the reference cannot be smoothed.")
    widened = [
        (int(a), int(b), float(lo), float(up), float(r))
        for a, b, lo, up, r in zip(
            i[outside], j[outside], lower[outside], upper[outside], distance[outside]
        )
        if (a, b) not in given
    ]
    return bounds, widened


def widened_message(mol: Chem.Mol, widened: Sequence[Widened], fallback: bool) -> str:
    """Which bounds were widened (the three farthest from the reference), and why."""
    worst = sorted(widened, key=lambda w: -max(w[4] - w[3], w[2] - w[4]))[:3]
    pairs = "; ".join(
        f"{a}-{b} ({mol.GetAtomWithIdx(a).GetSymbol()}"
        f"...{mol.GetAtomWithIdx(b).GetSymbol()}): {lo:.2f}-{up:.2f} A, "
        f"reference {r:.2f} A"
        for a, b, lo, up, r in worst
    )
    message = f"{len(widened)} distance bounds widened to the reference, e.g. {pairs}"
    if not fallback:
        return message
    return (
        "RDKit's distance bounds contradict the reference geometry near the frozen "
        "atoms, so triangle smoothing fails (e.g. the graph has a bond or a metal "
        "coordination that the reference lacks; the SMILES of the other side of the "
        f"reaction may fit better): {message}"
    )


def smoothed_bounds(
    mol: Chem.Mol,
    reference: Optional[Chem.Mol] = None,
    pairs: Sequence[Tuple[int, int]] = (),
    max_tolerance: float = 0.4,
    windows: Sequence = (),
    reference_bounds: str = "never",
) -> Tuple[np.ndarray, List[Widened]]:
    """
    bounds_matrix, and the bounds that were widened to the reference geometry (see
    widened_bounds): with reference_bounds="always" wherever they exclude it, with
    "fallback" only if the bounds without windows cannot be smoothed within
    max_tolerance (e.g. RDKit's bounds for a metal contradict the TS core), with a
    warning.
    """
    check_reference_bounds(reference_bounds)
    if reference is not None and reference_bounds == "always":
        return _widened(mol, reference, pairs, windows, fallback=False)
    try:
        return _smoothed(mol, reference, pairs, max_tolerance, windows), []
    except SmoothingError:
        if reference is None or reference_bounds != "fallback":
            raise
    return _widened(mol, reference, pairs, windows, fallback=True)


def _widened(mol, reference, pairs, windows, fallback):
    bounds, widened = widened_bounds(mol, reference, pairs, windows)
    if widened:
        level = logging.WARNING if fallback else logging.INFO
        logger.log(level, "%s.", widened_message(mol, widened, fallback))
    return bounds, widened


class InconsistentRestraints(ValueError):
    """The windows contradict each other or the distances that are fixed."""

    def __init__(self, message: str = INCONSISTENT_RESTRAINTS):
        super().__init__(message)


def hard_pairs(frozen: FrozenSet, reference: Optional[Chem.Mol]) -> list:
    """
    All pairs of hard atoms, whose distances the coordinate map fixes (none without a
    reference).
    """
    hard = list(frozen.hard) if reference is not None else []
    return [(a, b) for k, a in enumerate(hard) for b in hard[k + 1 :]]


def windows_fit(
    mol: Chem.Mol, reference: Optional[Chem.Mol], frozen: FrozenSet, windows: Sequence
) -> bool:
    """
    Whether the bounds of mol take the windows together, with the distances among the
    hard atoms fixed at the reference, and the bounds widened to the reference if
    smoothing fails without (as the embedders do by default).
    """
    try:
        bounds_matrix(
            mol,
            reference,
            hard_pairs(frozen, reference),
            windows=windows,
            reference_bounds="fallback",
        )
    except InconsistentRestraints:
        return False
    return True


def bounds_matrix(
    mol: Chem.Mol,
    reference: Optional[Chem.Mol] = None,
    pairs: Sequence[Tuple[int, int]] = (),
    max_tolerance: float = 0.4,
    windows: Sequence = (),
    reference_bounds: str = "never",
) -> np.ndarray:
    """
    The bounds matrix of mol, with the distances of the given pairs fixed at the
    reference geometry and the windows (DistanceRestraint) replacing the bounds of
    their pairs, after triangle smoothing.

    Smoothing starts without tolerance; while it fails, the tolerance grows (see
    tol_function) up to max_tolerance. A tolerance lets smoothing repair bounds, e.g.
    by stretching a bond, so the windows may not need more of it than the bounds
    without them. reference_bounds: "fallback" or "always" widens the bounds that
    exclude a distance of the reference (see smoothed_bounds and widened_bounds).

    Raises:
        InconsistentRestraints (a ValueError): If the windows need more tolerance than
            the bounds without them.
        SmoothingError: If the triangle smoothing needs more than max_tolerance.
    """
    return smoothed_bounds(
        mol, reference, pairs, max_tolerance, windows, reference_bounds
    )[0]


def _smoothed(mol, reference, pairs, max_tolerance, windows) -> np.ndarray:
    bounds = AllChem.GetMoleculeBoundsMatrix(mol)  # type: ignore[attr-defined]
    if pairs:
        if reference is None:
            raise ValueError("Fixed distances need a reference geometry.")
        dm = distance_matrix(reference.GetConformer().GetPositions())
        for atom, other in pairs:
            bounds[atom, other] = dm[atom, other]
            bounds[other, atom] = dm[other, atom]
    without_windows = bounds.copy()
    for window in windows:  # the upper triangle holds the upper bounds
        bounds[window.first, window.second] = window.upper
        bounds[window.second, window.first] = window.lower

    # Do triangle smoothing of the BM
    bounds_backup = bounds.copy()
    tol = 0
    failed_tol = None
    failures = 0
    smoothing = DoTriangleSmoothing(bounds, tol=tol)
    # Gradually increase tolerance up to a certain limit, until smoothing is True
    while not smoothing:
        failures += 1
        failed_tol = tol
        tol = tol_function(tol, a=1.2, b=0.02)
        bounds = bounds_backup.copy()
        smoothing = DoTriangleSmoothing(bounds, tol=tol)
        if tol > max_tolerance:
            if windows and _smooths(without_windows, failed_tol):
                raise InconsistentRestraints()
            raise SmoothingError(
                "Triangle smoothing error: tolerance above threshold "
                f"({tol:.3f} > {max_tolerance})"
            )
    if windows and failed_tol is not None and _smooths(without_windows, failed_tol):
        raise InconsistentRestraints()
    logger.debug("Triangle smoothing: %d failures, tolerance %s", failures, float(tol))
    return bounds


def _smooths(bounds: np.ndarray, tol: float) -> bool:
    return bool(DoTriangleSmoothing(bounds.copy(), tol=tol))


def log_inconsistent_bounds(bounds: np.ndarray) -> None:
    """Warn about pairs whose lower bound (bounds[i, j], i > j) exceeds the upper."""
    inconsistent = [
        (i, j, float(bounds[i][j]), float(bounds[j][i]))
        for i in range(len(bounds))
        for j in range(i)
        if bounds[i][j] > bounds[j][i]
    ]
    if inconsistent:
        logger.warning(
            "Bounds matrix with lower > upper bound (i, j, lower, upper): %s",
            inconsistent,
        )
