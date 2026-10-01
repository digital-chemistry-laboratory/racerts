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


def bounds_matrix(
    mol: Chem.Mol,
    reference: Optional[Chem.Mol] = None,
    pairs: Sequence[Tuple[int, int]] = (),
    max_tolerance: float = 0.4,
    windows: Sequence = (),
) -> np.ndarray:
    """
    The bounds matrix of mol, with the distances of the given pairs fixed at the
    reference geometry and the windows (DistanceRestraint) replacing the bounds of
    their pairs, after triangle smoothing.

    Smoothing starts without tolerance; while it fails, the tolerance grows (see
    tol_function) up to max_tolerance. A tolerance lets smoothing repair bounds, e.g.
    by stretching a bond, so the windows may not need more of it than the bounds
    without them.

    Raises:
        ValueError: If the windows need more tolerance than the bounds without them.
        Exception: If the triangle smoothing needs more than max_tolerance.
    """
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
                raise ValueError(INCONSISTENT_RESTRAINTS)
            raise Exception(
                "Triangle smoothing error: tolerance above threshold "
                f"({tol:.3f} > {max_tolerance})"
            )
    if windows and failed_tol is not None and _smooths(without_windows, failed_tol):
        raise ValueError(INCONSISTENT_RESTRAINTS)
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
