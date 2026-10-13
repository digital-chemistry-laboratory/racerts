"""
racerts.embedder.utils of legacy racerts. The bounds matrix is built by
racerts.embed.bounds; distance_matrix and the Minkowski functions are the copies of the
scipy functions that legacy racerts had.
"""

from typing import List, Optional

import numpy as np
from rdkit import Chem

from racerts.embed.bounds import (
    bounds_matrix,
    fixed_distance_pairs,
    log_inconsistent_bounds,
    tol_function,
)
from racerts.task import FrozenSet

__all__ = [
    "distance_matrix",
    "get_bounds_matrix",
    "minkowski_distance",
    "minkowski_distance_p",
    "print_bounds_matrix_errors",
    "tol_function",
]


def minkowski_distance_p(x, y, p=2):
    """The pth power of the L**p distance between two arrays (scipy.spatial)."""
    x = np.asarray(x)
    y = np.asarray(y)
    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype), "float64")
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)
    if p == np.inf:
        return np.amax(np.abs(y - x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y - x), axis=-1)
    else:
        return np.sum(np.abs(y - x) ** p, axis=-1)


def minkowski_distance(x, y, p=2):
    """The L**p distance between two arrays (scipy.spatial)."""
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf or p == 1:
        return minkowski_distance_p(x, y, p)
    else:
        return minkowski_distance_p(x, y, p) ** (1.0 / p)


def distance_matrix(x, y, p=2, threshold=1000000):
    """All pairwise distances between the vectors of x and y (scipy.spatial)."""
    x = np.asarray(x)
    m, k = x.shape
    y = np.asarray(y)
    n, kk = y.shape

    if k != kk:
        raise ValueError(
            f"x contains {k}-dimensional vectors but y contains "
            f"{kk}-dimensional vectors"
        )

    if m * n * k <= threshold:
        return minkowski_distance(x[:, np.newaxis, :], y[np.newaxis, :, :], p)
    else:
        result = np.empty((m, n), dtype=float)
        if m < n:
            for i in range(m):
                result[i, :] = minkowski_distance(x[i], y, p)
        else:
            for j in range(n):
                result[:, j] = minkowski_distance(x, y[j], p)
        return result


def get_bounds_matrix(
    mol_ts: Chem.Mol,
    new_mol: Chem.Mol,
    frozen_atoms: Optional[List[int]] = None,
    reacting_atoms: Optional[List[int]] = None,
    max_tolerance: float = 0.4,
    verbose: bool = False,
) -> np.ndarray:
    """
    The smoothed bounds matrix of new_mol with the distances between each reacting
    atom and every frozen atom fixed at the geometry of mol_ts.
    """
    if (
        frozen_atoms is None
        or reacting_atoms is None
        or mol_ts is None
        or new_mol is None
    ):
        raise ValueError("Missing input values for bound matrix generation")
    frozen = FrozenSet(hard=tuple(frozen_atoms), core=tuple(reacting_atoms))
    return bounds_matrix(new_mol, mol_ts, fixed_distance_pairs(frozen), max_tolerance)


def print_bounds_matrix_errors(bounds_matrix):
    """Warn about pairs whose lower bound exceeds the upper bound (now logged)."""
    log_inconsistent_bounds(bounds_matrix)
