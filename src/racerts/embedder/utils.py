from rdkit import Chem
from rdkit.Chem import AllChem

import numpy as np
from rdkit.DistanceGeometry import DoTriangleSmoothing
from typing import Optional, List


def minkowski_distance_p(x, y, p=2):
    """COPIED FROM SCIPY SOURCE CODE

    Compute the pth power of the L**p distance between two arrays.

    For efficiency, this function computes the L**p distance but does
    not extract the pth root. If `p` is 1 or infinity, this is equal to
    the actual L**p distance.

    The last dimensions of `x` and `y` must be the same length.  Any
    other dimensions must be compatible for broadcasting.

    Parameters
    ----------
    x : (..., K) array_like
        Input array.
    y : (..., K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.

    Returns
    -------
    dist : ndarray
        pth power of the distance between the input arrays.

    Examples
    --------
    >>> from scipy.spatial import minkowski_distance_p
    >>> minkowski_distance_p([[0, 0], [0, 0]], [[1, 1], [0, 1]])
    array([2., 1.])

    """
    x = np.asarray(x)
    y = np.asarray(y)

    # Find smallest common datatype with float64 (return type of this
    # function) - addresses #10262.
    # Don't just cast to float64 for complex input case.
    common_datatype = np.promote_types(np.promote_types(x.dtype, y.dtype), "float64")

    # Make sure x and y are NumPy arrays of correct datatype.
    x = x.astype(common_datatype)
    y = y.astype(common_datatype)

    if p == np.inf:
        return np.amax(np.abs(y - x), axis=-1)
    elif p == 1:
        return np.sum(np.abs(y - x), axis=-1)
    else:
        return np.sum(np.abs(y - x) ** p, axis=-1)


def minkowski_distance(x, y, p=2):
    """COPIED FROM SCIPY SOURCE CODE

    Compute the L**p distance between two arrays.

    The last dimensions of `x` and `y` must be the same length.  Any
    other dimensions must be compatible for broadcasting.

    Parameters
    ----------
    x : (..., K) array_like
        Input array.
    y : (..., K) array_like
        Input array.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.

    Returns
    -------
    dist : ndarray
        Distance between the input arrays.

    Examples
    --------
    >>> from scipy.spatial import minkowski_distance
    >>> minkowski_distance([[0, 0], [0, 0]], [[1, 1], [0, 1]])
    array([ 1.41421356,  1.        ])

    """
    x = np.asarray(x)
    y = np.asarray(y)
    if p == np.inf or p == 1:
        return minkowski_distance_p(x, y, p)
    else:
        return minkowski_distance_p(x, y, p) ** (1.0 / p)


def tol_function(
    tol: float, a: float = 1.2, b: float = 0.002, verbose: bool = False
) -> float:
    """
    Gradually increases the tolerance value for triangle smoothing.

    Args:
        tol (float): Current tolerance value.
        a (float): Multiplicative factor for tolerance adjustment.
        b (float): Additive factor for tolerance adjustment.
        verbose (bool): Prints the updated tolerance if set to True.

    Returns:
        float: Updated tolerance value.
    """
    tol = a * tol + b

    if verbose is True:
        print(tol)
    return tol


def distance_matrix(x, y, p=2, threshold=1000000):
    """COPIED FROM SCIPY SOURCE CODE

    Compute the distance matrix.

    Returns the matrix of all pair-wise distances.

    Parameters
    ----------
    x : (M, K) array_like
        Matrix of M vectors in K dimensions.
    y : (N, K) array_like
        Matrix of N vectors in K dimensions.
    p : float, 1 <= p <= infinity
        Which Minkowski p-norm to use.
    threshold : positive int
        If ``M * N * K`` > `threshold`, algorithm uses a Python loop instead
        of large temporary arrays.

    Returns
    -------
    result : (M, N) ndarray
        Matrix containing the distance from every vector in `x` to every vector
        in `y`.

    Examples
    --------
    >>> from scipy.spatial import distance_matrix
    >>> distance_matrix([[0,0],[0,1]], [[1,0],[1,1]])
    array([[ 1.        ,  1.41421356],
           [ 1.41421356,  1.        ]])

    """

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
        result = np.empty((m, n), dtype=float)  # FIXME: figure out the best dtype
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
    Generates a bounds matrix for a molecule based on inter-atomic distances and specified tolerances.
    All distances between reacting atoms will be fixed, as well as all distances between each neighbor
    of a reacting atoms with all reacting atoms (However, no neighbor <-> other neighbor distance constraint
    unless the other neighbor is part of the reacting atoms).

    Args:
        mol_ts (Chem.Mol): RDKit molecule object from which to take conformer positions.
        new_mol (Chem.Mol): RDKit molecule object to which the bounds matrix is applied.
        frozen_atoms (List[int], optional): List of indices of frozen atoms to consider. Defaults to None.
        reacting_atoms (List[int], optional): List of indices of reacting atoms to consider. Defaults to None.
        max_tolerance (float): Maximum tolerance limit for triangle smoothing.
        verbose (bool): Enables verbose output.

    Returns:
        np.ndarray: The resulting bounds matrix after applying triangle smoothing.

    Raises:
        Exception: If essential parameters are missing or if the triangle smoothing exceeds max tolerance.
    """

    if (
        frozen_atoms is None
        or reacting_atoms is None
        or mol_ts is None
        or new_mol is None
    ):
        raise Exception("Missing input values for bound matrix generation")

    coordinates_ts = mol_ts.GetConformer().GetPositions()
    # dm = scipy.spatial.distance_matrix(coordinates_ts, coordinates_ts)
    dm = distance_matrix(coordinates_ts, coordinates_ts)

    bounds_matrix = AllChem.GetMoleculeBoundsMatrix(new_mol)  # type: ignore[attr-defined]

    for atom in reacting_atoms:
        for other_atom in frozen_atoms:
            if other_atom != atom:
                bounds_matrix[atom, other_atom] = dm[atom, other_atom]
                bounds_matrix[other_atom, atom] = dm[other_atom, atom]

    # Do triangle smoothing of the BM
    bounds_matrix_backup = bounds_matrix.copy()
    tol = 0

    smoothing = DoTriangleSmoothing(bounds_matrix, tol=tol)
    triangle_error_log = {"number of failures": 0, "tolerance": 0, "tolerance": 0.0}

    # Gradually increase tolerance up to a certain limit, until smoothing is True

    while True:
        if not smoothing:
            triangle_error_log["number of failures"] = (
                triangle_error_log["number of failures"] + 1
            )
            tol = tol_function(tol, a=1.2, b=0.02, verbose=verbose)
            bounds_matrix = bounds_matrix_backup.copy()
            smoothing = DoTriangleSmoothing(bounds_matrix, tol=tol)

        else:
            break

        if tol > max_tolerance:
            print("Error in triangle smoothing")
            raise Exception("Triangle smoothing error: tolerance above threshold")

    triangle_error_log["tolerance"] = tol

    if verbose is True:
        print("Triangle Smoothing Error Log: ", triangle_error_log)

    return bounds_matrix


def print_bounds_matrix_errors(bounds_matrix):
    """
    Prints bounds matrix errors, specifically if bounds[i][j] > bounds[j][i] for any i and j.

    Args:
        bounds_matrix (List[List[float]]): A square matrix of bounds.
    """
    for i in range(0, len(bounds_matrix)):
        for j in range(0, i):

            if i != j and bounds_matrix[i][j] > bounds_matrix[j][i]:
                print(i, j, bounds_matrix[i][j], bounds_matrix[j][i])
