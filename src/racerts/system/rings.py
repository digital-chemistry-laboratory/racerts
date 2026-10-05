"""Ring puckers: the canonical Cremer-Pople forms of five- and six-membered rings."""

from typing import List, Tuple

import numpy as np


def mean_plane(positions) -> Tuple[np.ndarray, np.ndarray]:
    """The center and the unit normal of the mean plane of positions (a ring)."""
    positions = np.asarray(positions, dtype=float)
    center = positions.mean(axis=0)
    _, _, vt = np.linalg.svd(positions - center)
    return center, vt[2]


def pucker_forms(size: int) -> List[Tuple[str, np.ndarray]]:
    """
    The canonical puckers of a ring of size 5 or 6 as (name, pattern): the
    out-of-plane displacements of the ring atoms in ring order, normalised to 1
    (Cremer and Pople, J. Am. Chem. Soc. 1975, 97, 1354: z_j ~ cos(phi + 4 pi j / N),
    and (-1)^j for the chairs).

    Six-membered rings: the two chairs and the six twist-boats (the boats are saddles
    between twist-boats). Five-membered rings: ten envelopes and ten twists.
    """
    j = np.arange(size)
    forms = []
    if size == 6:
        chair = (-1.0) ** j
        forms += [("chair", chair), ("inverted chair", -chair)]
        for phi in range(30, 360, 60):
            forms.append(
                (f"twist-boat {phi}", np.cos(np.radians(phi) + 4 * np.pi * j / 6))
            )
    elif size == 5:
        for phi in range(0, 360, 18):
            name = "envelope" if phi % 36 == 0 else "twist"
            forms.append((f"{name} {phi}", np.cos(np.radians(phi) + 4 * np.pi * j / 5)))
    else:
        raise ValueError(f"No pucker forms for rings of size {size} (only 5 and 6).")
    return [(name, z / np.linalg.norm(z)) for name, z in forms]


def closest_form(displacements, forms) -> int:
    """The index of the form in forms that is closest to the displacements."""
    z = np.asarray(displacements, dtype=float)
    norm = np.linalg.norm(z)
    if norm == 0:
        return 0
    return int(np.argmax([np.dot(z / norm, pattern) for _, pattern in forms]))
