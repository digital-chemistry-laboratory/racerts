"""
Superposition and RMSD of conformers: over chosen atoms (e.g. the heavy atoms), with
or without superposition, and symmetry-aware (the smallest RMSD over the automorphisms
of the graph).

    from racerts.geometry import rmsd, symmetry_maps

    symmetry = symmetry_maps(mol)  # once per graph: the heavy atoms and their maps
    a = mol.GetConformer(0).GetPositions()
    b = mol.GetConformer(1).GetPositions()
    rmsd(a, b, symmetry.atoms, symmetry.maps)
"""

import logging
from numbers import Integral
from typing import List, NamedTuple, Optional, Sequence

import numpy as np
from rdkit import Chem

logger = logging.getLogger(__name__)

# Terminal O or N (degree 1) in X-*=X or X=*-X, e.g. carboxylate or nitro oxygens.
_TERMINAL = "O,N;D1"
_TERMINAL_O_N = Chem.MolFromSmarts(
    f"[{_TERMINAL};$([{_TERMINAL}]-[*]=[{_TERMINAL}]),$([{_TERMINAL}]=[*]-[{_TERMINAL}])]"
    "~[*]"
)
_INDEX = "_racerts_index"  # the atom indices of mol, through Chem.RemoveHs


class SymmetryMaps(NamedTuple):
    """The atoms of a symmetry-aware RMSD and the automorphisms of the graph on them."""

    atoms: List[int]
    maps: np.ndarray  # (n_maps, len(atoms)), the identity first


def heavy_atoms(mol: Chem.Mol) -> List[int]:
    """The heavy atoms of mol, or all atoms if it has none (e.g. H2)."""
    heavy = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]
    return heavy or list(range(mol.GetNumAtoms()))


def superpose(positions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """positions centred and rotated (Kabsch, proper rotations) onto the centred
    reference."""
    p = positions - positions.mean(axis=0)
    q = reference - reference.mean(axis=0)
    if len(p) < 2:
        return p
    u, _, vt = np.linalg.svd(p.T @ q)
    d = np.sign(np.linalg.det(u @ vt)) or 1.0
    return p @ (u @ np.diag([1.0, 1.0, d]) @ vt)


def rmsd(
    positions_a,
    positions_b,
    atoms: Optional[Sequence[int]] = None,
    maps=None,
    align: bool = True,
) -> float:
    """
    The RMSD between two geometries of the same atoms.

    Args:
        positions_a, positions_b: (n_atoms, 3) coordinates.
        atoms: The atoms compared and superposed (default: all), e.g. heavy_atoms(mol).
        maps: Atom maps to minimise over, e.g. symmetry_maps(mol).maps: rows of
            indices into atoms; row m pairs atoms[i] of a with atoms[maps[m][i]] of b.
            Default: every atom with itself.
        align: Superpose first, by translation and proper rotation (Kabsch), so that
            mirror images stay apart. False compares the coordinates as they are, e.g.
            of conformers that share the frame of a frozen core.
    """
    a, b = _selected(positions_a, positions_b, atoms)
    mapped = _mapped(b, maps)
    best = mapped[int(np.argmin(_msd(a, mapped, align))) if len(mapped) > 1 else 0]
    # The best map again, rotated explicitly: _msd is exact to ~1e-15 A^2 only, which
    # would leave ~1e-8 A for identical geometries.
    delta = superpose(best, a) - (a - a.mean(axis=0)) if align else best - a
    return float(np.sqrt(np.mean(np.sum(delta * delta, axis=1))))


def rmsd_within(
    positions_a,
    positions_b,
    threshold: float,
    atoms: Optional[Sequence[int]] = None,
    maps=None,
    align: bool = True,
) -> bool:
    """
    Whether rmsd(positions_a, positions_b, atoms, maps, align) <= threshold. The first
    map (the identity of symmetry_maps) is compared first: if it is within the
    threshold, so is the smallest RMSD over the maps, and the others are not needed.
    """
    a, b = _selected(positions_a, positions_b, atoms)
    if maps is not None and len(maps) > 1:
        first = np.sqrt(_msd(a, _mapped(b, maps[:1]), align)[0])
        if first <= threshold:
            return True
    return bool(np.sqrt(_msd(a, _mapped(b, maps), align).min()) <= threshold)


def symmetry_maps(
    mol: Chem.Mol, include_hs: bool = False, max_matches: int = 10000
) -> SymmetryMaps:
    """
    The atoms of a symmetry-aware RMSD of mol and the automorphisms of its graph on
    them, the identity first. The atoms are those that Chem.RemoveHs keeps (e.g. also
    hydrogens bonded to two atoms), or all with include_hs; the terminal O or N of
    X-*=X groups (carboxylate, nitro) are equivalent (symmetrize_terminal_atoms), and
    chirality is respected. The maps depend on the graph only: compute them once for
    all conformers. Beyond max_matches automorphisms (e.g. several identical solvent
    molecules with their hydrogens), the others are left out, with a warning, and the
    RMSD can miss equivalent structures.
    """
    graph = Chem.Mol(mol)
    graph.RemoveAllConformers()
    for atom in graph.GetAtoms():
        atom.SetIntProp(_INDEX, atom.GetIdx())
    if not include_hs:
        try:
            graph = Chem.RemoveHs(graph, sanitize=True)
        except Exception:
            graph = Chem.RemoveHs(graph, sanitize=False)
    atoms = [atom.GetIntProp(_INDEX) for atom in graph.GetAtoms()]
    matches = atom_matches(graph, graph, max_matches)
    if len(matches) >= max_matches:
        logger.warning(
            "The symmetry matches reach maxMatches=%d, so the RMSD may miss "
            "equivalent atom mappings, and duplicates of symmetric structures (e.g. "
            "identical solvent molecules) can remain; increase maxMatches.",
            max_matches,
        )
    identity = np.arange(len(atoms))
    maps = np.array(matches, dtype=np.intp).reshape(-1, len(atoms))
    maps = np.vstack([identity, maps[(maps != identity).any(axis=1)]])
    return SymmetryMaps(atoms, maps)


def atom_matches(
    mol1: Chem.Mol, mol2: Chem.Mol, max_matches: int = 10000, symmetrize: bool = True
) -> tuple:
    """
    The matches of the graph mol2 in mol1 (tuples of mol1 atom indices, one per mol2
    atom; for mol2 = mol1 the automorphisms), with chirality, after
    symmetrize_terminal_atoms unless not symmetrize.
    """
    if symmetrize:
        same = mol2 is mol1
        mol1 = symmetrize_terminal_atoms(mol1)
        mol2 = mol1 if same else symmetrize_terminal_atoms(mol2)
    return mol1.GetSubstructMatches(
        mol2,
        maxMatches=max_matches,
        uniquify=False,
        useChirality=True,
        useQueryQueryMatches=False,
    )


def symmetrize_terminal_atoms(mol: Chem.Mol) -> Chem.RWMol:
    """
    A copy of mol whose terminal O or N atoms (degree 1) in X-*=X or X=*-X groups
    have no formal charge and an unspecified bond, so that the two ends match either
    way (e.g. carboxylate or nitro oxygens).
    """
    rw_mol = Chem.RWMol(mol)
    for match in rw_mol.GetSubstructMatches(_TERMINAL_O_N):
        atom_idx, nbr_idx = match[0], match[1]
        rw_mol.GetAtomWithIdx(atom_idx).SetFormalCharge(0)
        if rw_mol.GetBondBetweenAtoms(atom_idx, nbr_idx) is None:
            raise RuntimeError("could not find expected bond")
        rw_mol.RemoveBond(atom_idx, nbr_idx)
        rw_mol.AddBond(atom_idx, nbr_idx, Chem.BondType.UNSPECIFIED)
    return rw_mol


def _selected(positions_a, positions_b, atoms):
    a = np.asarray(positions_a, dtype=float)
    b = np.asarray(positions_b, dtype=float)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError("RMSD inputs must have matching (n_atoms, 3) shapes.")
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("RMSD coordinates must be finite.")
    if atoms is None:
        return a, b
    indices = list(atoms)
    if any(isinstance(i, bool) or not isinstance(i, Integral) for i in indices):
        raise TypeError("RMSD atom indices must be integers.")
    if not indices or len(set(indices)) != len(indices):
        raise ValueError("RMSD atom indices must be nonempty and unique.")
    if min(indices) < 0 or max(indices) >= len(a):
        raise IndexError(f"RMSD atom indices must lie between 0 and {len(a) - 1}.")
    return a[indices], b[indices]


def _mapped(b: np.ndarray, maps) -> np.ndarray:
    """b in the atom order of each map: (n_maps, n_atoms, 3)."""
    if maps is None:
        return b[np.newaxis]
    maps = np.asarray(maps)
    if maps.ndim != 2 or maps.shape[1] != len(b) or not len(maps):
        raise ValueError(f"The maps must be rows of {len(b)} atom indices.")
    if not np.issubdtype(maps.dtype, np.integer):
        raise TypeError("The maps must hold integer atom indices.")
    if maps.min() < 0 or maps.max() >= len(b):
        raise IndexError(f"Map indices must lie between 0 and {len(b) - 1}.")
    return b[maps]


def _msd(a: np.ndarray, b: np.ndarray, align: bool) -> np.ndarray:
    """
    The mean squared deviation of a ((n, 3) or (k, n, 3)) from each b[k], after the
    best superposition by proper rotations if align (Kabsch: the largest trace of R H
    for H = a^T b is s1 + s2 + sign(det H) s3, s the singular values of H).
    """
    if not align:
        return ((b - a) ** 2).sum(axis=-1).mean(axis=-1)
    a = a - a.mean(axis=-2, keepdims=True)
    b = b - b.mean(axis=-2, keepdims=True)
    h = np.matmul(np.swapaxes(a, -1, -2), b)
    s = np.linalg.svd(h, compute_uv=False)
    sign = np.where(np.linalg.det(h) < 0, -1.0, 1.0)  # proper rotations only
    trace = s[..., 0] + s[..., 1] + sign * s[..., 2]
    squares = (a * a).sum(axis=(-2, -1)) + (b * b).sum(axis=(-2, -1))
    return np.maximum(squares - 2.0 * trace, 0.0) / a.shape[-2]
