"""
RMSDs of conformers, shared by pruning, validation, the search stages and the
comparison of runs:

- listed (racerts.rmsd.listed): superposition and the RMSD over chosen atoms (e.g. the
  heavy atoms), the smallest over a list of symmetry maps, as legacy racerts computes it;
- symmetric (racerts.rmsd.symmetric): the symmetry-aware RMSD without listing every
  equivalent atom mapping (SymmetricRMSD).

    from racerts.rmsd import rmsd, symmetry_maps

    symmetry = symmetry_maps(mol)  # once per graph: the heavy atoms and their maps
    a = mol.GetConformer(0).GetPositions()
    b = mol.GetConformer(1).GetPositions()
    rmsd(a, b, symmetry.atoms, symmetry.maps)
"""

from .listed import (
    SymmetryMaps,
    atom_matches,
    heavy_atoms,
    rmsd,
    rmsd_within,
    superpose,
    symmetrize_terminal_atoms,
    symmetry_maps,
)
from .symmetric import (
    LISTED_MAPS,
    POLAR_PARENTS,
    Prepared,
    SymmetricRMSD,
    bound_descriptors,
    matching_graph,
    symmetry_classes,
)

__all__ = [
    "LISTED_MAPS",
    "POLAR_PARENTS",
    "Prepared",
    "SymmetricRMSD",
    "SymmetryMaps",
    "atom_matches",
    "bound_descriptors",
    "heavy_atoms",
    "matching_graph",
    "rmsd",
    "rmsd_within",
    "superpose",
    "symmetrize_terminal_atoms",
    "symmetry_classes",
    "symmetry_maps",
]
