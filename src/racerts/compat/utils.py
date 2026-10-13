"""
racerts.utils of legacy racerts; racerts.utils (now the helper package) forwards these
names here.
"""

from typing import List

from rdkit import Chem

from racerts.system import count_electrons, infer_charge_and_multiplicity
from racerts.system.build import suppress_std
from racerts.task.transition_state import get_frozen_atoms
from racerts.utils.units import EV_TO_KCAL_MOL

__all__ = [
    "EV_TO_KCAL_MOL",
    "atom_idx_input_validation",
    "count_electrons",
    "get_frozen_atoms",
    "infer_charge_and_multiplicity",
    "suppress_std",
]


def atom_idx_input_validation(mol: Chem.Mol, reacting_atoms: List[int]) -> bool:
    """
    Validate whether all atom indices in the list exist in the molecule.

    Args:
        mol (Chem.Mol): The RDKit molecule object.
        reacting_atoms (list of int): A list of atom indices to validate.

    Returns:
        bool: True if all indices are valid, False otherwise.
    """
    idx = set(reacting_atoms)
    mol_atom_idx = set([atom.GetIdx() for atom in mol.GetAtoms()])
    return idx - mol_atom_idx == set()
