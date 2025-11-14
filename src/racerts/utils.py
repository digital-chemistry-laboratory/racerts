from rdkit import Chem

from typing import List
import os


class suppress_std:
    """
    Suppress both stdout (fd 1) and stderr (fd 2).
    Hides all console output from C/C++ and Python code.
    """

    def __enter__(self):
        self.old_stdout_fd = os.dup(1)
        self.old_stderr_fd = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull, 1)
        os.dup2(devnull, 2)
        os.close(devnull)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        os.dup2(self.old_stdout_fd, 1)
        os.dup2(self.old_stderr_fd, 2)
        os.close(self.old_stdout_fd)
        os.close(self.old_stderr_fd)
        return False  # don't suppress exceptions


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


def get_frozen_atoms(
    mol_ts: Chem.Mol, reacting_atoms: List, frozen_atoms: List = [], verbose=False
):
    """
    Retrieve the atoms to be fixed from the molecular graph and the reacting atoms.

    Args:
        mol_ts (Chem.Mol): RDKit mol object.
        reacting_atoms (List): atom indeces of reacting atoms (atoms that change connectivity during reaction)
        (Optional) frozen_atoms (List): atom indeces of atoms to be fixed, if these should not be inferred from the graph.

    Returns:
        List: atom indeces of atoms to be fixed
    """
    frozen_atoms_new = []
    for atom_idx in reacting_atoms:
        for neighbor in mol_ts.GetAtomWithIdx(atom_idx).GetNeighbors():
            id = neighbor.GetIdx()
            if id not in frozen_atoms_new:
                frozen_atoms_new.append(id)
        if atom_idx not in frozen_atoms_new:
            frozen_atoms_new.append(atom_idx)
    if frozen_atoms is None or len(frozen_atoms) == 0:
        if verbose is True:
            print(
                "No frozen atoms are given by the user. The following frozen atoms are considered: "
                + str(frozen_atoms_new)
            )
        frozen_atoms = frozen_atoms_new
    else:
        print("Following frozen atoms are given by the user: " + str(frozen_atoms))
        print(
            "Detected frozen atoms (not further used) would have been: "
            + str(frozen_atoms_new)
        )

    return frozen_atoms
