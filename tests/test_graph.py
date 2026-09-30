"""Graphs from a geometry and an explicit-hydrogen SMILES; multiplicity of radicals."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from racerts.system import (
    bondless_mol,
    mol_from_explicit_h_smiles,
    mol_from_geometry,
    radical_multiplicity,
)


def _geometry(smiles, seed=3):
    """A molecule with explicit hydrogens and one conformer, in shuffled order."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    order = list(range(mol.GetNumAtoms()))[::-1]
    return Chem.RenumberAtoms(mol, order)


def test_explicit_h_smiles_keeps_radicals():
    mol = mol_from_explicit_h_smiles("[H]C([H])([H])[C]([H])[H]")  # ethyl radical
    assert mol.GetNumAtoms() == 7
    assert radical_multiplicity(mol) == 2
    assert all(atom.GetNoImplicit() for atom in mol.GetAtoms())
    with pytest.raises(ValueError, match="Invalid SMILES"):
        mol_from_explicit_h_smiles("C(")


def test_graph_from_geometry_takes_bond_orders_and_keeps_the_atom_order():
    geometry = _geometry("C=CC(=O)O")
    smiles = Chem.MolToSmiles(Chem.AddHs(Chem.MolFromSmiles("C=CC(=O)O")))
    mol = mol_from_geometry(geometry, smiles, charge=0, multiplicity=1)

    assert [a.GetSymbol() for a in mol.GetAtoms()] == [
        a.GetSymbol() for a in geometry.GetAtoms()
    ]
    assert np.allclose(
        mol.GetConformer().GetPositions(), geometry.GetConformer().GetPositions()
    )
    assert Chem.MolToSmiles(Chem.RemoveHs(mol)) == Chem.MolToSmiles(
        Chem.MolFromSmiles("C=CC(=O)O")
    )


def test_graph_from_geometry_with_a_radical():
    # The ethyl radical: the CH2 carbon is under-coordinated in the geometry.
    ethane = _geometry("CC")
    rw = Chem.RWMol(ethane)
    h = next(a.GetIdx() for a in rw.GetAtoms() if a.GetSymbol() == "H")
    rw.RemoveAtom(h)
    radical = rw.GetMol()
    mol = mol_from_geometry(radical, "[H]C([H])([H])[C]([H])[H]", multiplicity=2)
    assert radical_multiplicity(mol) == 2
    assert Chem.GetFormalCharge(mol) == 0


def test_graph_from_geometry_rejects_mismatches():
    geometry = _geometry("CCO")
    with pytest.raises(ValueError, match="connectivity"):
        mol_from_geometry(geometry, "[H]C([H])([H])OC([H])([H])[H]")  # dimethyl ether
    with pytest.raises(ValueError, match="Charge mismatch"):
        mol_from_geometry(geometry, "[H]OC([H])([H])C([H])([H])[H]", charge=-1)
    with pytest.raises(ValueError, match="Multiplicity mismatch"):
        mol_from_geometry(geometry, "[H]OC([H])([H])C([H])([H])[H]", multiplicity=3)
    with pytest.raises(ValueError, match="atoms"):
        mol_from_geometry(geometry, "[H]OC([H])([H])[H]")


def test_bondless_mol():
    mol = bondless_mol(["O", "H", "H"], [[0, 0, 0], [0, 0, 0.96], [0.93, 0, -0.24]])
    assert mol.GetNumBonds() == 0 and mol.GetNumConformers() == 1
    with pytest.raises(ValueError, match="shape"):
        bondless_mol(["O"], [[0, 0]])
