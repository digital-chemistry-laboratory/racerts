"""Graphs from a geometry and an explicit-hydrogen SMILES; multiplicity of radicals."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import racerts
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


@pytest.mark.parametrize(
    "smiles", ["[H]C([H])([H])[C]([H])[H]", "[H]C([H])([H])C([H])[H]"]
)
def test_explicit_h_smiles_keeps_radicals(smiles):
    # The ethyl radical, its CH2 carbon in brackets or not: no hydrogen is added.
    mol = mol_from_explicit_h_smiles(smiles)
    assert mol.GetNumAtoms() == 7
    assert radical_multiplicity(mol) == 2
    assert all(atom.GetNoImplicit() for atom in mol.GetAtoms())
    assert radical_multiplicity(mol_from_explicit_h_smiles("[H]C([H])([H])[H]")) == 1
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
    # The given state is stored, for the Context and the calculators.
    assert mol.GetIntProp("multiplicity") == 2 and not mol.HasProp("charge")
    ctx = racerts.Context.create(mol, racerts.GroundState())
    assert ctx.mol.GetIntProp("multiplicity") == 2


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


def test_map_numbers_count_the_atoms_of_the_geometry_from_one():
    from racerts.system.match import mapped_atoms

    geometry = Chem.MolFromSmiles("CCO")
    template = Chem.MolFromSmiles("[CH3:1]C[OH:3]")
    assert mapped_atoms(template, geometry) == {0: 0, 2: 2}  # the last atom included
    for smiles, message in [
        ("[CH3:1]C[OH:4]", "atom map number 4 is not an atom"),
        ("[CH3:3]CO", "is C in the SMILES but O in the xyz file"),
        ("[CH3:1][CH2:1]O", "repeated"),
    ]:
        with pytest.raises(ValueError, match=message):
            mapped_atoms(Chem.MolFromSmiles(smiles), geometry)


def test_bondless_mol():
    mol = bondless_mol(["O", "H", "H"], [[0, 0, 0], [0, 0, 0.96], [0.93, 0, -0.24]])
    assert mol.GetNumBonds() == 0 and mol.GetNumConformers() == 1
    with pytest.raises(ValueError, match="shape"):
        bondless_mol(["O"], [[0, 0]])


def test_ground_states_with_radicals_get_their_multiplicity():
    config = racerts.PipelineConfig(embed=racerts.EmbedConfig(n_conformers=2))
    carbene = racerts.generate_gs("[CH2]", config=config)  # 2 radical electrons
    assert carbene.mol.GetIntProp("multiplicity") == 3
    assert racerts.generate_gs("C", config=config).mol.GetIntProp("multiplicity") == 1
    singlet = racerts.generate_gs("[CH2]", multiplicity=1, config=config)
    assert singlet.mol.GetIntProp("multiplicity") == 1
