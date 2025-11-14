import pytest  # noqa
import os
from rdkit import Chem
from racerts import ConformerGenerator
from racerts.utils import atom_idx_input_validation, get_frozen_atoms
from racerts.mol_getter import (
    MolGetterBonds,
    MolGetterConnectivity,
    MolGetterSMILES,
)
from racerts.embedder import BoundsMatrixEmbedder
from racerts.optimizer import UFFOptimizer


filename = os.path.join(os.path.dirname(__file__), "data", "ex.xyz")
if not os.path.isfile(filename):
    raise FileNotFoundError(
        f"File {filename} not found. Please make sure the test data is available."
    )

charge = 0
reacting_atoms = [3, 4, 5]
input_smiles = ["CCCCCC=C"]


def _test_default_getter(cg):
    # determine bonds
    mol = cg.get_mol(filename, charge, reacting_atoms, input_smiles=input_smiles)
    assert isinstance(mol, Chem.rdchem.Mol)
    assert isinstance(cg.mol_getter, MolGetterSMILES)

    # determine bonds
    mol1 = cg.get_mol(filename, charge, reacting_atoms)
    assert isinstance(mol1, Chem.rdchem.Mol)
    assert isinstance(cg.mol_getter, MolGetterSMILES)  # no change in default setting

    # determine connectivity
    mol2 = cg.get_mol(filename, charge + 5, reacting_atoms)
    assert isinstance(mol2, Chem.rdchem.Mol)
    assert False not in [
        b.GetBondType() == Chem.rdchem.BondType.SINGLE for b in list(mol2.GetBonds())
    ]  # only single bonds
    assert isinstance(cg.mol_getter, MolGetterSMILES)  # no change in default setting
    return True


def _test_conformer_generator(cg):
    mol = cg.get_mol(filename, charge, reacting_atoms, input_smiles=input_smiles)
    assert isinstance(mol, Chem.rdchem.Mol)

    # checks if the idx are valid
    assert atom_idx_input_validation(mol, [mol.GetNumAtoms() - 1]) is True
    assert atom_idx_input_validation(mol, [mol.GetNumAtoms()]) is False
    try:
        cg.generate_conformers(
            filename, charge, reacting_atoms=[100]
        )  # should throw value error
        assert False
    except Exception as e:
        assert isinstance(e, ValueError)

    # finds neighbors to be frozen if not provided
    assert get_frozen_atoms(mol, reacting_atoms=[]) == []
    assert len(get_frozen_atoms(mol, reacting_atoms=[1])) != 0
    frozen = [1, 100, 1000]
    assert frozen == get_frozen_atoms(
        mol, reacting_atoms=[1], frozen_atoms=frozen
    )  # provided frozen atoms are not overwritten!

    # Embed
    n = 5
    m = mol
    new_m = Chem.Mol(m)
    new_m.RemoveAllConformers()

    valid = cg.embed_TS(
        mol_ts=m,
        new_mol=new_m,
        reacting_atoms=reacting_atoms,
        frozen_atoms=get_frozen_atoms(mol, reacting_atoms),
        number_of_conformers=n,
        conf_factor=1,
    )
    assert isinstance(valid, Chem.rdchem.Mol)
    assert m.GetNumConformers() == 1
    assert valid.GetNumConformers() == n
    assert valid == new_m

    new_m = Chem.Mol(m)
    new_m.RemoveAllConformers()

    all_constrained = cg.embed_TS(
        mol_ts=m,
        new_mol=new_m,
        reacting_atoms=[a.GetIdx() for a in m.GetAtoms()],
        frozen_atoms=[a.GetIdx() for a in m.GetAtoms()],
        number_of_conformers=n,
    )
    assert isinstance(valid, Chem.rdchem.Mol)
    assert m.GetNumConformers() == 1
    assert all_constrained.GetNumConformers() == n
    assert all_constrained == new_m

    # FF
    new_mol = cg.optimize(
        new_mol=valid, mol_ts=m, frozen_atoms=get_frozen_atoms(mol, reacting_atoms)
    )
    assert new_mol.GetNumConformers() == valid.GetNumConformers()

    all_constrained_new_mol = cg.optimize(
        new_mol=all_constrained,
        mol_ts=m,
        frozen_atoms=get_frozen_atoms(mol, reacting_atoms),
    )
    assert (
        all_constrained_new_mol.GetNumConformers() == all_constrained.GetNumConformers()
    )

    # Prune
    pruned_new_mol = cg.prune(new_mol)
    assert pruned_new_mol.GetNumConformers() <= new_mol.GetNumConformers()

    pruned_all_constrained_new_mol = cg.prune(all_constrained_new_mol)
    assert pruned_all_constrained_new_mol.GetNumConformers() == 1

    return True


def test():

    cg = ConformerGenerator()
    assert _test_default_getter(cg)
    assert _test_conformer_generator(cg)

    cg = ConformerGenerator()
    cg.embedder = BoundsMatrixEmbedder()
    assert _test_conformer_generator(cg)

    cg = ConformerGenerator()
    cg.mol_getter = MolGetterBonds()
    assert _test_conformer_generator(cg)

    cg = ConformerGenerator()
    cg.mol_getter = MolGetterConnectivity()
    assert _test_conformer_generator(cg)

    cg = ConformerGenerator()
    cg.ff_optimizer = UFFOptimizer()
    assert _test_conformer_generator(cg)
