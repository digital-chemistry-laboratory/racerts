import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

from racerts.pruner import RMSDPruner


@pytest.mark.parametrize(
    ("smiles", "positions"),
    [
        ("[Br-]", [(0.0, 0.0, 0.0)]),
        (
            "O=C=O",
            [(-1.0, 0.0, 0.0), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)],
        ),
    ],
)
def test_rmsd_pruner_handles_zero_principal_moments(smiles, positions):
    mol = Chem.MolFromSmiles(smiles)
    for shift in (0.0, 2.0):
        conformer = Chem.Conformer(mol.GetNumAtoms())
        for atom_index, (x, y, z) in enumerate(positions):
            conformer.SetAtomPosition(atom_index, Point3D(x + shift, y, z))
        mol.AddConformer(conformer, assignId=True)

    RMSDPruner().prune(mol)

    assert mol.GetNumConformers() == 1
