import logging

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdEHTTools, rdMolTransforms
from rdkit.Geometry import Point3D

from racerts.pruner import EnergyPruner, RMSDPruner

EV_TO_KCAL_MOL = 23.06054783061903


def _conformers(smiles, energies, seed=7):
    """Embed one conformer per entry of energies; None leaves a conformer without energy."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(mol, len(energies), randomSeed=seed)
    for conf, energy in zip(mol.GetConformers(), energies):
        if energy is not None:
            conf.SetDoubleProp("energy", energy)
    return mol


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


def test_eht_energies_are_set_per_conformer_in_kcal_per_mol(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # some RDKit versions write YAeHMOP output files
    mol = _conformers("CCO", [None, None])

    EnergyPruner().set_QM_energies(mol)

    for conf in mol.GetConformers():
        _, res = rdEHTTools.RunMol(mol, confId=conf.GetId())
        assert conf.GetDoubleProp("energy") == pytest.approx(
            res.totalEnergy * EV_TO_KCAL_MOL
        )


def test_failed_eht_calculation_leaves_no_stale_energy(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # some RDKit versions write YAeHMOP output files
    mol = _conformers("CCO", [5.0, 5.0])
    run_mol = rdEHTTools.RunMol

    def fail_for_second_conformer(mol, confId=-1, **kwargs):
        if confId == 1:
            return False, None
        return run_mol(mol, confId=confId, **kwargs)

    monkeypatch.setattr(rdEHTTools, "RunMol", fail_for_second_conformer)
    EnergyPruner().set_QM_energies(mol)

    assert mol.GetConformer(0).HasProp("energy")
    assert not mol.GetConformer(1).HasProp("energy")


def test_minimal_energy_follows_updated_energies():
    mol = _conformers("CCCO", [1.0, 2.0, 3.0])
    pruner = EnergyPruner()
    assert pruner.get_minimal_energy(mol) == 1.0

    for conf, energy in zip(mol.GetConformers(), [5.0, 6.0, 0.5]):
        conf.SetDoubleProp("energy", energy)

    assert pruner.get_minimal_energy(mol, verbose=True) == 0.5


def test_energy_pruner_drops_conformers_without_energy(caplog):
    mol = _conformers("CCCCO", [0.0, 1.0, None, 30.0])

    with caplog.at_level(logging.WARNING):
        EnergyPruner(threshold=20.0).prune(mol)

    assert [conf.GetId() for conf in mol.GetConformers()] == [0, 1]
    assert "without an energy" in caplog.text


def test_energy_pruner_raises_if_no_conformer_has_an_energy():
    with pytest.raises(ValueError):
        EnergyPruner().prune(_conformers("CCO", [None, None]))


def test_rmsd_pruner_keeps_distinct_conformer_zero():
    # Conformer 0 is higher in energy than conformer 1 and clearly different from it.
    mol = _conformers("CCCC", [5.0, 0.0], seed=1)
    for conf_id, dihedral in ((0, 60.0), (1, 180.0)):
        rdMolTransforms.SetDihedralDeg(mol.GetConformer(conf_id), 0, 1, 2, 3, dihedral)

    RMSDPruner(filter_energies=False, filter_rotations=False).prune(mol)

    assert mol.GetNumConformers() == 2


def test_rmsd_pruner_drops_conformers_without_energy(caplog):
    mol = _conformers("CCCCO", [0.0, 1.0, None, 2.0])

    with caplog.at_level(logging.WARNING):
        RMSDPruner(threshold=0.01).prune(mol)

    assert 2 not in [conf.GetId() for conf in mol.GetConformers()]
    assert "without an energy" in caplog.text
