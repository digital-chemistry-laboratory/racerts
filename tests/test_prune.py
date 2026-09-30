"""Pruning details: RMSD symmetry maps."""

import logging

from rdkit import Chem
from rdkit.Chem import AllChem

from racerts.prune import RMSDPruner


def _conformers(smiles, n, seed=3):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(mol, n, randomSeed=seed)
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    for conf in mol.GetConformers():
        props = AllChem.MMFFGetMoleculeProperties(mol)
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf.GetId())
        conf.SetDoubleProp("energy", ff.CalcEnergy())
    return mol


def test_symmetry_maps_are_computed_once(monkeypatch):
    mol = _conformers("CCCCC(C)(C)O", 12)
    calls = []
    original = RMSDPruner.get_atom_maps

    def counting(self, *args, **kwargs):
        calls.append(1)
        return original(self, *args, **kwargs)

    monkeypatch.setattr(RMSDPruner, "get_atom_maps", counting)
    # No energy prefilter: every pair is compared by RMSD.
    pruned = RMSDPruner(filter_energies=False).prune(Chem.Mol(mol))
    assert len(calls) == 1
    assert 0 < pruned.GetNumConformers() <= mol.GetNumConformers()


def test_too_many_symmetry_matches_are_reported(caplog):
    # Three identical water molecules (hydrogens included): 3! * 2^3 = 48 maps.
    mol = _conformers("O.O.O", 3)
    with caplog.at_level(logging.WARNING):
        RMSDPruner(include_hs=True, maxMatches=10).prune(Chem.Mol(mol))
    assert "maxMatches=10" in caplog.text


# Ported from catmlp (test_conformer_pruning, test_conformer_selection at e1547eb).

import math  # noqa: E402

import pytest  # noqa: E402

import racerts  # noqa: E402
from racerts.prune import (  # noqa: E402
    EnergyPruner,
)


@pytest.fixture
def ensemble():
    """Five conformers of pentanol with energies 3, 1, 4, 1.5, 2 kcal/mol."""
    mol = _conformers("CCCCCO", 5)
    for conf, energy in zip(mol.GetConformers(), [3.0, 1.0, 4.0, 1.5, 2.0]):
        conf.SetDoubleProp("energy", energy)
    return racerts.ConformerEnsemble(mol)


@pytest.mark.parametrize(
    "make, error",
    [
        (lambda: EnergyPruner(threshold=-1), ValueError),
        (lambda: EnergyPruner(threshold=math.inf), ValueError),
        (lambda: RMSDPruner(threshold=math.nan), ValueError),
        (lambda: RMSDPruner(threshold="0.1"), TypeError),
        (lambda: RMSDPruner(energy_threshold=-0.1), ValueError),
        (lambda: RMSDPruner(threshold=True), TypeError),
    ],
)
def test_pruner_thresholds_are_checked(make, error):
    with pytest.raises(error):
        make()


def test_energy_pruner_drops_non_finite_energies(ensemble, caplog):
    ensemble.mol.GetConformer(2).SetDoubleProp("energy", math.nan)
    with caplog.at_level(logging.WARNING):
        EnergyPruner(threshold=10).prune(ensemble.mol)
    assert ensemble.conf_ids == [0, 1, 3, 4]
    assert "non-finite one): [2]" in caplog.text


def test_legacy_rmsd_pruners_without_maps_still_work():
    class LegacySubclass(RMSDPruner):
        def check_similarity(self, mol, id, j_s, filter_energies=True,
                             filter_rotations=True, energy_threshold=0.05,
                             rot_fraction_threshold=0.03, maxMatches=100000):  # fmt: skip
            return super().check_similarity(
                mol, id, j_s, filter_energies, filter_rotations, energy_threshold,
                rot_fraction_threshold, maxMatches,
            )  # fmt: skip

    mol = _conformers("CCCCCO", 6)
    expected = RMSDPruner().prune(Chem.Mol(mol)).GetNumConformers()
    assert LegacySubclass().prune(Chem.Mol(mol)).GetNumConformers() == expected
