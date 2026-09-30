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
