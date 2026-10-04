"""Pruning details: the RMSD of RMSDPruner (racerts.geometry), thresholds, energies."""

import logging
import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

import racerts
import racerts.prune.rmsd
from racerts.geometry import symmetry_maps
from racerts.prune import EnergyPruner, PruneCount, RMSDPruner


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
    original = racerts.prune.rmsd.symmetry_maps

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(racerts.prune.rmsd, "symmetry_maps", counting)
    # No energy prefilter: every pair is compared by RMSD.
    pruned = RMSDPruner(filter_energies=False).prune(Chem.Mol(mol))
    assert len(calls) == 1
    assert 0 < pruned.GetNumConformers() <= mol.GetNumConformers()


def test_too_many_symmetry_matches_are_reported(caplog):
    # Three identical water molecules (hydrogens included): 3! * 2^3 = 48 maps.
    mol = _conformers("O.O.O", 3)
    with caplog.at_level(logging.WARNING):
        RMSDPruner(include_hs=True, maxMatches=10).prune(Chem.Mol(mol))
    assert "limit of 10" in caplog.text


def test_as_many_symmetry_maps_as_the_limit_are_complete(caplog):
    # 2-methylpropane: 3! = 6 maps of the heavy atoms.
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)C"))
    with caplog.at_level(logging.WARNING):
        assert len(symmetry_maps(mol, max_matches=6).maps) == 6
    assert "limit" not in caplog.text


def test_superposition_can_be_left_out():
    # Conformer 1 is conformer 0 turned by 90 degrees: a duplicate after
    # superposition, not in the frame of the coordinates (e.g. of a frozen core).
    mol = _conformers("CCCCO", 1)
    turned = Chem.Conformer(mol.GetConformer(0))
    positions = turned.GetPositions() @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]).T
    for i, position in enumerate(positions):
        turned.SetAtomPosition(i, position.tolist())
    mol.AddConformer(turned, assignId=True)
    assert RMSDPruner().prune(Chem.Mol(mol)).GetNumConformers() == 1
    assert RMSDPruner(align=False).prune(Chem.Mol(mol)).GetNumConformers() == 2


def test_threshold_zero_keeps_every_conformer():
    # The keeper's RMSD to itself is a rounding error above zero.
    mol = _conformers("CCCCO", 3)
    assert RMSDPruner(threshold=0.0).prune(Chem.Mol(mol)).GetNumConformers() == 3


def test_calc_rmsd_gives_rdkits_best_rms():
    mol = Chem.RemoveHs(_conformers("CC(C)(C)CC(=O)[O-]", 4))
    pruner = RMSDPruner()
    for i, j in [(0, 1), (2, 3)]:
        best = rdMolAlign.GetBestRMS(Chem.Mol(mol), mol, prbId=i, refId=j)
        assert abs(pruner.calc_rmsd(mol, mol, i, j) - best) < 1e-6


@pytest.fixture
def ensemble():
    """Five conformers of pentanol with energies 3, 1, 4, 1.5, 2 kcal/mol."""
    mol = _conformers("CCCCCO", 5)
    for conf, energy in zip(mol.GetConformers(), [3.0, 1.0, 4.0, 1.5, 2.0]):
        conf.SetDoubleProp("energy", energy)
    return racerts.ConformerEnsemble(mol)


@pytest.mark.parametrize("renumber", [False, True])
def test_prune_count_keeps_the_lowest_in_energy_order(ensemble, renumber):
    kept = PruneCount(3, renumber=renumber).run(None, ensemble)
    assert kept.energies().tolist() == [1.0, 1.5, 2.0]
    assert kept.conf_ids == ([0, 1, 2] if renumber else [1, 3, 4])
    assert len(PruneCount(10).run(None, ensemble)) == 5


@pytest.mark.parametrize("n, error", [(0, ValueError), (-1, ValueError),
                                      (1.5, TypeError), (True, TypeError)])  # fmt: skip
def test_prune_count_checks_n(n, error):
    with pytest.raises(error):
        PruneCount(n)


def test_prune_count_needs_finite_energies(ensemble):
    ensemble.mol.GetConformer(2).SetDoubleProp("energy", math.nan)
    ensemble.mol.GetConformer(4).ClearProp("energy")
    before = ensemble.mol.ToBinary()
    with pytest.raises(ValueError, match=r"Conformers \[2, 4\] have no finite energy"):
        PruneCount(2).run(None, ensemble)
    assert ensemble.mol.ToBinary() == before


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


def test_prune_stages_take_a_pruner_object():
    with pytest.raises(TypeError, match="pruner"):
        racerts.PruneRMSD(0.2)  # a threshold: PruneRMSD(RMSDPruner(threshold=0.2))


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
