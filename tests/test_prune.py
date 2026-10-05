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


@pytest.mark.parametrize("smiles", ["C#C", "C#N", "[CH2]"])
def test_copies_of_a_linear_molecule_are_duplicates(smiles):
    # The moment of inertia about the axis of a linear molecule is rounding noise
    # (1e-16 to 1e-10): its relative difference between two copies says nothing.
    ensemble = racerts.generate_gs(smiles)
    assert len(ensemble) == 1


def test_a_bent_conformer_differs_from_a_linear_one():
    mol = Chem.AddHs(Chem.MolFromSmiles("O"))
    for positions in (
        [[0, 0, 0], [0.96, 0, 0], [-0.96, 0, 0]],
        [[0, 0, 0], [0.96, 0, 0], [-0.24, 0.93, 0]],
    ):
        conf = Chem.Conformer(3)
        for i, p in enumerate(positions):
            conf.SetAtomPosition(i, p)
        conf.SetDoubleProp("energy", 0.0)
        mol.AddConformer(conf, assignId=True)
    for order in ((0, 1), (1, 0)):  # either one as the kept conformer
        copy = Chem.Mol(mol)
        for new_id, conf_id in enumerate(order):
            copy.GetConformer(conf_id).SetId(10 + new_id)
        pruned = RMSDPruner(include_hs=True).prune(copy)
        assert pruned.GetNumConformers() == 2


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


# ---- without the prefilters: every pair decided by its RMSD


def _twins(smiles, n, shift=1.0, seed=4):
    """Conformers of a molecule, each followed by a twin: the same geometry turned in
    space, shift kcal/mol higher in energy (as two optimizations of one minimum that
    stopped at different points)."""
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = list(AllChem.EmbedMultipleConfs(mol, n, randomSeed=seed))
    energies = [e for _, e in AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=500)]
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    for conf_id, energy in zip(ids, energies):
        conf = mol.GetConformer(conf_id)
        conf.SetDoubleProp("energy", energy)
        twin = Chem.Conformer(conf)
        positions = conf.GetPositions() @ turn + 1.0
        for i, position in enumerate(positions):
            twin.SetAtomPosition(i, position.tolist())
        twin.SetDoubleProp("energy", energy + shift)
        mol.AddConformer(twin, assignId=True)
    return mol, len(ids)


def test_without_the_prefilters_the_rmsd_alone_decides():
    # A twin 1 kcal/mol above its conformer is a duplicate by its RMSD (0 A). The
    # energy prefilter of legacy racerts never compares the two (0.1 kcal/mol).
    mol, n = _twins("OCCCCCO", 12)
    legacy = RMSDPruner().prune(Chem.Mol(mol))
    alone = RMSDPruner(filter_energies=False, filter_rotations=False).prune(
        Chem.Mol(mol)
    )
    originals = set(range(n))
    assert {c.GetId() for c in alone.GetConformers()} <= originals
    assert not {c.GetId() for c in legacy.GetConformers()} <= originals
    # The same conformers as the comparison of every pair over all maps.
    assert {c.GetId() for c in alone.GetConformers()} == _every_pair(mol, 0.125)


def _every_pair(mol, threshold):
    """The conformers that stay when each, in the order of the energy, is compared with
    every kept one by the RMSD over all maps."""
    from racerts.geometry import rmsd_within, symmetry_maps

    found = symmetry_maps(mol)
    order = sorted(mol.GetConformers(), key=lambda c: c.GetDoubleProp("energy"))
    kept = []
    for conf in order:
        x = conf.GetPositions()
        if not any(
            rmsd_within(k.GetPositions(), x, threshold, found.atoms, found.maps)
            for k in kept
        ):
            kept.append(conf)
    return {c.GetId() for c in kept}


def test_which_hydrogens_count():
    # Two conformers of glycerol that differ in the rotor of one O-H.
    mol = Chem.AddHs(Chem.MolFromSmiles("OCC(O)CO"))
    AllChem.EmbedMolecule(mol, randomSeed=5)
    AllChem.MMFFOptimizeMolecule(mol)
    turned = Chem.Conformer(mol.GetConformer())
    hydroxyl = mol.GetSubstructMatch(Chem.MolFromSmarts("[#6][#6][OX2][H]"))
    angle = Chem.rdMolTransforms.GetDihedralDeg(turned, *hydroxyl)
    Chem.rdMolTransforms.SetDihedralDeg(turned, *hydroxyl, angle + 120)
    mol.AddConformer(turned, assignId=True)

    def kept(**settings):
        pruner = RMSDPruner(filter_energies=False, filter_rotations=False, **settings)
        return pruner.prune(Chem.Mol(mol)).GetNumConformers()

    assert kept() == kept(hydrogens="none") == 1
    assert (
        kept(hydrogens="polar") == kept(hydrogens="all") == kept(include_hs=True) == 2
    )
    with pytest.raises(ValueError, match="hydrogens must be one of"):
        RMSDPruner(hydrogens="some")
    with pytest.raises(ValueError, match="filter_energies=False"):
        RMSDPruner(hydrogens="polar").prune(Chem.Mol(mol))  # the legacy prefilters


def test_all_hydrogens_of_a_molecule_with_many_equal_branches():
    # Tri-tert-butylphenol with its hydrogens has more equivalent atom mappings than
    # any list holds. Its twins are found all the same, with few maps listed.
    mol, n = _twins("CC(C)(C)c1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1", 3)
    pruner = RMSDPruner(
        hydrogens="all", maxMatches=100, filter_energies=False, filter_rotations=False
    )
    pruned = pruner.prune(Chem.Mol(mol))
    assert {c.GetId() for c in pruned.GetConformers()} <= set(range(n))
