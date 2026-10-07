"""Pruning details: the RMSD of RMSDPruner (racerts.geometry), thresholds, energies."""

import logging
import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

import racerts
import racerts.compat.pruner.pruner
from racerts.compat.pruner import RMSDPruner as LegacyRMSDPruner
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


def test_the_rmsd_pruner_of_legacy_racerts(caplog, monkeypatch):
    # -- too many symmetry matches are reported
    # Three identical water molecules (hydrogens included): 3! * 2^3 = 48 maps.
    mol = _conformers("O.O.O", 3)
    with caplog.at_level(logging.WARNING):
        LegacyRMSDPruner(include_hs=True, maxMatches=10).prune(Chem.Mol(mol))
    assert "limit of 10" in caplog.text

    # -- as many symmetry maps as the limit are complete
    caplog.clear()
    # 2-methylpropane: 3! = 6 maps of the heavy atoms.
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)C"))
    with caplog.at_level(logging.WARNING):
        assert len(symmetry_maps(mol, max_matches=6).maps) == 6
    assert "limit" not in caplog.text

    # -- a bent conformer differs from a linear one
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
        pruned = LegacyRMSDPruner(include_hs=True).prune(copy)
        assert pruned.GetNumConformers() == 2

    # -- calc rmsd gives rdkits best rms
    mol = Chem.RemoveHs(_conformers("CC(C)(C)CC(=O)[O-]", 4))
    pruner = LegacyRMSDPruner()
    for i, j in [(0, 1), (2, 3)]:
        best = rdMolAlign.GetBestRMS(Chem.Mol(mol), mol, prbId=i, refId=j)
        assert abs(pruner.calc_rmsd(mol, mol, i, j) - best) < 1e-6

    # -- legacy rmsd pruners without maps still work
    class LegacySubclass(RMSDPruner):
        def check_similarity(self, mol, id, j_s, filter_energies=True,
                             filter_rotations=True, energy_threshold=0.05,
                             rot_fraction_threshold=0.03, maxMatches=100000):  # fmt: skip
            return super().check_similarity(
                mol, id, j_s, filter_energies, filter_rotations, energy_threshold,
                rot_fraction_threshold, maxMatches,
            )  # fmt: skip

    mol = _conformers("CCCCCO", 6)
    expected = LegacyRMSDPruner().prune(Chem.Mol(mol)).GetNumConformers()
    assert LegacySubclass().prune(Chem.Mol(mol)).GetNumConformers() == expected

    # -- symmetry maps are computed once
    mol = _conformers("CCCCC(C)(C)O", 12)
    calls = []
    original = racerts.compat.pruner.pruner.symmetry_maps

    def counting(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(racerts.compat.pruner.pruner, "symmetry_maps", counting)
    # The pruner of legacy racerts without its energy prefilter: every pair that the
    # inertia prefilter lets through is compared by RMSD.
    pruned = LegacyRMSDPruner(filter_energies=False).prune(Chem.Mol(mol))
    assert len(calls) == 1
    assert 0 < pruned.GetNumConformers() <= mol.GetNumConformers()


def test_duplicates_by_the_rmsd_alone():
    # -- superposition can be left out
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

    # -- threshold zero keeps every conformer
    # The keeper's RMSD to itself is a rounding error above zero.
    mol = _conformers("CCCCO", 3)
    assert RMSDPruner(threshold=0.0).prune(Chem.Mol(mol)).GetNumConformers() == 3

    # -- without the prefilters the rmsd alone decides
    # A twin 1 kcal/mol above its conformer is a duplicate by its RMSD (0 A). The
    # energy prefilter of legacy racerts never compares the two (0.1 kcal/mol).
    mol, n = _twins("OCCCCCO", 12)
    legacy = LegacyRMSDPruner().prune(Chem.Mol(mol))
    alone = RMSDPruner(hydrogens="none").prune(Chem.Mol(mol))
    originals = set(range(n))
    assert {c.GetId() for c in alone.GetConformers()} <= originals
    assert not {c.GetId() for c in legacy.GetConformers()} <= originals
    # The same conformers as the comparison of every pair over all maps, and as the
    # pruner of legacy racerts gives with its two prefilters switched off.
    assert {c.GetId() for c in alone.GetConformers()} == _every_pair(mol, 0.125)
    unfiltered = LegacyRMSDPruner(filter_energies=False, filter_rotations=False)
    assert unfiltered.prune(Chem.Mol(mol)).ToBinary() == alone.ToBinary()

    # -- which hydrogens count
    # Two conformers of glycerol that differ in the rotor of one O-H.
    mol = Chem.AddHs(Chem.MolFromSmiles("OCC(O)CO"))
    AllChem.EmbedMolecule(mol, randomSeed=5)
    AllChem.MMFFOptimizeMolecule(mol)
    turned = Chem.Conformer(mol.GetConformer())
    hydroxyl = mol.GetSubstructMatch(Chem.MolFromSmarts("[#6][#6][OX2][H]"))
    angle = Chem.rdMolTransforms.GetDihedralDeg(turned, *hydroxyl)
    Chem.rdMolTransforms.SetDihedralDeg(turned, *hydroxyl, angle + 120)
    mol.AddConformer(turned, assignId=True)

    def kept(pruner):
        return pruner.prune(Chem.Mol(mol)).GetNumConformers()

    assert kept(RMSDPruner(hydrogens="none")) == 1
    assert kept(RMSDPruner()) == kept(RMSDPruner(hydrogens="polar")) == 2
    assert kept(RMSDPruner(hydrogens="all")) == 2
    with pytest.raises(ValueError, match="hydrogens must be one of"):
        RMSDPruner(hydrogens="some")
    # The pruner of legacy racerts: heavy atoms unless asked; no polar hydrogens with
    # its prefilters; its keywords are not those of the current pruner.
    off = dict(filter_energies=False, filter_rotations=False)
    assert kept(LegacyRMSDPruner(**off)) == 1
    assert kept(LegacyRMSDPruner(include_hs=True, **off)) == 2
    with pytest.raises(ValueError, match="filter_energies=False"):
        LegacyRMSDPruner(hydrogens="polar").prune(Chem.Mol(mol))
    with pytest.raises(TypeError, match="racerts.compat.pruner.RMSDPruner"):
        RMSDPruner(filter_energies=False)

    # -- all hydrogens of a molecule with many equal branches
    # Tri-tert-butylphenol with its hydrogens has more equivalent atom mappings than
    # any list holds. Its twins are found all the same, with few maps listed.
    mol, n = _twins("CC(C)(C)c1cc(C(C)(C)C)c(O)c(C(C)(C)C)c1", 3)
    pruner = RMSDPruner(hydrogens="all", max_maps=100)
    pruned = pruner.prune(Chem.Mol(mol))
    assert {c.GetId() for c in pruned.GetConformers()} <= set(range(n))


@pytest.mark.parametrize("smiles", ["C#C", "C#N", "[CH2]"])
def test_copies_of_a_linear_molecule_are_duplicates(smiles):
    # The moment of inertia about the axis of a linear molecule is rounding noise
    # (1e-16 to 1e-10): its relative difference between two copies says nothing.
    ensemble = racerts.generate_gs(smiles)
    assert len(ensemble) == 1


def _without_bonds(mol):
    """mol and its conformers without a bond, as read from coordinates alone."""
    bare = Chem.RWMol(mol)
    for bond in list(bare.GetBonds()):
        bare.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    return bare.GetMol()


def _with_exchanged(mol, pairs):
    """mol with a second conformer: the first with the places of pairs exchanged."""
    mol = Chem.Mol(mol)
    positions = mol.GetConformer().GetPositions()
    copy = Chem.Conformer(mol.GetConformer())
    for a, b in pairs:
        copy.SetAtomPosition(a, positions[b].tolist())
        copy.SetAtomPosition(b, positions[a].tolist())
    mol.AddConformer(copy, assignId=True)
    return mol


def test_duplicates_of_a_structure_without_bonds_need_its_graph(caplog):
    def kept(mol, **settings):
        return RMSDPruner(**settings).prune(Chem.Mol(mol)).GetNumConformers()

    # -- a copy is not found
    # Butanol and a copy in which two hydrogens of the methyl group changed places:
    # one conformer. Without bonds every hydrogen may take the place of every other,
    # which is more mappings than any list holds: the copy is not recognised.
    butanol = _conformers("CCCCO", 1)
    methyl = [a.GetIdx() for a in butanol.GetAtomWithIdx(0).GetNeighbors()]
    hydrogens = [i for i in methyl if butanol.GetAtomWithIdx(i).GetAtomicNum() == 1]
    copies = _with_exchanged(butanol, [hydrogens[:2]])
    assert kept(copies, hydrogens="all") == 1
    with caplog.at_level(logging.WARNING, logger="racerts"):
        assert kept(_without_bonds(copies), hydrogens="all") == 2
    assert "has no bonds" in caplog.text and "graph=" in caplog.text
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="racerts"):
        assert kept(_without_bonds(copies), hydrogens="all", graph=butanol) == 1
    assert not caplog.text

    # -- two structures are taken for one
    # Ethanol, and ethanol whose hydroxyl hydrogen sits where a hydrogen of the methyl
    # group was and the other way round: another structure on the graph (which
    # hydrogen is on the oxygen), the same one without bonds.
    ethanol = _conformers("CCO", 1)
    pattern = Chem.MolFromSmarts("[H][CH3][CH2][OH][H]")
    on_carbon, _, _, _, on_oxygen = ethanol.GetSubstructMatch(pattern)
    two = _with_exchanged(ethanol, [(on_carbon, on_oxygen)])
    assert kept(two) == kept(two, hydrogens="all") == 2
    assert kept(_without_bonds(two)) == 1
    assert kept(_without_bonds(two), graph=ethanol) == 2

    # -- which hydrogens count is read from the graph
    # Without bonds no hydrogen is known as polar, and none can be left out.
    assert kept(_without_bonds(copies), hydrogens="none", graph=butanol) == 1
    assert kept(_without_bonds(two), hydrogens="none", graph=ethanol) == 1
    assert kept(_without_bonds(two), hydrogens="polar", graph=ethanol) == 2

    # -- the pruner of legacy racerts takes the graph too
    # It ignores keywords it does not know; the graph must not be one of them. With
    # its two prefilters (the copy has the energy and the moments of its original)
    # and without them.
    for settings in ({}, dict(filter_energies=False, filter_rotations=False)):

        def legacy(mol, **more):
            pruner = LegacyRMSDPruner(include_hs=True, **settings, **more)
            return pruner.prune(Chem.Mol(mol)).GetNumConformers()

        assert legacy(_without_bonds(copies)) == 2
        assert legacy(_without_bonds(copies), graph=butanol) == 1
        assert legacy(_without_bonds(two)) == 1
        assert legacy(_without_bonds(two), graph=ethanol) == 2

    # -- the graph has the atoms of the conformers, in their order
    with pytest.raises(ValueError, match="same elements in the same order"):
        kept(_without_bonds(two), graph=butanol)
    with pytest.raises(ValueError, match="same elements in the same order"):
        LegacyRMSDPruner(graph=butanol).prune(_without_bonds(two))
    with pytest.raises(TypeError, match="graph must be an RDKit molecule"):
        RMSDPruner(graph="CCO")


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
        (lambda: RMSDPruner(threshold=True), TypeError),
        (lambda: RMSDPruner(max_maps=0), ValueError),
        (lambda: LegacyRMSDPruner(energy_threshold=-0.1), ValueError),
        (lambda: LegacyRMSDPruner(threshold=math.nan), ValueError),
    ],
)
def test_pruner_thresholds_are_checked(make, error):
    with pytest.raises(error):
        make()


def test_prune_stages_take_a_pruner_object():
    with pytest.raises(TypeError, match="pruner"):
        racerts.PruneRMSD(0.2)  # a threshold: PruneRMSD(RMSDPruner(threshold=0.2))


def test_the_energy_window_includes_its_edge(ensemble):
    for conf_id, energy in zip(ensemble.conf_ids, (0.0, 20.0, 20.001, 5.0, 30.0)):
        ensemble.mol.GetConformer(conf_id).SetDoubleProp("energy", energy)
    EnergyPruner(threshold=20).prune(ensemble.mol)
    assert ensemble.conf_ids == [0, 1, 3]


def test_energy_pruner_drops_non_finite_energies(ensemble, caplog):
    ensemble.mol.GetConformer(2).SetDoubleProp("energy", math.nan)
    with caplog.at_level(logging.WARNING):
        EnergyPruner(threshold=10).prune(ensemble.mol)
    assert ensemble.conf_ids == [0, 1, 3, 4]
    assert "non-finite one): [2]" in caplog.text


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
