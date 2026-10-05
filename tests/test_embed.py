"""Embedding details: the chirality fallback of legacy racerts and the bounds matrix."""

import logging
import re

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedFailureCauses

import racerts
import racerts.embed.dg as dg
from racerts import Constrained, EmbedConfig, PipelineConfig
from racerts.embed.bounds import distance_matrix
from racerts.utils import seeds

SMALL = PipelineConfig(embed=EmbedConfig(n_conformers=4))


def test_an_embedder_implements_embed():
    class Empty(racerts.embed.BaseEmbedder):
        pass

    with pytest.raises(TypeError):
        Empty()


def _counts(**failures):
    counts = [0] * 12
    for name, value in failures.items():
        counts[getattr(EmbedFailureCauses, name)] = value
    return tuple(counts)


def test_the_legacy_chirality_rule():
    rule = dg.needed_fallback
    assert rule(0, _counts(FIRST_MINIMIZATION=3), 3, 0) == "strip_tags"
    assert rule(1, _counts(FINAL_CHIRAL_BOUNDS=1), 3, 0) == "no_enforce"
    assert rule(0, _counts(FINAL_CENTER_IN_VOLUME=2), 3, 0) == "no_enforce"
    assert rule(2, _counts(FIRST_MINIMIZATION=3), 3, 0) is None  # enough embedded
    assert rule(0, _counts(), 3, 0) is None  # failed for other reasons
    # With maxIterations set (RDKit's default is 0), more failures are needed.
    assert rule(0, _counts(FIRST_MINIMIZATION=15), 3, 10) is None
    assert rule(0, _counts(FIRST_MINIMIZATION=16), 3, 10) == "strip_tags"


def test_embedders_do_not_swallow_unknown_settings(hept_1_ene_ts):
    with pytest.raises(TypeError, match="sequential_seed"):
        racerts.embed.CmapEmbedder(sequential_seed=True)  # misspelt
    task = racerts.TransitionState([3, 4, 5])
    with pytest.raises(TypeError, match="squential_seeds"):
        racerts.embed.default_embedder(task, 12, squential_seeds=True)
    with pytest.raises(ValueError, match="embed mode 'dm'"):
        racerts.embed.default_embedder(task, 12, mode="dm")
    assert racerts.embed.default_embedder(task, 12, num_threads=2).num_threads == 2


def test_seed_zero_gives_identical_conformers_and_a_warning(hept_1_ene_ts, caplog):
    # RDKit seeds conformer i with (i + 1) * seed, unless the seeds are sequential.
    legacy = racerts.embed.CmapEmbedder(randomSeed=0, sequential_seeds=False)
    with caplog.at_level(logging.WARNING):
        same = _embed(legacy, hept_1_ene_ts, 4)
    assert len(_duplicates(same)) == 6 and "randomSeed 0" in caplog.text
    caplog.clear()
    sequential = racerts.embed.CmapEmbedder(randomSeed=0, sequential_seeds=True)
    with caplog.at_level(logging.WARNING):
        assert _duplicates(_embed(sequential, hept_1_ene_ts, 4)) == []
    assert "randomSeed 0" not in caplog.text


@pytest.mark.parametrize("n", [0, -3, 2.5, True])
def test_embed_checks_the_number_of_conformers(n):
    with pytest.raises((TypeError, ValueError), match="n_conformers"):
        racerts.Embed(n_conformers=n)


@pytest.fixture
def butanol():
    """(S)-butan-2-ol with a geometry."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)CC"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    return mol


def _chiral_tags(mol):
    return [a.GetChiralTag() for a in mol.GetAtoms() if a.GetAtomicNum() > 1]


def test_ground_states_never_give_up_stereocentres(butanol, monkeypatch, caplog):
    # Pretend every embedding fails on chirality: the legacy rule would strip the tags.
    monkeypatch.setattr(dg, "needed_fallback", lambda *args: "strip_tags")

    with caplog.at_level(logging.WARNING):
        ensemble = racerts.generate_gs(butanol, config=SMALL)
    assert _chiral_tags(ensemble.mol) == _chiral_tags(butanol)
    assert "failed on chirality" not in caplog.text

    # With atoms held at the reference, a fallback applies, with a warning: the legacy
    # one strips the tags; frozen_first (the default) takes the frozen stereocentre's
    # configuration from the reference.
    legacy = PipelineConfig.legacy(embed={"n_conformers": 4})
    with caplog.at_level(logging.WARNING):
        ensemble = racerts.generate(butanol, Constrained(hard=[0, 1, 2]), config=legacy)
    assert set(_chiral_tags(ensemble.mol)) == {Chem.ChiralType.CHI_UNSPECIFIED}
    assert "failed on chirality" in caplog.text
    ensemble = racerts.generate(butanol, Constrained(hard=[0, 1, 2]), config=SMALL)
    assert _chiral_tags(ensemble.mol) == _chiral_tags(butanol)


def test_distance_matrix_row_by_row_is_identical():
    coordinates = np.random.default_rng(1).normal(size=(40, 3)) * 5

    assert np.array_equal(
        distance_matrix(coordinates), distance_matrix(coordinates, max_elements=0)
    )


def _positions(mol):
    return [conf.GetPositions() for conf in mol.GetConformers()]


def _embed(embedder, mol, n):
    """Embed n conformers of the TS of ex.xyz into a copy of its graph."""
    frozen = racerts.TransitionState([3, 4, 5]).frozen_atoms(mol)
    graph = Chem.Mol(mol)
    graph.RemoveAllConformers()
    embedder.embed(graph, mol, frozen, n)
    return graph


def _duplicates(mol):
    positions = _positions(mol)
    return [
        (i, j)
        for i in range(len(positions))
        for j in range(i)
        if np.allclose(positions[i], positions[j])
    ]


def test_only_legacy_seeds_repeat_the_first_three_conformers(hept_1_ene_ts):
    # A seed per conformer is the default; the legacy seeds are for reproducing legacy
    # racerts (its embedders, PipelineConfig.legacy()).
    assert racerts.embed.CmapEmbedder().sequential_seeds is True
    assert racerts.embedder.CmapEmbedder().sequential_seeds is False
    assert racerts.PipelineConfig.legacy().embed.sequential_seeds is False
    assert _duplicates(_embed(racerts.embed.CmapEmbedder(), hept_1_ene_ts, 8)) == []
    embedded = racerts.Embed(n_conformers=8).run(
        racerts.Context.create(hept_1_ene_ts, racerts.TransitionState([3, 4, 5]))
    )
    assert _duplicates(embedded.mol) == []

    legacy = racerts.embed.CmapEmbedder(sequential_seeds=False)
    assert _duplicates(_embed(legacy, hept_1_ene_ts, 8)) == [(3, 0), (4, 1), (5, 2)]


def test_batches_never_repeat_conformers(hept_1_ene_ts):
    # Legacy racerts has no batches, so there is nothing to reproduce: every batch
    # embeds with a seed per conformer, whatever the embedder says.
    mol = Chem.Mol(hept_1_ene_ts)
    mol.AddConformer(Chem.Conformer(hept_1_ene_ts.GetConformer()), assignId=True)
    ctx = racerts.Context.create(mol, racerts.TransitionState([3, 4, 5]))
    legacy = racerts.embed.CmapEmbedder(sequential_seeds=False)
    embedded = racerts.Embed(legacy, n_conformers=6, references="all").run(ctx)
    assert len(embedded) == 12 and legacy.sequential_seeds is False
    by_reference = {}
    for conf_id in embedded.conf_ids:
        reference = embedded.provenance(conf_id)["reference"]
        by_reference.setdefault(reference, []).append(
            embedded.mol.GetConformer(conf_id).GetPositions()
        )
    for positions in by_reference.values():
        assert not any(
            np.allclose(a, b) for k, a in enumerate(positions) for b in positions[:k]
        )


def test_sequential_seeds_are_one_seed_stream(hept_1_ene_ts):
    # Conformer i gets start + i, whether it is embedded in the check of the first
    # three or with the rest.
    embedded = _embed(
        racerts.embed.CmapEmbedder(sequential_seeds=True), hept_1_ene_ts, 8
    )
    later = Chem.Mol(hept_1_ene_ts)
    later.RemoveAllConformers()
    params = AllChem.EmbedParameters()
    params.randomSeed = seeds.derive(12) + 3
    params.enableSequentialRandomSeeds = True
    params.useRandomCoords = True
    params.embedFragmentsSeparately = False
    frozen = racerts.TransitionState([3, 4, 5]).frozen_atoms(hept_1_ene_ts)
    params.SetCoordMap(
        {i: hept_1_ene_ts.GetConformer().GetAtomPosition(i) for i in frozen.hard}
    )
    AllChem.EmbedMultipleConfs(later, 5, params)
    assert np.allclose(np.array(_positions(embedded)[3:]), np.array(_positions(later)))


def test_neighbouring_seeds_give_different_streams(hept_1_ene_ts):
    def run(seed):
        embedder = racerts.embed.CmapEmbedder(randomSeed=seed, sequential_seeds=True)
        return _positions(_embed(embedder, hept_1_ene_ts, 6))

    one, two = run(1), run(2)
    shared = [
        (i, j)
        for i, a in enumerate(one)
        for j, b in enumerate(two)
        if np.allclose(a, b)
    ]
    assert shared == []
    assert seeds.derive(1) != seeds.derive(2) and seeds.derive(1) >= 0
    # The stream of a seed is fixed for good: a hash of the seed, which no library
    # version changes.
    assert seeds.derive(12) == 1051840539
    assert seeds.derive(np.int64(12)) == 1051840539
    assert all(
        0 <= seeds.derive(seed) < 2**31 - 2**24 for seed in (0, 1, 2**31, 10**12)
    )


@pytest.fixture
def pentanediol_with_a_wrong_tag():
    """
    (2R,4R)-pentane-2,4-diol geometry with the graph of (2S,4R), C2 (atom 1) held with
    its neighbours: its tag cannot be met. C4 (atom 4) is free.
    """
    geometry = Chem.AddHs(Chem.MolFromSmiles("C[C@@H](O)C[C@@H](C)O"))
    AllChem.EmbedMolecule(geometry, randomSeed=5)
    AllChem.MMFFOptimizeMolecule(geometry)
    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)C[C@@H](C)O"))
    mol.AddConformer(geometry.GetConformer(), assignId=True)
    held = [1] + [n.GetIdx() for n in mol.GetAtomWithIdx(1).GetNeighbors()]
    return mol, Constrained(hard=held)


def _cip_codes(mol, atom):
    """The CIP label of atom in every conformer, from the geometry alone."""
    codes = ""
    for conf in mol.GetConformers():
        one = Chem.Mol(mol)
        one.RemoveAllConformers()
        one.AddConformer(Chem.Conformer(conf), assignId=True)
        for a in one.GetAtoms():
            a.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
        Chem.AssignStereochemistryFrom3D(one)
        codes += one.GetAtomWithIdx(atom).GetPropsAsDict().get("_CIPCode", "?")
    return codes


def test_frozen_first_keeps_the_free_stereocentres(
    pentanediol_with_a_wrong_tag, caplog
):
    mol, task = pentanediol_with_a_wrong_tag

    def run(fallback):
        config = PipelineConfig(
            embed=EmbedConfig(n_conformers=10, chirality_fallback=fallback)
        )
        with caplog.at_level(logging.WARNING):
            return racerts.generate(mol, task, config=config)

    # The legacy fallback drops every chiral tag, so the free C4 comes out S in some
    # conformers.
    legacy = run("legacy")
    assert "embedding without chiral tags" in caplog.text
    assert "S" in _cip_codes(legacy.mol, 4)

    # frozen_first drops only the tag of the held C2, whose configuration (R) comes
    # from the reference.
    caplog.clear()
    ensemble = run("frozen_first")
    assert "without the chiral tags of the frozen atoms [1]" in caplog.text
    assert "embedding without chiral tags" not in caplog.text
    assert set(_cip_codes(ensemble.mol, 4)) == {"R"}
    assert set(_cip_codes(ensemble.mol, 1)) == {"R"}


def test_conformer_count_policies():
    from rdkit.Chem import Descriptors

    from racerts.embed import conformer_count
    from racerts.task import FrozenSet

    # Pentane with a water and a chloride that are not reacting. (RDKit counts the
    # rotatable bonds of pentane with explicit hydrogens as 4 up to 2025.03, 2 since.)
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCC.O.[Cl-]"))
    n_rot = Descriptors.NumRotatableBonds(mol)
    core = FrozenSet(hard=(0, 1), core=(0,))

    assert conformer_count(mol, 12) == 12
    assert conformer_count(mol, conf_factor=10) == n_rot * 10 + 30
    # Water: 6 rigid-body degrees of freedom; chloride: 3.
    assert conformer_count(mol, conf_factor=10, policy="fragments", frozen=core) == (
        (n_rot + 6 + 3) * 10 + 30
    )
    # Without a core, the largest fragment is the reference.
    assert conformer_count(mol, conf_factor=10, policy="fragments") == (
        (n_rot + 6 + 3) * 10 + 30
    )
    # With the chloride (atom 6) as the core, pentane and the water move: 6 + 6.
    chloride = FrozenSet(hard=(6,))
    assert conformer_count(
        mol, conf_factor=10, policy="fragments", frozen=chloride
    ) == ((n_rot + 6 + 6) * 10 + 30)
    assert conformer_count(mol, policy="per_bond") == max(7, 10 * n_rot)
    assert conformer_count(Chem.MolFromSmiles("C"), policy="per_bond") == 7
    assert conformer_count(mol, policy=lambda mol, frozen: 5) == 5
    with pytest.raises(ValueError, match="policy"):
        conformer_count(mol, policy="many")


def test_frozen_first_removes_conformers_with_inverted_stereo(
    butanol, monkeypatch, caplog
):
    # Pretend the chirality checks fail at the first minimization. No frozen atom has
    # a tag, so all tags are dropped (as in legacy racerts): the free C2 comes out
    # inverted in some conformers, which frozen_first then removes.
    monkeypatch.setattr(dg, "needed_fallback", lambda *args: "strip_tags")
    task = Constrained(hard=[3, 4, 12, 13])  # the ethyl CH2 and CH3 carbons, 2 H
    assert butanol.GetAtomWithIdx(1).GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED

    def run(fallback):
        config = PipelineConfig(
            embed=EmbedConfig(n_conformers=20, chirality_fallback=fallback)
        )
        with caplog.at_level(logging.WARNING):
            return racerts.generate(butanol, task, config=config)

    legacy = run("legacy")
    codes = _cip_codes(legacy.mol, 1)
    assert "R" in codes and "S" in codes  # inverted and not

    caplog.clear()
    ensemble = run("frozen_first")
    assert "whose stereo is inverted after the chirality fallback" in caplog.text
    assert _chiral_tags(ensemble.mol) == _chiral_tags(butanol)  # tags kept
    reference = _cip_codes(butanol, 1)
    assert set(_cip_codes(ensemble.mol, 1)) == set(reference)
    assert 0 < len(ensemble) < 20


@pytest.mark.parametrize("task_atoms", ["ts_neighbour", "constrained"])
def test_frozen_first_takes_frozen_stereo_from_the_reference(
    pentanediol_with_a_wrong_tag, task_atoms, caplog
):
    # C2 (atom 1) is frozen with a tag that contradicts the reference: as a
    # neighbour of reacting atoms of a TS, or held by Constrained, with the stereo
    # check after refinement.
    mol, constrained = pentanediol_with_a_wrong_tag
    task = constrained
    if task_atoms == "ts_neighbour":
        task = racerts.TransitionState([0, 2, 3])  # C1, O, C3: C2 is a neighbour
    config = PipelineConfig(
        embed=EmbedConfig(n_conformers=10, chirality_fallback="frozen_first"),
        prune={"check_stereo": True},
    )
    with caplog.at_level(logging.WARNING):
        ensemble = racerts.generate(mol, task, config=config)
    assert len(ensemble) > 0
    assert set(_cip_codes(ensemble.mol, 1)) == {"R"}  # the reference's
    assert set(_cip_codes(ensemble.mol, 4)) == {"R"}  # the free centre: as the graph
    if task_atoms == "ts_neighbour":
        assert "contradict the reference geometry" in caplog.text


def test_frozen_first_keeps_the_tag_of_a_frozen_atom_with_free_neighbours(
    pentanediol_with_a_wrong_tag,
):
    # C4 (atom 4) is held as well, but not its neighbours: the reference does not fix
    # its configuration, so its tag stays and no conformer comes out inverted.
    mol, task = pentanediol_with_a_wrong_tag
    config = PipelineConfig(
        embed=EmbedConfig(n_conformers=20, chirality_fallback="frozen_first")
    )
    ensemble = racerts.generate(mol, Constrained(hard=[*task.hard, 4]), config=config)
    assert set(_cip_codes(ensemble.mol, 4)) == {"R"}
    assert set(_cip_codes(ensemble.mol, 1)) == {"R"}  # held with its neighbours


def test_embed_needs_named_references(hept_1_ene_ts):
    ctx = racerts.Context.create(hept_1_ene_ts, racerts.TransitionState([3, 4, 5]))
    with pytest.raises(ValueError, match="at least one"):
        racerts.Embed(references=[]).run(ctx)


def test_frozen_first_explains_when_every_conformer_is_inverted(
    butanol, monkeypatch, caplog
):
    # Frozen stereocentres whose one free methyl sets their configuration can come out
    # inverted in every conformer (legacy racerts returns only the wrong
    # stereoisomer); frozen_first removes them all and says why.
    from racerts.system.stereo import StereoCheck

    monkeypatch.setattr(dg, "needed_fallback", lambda *args: "strip_tags")
    monkeypatch.setattr(
        StereoCheck, "mismatch", lambda self, conf: "stereo of atom 1 inverted"
    )
    config = PipelineConfig(
        embed=EmbedConfig(n_conformers=5, chirality_fallback="frozen_first")
    )
    with (
        caplog.at_level(logging.WARNING),
        pytest.raises(RuntimeError, match="no conformers"),
    ):
        racerts.generate(butanol, Constrained(hard=[3, 4, 12, 13]), config=config)
    assert re.search(r"All \d+ conformers have inverted stereo", caplog.text)
    assert "chirality_fallback='legacy' keeps them" in caplog.text


def test_frozen_first_holds_substituents_that_set_frozen_stereo():
    # A frozen stereocentre whose one free methyl sets its configuration: with
    # frozen_first, the methyl starts at the reference too (legacy mode is unchanged).
    from racerts.system.stereo import stereo_anchors
    from racerts.task import FrozenSet

    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)CC"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    # C1 frozen with O2, C3 and its H: the methyl C0 alone sets its configuration.
    h1 = next(
        n.GetIdx()
        for n in mol.GetAtomWithIdx(1).GetNeighbors()
        if n.GetAtomicNum() == 1
    )
    frozen = FrozenSet(hard=(1, 2, 3, h1))
    assert stereo_anchors(mol, frozen) == [0]
    assert stereo_anchors(mol, FrozenSet(hard=(1, 2))) == []  # two free neighbours

    ctx = racerts.Context.create(mol, Constrained(hard=[1, 2, 3, h1]))
    reference = mol.GetConformer().GetPositions()
    for fallback, pinned in (("frozen_first", True), ("legacy", False)):
        embedder = dg.CmapEmbedder(randomSeed=7, chirality_fallback=fallback)
        ensemble = racerts.Embed(embedder, n_conformers=4).run(ctx)
        moved = max(
            np.linalg.norm(
                ensemble.mol.GetConformer(c).GetPositions()[0] - reference[0]
            )
            for c in ensemble.conf_ids
        )
        assert (moved < 1e-3) == pinned


def test_refine_holds_stereo_anchors_when_asked():
    from racerts.refine import MMFFOptimizer
    from racerts.system.stereo import stereo_anchors

    mol = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)CC"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    h1 = next(
        n.GetIdx()
        for n in mol.GetAtomWithIdx(1).GetNeighbors()
        if n.GetAtomicNum() == 1
    )
    ctx = racerts.Context.create(mol, Constrained(hard=[1, 2, 3, h1]))
    assert stereo_anchors(ctx.mol, ctx.frozen) == [0]
    seen = []

    class Spy(MMFFOptimizer):
        def _refine(self, mol, reference, anchors, restraints=()):
            seen.append(list(anchors))
            return super()._refine(mol, reference, anchors, restraints)

    ensemble = racerts.Embed(n_conformers=2).run(ctx)
    racerts.Refine(Spy(), stereo_anchors=True).run(ctx, ensemble.copy())
    racerts.Refine(Spy()).run(ctx, ensemble.copy())
    assert seen == [[1, 2, 3, h1, 0], [1, 2, 3, h1]]
    # The default pipeline holds them with the frozen_first fallback only.
    from racerts.config import PipelineConfig as Config

    def refine_stage(fallback):
        stages = (
            Config.from_dict({"embed": {"chirality_fallback": fallback}})
            .build(Constrained(hard=[1]))
            .stages
        )
        return next(s for s in stages if s.name == "refine")

    assert refine_stage("frozen_first").stereo_anchors
    assert not refine_stage("legacy").stereo_anchors


def test_every_reference_is_embedded_with_seeds_of_its_own(hept_1_ene_ts):
    # Two references with the same placed atoms (here: the same conformer twice) gave
    # the same conformers twice: the batch numbers, and so the seeds, restarted for
    # every reference.
    mol = Chem.Mol(hept_1_ene_ts)
    mol.AddConformer(Chem.Conformer(mol.GetConformer()), assignId=True)
    ctx = racerts.Context.create(mol, racerts.TransitionState([3, 4, 5]))
    ensemble = racerts.Embed(n_conformers=6, references="all").run(ctx)
    groups = {}
    for conf_id in ensemble.conf_ids:
        reference = ensemble.provenance(conf_id)["reference"]
        groups.setdefault(reference, []).append(
            ensemble.mol.GetConformer(conf_id).GetPositions()
        )
    first, second = groups[0], groups[1]
    assert len(first) == len(second) == 6
    assert not any(np.allclose(a, b, atol=1e-3) for a in first for b in second)
