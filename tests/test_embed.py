"""Embedding details: the chirality fallback of legacy racerts and the bounds matrix."""

import logging

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.rdDistGeom import EmbedFailureCauses

import racerts
import racerts.embed.dg as dg
from racerts import Constrained, EmbedConfig, PipelineConfig
from racerts.embed.bounds import distance_matrix

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
    rule = dg.chirality_fallback
    assert rule(0, _counts(FIRST_MINIMIZATION=3), 3, 0) == "strip_tags"
    assert rule(1, _counts(FINAL_CHIRAL_BOUNDS=1), 3, 0) == "no_enforce"
    assert rule(0, _counts(FINAL_CENTER_IN_VOLUME=2), 3, 0) == "no_enforce"
    assert rule(2, _counts(FIRST_MINIMIZATION=3), 3, 0) is None  # enough embedded
    assert rule(0, _counts(), 3, 0) is None  # failed for other reasons


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
    monkeypatch.setattr(dg, "chirality_fallback", lambda *args: "strip_tags")

    with caplog.at_level(logging.WARNING):
        ensemble = racerts.generate_gs(butanol, config=SMALL)
    assert _chiral_tags(ensemble.mol) == _chiral_tags(butanol)
    assert "failed on chirality" not in caplog.text

    # With atoms held at the reference, the legacy fallback applies, with a warning.
    with caplog.at_level(logging.WARNING):
        ensemble = racerts.generate(butanol, Constrained(hard=[0, 1, 2]), config=SMALL)
    assert set(_chiral_tags(ensemble.mol)) == {Chem.ChiralType.CHI_UNSPECIFIED}
    assert "failed on chirality" in caplog.text


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


def test_legacy_embedding_repeats_its_first_three_conformers(hept_1_ene_ts):
    legacy = _embed(racerts.embed.CmapEmbedder(), hept_1_ene_ts, 8)
    assert _duplicates(legacy) == [(3, 0), (4, 1), (5, 2)]

    sequential = racerts.embed.CmapEmbedder(sequential_seeds=True)
    assert _duplicates(_embed(sequential, hept_1_ene_ts, 8)) == []


def test_sequential_seeds_are_one_seed_stream(hept_1_ene_ts):
    # Conformer i gets start + i, whether it is embedded in the check of the first
    # three or with the rest.
    embedded = _embed(
        racerts.embed.CmapEmbedder(sequential_seeds=True), hept_1_ene_ts, 8
    )
    later = Chem.Mol(hept_1_ene_ts)
    later.RemoveAllConformers()
    params = AllChem.EmbedParameters()
    params.randomSeed = dg.stream_start(12) + 3
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
    assert dg.stream_start(1) != dg.stream_start(2) and dg.stream_start(1) >= 0


def test_embed_needs_named_references(hept_1_ene_ts):
    ctx = racerts.Context.create(hept_1_ene_ts, racerts.TransitionState([3, 4, 5]))
    with pytest.raises(ValueError, match="at least one"):
        racerts.Embed(references=[]).run(ctx)
