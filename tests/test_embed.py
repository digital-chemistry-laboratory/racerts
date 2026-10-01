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


def test_embed_needs_named_references(hept_1_ene_ts):
    ctx = racerts.Context.create(hept_1_ene_ts, racerts.TransitionState([3, 4, 5]))
    with pytest.raises(ValueError, match="at least one"):
        racerts.Embed(references=[]).run(ctx)
