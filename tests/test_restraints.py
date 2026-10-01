"""Distance restraints: model, sources, embedding windows, flat-bottom refinement."""

import numpy as np
import pytest

import racerts
import racerts.embed.dg as dg
from racerts import EmbedConfig, PipelineConfig, TransitionState
from racerts.restraints import (
    DistanceRestraint,
    RestraintSet,
)
from racerts.system import build_mol

# SN2 TS with a water on the nucleophile (conftest): keep Cl2...H7-O6.
SN2_SMILES = ["CCl", "[Cl-]", "O"]
REACTING = [0, 1, 2]
CONTACT = [(2, 7, 2.20), (2, 6, 3.16)]


def _in_windows(mol, triplets, half_width=0.25, slack=0.05):
    """Fraction of conformers with every restrained distance in its window."""
    inside = []
    for conf in mol.GetConformers():
        p = conf.GetPositions()
        inside.append(
            all(
                abs(np.linalg.norm(p[i] - p[j]) - target) <= half_width + slack
                for i, j, target in triplets
            )
        )
    return float(np.mean(inside))


def _sn2_water(path):
    return build_mol(path, -1, REACTING, input_smiles=SN2_SMILES)


# ---- model ----


def test_a_restraint_is_a_window():
    r = DistanceRestraint.around(7, 2, 2.2)
    assert (r.first, r.second) == (2, 7)
    assert (r.lower, r.upper) == pytest.approx((1.95, 2.45))
    assert r.force_constant == 20.0 and r.stage == "both" and r.label == "user:2-7"
    positions = np.zeros((8, 3))
    positions[7] = [2.0, 0, 0]
    assert r.violation(positions) == 0
    positions[7] = [3.0, 0, 0]
    assert r.violation(positions) == pytest.approx(0.55)
    assert DistanceRestraint.around(0, 1, 0.1).lower == 0.0


@pytest.mark.parametrize(
    "args, error",
    [
        ((0, 0, 1.0, 2.0), ValueError),
        ((0, 1, 2.0, 1.0), ValueError),
        ((0, 1, -1.0, 1.0), ValueError),
        ((0, 1.5, 1.0, 2.0), TypeError),
        ((True, 1, 1.0, 2.0), TypeError),
    ],
)
def test_invalid_restraints_raise(args, error):
    with pytest.raises(error):
        DistanceRestraint(*args)
    with pytest.raises(ValueError, match="stage"):
        DistanceRestraint(0, 1, 1.0, 2.0, stage="late")


def test_restraint_sets():
    a = DistanceRestraint.around(0, 1, 2.0)
    b = DistanceRestraint.around(1, 0, 3.0, stage="embed", source="hbond")
    with pytest.raises(ValueError, match="Conflicting restraints for atom pair"):
        RestraintSet([a, b])
    merged = RestraintSet([a]).merge([b])
    assert list(merged) == [b]
    assert merged.for_stage("refine") == [] and merged.for_stage("embed") == [b]
    both = RestraintSet([a, DistanceRestraint.around(2, 3, 2.5, source="contact")])
    assert [r.pair for r in both.by_source("contact")] == [(2, 3)]
    assert len(both.without_pairs_within([0, 1])) == 1
    remapped = both.remap({0: 10, 1: 11, 2: 12})
    assert [r.pair for r in remapped] == [(10, 11)] and list(remapped)[
        0
    ].label == "user:10-11"
    assert RestraintSet.from_json(both.to_json()) == both
    sampled = [len(both.sample(np.random.default_rng(i), 0.5)) for i in range(200)]
    assert 0.4 < np.mean(sampled) / 2 < 0.6


# ---- embedding ----


@pytest.mark.parametrize("mode", ["cmap", "bounds"])
def test_embedding_places_conformers_in_the_windows(sn2_ts_water, mode):
    mol = _sn2_water(sn2_ts_water)
    config = PipelineConfig(embed=EmbedConfig(mode=mode, n_conformers=20))
    embed = config.build(TransitionState(REACTING)).stages[0]
    restraints = RestraintSet(DistanceRestraint.around(*t) for t in CONTACT)

    def run(restraints):
        ctx = racerts.Context.create(
            mol, TransitionState(REACTING), restraints=restraints
        )
        return embed.run(ctx)

    assert _in_windows(run(None).mol, CONTACT) < 0.2
    embedded = run(restraints)
    assert _in_windows(embedded.mol, CONTACT) == 1.0
    assert embedded.provenance(0)["restraints"] == ["user:2-7", "user:2-6"]


def test_embedding_keeps_the_frozen_atoms_with_restraints(hept_1_ene_ts):
    task = TransitionState([3, 4, 5])
    seed = hept_1_ene_ts.GetConformer().GetPositions()
    target = float(np.linalg.norm(seed[0] - seed[6]))
    ctx = racerts.Context.create(
        hept_1_ene_ts,
        task,
        restraints=RestraintSet([DistanceRestraint.around(0, 6, target)]),
    )
    embedded = racerts.Embed(n_conformers=10).run(ctx)
    frozen = list(ctx.frozen.hard)
    for conf in embedded.mol.GetConformers():
        assert np.abs(conf.GetPositions()[frozen] - seed[frozen]).max() < 1e-3
    # Plain distance geometry honours windows in most conformers (refinement then
    # holds them).
    assert _in_windows(embedded.mol, [(0, 6, target)]) >= 0.8


def test_without_restraints_the_cmap_embedder_uses_no_bounds_matrix(
    sn2_ts_water, monkeypatch
):
    def no_bounds(*args, **kwargs):
        raise AssertionError("bounds matrix built without restraints")

    monkeypatch.setattr(dg, "bounds_matrix", no_bounds)
    mol = _sn2_water(sn2_ts_water)
    config = PipelineConfig(embed=EmbedConfig(n_conformers=5))
    assert len(racerts.generate(mol, TransitionState(REACTING), config=config)) > 0


def test_legacy_embedders_cannot_take_restraints(sn2_ts_water):
    from racerts.embedder import CmapEmbedder as LegacyCmap

    mol = _sn2_water(sn2_ts_water)
    ctx = racerts.Context.create(
        mol,
        TransitionState(REACTING),
        restraints=RestraintSet(DistanceRestraint.around(*t) for t in CONTACT),
    )
    with pytest.raises(ValueError, match="takes no restraints"):
        racerts.Embed(LegacyCmap(), n_conformers=3).run(ctx)
