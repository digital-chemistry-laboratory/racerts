"""Distance restraints: model, sources, embedding windows, flat-bottom refinement."""

import logging

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import racerts
import racerts.embed.dg as dg
from racerts import EmbedConfig, PipelineConfig, TransitionState
from racerts.restraints import (
    DistanceRestraint,
    RestraintSet,
    build_restraints,
    sources,
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


# ---- sources ----


def test_hydrogen_bonds_of_the_seed_become_two_windows(sn2_ts_water):
    mol = _sn2_water(sn2_ts_water)
    triplets = sources.hydrogen_bonds(mol)
    assert [(i, j) for i, j, _ in triplets] == [(7, 2), (6, 2)]
    assert [round(d, 2) for _, _, d in triplets] == [2.2, 3.16]


def test_contacts_include_the_neighbours_for_orientation(sn2_ts_water):
    mol = _sn2_water(sn2_ts_water)
    pairs = [(i, j) for i, j, _ in sources.contacts(mol, [(2, 6)])]
    assert sorted(pairs) == [(2, 6), (2, 7), (2, 8)]
    with pytest.raises(ValueError, match="bonded or share a neighbour"):
        sources.contacts(mol, [(6, 8)])


def test_fragments_keep_their_closest_contact(sn2_ts_water, sn2_ts_two_waters):
    assert sources.fragment_contacts(_sn2_water(sn2_ts_water), REACTING) == [(2, 7)]
    two = build_mol(sn2_ts_two_waters, -1, REACTING, input_smiles=SN2_SMILES + ["O"])
    # The second water is attached to the first one, not to the nucleophile.
    assert sources.fragment_contacts(two, REACTING) == [(2, 7), (6, 10)]


def test_build_restraints_precedence(sn2_ts_water, caplog):
    mol = _sn2_water(sn2_ts_water)
    frozen = TransitionState(REACTING).frozen_atoms(mol)
    with caplog.at_level(logging.WARNING):
        restraints = build_restraints(
            mol, frozen, user=[(2, 7, 2.5), (0, 1, 2.0)], hbonds=True
        )
    by_pair = {r.pair: r for r in restraints}
    assert by_pair[(2, 7)].source == "user" and by_pair[(2, 7)].upper == 2.75
    assert by_pair[(2, 6)].source == "hbond"
    assert (0, 1) not in by_pair and "both atoms are frozen" in caplog.text
    with pytest.raises(ValueError, match="Conflicting"):
        build_restraints(mol, frozen, user=[(2, 7, 2.5), (7, 2, 2.6)])
    with pytest.raises(ValueError, match="list of \\(atom, atom, distance\\) triplets"):
        build_restraints(mol, frozen, user=(2, 7, 2.5))
    with pytest.raises(ValueError, match="outside the 9-atom molecule"):
        build_restraints(mol, frozen, user=[(2, 70, 2.5)])


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


@pytest.mark.parametrize("cl_o", [5.0, 3.75])
def test_inconsistent_windows_raise(sn2_ts_water, cl_o):
    # H7 is bonded to O6, so Cl2...H7 = 2.2 +/- 0.25 A keeps Cl2...O6 below about
    # 3.45 A. Triangle smoothing could repair a window at 3.75 A by stretching the bond.
    mol = _sn2_water(sn2_ts_water)
    config = PipelineConfig(
        embed=EmbedConfig(n_conformers=3),
        restraints={"user": [[2, 7, 2.2], [2, 6, cl_o]]},
    )
    with pytest.raises(ValueError, match="restraints are inconsistent"):
        racerts.generate(mol, TransitionState(REACTING), config=config)


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


# ---- refinement ----


def _butanediol(n=20):
    mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCO"))
    AllChem.EmbedMultipleConfs(mol, n, randomSeed=5)
    return mol


@pytest.mark.parametrize(
    "optimizer_cls", [racerts.refine.MMFFOptimizer, racerts.refine.UFFOptimizer]
)
def test_flat_bottom_terms_pull_conformers_into_the_window(optimizer_cls):
    window = RestraintSet([DistanceRestraint(0, 5, 2.6, 3.0)])

    def refined(restraints):
        mol = _butanediol()
        optimizer_cls().refine(mol, restraints=list(restraints))
        return [window.violations(c.GetPositions())[0] for c in mol.GetConformers()]

    # UFF has no electrostatics: a few conformers stay in other minima.
    assert np.mean(np.array(refined(window)) < 0.1) >= 0.8
    assert np.mean(np.array(refined(RestraintSet())) < 0.1) < 0.5


def test_reported_energies_leave_out_the_restraints():
    mol = _butanediol(5)
    racerts.refine.MMFFOptimizer().refine(
        mol, restraints=[DistanceRestraint(0, 5, 2.6, 3.0, force_constant=100.0)]
    )
    props = AllChem.MMFFGetMoleculeProperties(mol)
    for conf in mol.GetConformers():
        ff = AllChem.MMFFGetMoleculeForceField(
            mol, props, confId=conf.GetId(), ignoreInterfragInteractions=False
        )
        assert conf.GetDoubleProp("energy") == pytest.approx(ff.CalcEnergy(), abs=1e-8)


@pytest.mark.ase
def test_ase_refinement_runs_without_the_restraints(caplog):
    pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones

    from racerts.refine import ASEOptimizer

    mol = _butanediol(2)
    with caplog.at_level(logging.INFO):
        ASEOptimizer(LennardJones(), max_steps=2).refine(
            mol, restraints=[DistanceRestraint(0, 5, 2.6, 3.0)]
        )
    assert "refines without the 1 restraints" in caplog.text


# ---- the pipeline ----


def test_generate_ts_keeps_the_restrained_contact(sn2_ts_water):
    config = PipelineConfig.from_dict(
        {
            "embed": {"n_conformers": 20},
            "restraints": {"user": [list(t) for t in CONTACT]},
        }
    )
    ensemble = racerts.generate_ts(
        sn2_ts_water, REACTING, charge=-1, smiles=SN2_SMILES, config=config
    )
    assert len(ensemble) > 0
    assert _in_windows(ensemble.mol, CONTACT) == 1.0
    assert PipelineConfig.from_dict(config.to_dict()) == config


@pytest.mark.parametrize("setting", ["hbonds", "keep_fragments"])
def test_seed_contacts_are_kept(sn2_ts_water, setting):
    config = PipelineConfig.from_dict(
        {"embed": {"n_conformers": 20}, "restraints": {setting: True}}
    )
    ensemble = racerts.generate_ts(
        sn2_ts_water, REACTING, charge=-1, smiles=SN2_SMILES, config=config
    )
    assert _in_windows(ensemble.mol, [(2, 7, 2.20)]) == 1.0


def test_restraints_between_frozen_atoms_are_dropped(sn2_ts, caplog):
    with caplog.at_level(logging.WARNING):
        ensemble = racerts.generate_ts(
            sn2_ts,
            REACTING,
            charge=-1,
            smiles="CCl.[Cl-]",
            config=PipelineConfig(embed=EmbedConfig(n_conformers=3)),
            restraints=RestraintSet([DistanceRestraint.around(0, 1, 2.0)]),
        )
    assert len(ensemble) > 0 and "both atoms are frozen" in caplog.text


# ---- fragment links (catmlp) ----


def test_fragment_links_join_every_fragment():
    # Acetate, ammonium and a water: the charged pair first, then the water.
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(=O)[O-].[NH4+].O"))
    links = sources.carrier_links(mol, sources.fallback_links(mol))
    fragments = Chem.GetMolFrags(mol)
    fragment_of = {i: k for k, f in enumerate(fragments) for i in f}
    assert len(links) == 2
    assert {fragment_of[a] for a, _ in links} | {fragment_of[b] for _, b in links} == {
        0,
        1,
        2,
    }
    charged = [(a, b) for a, b in links if {fragment_of[a], fragment_of[b]} == {0, 1}]
    assert charged and all(
        mol.GetAtomWithIdx(i).GetFormalCharge() != 0 for i in charged[0]
    )
    with pytest.raises(ValueError, match="do not join every fragment"):
        sources.carrier_links(mol, [links[0]])


def test_ground_state_complexes_are_embedded_together():
    config = PipelineConfig.from_dict(
        {"embed": {"n_conformers": 10}, "restraints": {"link_fragments": True}}
    )
    ensemble = racerts.generate_gs("CC(=O)[O-].[NH4+]", config=config)
    labels = ensemble.provenance(ensemble.conf_ids[0])["restraints"]
    assert len(labels) == 1 and labels[0].startswith("link:")
    # The fragments stay together; MMFF's Coulomb attraction pulls the salt bridge
    # below the vdW window, which k = 20 does not prevent.
    (link,) = [r for r in build_restraints(ensemble.mol, link_fragments=True)]
    for conf in ensemble.mol.GetConformers():
        p = conf.GetPositions()
        assert np.linalg.norm(p[link.first] - p[link.second]) < link.upper + 0.3


# ---- graph hints ----


@pytest.mark.parametrize(
    "smiles, expected",
    [
        ("OCCCCCCCO", [(9, 8), (24, 0)]),  # each O-H to the other O
        ("CC(=O)NCCCO", [(11, 7), (18, 2)]),  # amide N-H to O-H; O-H to C=O
        ("CC(=O)OCCO", [(14, 2)]),  # not to the ester alkoxy O
        ("Nc1ccccc1CCO", [(10, 9), (11, 9)]),  # aniline N-H donates, never accepts
        ("OC(=O)[C@@H]1CCCN1C(C)=C", []),  # a 5-membered pseudo-ring only
        ("[NH3+]CCCCC(=O)[O-]", []),  # charged partners: off
    ],
)
def test_graph_hints(smiles, expected):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    hints = sources.graph_hints(mol)
    assert [(h, a) for h, a, *_ in hints] == expected
    assert all(window == (1.7, 2.3) for *_, lo, hi in hints for window in [(lo, hi)])
    if smiles.startswith("[NH3+]"):
        assert sources.graph_hints(mol, charged=True)


def test_hint_batches_and_their_provenance():
    config = PipelineConfig.from_dict(
        {"seed": 3, "embed": {"n_conformers": 20}, "restraints": {"hints": True}}
    )
    pipeline = racerts.Pipeline([racerts.Embed(n_conformers=20)])
    ensemble = racerts.generate_gs("OCCCCCCCO", config=config, pipeline=pipeline)
    active = [
        tuple(ensemble.provenance(i)["active_restraints"]) for i in ensemble.conf_ids
    ]
    assert active.count(()) == 14
    assert active.count(("hint:8-9",)) == 2 and active.count(("hint:0-24",)) == 2
    assert active.count(("hint:8-9", "hint:0-24")) == 2
    seeds = {ensemble.provenance(i)["seed"] for i in ensemble.conf_ids}
    assert len(seeds) == 4  # one per batch
    # Hinted conformers have their hydrogen bond.
    hinted = [i for i, a in zip(ensemble.conf_ids, active) if a]
    positions = [ensemble.mol.GetConformer(i).GetPositions() for i in hinted]
    windows = {"hint:8-9": (8, 9), "hint:0-24": (0, 24)}
    inside = [
        all(
            1.6 <= np.linalg.norm(p[windows[h][0]] - p[windows[h][1]]) <= 2.4 for h in a
        )
        for p, a in zip(positions, [a for a in active if a])
    ]
    assert np.mean(inside) >= 0.8
    assert "restraints" not in ensemble.provenance(ensemble.conf_ids[0])


def test_hints_are_released_in_refinement():
    restraints = build_restraints(
        Chem.AddHs(Chem.MolFromSmiles("OCCCCCCCO")), hints=True
    )
    assert {r.stage for r in restraints} == {"embed"}
    assert RestraintSet(restraints).for_stage("refine") == []


def test_generated_windows_that_do_not_fit_are_left_out(sn2_ts_water, caplog):
    # Cl2...O6 at 3.75 A does not fit with Cl2...H7 at 2.2 A (O6-H7 is a bond): from
    # the user it raises (test_inconsistent_windows_raise); generated, it is left out.
    from racerts.restraints.build import _consistent

    mol = _sn2_water(sn2_ts_water)
    frozen = TransitionState(REACTING).frozen_atoms(mol)
    user = RestraintSet([DistanceRestraint.around(2, 7, 2.2)])
    generated = RestraintSet(
        [
            DistanceRestraint.around(2, 6, 3.75, source="contact"),
            DistanceRestraint.around(0, 6, 5.66, source="contact"),
        ]
    )
    with caplog.at_level(logging.WARNING):
        kept = _consistent(mol, frozen, user, generated)
    assert [r.pair for r in kept] == [(0, 6)]
    assert "contact:2-6" in caplog.text and "left out" in caplog.text


# ---- ported from catmlp test_embedding_constraints (e1547eb) ----


def test_user_triplets_become_windows_and_flat_bottom_terms():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    (r,) = build_restraints(mol, user=[(0, 3, 2.8)])
    assert (r.pair, r.force_constant) == ((0, 3), 20.0)
    assert (r.lower, r.upper) == pytest.approx((2.55, 3.05))
    assert len(build_restraints(Chem.AddHs(Chem.MolFromSmiles("CC")))) == 0

    config = PipelineConfig.from_dict(
        {"embed": {"n_conformers": 3}, "restraints": {"user": [[0, 3, 2.8]]}}
    )
    ensemble = racerts.generate_gs("CCCC", config=config)
    for conf in ensemble.mol.GetConformers():
        assert r.violation(conf.GetPositions()) < 0.1


def test_a_user_contact_between_fragments_is_the_carrier():
    mol = Chem.AddHs(Chem.MolFromSmiles("C.C"))
    restraints = build_restraints(mol, user=[(0, 1, 2.0)], link_fragments=True)
    (r,) = restraints
    assert r.pair == (0, 1) and r.source == "user"  # the user window wins
    assert (r.lower, r.upper) == pytest.approx((1.75, 2.25))


def test_parallel_restrained_refinement_reports_physical_energies():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    ids = list(AllChem.EmbedMultipleConfs(mol, numConfs=3, randomSeed=61453))
    restraint = DistanceRestraint(0, 3, 2.7, 2.9, force_constant=1000.0)
    racerts.refine.MMFFOptimizer(num_threads=2, converge=True).refine(
        mol, restraints=[restraint]
    )
    props = AllChem.MMFFGetMoleculeProperties(mol)
    for conf_id in ids:
        conf = mol.GetConformer(conf_id)
        p = conf.GetPositions()
        assert np.linalg.norm(p[0] - p[3]) < 2.93
        ff = AllChem.MMFFGetMoleculeForceField(mol, props, confId=conf_id)
        assert conf.GetDoubleProp("energy") == pytest.approx(ff.CalcEnergy(), abs=1e-8)


@pytest.mark.parametrize(
    "user, error",
    [
        ([(0, 0, 2.0)], ValueError),
        ([(0, 99, 2.0)], ValueError),
        ([(0, 1, -1.0)], ValueError),
    ],
)
def test_invalid_user_restraints_fail_early(user, error):
    with pytest.raises(error):
        build_restraints(Chem.AddHs(Chem.MolFromSmiles("CC")), user=user)


# ---- regression tests ----


def test_the_combined_hint_batch_is_checked_with_the_frozen_atoms(sn2_ts_water):
    # Each hint fits alone with the frozen core; together they fit only without it.
    mol = _sn2_water(sn2_ts_water)
    hints = RestraintSet(
        DistanceRestraint(i, j, 1.9, 2.3, stage="embed", source="hint")
        for i, j in ((2, 6), (1, 6))
    )
    config = PipelineConfig(embed=EmbedConfig(n_conformers=12))
    ensemble = racerts.generate(
        mol, TransitionState(REACTING), config=config, restraints=hints
    )
    active = {
        tuple(ensemble.provenance(i)["active_restraints"]) for i in ensemble.conf_ids
    }
    assert ("hint:2-6", "hint:1-6") not in active  # no combined batch


def test_hint_batches_with_small_budgets_and_extreme_seeds():
    def run(n, share, seed=3):
        config = PipelineConfig.from_dict(
            {"seed": seed, "restraints": {"hints": True, "hint_share": share}}
        )
        pipeline = racerts.Pipeline([racerts.Embed(n_conformers=n, hint_share=share)])
        return racerts.generate_gs("OCCCCCCCO", config=config, pipeline=pipeline)

    none = run(10, 0.0)
    assert {tuple(none.provenance(i)["active_restraints"]) for i in none.conf_ids} == {
        ()
    }
    assert len(run(1, 0.3)) == 1  # one conformer: no hint batch
    assert len(run(2, 1.0)) == 2  # two hint batches of one, no more
    assert len(run(10, 0.3, seed=2**31 - 1)) == 10  # batch seeds stay in range
