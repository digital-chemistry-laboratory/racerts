"""Active-bond windows of TSs and the attack-face filter, on the aldol TS."""

import collections
import os

import numpy as np
import pytest

import racerts
from racerts import PipelineConfig, TransitionState
from racerts.system import build_mol
from racerts.validate import AttackFace

from .conftest import DATA

ALDOL = os.path.join(DATA, "aldol_ts.xyz")  # tests/data/make_aldol_ts.py
REACTING = [0, 10, 11, 12, 19]
SMILES = ["OC(=O)[C@@H]1CCCN1C(C)=C", "O=Cc1ccccc1"]
CC = (10, 12)  # the forming C-C bond, 2.2 A in the seed


@pytest.fixture(scope="module")
def aldol():
    return build_mol(ALDOL, 0, REACTING, input_smiles=SMILES)


def _run(mol, task, n=30, pipeline=None):
    ctx = racerts.Context.create(mol, task)
    pipeline = pipeline or racerts.Pipeline(
        [racerts.Embed(n_conformers=n), racerts.Refine()]
    )
    return pipeline.run(ctx), ctx


def _length(ensemble, conf_id, pair=CC):
    p = ensemble.mol.GetConformer(conf_id).GetPositions()
    return float(np.linalg.norm(p[pair[0]] - p[pair[1]]))


def test_the_legacy_ts_has_no_windows(aldol):
    legacy = TransitionState(REACTING)
    assert not legacy.windowed and len(legacy.restraints(aldol)) == 0
    assert legacy.frozen_atoms(aldol).core == tuple(REACTING)


def test_active_bonds_are_the_forming_bonds(aldol):
    task = TransitionState(REACTING, active_window=0.25)
    assert task.active_pairs(aldol) == [CC, (11, 19)]  # C-C, and H...O of the transfer
    frozen = task.frozen_atoms(aldol)
    assert not set(frozen.hard) & set(REACTING)  # the reacting atoms are free
    sources = collections.Counter(r.source for r in task.restraints(aldol))
    assert sources["active"] == 2 and sources["neighbor"] == 8 and sources["core"] > 0


def test_uniform_window(aldol):
    ensemble, ctx = _run(aldol, TransitionState(REACTING, active_window=0.25))
    lengths = [_length(ensemble, i) for i in ensemble.conf_ids]
    assert min(lengths) > 2.2 - 0.25 - 0.05 and max(lengths) < 2.2 + 0.25 + 0.05
    assert np.std(lengths) > 0.05  # it varies
    hard = list(ctx.frozen.hard)
    seed = aldol.GetConformer().GetPositions()[hard]
    for conf_id in ensemble.conf_ids:
        p = ensemble.mol.GetConformer(conf_id).GetPositions()
        assert np.abs(p[hard] - seed).max() < 1e-3
        # Refinement holds the embedded length (the target).
        provenance = ensemble.provenance(conf_id)
        assert (
            abs(
                provenance["active_bond_lengths"]["10-12"]
                - provenance["active_bond_targets"]["10-12"]
            )
            < 0.05
        )
    neighbor = [r for r in ctx.restraints if r.source == "neighbor"]
    worst = max(
        r.violation(ensemble.mol.GetConformer(i).GetPositions())
        for r in neighbor
        for i in ensemble.conf_ids
    )
    assert worst < 0.02


def test_stratified_targets(aldol):
    task = TransitionState(
        REACTING, active_bonds=[CC], active_window=(2.0, 2.9), stratify=5
    )
    ensemble, _ = _run(aldol, task, n=25)
    targets = collections.Counter()
    for conf_id in ensemble.conf_ids:
        target = ensemble.provenance(conf_id)["active_bond_targets"]["10-12"]
        assert abs(_length(ensemble, conf_id) - target) < 0.05
        targets[target] += 1
    assert sorted(targets) == pytest.approx([2.0, 2.225, 2.45, 2.675, 2.9])

    # The default pipeline prunes each target on its own: every target stays.
    config = PipelineConfig.from_dict({"embed": {"n_conformers": 25}})
    pruned = racerts.generate_ts(
        ALDOL, REACTING, smiles=SMILES, config=config, active_window=(2.0, 2.9),
        active_bonds=[CC], stratify=5,
    )  # fmt: skip
    survived = {
        pruned.provenance(i)["active_bond_targets"]["10-12"] for i in pruned.conf_ids
    }
    assert len(survived) == 5
    summary = pruned.summary()
    assert "active bond 10-12: min 2.0" in summary and "max 2.8" in summary


def test_attack_face_filter(aldol):
    ensemble, ctx = _run(aldol, TransitionState(REACTING, active_window=0.25), n=4)
    assert AttackFace().validate(ctx, ensemble) == {}
    # Mirror C10 through the plane of C12's neighbours: the other face.
    conf_id = ensemble.conf_ids[1]
    conf = ensemble.mol.GetConformer(conf_id)
    p = conf.GetPositions()
    a, b, c = (p[i] for i in (11, 13, 32))
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    mirrored = p[10] - 2 * np.dot(p[10] - a, normal) * normal
    conf.SetAtomPosition(10, mirrored.tolist())
    reasons = AttackFace().validate(ctx, ensemble)
    assert list(reasons) == [conf_id] and "10 on 12" in reasons[conf_id]
    # The default pipeline of a windowed TS includes the filter.
    names = [
        s.name
        for s in PipelineConfig()
        .build(TransitionState(REACTING, active_window=0.2))
        .stages
    ]
    assert names[:3] == ["embed", "refine", "validate"]  # on refined geometries
    no_filter = TransitionState(REACTING, active_window=0.2, stereo_filter=False)
    assert "validate" not in [s.name for s in PipelineConfig().build(no_filter).stages]


@pytest.mark.parametrize(
    "settings, message",
    [
        (dict(stratify=3), "stratify needs an active_window"),
        (dict(active_window=(2.9, 2.0)), "active_window"),
        (dict(active_window=-0.1), "active_window must be positive"),
        (dict(active_window=0.2, stratify=1), "at least 2 targets"),
        (dict(active_window=0.2, frozen_atoms=[1, 2]), "either frozen_atoms"),
    ],
)
def test_invalid_window_settings(settings, message):
    with pytest.raises(ValueError, match=message):
        TransitionState(REACTING, **settings)


def test_cli_writes_active_bonds(tmp_path):
    from racerts.cli import run_subcommand

    out = tmp_path / "ts.xyz"
    run_subcommand(
        ["ts", ALDOL, "-r", *map(str, REACTING), "-s", *SMILES, "-n", "10",
         "--active-window", "2.0", "2.9", "--active-bond", "10", "12", "--stratify", "2",
         "-o", str(out)]
    )  # fmt: skip
    lines = (tmp_path / "active_bonds.csv").read_text().splitlines()
    assert lines[0] == "conf_id,energy_kcal_mol,target_10-12,length_10-12"
    assert {line.split(",")[2] for line in lines[1:]} == {"2.0000", "2.9000"}


# ---- regression tests ----


def test_uniform_windows_are_pruned_per_length(aldol):
    # With one energy window over all lengths, only the long end survived.
    config = PipelineConfig.from_dict({"embed": {"n_conformers": 40}})
    ensemble = racerts.generate_ts(
        ALDOL, REACTING, smiles=SMILES, config=config, active_window=(2.0, 2.9),
        active_bonds=[CC],
    )  # fmt: skip
    lengths = [_length(ensemble, i) for i in ensemble.conf_ids]
    assert min(lengths) < 2.3 and max(lengths) > 2.6


def test_task_windows_win_over_generated_restraints(aldol, sn2_ts_water, caplog):
    import logging

    from racerts.restraints import DistanceRestraint, RestraintSet

    # A graph hint on the proton transfer H19...O11, an active bond: left out.
    config = PipelineConfig.from_dict(
        {"embed": {"n_conformers": 12}, "restraints": {"hints": True}}
    )
    with caplog.at_level(logging.WARNING):
        ensemble = racerts.generate(
            aldol, TransitionState(REACTING, active_window=0.25), config=config
        )
    for conf_id in ensemble.conf_ids:
        provenance = ensemble.provenance(conf_id)
        assert "hint:11-19" not in provenance.get("active_restraints", [])
        length = provenance["active_bond_lengths"]["11-19"]
        assert abs(length - provenance["active_bond_targets"]["11-19"]) < 0.05

    # keep_fragments in window mode: the nucleophile belongs to the core.
    mol = build_mol(sn2_ts_water, -1, [0, 1, 2], input_smiles=["CCl", "[Cl-]", "O"])
    task = TransitionState([0, 1, 2], active_window=0.3)
    ctx = racerts.Context.create(
        mol, task, restraints=racerts.PipelineConfig.from_dict(
            {"restraints": {"keep_fragments": True}}
        ).restraints.build(mol, task.frozen_atoms(mol)),
    )  # fmt: skip
    by_pair = {r.pair: r.source for r in ctx.restraints}
    assert by_pair[(0, 2)] == "active" and by_pair[(2, 7)] == "fragment"

    # A user restraint on an active bond is a contradiction.
    with pytest.raises(ValueError, match="active-bond windows"):
        racerts.Context.create(
            aldol,
            TransitionState(REACTING, active_window=0.25),
            restraints=RestraintSet([DistanceRestraint.around(10, 12, 2.5)]),
        )


@pytest.mark.ase
def test_windowed_refinement_needs_restraints(aldol):
    pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones

    from racerts.refine import ASEOptimizer

    task = TransitionState(REACTING, active_window=0.25)
    ctx = racerts.Context.create(aldol, task)
    ensemble = racerts.Embed(n_conformers=2).run(ctx)
    lj = ASEOptimizer(LennardJones(), max_steps=2)
    with pytest.raises(ValueError, match="anchors=False"):
        racerts.Refine(lj).run(ctx, ensemble.copy())
    racerts.Refine(lj, anchors=False).run(ctx, ensemble.copy())  # a free search: fine


def test_attack_face_with_an_atom_in_two_active_bonds(sn2_ts):
    # SN2: C0 forms a bond to Cl2 and breaks the one to Cl1.
    mol = build_mol(sn2_ts, -1, [0, 1, 2], input_smiles=["CCl", "[Cl-]"])
    task = TransitionState([0, 1, 2], active_bonds=[(0, 1), (0, 2)], active_window=0.2)
    ctx = racerts.Context.create(mol, task)
    seed_copy = racerts.ConformerEnsemble(racerts.Context.create(mol, task).mol)
    assert AttackFace().validate(ctx, seed_copy) == {}


@pytest.mark.parametrize(
    "bonds, error",
    [([(10, 99)], ValueError), ([(1, 8)], ValueError), ([(10, 10)], ValueError)],
)
def test_active_bonds_are_checked(aldol, bonds, error):
    with pytest.raises(error):
        TransitionState(REACTING, active_bonds=bonds, active_window=0.2).active_pairs(
            aldol
        )


def test_cli_stratify_needs_a_window():
    from racerts.cli import run_subcommand

    with pytest.raises(SystemExit):
        run_subcommand(["ts", ALDOL, "-r", *map(str, REACTING), "--stratify", "3"])


def test_ring_13_pairs_are_not_forming_bonds():
    # Benchmark SNAr and cyclization TSs: reacting ring atoms 1,3 apart (2.4 A) were
    # taken for forming bonds, and their in-plane "faces" flipped at random.
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCC1"))  # 1,3 C...C about 2.2 A
    AllChem.EmbedMolecule(mol, randomSeed=1)
    assert TransitionState([0, 1, 2], active_window=0.3).active_pairs(mol) == []
    explicit = TransitionState([0, 1, 2], active_bonds=[(0, 2)], active_window=0.3)
    assert explicit.active_pairs(mol) == [(0, 2)]


def test_three_membered_forming_bonds_stay_active():
    # A reductive-elimination TS C-Pd-C: C...C 1.88 A at a 56 degree angle forms a
    # bond although the two carbons share Pd.
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.RWMol(Chem.MolFromSmiles("C[Pd]C"))
    conf = Chem.Conformer(3)
    half = np.radians(28)
    for i, (x, y) in enumerate(
        [(np.cos(half), np.sin(half)), (0, 0), (np.cos(half), -np.sin(half))]
    ):
        conf.SetAtomPosition(i, Point3D(2.0 * x, 2.0 * y, 0.0))
    mol.AddConformer(conf)
    assert TransitionState([0, 1, 2], active_window=0.3).active_pairs(mol) == [(0, 2)]


def test_attack_face_needs_a_clear_height(aldol):
    ensemble, ctx = _run(aldol, TransitionState(REACTING, active_window=0.25), n=2)
    conf_id = ensemble.conf_ids[0]
    conf = ensemble.mol.GetConformer(conf_id)
    p = conf.GetPositions()
    a, b, c = (p[i] for i in (11, 13, 32))  # the plane of C12's other neighbours
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    height = np.dot(p[10] - p[12], normal)
    # C10 just across the plane (0.1 A): no clear face, not flagged ...
    conf.SetAtomPosition(
        10, (p[10] - (height + 0.1 * np.sign(height)) * normal).tolist()
    )
    assert AttackFace().validate(ctx, ensemble) == {}
    # ... 0.5 A across: flagged.
    conf.SetAtomPosition(
        10, (p[10] - (height + 0.5 * np.sign(height)) * normal).tolist()
    )
    assert "10 on 12" in AttackFace().validate(ctx, ensemble)[conf_id]
    assert AttackFace(min_height=0.6).validate(ctx, ensemble) == {}


def test_window_problems_are_explained(sn2_ts, monkeypatch, caplog):
    import racerts.embed.dg as dg
    from racerts.embed.bounds import INCONSISTENT_RESTRAINTS

    mol = build_mol(sn2_ts, -1, [0, 1, 2], input_smiles=["CCl", "[Cl-]"])
    with caplog.at_level("WARNING"):  # a C-Cl window down to 0.5 A
        TransitionState(
            [0, 1, 2], active_bonds=[(0, 2)], active_window=(0.5, 0.8)
        ).restraints(mol)
    assert "below 0.9 times its covalent length" in caplog.text

    def inconsistent(*args, **kwargs):
        raise ValueError(INCONSISTENT_RESTRAINTS)

    monkeypatch.setattr(dg, "bounds_matrix", inconsistent)
    ctx = racerts.Context.create(
        mol, TransitionState([0, 1, 2], active_bonds=[(0, 2)], active_window=0.3)
    )
    with pytest.raises(ValueError, match="narrower active_window"):
        racerts.Embed(n_conformers=2).run(ctx)
