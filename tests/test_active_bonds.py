"""Active-bond windows of TSs and the attack-face filter, on the aldol TS."""

import collections
import os
import re

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Geometry import Point3D

import racerts
from racerts import PipelineConfig, TransitionState
from racerts.prune import FamilySelector
from racerts.prune.targets import window_bins
from racerts.restraints import DistanceRestraint, RestraintSet
from racerts.system import build_mol
from racerts.validate import AttackFace

from .conftest import DATA

ALDOL = os.path.join(DATA, "aldol_ts.xyz")  # tests/data/make_aldol_ts.py
REACTING = [0, 10, 11, 12, 19]
SMILES = ["OC(=O)[C@@H]1CCCN1C(C)=C", "O=Cc1ccccc1"]
CC = (10, 12)  # the forming C-C bond, 2.2 A in the reference


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


def test_a_window_is_sampled_at_five_targets_by_default(aldol, caplog):
    assert TransitionState(REACTING).stratify == 0  # no window, no targets
    task = TransitionState(REACTING, active_bonds=[CC], active_window=0.25)
    assert task.stratify == 5
    assert TransitionState(REACTING, active_window=0.25, stratify=0).stratify == 0
    ensemble = racerts.Embed(n_conformers=10).run(racerts.Context.create(aldol, task))
    targets = collections.Counter(
        ensemble.provenance(i)["active_bond_targets"]["10-12"]
        for i in ensemble.conf_ids
    )
    assert sorted(targets) == pytest.approx([2.0, 2.1, 2.2, 2.3, 2.4], abs=1e-3)
    assert set(targets.values()) == {2}
    # Fewer conformers than the default targets: one target each, and no warning (the
    # number of targets was not asked for).
    with caplog.at_level("WARNING"):
        few = racerts.Embed(n_conformers=3).run(racerts.Context.create(aldol, task))
    assert [
        few.provenance(i)["active_bond_targets"]["10-12"] for i in few.conf_ids
    ] == pytest.approx([2.0333, 2.2, 2.3667], abs=1e-3)
    assert "targets" not in caplog.text
    # The same task for renumbered atoms keeps the choice.
    same = dict(zip(range(aldol.GetNumAtoms()), range(aldol.GetNumAtoms())))
    assert task.remap(same).stratify == 5 and not task.remap(same).stratify_given
    assert (
        TransitionState(REACTING, active_window=0.2, stratify=3)
        .remap(same)
        .stratify_given
    )


def test_unstratified_window(aldol):
    ensemble, ctx = _run(
        aldol, TransitionState(REACTING, active_window=0.25, stratify=0)
    )
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
    # The midpoints of five equal parts of the window: none on its edges.
    assert sorted(targets) == pytest.approx([2.09, 2.27, 2.45, 2.63, 2.81])
    assert all(count == 5 for count in targets.values())

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
    lengths = re.search(
        r"active bond 10-12: min (\S+), median \S+, max (\S+) A", pruned.summary()
    )
    assert (
        lengths and float(lengths[1]) < 2.2 and float(lengths[2]) > 2.7
    )  # all targets


def test_targets_are_the_midpoints_of_equal_parts(aldol):
    def lengths(task, count=None):
        return [round(t[CC], 4) for t in task.targets(aldol, count)]

    # +/- 0.25 A around the reference (2.2 A): the reference is a target for odd k.
    around = TransitionState(
        REACTING, active_bonds=[CC], active_window=0.25, stratify=5
    )
    assert lengths(around) == pytest.approx([2.0, 2.1, 2.2, 2.3, 2.4], abs=1e-3)
    three = TransitionState(REACTING, active_bonds=[CC], active_window=0.3, stratify=3)
    assert lengths(three) == pytest.approx([2.0, 2.2, 2.4], abs=1e-3)
    two = TransitionState(
        REACTING, active_bonds=[CC], active_window=(2.0, 2.8), stratify=2
    )
    assert lengths(two) == pytest.approx([2.2, 2.6])
    # Fewer conformers than targets: as many parts as conformers, not the first ones.
    assert lengths(around, 2) == pytest.approx([2.075, 2.325], abs=1e-3)
    assert lengths(around, 1) == pytest.approx([2.2], abs=1e-3)
    assert lengths(around, 9) == lengths(around)


def test_several_bonds_take_their_targets_as_a_latin_hypercube(aldol):
    from racerts.task.transition_state import target_design

    # Two bonds, five targets each: with target i of every bond in batch i the two
    # lengths would only vary together (short-short to long-long). Every bond still
    # takes each of its targets once; which ones go together spreads the batches over
    # the box of the two windows.
    task = TransitionState(REACTING, active_window=0.25, stratify=5)
    pairs = task.active_pairs(aldol)
    assert pairs == [CC, (11, 19)]
    positions = aldol.GetConformer().GetPositions()
    reference = {(a, b): np.linalg.norm(positions[a] - positions[b]) for a, b in pairs}
    offsets = [
        tuple(round(float(batch[pair] - reference[pair]), 3) for pair in pairs)
        for batch in task.targets(aldol)
    ]
    assert offsets == [(-0.2, -0.1), (-0.1, 0.2), (0.0, 0.0), (0.1, -0.2), (0.2, 0.1)]

    # The table behind it is fixed for the numbers of targets and bonds.
    assert target_design(5, 2) == ((0, 1), (1, 4), (2, 2), (3, 0), (4, 3))
    assert target_design(5, 1) == ((0,), (1,), (2,), (3,), (4,))  # one bond: as before
    assert target_design(1, 3) == ((0, 0, 0),)

    def nearest(design):
        return min(
            float(np.linalg.norm(a - b))
            for i, a in enumerate(design)
            for b in design[i + 1 :]
        )

    for k, m in ((2, 2), (3, 2), (4, 2), (5, 3), (6, 2), (7, 3), (9, 2), (12, 3)):
        design = np.array(target_design(k, m), dtype=float)
        assert design.shape == (k, m)
        assert all(sorted(column) == list(range(k)) for column in design.T)
        # No two batches closer than neighbours on the diagonal are.
        assert nearest(design) >= np.sqrt(m) - 1e-9
    spread = np.array(target_design(5, 2), dtype=float)
    assert nearest(spread) == pytest.approx(np.sqrt(5.0))  # the diagonal: sqrt(2)
    assert abs(np.corrcoef(spread.T)[0, 1]) < 1e-9

    # Fewer conformers than targets: as many batches as conformers.
    few = task.targets(aldol, 3)
    assert len(few) == 3 and len({round(batch[CC], 4) for batch in few}) == 3

    # Every conformer is embedded at the targets of its batch.
    ensemble, _ = _run(aldol, task, n=10)
    batches = collections.Counter(
        tuple(sorted(ensemble.provenance(i)["active_bond_targets"].items()))
        for i in ensemble.conf_ids
    )
    assert len(batches) == 5 and set(batches.values()) == {2}
    for conf_id in ensemble.conf_ids:
        for bond, target in ensemble.provenance(conf_id)["active_bond_targets"].items():
            pair = tuple(int(i) for i in bond.split("-"))
            assert abs(_length(ensemble, conf_id, pair) - target) < 0.05


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

    # The default pipeline of a windowed TS includes the filter, on refined geometries.
    def filters(task):
        stages = PipelineConfig().build(task).stages
        return [
            i
            for i, stage in enumerate(stages)
            for check in getattr(stage, "validators", ())
            if isinstance(check, AttackFace)
        ]

    assert filters(TransitionState(REACTING, active_window=0.2)) == [2]
    no_filter = TransitionState(REACTING, active_window=0.2, stereo_filter=False)
    assert filters(no_filter) == []


@pytest.mark.parametrize(
    "settings, message",
    [
        (dict(stratify=3), "stratify needs an active_window"),
        (dict(active_window=(2.9, 2.0)), "active_window"),
        (dict(active_window=0.2, neighbor_window=0.0), "neighbor_window"),
        (dict(active_window=-0.1), "active_window must be positive"),
        (dict(active_window=0.2, stratify=1), "at least 2 targets"),
        (dict(active_window=0.2, frozen_atoms=[1, 2]), "either frozen_atoms"),
        (dict(active_window=True), "active_window must be a number or"),
        (dict(active_window=0.2, stratify=2.9), "stratify must be an integer"),
        (dict(active_window=0.2, target_force_constant=0), "target_force_constant"),
        (dict(window_force_constant=-1.0), "window_force_constant must be positive"),
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
    # Named after the output, so that runs in one folder do not overwrite each other.
    lines = (tmp_path / "ts.active_bonds.csv").read_text().splitlines()
    assert lines[0] == "conf_id,energy_kcal_mol,target_10-12,length_10-12"
    assert {line.split(",")[2] for line in lines[1:]} == {"2.2250", "2.6750"}
    assert not (tmp_path / "active_bonds.csv").exists()


def test_the_ensemble_writes_its_active_bonds(aldol, tmp_path):
    ctx, ensemble = _two_targets(aldol, ENERGIES)
    path = tmp_path / "bonds.csv"
    ensemble.write_active_bonds(str(path))
    lines = path.read_text().splitlines()
    assert lines[0] == "conf_id,energy_kcal_mol,target_10-12,length_10-12"
    first = lines[1].split(",")
    assert first[:3] == [str(ensemble.conf_ids[0]), "137.7000", "2.2250"]
    # The length is that of the conformer (embedded, so only near its target).
    assert float(first[3]) == pytest.approx(
        _length(ensemble, ensemble.conf_ids[0]), abs=1e-4
    )
    # After a free refinement the targets are released: empty cells, the lengths stay.
    released = racerts.Refine(_Recording(), fallback=False, anchors=False).run(
        ctx, ensemble
    )
    released.write_active_bonds(str(path))
    assert path.read_text().splitlines()[1].split(",")[2] == ""


class _Recording(racerts.refine.BaseOptimizer):
    """Records the sources of the restraints it gets; leaves the conformers."""

    def __init__(self):
        self.sources = set()

    def _refine(self, mol, reference, anchors, restraints=()):
        self.sources.update(getattr(r, "source", "position") for r in restraints)
        for conf in mol.GetConformers():
            conf.SetDoubleProp("energy", 0.0)
        return 0


def _two_targets(aldol, energies):
    """Three conformers at each of two targets (2.225 and 2.675 A), with these
    energies."""
    task = TransitionState(
        REACTING, active_window=(2.0, 2.9), active_bonds=[CC], stratify=2
    )
    # A restraint of the user between two free atoms, at their reference distance.
    user = RestraintSet([DistanceRestraint.around(14, 2, 6.62)])
    ctx = racerts.Context.create(aldol, task, restraints=user)
    ensemble = racerts.Embed(n_conformers=6).run(ctx)
    targets = [
        ensemble.provenance(i)["active_bond_targets"]["10-12"]
        for i in ensemble.conf_ids
    ]
    assert targets == [2.225] * 3 + [2.675] * 3
    for conf_id, energy in zip(ensemble.conf_ids, energies):
        ensemble.mol.GetConformer(conf_id).SetDoubleProp("energy", energy)
    return ctx, ensemble


ENERGIES = [137.7, 158.9, 160.7, 82.8, 72.0, 94.9]  # the short target, then the long


def test_counts_and_families_are_chosen_per_target(aldol):
    # Energies at different held lengths do not compare: the best of every target
    # first, then the second best, ...
    ctx, ensemble = _two_targets(aldol, ENERGIES)
    first, second = ensemble.conf_ids[:3], ensemble.conf_ids[3:]

    kept = racerts.PruneCount(2).run(ctx, ensemble.copy())
    assert kept.conf_ids == [first[0], second[1]]
    kept = racerts.PruneCount(3).run(ctx, ensemble.copy())
    assert kept.conf_ids == [first[0], second[1], first[1]]
    kept = racerts.PruneCount(2, renumber=True).run(ctx, ensemble.copy())
    assert kept.conf_ids == [0, 1] and kept.energies().tolist() == [137.7, 72.0]
    # Without a context (or a windowed task) there is one group, ranked by energy.
    assert racerts.PruneCount(2).run(None, ensemble.copy()).conf_ids == [
        second[1],
        second[0],
    ]

    families = FamilySelector(2).run(ctx, ensemble.copy())
    assert sorted(families.conf_ids) == sorted([first[0], second[1]])


def test_a_free_refinement_releases_the_windows_of_the_task(aldol):
    ctx, ensemble = _two_targets(aldol, ENERGIES)
    held = _Recording()
    racerts.Refine(held, fallback=False).run(ctx, ensemble.copy())
    assert held.sources == {"target", "neighbor", "user"}

    free = _Recording()
    released = racerts.Refine(free, fallback=False, anchors=False).run(
        ctx, ensemble.copy()
    )
    # A free search (e.g. for the saddle point) is no longer held at the targets or
    # the neighbour windows; restraints from other sources stay.
    assert free.sources == {"user"}
    provenance = released.provenance(released.conf_ids[0])
    assert provenance["active_bond_targets"] is None
    assert provenance["released_targets"] == {"10-12": 2.225}
    assert "10-12" in provenance["active_bond_lengths"]
    # Afterwards the conformers are one group: a TS reached from two targets is one.
    assert len(set(window_bins(ctx, released).values())) == 1
    for conf_id, energy in zip(released.conf_ids, ENERGIES):
        released.mol.GetConformer(conf_id).SetDoubleProp("energy", energy)
    assert racerts.PruneCount(2).run(ctx, released).energies().tolist() == [72.0, 82.8]


# ---- regression tests ----


def test_unstratified_windows_are_pruned_per_length(aldol):
    # With one energy window over all lengths, only the long end survived.
    config = PipelineConfig.from_dict({"embed": {"n_conformers": 40}})
    ensemble = racerts.generate_ts(
        ALDOL, REACTING, smiles=SMILES, config=config, active_window=(2.0, 2.9),
        active_bonds=[CC], stratify=0,
    )  # fmt: skip
    lengths = [_length(ensemble, i) for i in ensemble.conf_ids]
    assert min(lengths) < 2.3 and max(lengths) > 2.6


def test_the_fifths_of_a_window_include_both_ends():
    from types import SimpleNamespace

    from racerts.prune.targets import window_bins

    targets = dict(enumerate([2.0, 2.19, 2.25, 2.99, 3.0]))
    task = SimpleNamespace(
        windowed=True, stratify=0, active_windows=lambda mol: {(0, 1): (2.0, 3.0)}
    )
    ensemble = SimpleNamespace(
        conf_ids=list(targets),
        provenance=lambda i: {"active_bond_targets": {"0-1": targets[i]}},
    )
    bins = window_bins(SimpleNamespace(task=task, mol=None), ensemble)
    assert [bins[i] for i in targets] == [(("0-1", k),) for k in (0, 0, 1, 4, 4)]


def test_task_windows_win_over_generated_restraints(aldol, sn2_ts_water, caplog):
    import logging

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
class _NoRestraints(racerts.refine.BaseOptimizer):
    """An optimizer that takes no restraints; it leaves the conformers as they are."""

    def _refine(self, mol, reference, anchors):
        for conf in mol.GetConformers():
            conf.SetDoubleProp("energy", 0.0)
        return 0


def test_windowed_refinement_needs_restraints(aldol):
    task = TransitionState(REACTING, active_window=0.25)
    ctx = racerts.Context.create(aldol, task)
    ensemble = racerts.Embed(n_conformers=2).run(ctx)
    plain = _NoRestraints()
    with pytest.raises(ValueError, match="anchors=False"):
        racerts.Refine(plain).run(ctx, ensemble.copy())
    racerts.Refine(plain, anchors=False).run(ctx, ensemble.copy())  # a free search


def test_attack_face_with_an_atom_in_two_active_bonds(sn2_ts):
    # SN2: C0 forms a bond to Cl2 and breaks the one to Cl1.
    mol = build_mol(sn2_ts, -1, [0, 1, 2], input_smiles=["CCl", "[Cl-]"])
    task = TransitionState([0, 1, 2], active_bonds=[(0, 1), (0, 2)], active_window=0.2)
    ctx = racerts.Context.create(mol, task)
    seed_copy = racerts.ConformerEnsemble(racerts.Context.create(mol, task).mol)
    assert AttackFace().validate(ctx, seed_copy) == {}


def test_attack_face_compares_with_the_reference_of_each_conformer(aldol):
    # Two references, the second the mirror image of the first: each is attacked
    # from its own face.
    mol = Chem.Mol(aldol)
    mirrored = Chem.Conformer(mol.GetConformer())
    for i, (x, y, z) in enumerate(mol.GetConformer().GetPositions()):
        mirrored.SetAtomPosition(i, Point3D(-x, y, z))
    ref_id = mol.AddConformer(mirrored, assignId=True)
    ctx = racerts.Context.create(mol, TransitionState(REACTING, active_window=0.25))
    ensemble = racerts.ConformerEnsemble(Chem.Mol(ctx.mol))  # the references themselves
    ensemble.add_provenance(ref_id, reference=ref_id)
    assert AttackFace().validate(ctx, ensemble) == {}
    # Compared with the first reference, the mirror image is attacked from the other face.
    ensemble.add_provenance(ref_id, reference=0)
    assert list(AttackFace().validate(ctx, ensemble)) == [ref_id]


@pytest.mark.parametrize(
    "bonds, error",
    [([(10, 99)], ValueError), ([(1, 8)], ValueError), ([(10, 10)], ValueError)],
)
def test_active_bonds_are_checked(aldol, bonds, error):
    with pytest.raises(error):
        TransitionState(REACTING, active_bonds=bonds, active_window=0.2).active_pairs(
            aldol
        )


def test_ring_13_pairs_are_not_forming_bonds():
    # SNAr and cyclization TSs: reacting ring atoms 1,3 apart (2.4 A) were taken for
    # forming bonds, and their in-plane "faces" flipped at random.
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


@pytest.mark.parametrize("stratify", [0, 5])
def test_a_window_out_of_reach_is_reported(aldol, caplog, stratify):
    # The frozen atoms let the C-C distance reach about 3.9 A.
    far = TransitionState(
        REACTING, active_bonds=[CC], active_window=(4.5, 5.5), stratify=stratify
    )
    with pytest.raises(ValueError, match="narrower active_window"):
        racerts.Embed(n_conformers=6).run(racerts.Context.create(aldol, far))
    # Partly in reach: the conformers embedded outside are named, with targets too.
    partly = TransitionState(
        REACTING, active_bonds=[CC], active_window=(3.5, 4.5), stratify=stratify
    )
    with caplog.at_level("WARNING"):
        ensemble = racerts.Embed(n_conformers=12).run(
            racerts.Context.create(aldol, partly)
        )
    assert len(ensemble) == 12
    assert re.search(r"\d+ of 12 conformers .* outside its window", caplog.text)
    held = "to their targets" if stratify else "at the edge of the window"
    assert held in caplog.text


def test_windows_take_one_reference(aldol):
    mol = Chem.Mol(aldol)
    mol.AddConformer(Chem.Conformer(mol.GetConformer()), assignId=True)
    ctx = racerts.Context.create(mol, TransitionState(REACTING, active_window=0.25))
    with pytest.raises(ValueError, match="one reference"):
        racerts.Embed(n_conformers=4, references="all").run(ctx)


def test_fewer_conformers_than_targets_is_reported(aldol, caplog):
    task = TransitionState(
        REACTING, active_bonds=[CC], active_window=(2.0, 2.9), stratify=5
    )
    with caplog.at_level("WARNING"):
        ensemble = racerts.Embed(n_conformers=3).run(
            racerts.Context.create(aldol, task)
        )
    assert len(ensemble) == 3
    assert "3 conformers for 5 targets" in caplog.text
    # They spread over the window: three parts, one conformer in the middle of each.
    targets = [
        ensemble.provenance(i)["active_bond_targets"]["10-12"]
        for i in ensemble.conf_ids
    ]
    assert targets == pytest.approx([2.15, 2.45, 2.75])


def test_window_problems_are_explained(sn2_ts, hept_1_ene_ts, caplog):
    mol = build_mol(sn2_ts, -1, [0, 1, 2], input_smiles=["CCl", "[Cl-]"])
    with caplog.at_level("WARNING"):  # a C-Cl window down to 0.5 A
        TransitionState(
            [0, 1, 2], active_bonds=[(0, 2)], active_window=(0.5, 0.8)
        ).restraints(mol)
    assert "below 0.9 times its covalent length" in caplog.text

    # C3...C5 of the ring-forming TS cannot be 3.5-4.0 A apart with its neighbours.
    task = TransitionState([3, 4, 5], active_bonds=[(3, 5)], active_window=(3.5, 4.0))
    ctx = racerts.Context.create(hept_1_ene_ts, task)
    with pytest.raises(ValueError, match="narrower active_window"):
        racerts.Embed(n_conformers=2).run(ctx)
