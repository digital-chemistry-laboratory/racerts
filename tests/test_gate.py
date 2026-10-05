"""The validity gate: clashes, broken restraints, and Validate's warning threshold."""

import logging

import numpy as np
import pytest
from rdkit import Chem

import racerts
from racerts import TransitionState
from racerts.restraints import DistanceRestraint, RestraintSet
from racerts.system import build_mol
from racerts.validate import Clash, RestraintViolation, Validate, gate

SN2_SMILES = ["CCl", "[Cl-]", "O"]
REACTING = [0, 1, 2]  # C0, Cl1, Cl2
WATER_O = 6


@pytest.fixture
def water_ts(sn2_ts_water):
    return build_mol(sn2_ts_water, -1, REACTING, input_smiles=SN2_SMILES)


def _ensemble(ctx, geometries):
    """An ensemble of ctx.mol with the given geometries as conformers 0, 1, ..."""
    mol = Chem.Mol(ctx.mol)
    mol.RemoveAllConformers()
    for positions in geometries:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, p in enumerate(positions):
            conf.SetAtomPosition(i, p.tolist())
        mol.AddConformer(conf, assignId=True)
    return racerts.ConformerEnsemble(mol)


def _seed(ctx):
    return ctx.mol.GetConformer().GetPositions()


def test_clash_between_fragments(water_ts):
    ctx = racerts.Context.create(water_ts, TransitionState(REACTING))
    seed = _seed(ctx)
    squeezed = seed.copy()
    # The water's O onto the nucleophile: 1.5 A from Cl2, below 0.7 x (1.52 + 1.75).
    direction = seed[WATER_O] - seed[2]
    squeezed[WATER_O:] += (seed[2] + 1.5 * direction / np.linalg.norm(direction)) - (
        seed[WATER_O]
    )
    reasons = Clash().validate(ctx, _ensemble(ctx, [seed, squeezed]))
    assert list(reasons) == [1]
    assert "atoms 2 and 6" in reasons[1]


def test_the_reacting_atoms_do_not_clash(sn2_ts_symmetric):
    # Both C...Cl (2.32 A, not bonds of the graph) are shorter than 0.7 x the vdW sum;
    # pairs of core atoms are exempt.
    mol = build_mol(sn2_ts_symmetric, -1, REACTING, input_smiles=["CCl", "[Cl-]"])
    ctx = racerts.Context.create(mol, TransitionState(REACTING))
    seed = _seed(ctx)
    assert np.linalg.norm(seed[0] - seed[2]) < 0.7 * (1.7 + 1.75)
    assert Clash().validate(ctx, _ensemble(ctx, [seed])) == {}
    graph = Chem.Mol(mol)
    graph.RemoveAllConformers()  # no reference either, whose contacts would count
    unexempt = racerts.Context.create(graph, racerts.GroundState())
    assert Clash().validate(unexempt, _ensemble(ctx, [seed]))  # a clash otherwise


def test_a_contact_of_the_reference_is_no_clash(water_ts):
    # Pairs as close in the reference (e.g. a coordination the graph lacks) count only
    # if they come more than 0.2 A closer.
    ctx = racerts.Context.create(water_ts, TransitionState(REACTING))
    seed = _seed(ctx)
    clash = Clash(factor=1.5)  # makes the seed's own contacts "clashes"
    assert clash.validate(ctx, _ensemble(ctx, [seed])) == {}
    closer = seed.copy()
    closer[WATER_O:] += (
        0.3 * (seed[2] - seed[WATER_O]) / np.linalg.norm(seed[2] - seed[WATER_O])
    )
    assert list(clash.validate(ctx, _ensemble(ctx, [seed, closer]))) == [1]


def test_restraint_violation(water_ts):
    contact = DistanceRestraint(2, WATER_O, 2.9, 3.4)
    ctx = racerts.Context.create(
        water_ts, TransitionState(REACTING), restraints=RestraintSet([contact])
    )
    seed = _seed(ctx)
    d = np.linalg.norm(seed[WATER_O] - seed[2])
    assert 2.9 <= d <= 3.4
    away = seed.copy()
    away[WATER_O:] += (seed[WATER_O] - seed[2]) / d * (3.4 + 0.6 - d)  # 0.6 beyond
    near = seed.copy()
    near[WATER_O:] += (seed[WATER_O] - seed[2]) / d * (3.4 + 0.3 - d)  # 0.3 beyond
    reasons = RestraintViolation(tolerance=0.5).validate(
        ctx, _ensemble(ctx, [seed, away, near])
    )
    assert list(reasons) == [1]
    assert "0.60 A" in reasons[1]


def test_embedding_only_restraints_are_not_checked(water_ts):
    hint = DistanceRestraint(2, WATER_O, 2.9, 3.0, stage="embed")
    ctx = racerts.Context.create(
        water_ts, TransitionState(REACTING), restraints=RestraintSet([hint])
    )
    away = _seed(ctx).copy()
    away[WATER_O:] += 5.0
    assert RestraintViolation().validate(ctx, _ensemble(ctx, [away])) == {}


def test_soft_atoms_too_far_from_the_reference(water_ts):
    task = racerts.Constrained(hard=[0, 1, 2, 3, 4, 5], soft=[6, 7, 8])
    ctx = racerts.Context.create(water_ts, task)
    moved = _seed(ctx).copy()
    moved[6:] += np.array([0.0, 0.0, 1.0])  # 1.0 A: 0.7 beyond the 0.3 A tolerance
    reasons = RestraintViolation().validate(ctx, _ensemble(ctx, [_seed(ctx), moved]))
    assert list(reasons) == [1]


def _failing(n_bad, n):
    """A validator that fails the first n_bad of n conformers."""

    class Fails:
        name = "fails"

        def validate(self, ctx, ensemble):
            return {i: "bad" for i in ensemble.conf_ids[:n_bad]}

    return Fails()


@pytest.mark.parametrize("n_bad, level", [(2, logging.INFO), (4, logging.WARNING)])
def test_warn_above(water_ts, caplog, n_bad, level):
    ctx = racerts.Context.create(water_ts, TransitionState(REACTING))
    ensemble = _ensemble(ctx, [_seed(ctx)] * 10)
    with caplog.at_level(logging.INFO, logger="racerts"):
        Validate(_failing(n_bad, 10), warn_above=0.3).run(ctx, ensemble)
    records = [r for r in caplog.records if "failed validation" in r.getMessage()]
    assert [r.levelno for r in records] == [level]
    assert "fails: " in records[0].getMessage()  # the count per check
    assert len(ensemble) == 10 - n_bad


def test_any_failure_warns_by_default(water_ts, caplog):
    ctx = racerts.Context.create(water_ts, TransitionState(REACTING))
    ensemble = _ensemble(ctx, [_seed(ctx)] * 10)
    with caplog.at_level(logging.INFO, logger="racerts"):
        Validate(_failing(1, 10)).run(ctx, ensemble)
    assert any(
        r.levelno == logging.WARNING and "failed validation" in r.getMessage()
        for r in caplog.records
    )


def test_warn_above_is_a_share():
    with pytest.raises(ValueError, match="warn_above"):
        Validate(Clash(), warn_above=1.5)


def test_the_gate(water_ts):
    ctx = racerts.Context.create(water_ts, TransitionState(REACTING))
    seed = _seed(ctx)
    moved_core = seed.copy()
    moved_core[3] += 0.5  # a hard hydrogen of the TS
    squeezed = seed.copy()
    squeezed[WATER_O:] += seed[2] - seed[WATER_O] + np.array([0.0, 0.0, 1.2])
    ensemble = _ensemble(ctx, [seed, moved_core, squeezed])
    checked = gate().run(ctx, ensemble)
    assert checked.conf_ids == [0]
    validation = checked.provenance(0)["validation"]
    assert set(validation) == {"frozen_core", "connectivity", "clash", "restraints"}


def test_a_graph_that_does_not_fit_the_reference_is_named(water_ts, caplog):
    # A bond in the graph that the reference geometry lacks: the graph is wrong
    # (e.g. a wrong atom mapping of the SMILES), not the conformers.
    wrong = Chem.RWMol(water_ts)
    wrong.AddBond(2, WATER_O, Chem.BondType.SINGLE)
    ctx = racerts.Context.create(wrong.GetMol(), TransitionState([0, 1]))
    with caplog.at_level(logging.WARNING, logger="racerts"):
        reasons = racerts.validate.Connectivity().validate(
            ctx, _ensemble(ctx, [_seed(ctx)] * 2)
        )
    assert len(reasons) == 2
    assert "reference geometry" in caplog.text and "(2, 6)" in caplog.text
