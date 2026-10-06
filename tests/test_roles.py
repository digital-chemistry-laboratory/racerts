"""Fragment roles (reactive, anchored, contained, free) and containment restraints."""

import logging

import numpy as np
import pytest
from rdkit import Chem

import racerts
from racerts import TransitionState
from racerts.exploit import rigid_moves
from racerts.restraints import DistanceRestraint, RestraintSet
from racerts.system import build_mol
from racerts.system.roles import (
    ANCHORED,
    CONTAINED,
    FREE,
    REACTIVE,
    containment,
    fragment_roles,
)

SMILES = ["CCl", "[Cl-]", "O", "O"]
REACTING = [0, 1, 2]
WATER_1, WATER_2 = (6, 7, 8), (9, 10, 11)


@pytest.fixture
def two_waters(sn2_ts_two_waters):
    return build_mol(sn2_ts_two_waters, -1, REACTING, input_smiles=SMILES)


def _ctx(mol, restraints=(), roles=None):
    return racerts.Context.create(
        mol, TransitionState(REACTING), restraints=RestraintSet(restraints), roles=roles
    )


def _roles(ctx):
    return {f.atoms: (f.role, f.anchors) for f in ctx.fragments}


def test_fragment_roles(two_waters):
    # -- without restraints the waters are free
    roles = _roles(_ctx(two_waters))
    assert roles[WATER_1] == (FREE, ()) and roles[WATER_2] == (FREE, ())
    assert roles[(2,)][0] == REACTIVE  # the nucleophile holds reacting atoms

    # -- anchored through a chain
    # water 1 donates to the nucleophile (H7...Cl2); water 2 donates to water 1.
    held = [DistanceRestraint(2, 7, 2.0, 2.4), DistanceRestraint(6, 10, 1.8, 2.2)]
    roles = _roles(_ctx(two_waters, held))
    assert roles[WATER_1] == (ANCHORED, (7,))
    assert roles[WATER_2] == (ANCHORED, (10,))

    # -- embedding only hints do not anchor
    hint = DistanceRestraint(2, 7, 2.0, 2.4, stage="embed", source="hint")
    assert _roles(_ctx(two_waters, [hint]))[WATER_1] == (FREE, ())

    # -- containment
    ctx = _ctx(two_waters)
    restraints = containment(ctx.mol, ctx.frozen, 6.0)
    assert len(restraints) == 2
    for r in restraints:
        assert r.source == "contain" and (r.lower, r.upper) == (0.0, 6.0)
        assert r.first in REACTING or r.second in REACTING  # to the core's center
    roles = _roles(_ctx(two_waters, restraints))
    assert roles[WATER_1][0] == CONTAINED and roles[WATER_2][0] == CONTAINED

    # -- an anchored fragment needs no containment
    ctx = _ctx(two_waters, [DistanceRestraint(2, 7, 2.0, 2.4)])
    contained = containment(ctx.mol, ctx.frozen, 6.0, ctx.restraints)
    assert len(contained) == 1 and set(next(iter(contained)).pair) & set(WATER_2)

    # -- override
    held = [DistanceRestraint(2, 7, 2.0, 2.4)]
    roles = _roles(_ctx(two_waters, held, roles={8: FREE, 9: CONTAINED}))
    assert roles[WATER_1][0] == FREE and roles[WATER_2][0] == CONTAINED
    with pytest.raises(ValueError, match="reactive"):
        _ctx(two_waters, roles={2: FREE})
    with pytest.raises(ValueError, match="role"):
        _ctx(two_waters, roles={6: "floating"})
    with pytest.raises(ValueError, match="role"):  # reactive follows from the task
        _ctx(two_waters, roles={6: REACTIVE})
    with pytest.raises(ValueError, match="not in the molecule"):
        _ctx(two_waters, roles={99: FREE})

    # -- ground states take the largest fragment as core
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCO.O"))
    roles = fragment_roles(mol, racerts.FrozenSet())
    assert [f.role for f in roles] == [REACTIVE, FREE]


def test_what_follows_the_roles(two_waters, sn2_ts_two_waters, caplog):
    from racerts.embed.stage import conformer_count

    # -- moves follow the roles
    held = [DistanceRestraint(2, 7, 2.0, 2.4)]
    ctx = _ctx(two_waters, held, roles={9: CONTAINED})
    frozen = {*ctx.frozen.hard, *ctx.frozen.core}
    moves = {
        m.atoms: m for m in rigid_moves(ctx.mol, frozen, ctx.restraints, ctx.roles)
    }
    assert moves[WATER_1].anchors == (7,) and not moves[WATER_1].shift
    assert moves[WATER_2].anchors == (9,) and moves[WATER_2].shift  # its center, O9
    positions = ctx.mol.GetConformer().GetPositions()
    rng = np.random.default_rng(0)
    moved = positions.copy()
    moves[WATER_1].apply(moved, rng)
    assert np.abs(moved[7] - positions[7]).max() < 1e-12  # turned about H7

    # -- the fragment count follows the roles
    free = _ctx(two_waters)
    held = _ctx(two_waters, [DistanceRestraint(2, 7, 2.0, 2.4)])
    counts = [
        conformer_count(c.mol, -1, 1, "fragments", c.frozen, c.fragments)
        for c in (free, held)
    ]
    assert counts[0] - counts[1] == 3  # an anchored water only turns

    # -- config contain and roles
    caplog.clear()
    config = racerts.PipelineConfig.from_dict(
        {
            "embed": {"n_conformers": 6},
            "restraints": {"contain": 5.0, "roles": {"9": "free"}},
        }
    )
    assert config.restraints.roles == {9: "free"}  # JSON and YAML keys are strings
    with caplog.at_level(logging.INFO, logger="racerts.pipeline.runner"):
        ensemble = racerts.generate_ts(
            sn2_ts_two_waters,
            REACTING,
            charge=-1,
            smiles=".".join(SMILES),
            config=config,
        )
    assert len(ensemble) > 0
    # The roles are logged once, with the restraints in place.
    lines = [r.getMessage() for r in caplog.records if "Fragments:" in r.getMessage()]
    assert len(lines) == 1
    assert "atoms 6-8: contained at 6" in lines[0]
    assert "atoms 9-11: free" in lines[0]
    with pytest.raises(ValueError, match="contain"):
        racerts.PipelineConfig.from_dict({"restraints": {"contain": -1.0}})
    for roles in ({"9": "reactive"}, {"x": "free"}, [9, "free"]):
        with pytest.raises(ValueError, match="restraints.roles"):
            racerts.PipelineConfig.from_dict({"restraints": {"roles": roles}})
