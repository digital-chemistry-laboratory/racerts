"""Validation: the Validate stage and the built-in validators."""

import logging

import pytest
from rdkit.Geometry import Point3D

import racerts
from racerts import TransitionState, Validate
from racerts.validate import (
    FrozenCore,
    validator,
)


@pytest.fixture
def ts_ensemble(hept_1_ene_ts):
    """Four MMFF conformers of the TS of ex.xyz, with their context."""
    ctx = racerts.Context.create(hept_1_ene_ts, TransitionState([3, 4, 5]))
    ensemble = racerts.Pipeline([racerts.Embed(n_conformers=4), racerts.Refine()]).run(
        ctx
    )
    return ensemble, ctx


def _move(ensemble, conf_id, atom, shift):
    conf = ensemble.mol.GetConformer(conf_id)
    p = conf.GetAtomPosition(atom)
    conf.SetAtomPosition(atom, Point3D(p.x + shift[0], p.y + shift[1], p.z + shift[2]))


def test_frozen_core(ts_ensemble):
    ensemble, ctx = ts_ensemble
    assert FrozenCore().validate(ctx, ensemble) == {}
    conf_id = ensemble.conf_ids[2]
    _move(ensemble, conf_id, ctx.frozen.hard[0], (0.2, 0.0, 0.0))
    reasons = FrozenCore(tolerance=0.01).validate(ctx, ensemble)
    assert list(reasons) == [conf_id] and "moved by up to 0.1" in reasons[conf_id]


def test_validate_drops_or_flags(ts_ensemble, caplog):
    ensemble, ctx = ts_ensemble
    bad = ensemble.conf_ids[0]

    def not_the_first(mol, conf_id):
        return conf_id != bad  # False: fails

    flagged = Validate(validator(not_the_first), FrozenCore(), on_fail="flag").run(
        ctx, ensemble.copy()
    )
    assert flagged.conf_ids == ensemble.conf_ids
    assert flagged.provenance(bad)["validation"] == {
        "not_the_first": "failed",
        "frozen_core": "ok",
    }

    with caplog.at_level(logging.WARNING):
        dropped = Validate(validator(not_the_first)).run(ctx, ensemble.copy())
    assert bad not in dropped.conf_ids and len(dropped) == len(ensemble) - 1
    assert "1 of 4 conformers failed validation" in caplog.text


def test_validate_raises_if_none_passes(ts_ensemble):
    ensemble, ctx = ts_ensemble
    always = validator(lambda mol, conf_id: "no", name="never")
    with pytest.raises(RuntimeError, match="No conformer passed validation.*never: no"):
        Validate(always).run(ctx, ensemble.copy())
    assert len(Validate(always, require_any=False).run(ctx, ensemble.copy())) == 0

    with pytest.raises(TypeError, match="not a Validator"):
        Validate(lambda mol, conf_id: None)
    with pytest.raises(ValueError, match="at least one"):
        Validate()
