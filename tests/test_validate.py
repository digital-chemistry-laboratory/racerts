"""Validation: the Validate stage and the built-in validators."""

import logging

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

import racerts
from racerts import TransitionState, Validate
from racerts.validate import (
    Connectivity,
    FrozenCore,
    IdentityFilter,
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


def _gs(smiles, n=3, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(mol, n, randomSeed=seed)
    ctx = racerts.Context.create(mol, racerts.GroundState())
    return racerts.ConformerEnsemble(Chem.Mol(mol)), ctx


def _move(ensemble, conf_id, atom, shift):
    conf = ensemble.mol.GetConformer(conf_id)
    p = conf.GetAtomPosition(atom)
    conf.SetAtomPosition(atom, Point3D(p.x + shift[0], p.y + shift[1], p.z + shift[2]))


def _mirror(ensemble, conf_id):
    conf = ensemble.mol.GetConformer(conf_id)
    for i, p in enumerate(conf.GetPositions()):
        conf.SetAtomPosition(i, Point3D(-p[0], p[1], p[2]))


def test_connectivity_passes_a_ts_ensemble(ts_ensemble):
    ensemble, ctx = ts_ensemble
    assert Connectivity().validate(ctx, ensemble) == {}


def test_bonds_between_reacting_atoms_are_exempt(ts_ensemble):
    # A graph without the C3-C4 bond, which the geometry has: 3 and 4 are reacting
    # atoms, whose bonds form or break, so the difference is not checked.
    ensemble, ctx = ts_ensemble
    graph = Chem.RWMol(ensemble.mol)
    graph.RemoveBond(3, 4)
    ensemble = racerts.ConformerEnsemble(graph.GetMol())
    assert Connectivity().validate(ctx, ensemble) == {}
    reasons = Connectivity(exempt=(), stereo=False).validate(ctx, ensemble)
    assert set(reasons.values()) == {"bonds [(3, 4)] extra"}


def test_connectivity_finds_a_broken_bond(ts_ensemble):
    ensemble, ctx = ts_ensemble
    conf_id = ensemble.conf_ids[1]
    _move(ensemble, conf_id, 0, (10.0, 0.0, 0.0))  # a terminal carbon pulled away
    reasons = Connectivity().validate(ctx, ensemble)
    assert list(reasons) == [conf_id]
    assert reasons[conf_id].startswith("bonds [(0, 1), ")
    assert reasons[conf_id].endswith("missing")


def test_connectivity_checks_the_specified_stereo():
    ensemble, ctx = _gs("C[C@H](O)CC")
    _mirror(ensemble, 1)
    assert Connectivity().validate(ctx, ensemble) == {1: "stereo of atom 1 inverted"}
    assert Connectivity(stereo=False).validate(ctx, ensemble) == {}

    # Unspecified stereo is a wildcard.
    ensemble, ctx = _gs("CC(O)CC")
    _mirror(ensemble, 1)
    assert Connectivity().validate(ctx, ensemble) == {}


def test_connectivity_checks_double_bonds():
    ensemble, ctx = _gs("C/C=C/C", n=1)
    z = Chem.AddHs(Chem.MolFromSmiles("C/C=C\\C"))  # the same atom order
    AllChem.EmbedMolecule(z, randomSeed=3)
    ensemble.mol.AddConformer(z.GetConformer(), assignId=True)
    reasons = Connectivity().validate(ctx, ensemble)
    assert list(reasons) == [1] and "bond 1" in reasons[1]


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


def test_identity_filter_in_a_pipeline(hept_1_ene_ts):
    pipeline = racerts.Pipeline(
        [racerts.Embed(n_conformers=5), racerts.Refine(), IdentityFilter()]
    )
    ensemble = racerts.generate(
        hept_1_ene_ts, TransitionState([3, 4, 5]), pipeline=pipeline
    )
    assert len(ensemble) == 5
    assert all(
        ensemble.provenance(i)["validation"] == {"connectivity": "ok"}
        for i in ensemble.conf_ids
    )


def test_stereo_only_check():
    ensemble, ctx = _gs("C[C@H](O)CC")
    _move(ensemble, 1, 4, (10.0, 0.0, 0.0))  # a broken bond ...
    _mirror(ensemble, 2)  # ... and a mirror image
    reasons = Connectivity(bonds=False).validate(ctx, ensemble)
    assert reasons == {2: "stereo of atom 1 inverted"}
    assert Connectivity(bonds=False).name == "stereo"
    with pytest.raises(ValueError, match="bonds or stereo"):
        Connectivity(bonds=False, stereo=False)
