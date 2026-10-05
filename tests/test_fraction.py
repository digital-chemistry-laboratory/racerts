"""restraint_fraction: biased and unbiased embedding batches in one ensemble."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import racerts
from racerts import TransitionState
from racerts.restraints import DistanceRestraint, RestraintSet
from racerts.restraints.model import applying
from racerts.system import build_mol
from racerts.validate import RestraintViolation

WINDOW = (2.6, 3.2)  # O...O of hexane-1,6-diol


def _diol_ctx(seed=3):
    mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCCCO"))
    AllChem.EmbedMolecule(mol, randomSeed=seed)
    contact = DistanceRestraint(0, 7, *WINDOW, source="contact")
    return racerts.Context.create(
        mol, racerts.GroundState(), restraints=RestraintSet([contact])
    )


def _o_o(ensemble, conf_id):
    p = ensemble.mol.GetConformer(conf_id).GetPositions()
    return float(np.linalg.norm(p[0] - p[7]))


def _with_contact(ensemble):
    return [
        i
        for i in ensemble.conf_ids
        if "contact:0-7" in ensemble.provenance(i)["restraint_subset"]
    ]


def test_batches_take_the_contact_at_random():
    ctx = _diol_ctx()
    ensemble = racerts.Embed(n_conformers=60, restraint_fraction=0.5).run(ctx)
    with_contact = _with_contact(ensemble)
    assert 0.2 < len(with_contact) / len(ensemble) < 0.8
    inside = [
        WINDOW[0] - 0.1 <= _o_o(ensemble, i) <= WINDOW[1] + 0.1
        for i in ensemble.conf_ids
    ]
    biased = np.mean(
        [inside[k] for k, i in enumerate(ensemble.conf_ids) if i in with_contact]
    )
    free = np.mean(
        [inside[k] for k, i in enumerate(ensemble.conf_ids) if i not in with_contact]
    )
    assert biased > 0.8 and free < 0.5
    # The fraction is the share of the batches with the contact, not without it.
    mostly = racerts.Embed(n_conformers=40, restraint_fraction=0.9).run(ctx)
    assert len(_with_contact(mostly)) / len(mostly) >= 0.7


def test_refinement_and_the_gate_follow_the_batches():
    ctx = _diol_ctx()
    ensemble = racerts.Embed(n_conformers=30, restraint_fraction=0.5).run(ctx)
    ensemble = racerts.Refine(racerts.refine.MMFFOptimizer(converge=True)).run(
        ctx, ensemble
    )
    held = _with_contact(ensemble)
    free = [i for i in ensemble.conf_ids if i not in held]
    assert free and any(_o_o(ensemble, i) > WINDOW[1] + 0.6 for i in free)
    # Only conformers whose batch took the contact can break it (k = 20 is soft).
    assert set(RestraintViolation().validate(ctx, ensemble)) <= set(held)
    inside = [_o_o(ensemble, i) < WINDOW[1] + 0.3 for i in held]
    assert np.mean(inside) > 0.7


def test_applying():
    contact = DistanceRestraint(0, 7, *WINDOW, source="contact")
    user = DistanceRestraint(1, 6, 3.0, 4.0)
    assert applying([contact, user], {}) == [contact, user]
    assert applying([contact, user], {"restraint_subset": []}) == [user]
    assert applying([contact, user], {"restraint_subset": ["contact:0-7"]}) == [
        contact,
        user,
    ]


def test_without_a_fraction_nothing_changes():
    ctx = _diol_ctx()
    ensemble = racerts.Embed(n_conformers=10).run(ctx)
    assert all(
        "restraint_subset" not in ensemble.provenance(i) for i in ensemble.conf_ids
    )
    with pytest.raises(ValueError, match="restraint_fraction"):
        racerts.Embed(restraint_fraction=0.0)


def test_targets_and_hints_combine(sn2_ts_water):
    # A water H...Cl hint next to stratified windows on the C-Cl bonds (graph hints
    # leave out charged partners, so it is given here).
    mol = build_mol(sn2_ts_water, -1, [0, 1, 2], input_smiles=["CCl", "[Cl-]", "O"])
    task = TransitionState(
        [0, 1, 2], active_bonds=[(0, 1), (0, 2)], active_window=0.2, stratify=2
    )
    hint = DistanceRestraint(2, 7, 2.0, 2.6, stage="embed", source="hint")
    ctx = racerts.Context.create(mol, task, restraints=RestraintSet([hint]))
    assert any(r.source == "hint" for r in ctx.restraints)
    ensemble = racerts.Embed(n_conformers=12).run(ctx)
    provenance = [ensemble.provenance(i) for i in ensemble.conf_ids]
    assert len({str(p["active_bond_targets"]) for p in provenance}) == 2
    assert any(p.get("active_restraints") for p in provenance)


def test_config_and_cli_take_the_fraction(sn2_ts_water, tmp_path):
    config = racerts.PipelineConfig.from_dict({"restraints": {"fraction": 0.5}})
    assert config.build(TransitionState([0, 1, 2])).stages[0].restraint_fraction == 0.5
    for bad in (0.0, 1.5):
        with pytest.raises(ValueError, match="fraction"):
            racerts.PipelineConfig.from_dict({"restraints": {"fraction": bad}})
    from racerts.cli import main

    out = str(tmp_path / "out.xyz")
    main(
        [
            "ts", sn2_ts_water, "-r", "0", "1", "2", "-c", "-1",
            "--smiles", "CCl.[Cl-].O", "--keep-hbonds", "--restraint-fraction", "0.5",
            "--n-conformers", "6", "-o", out,
        ]
    )  # fmt: skip
    assert open(out).read()
