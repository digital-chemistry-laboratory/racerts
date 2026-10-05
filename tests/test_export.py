"""Restraints for other programs: xtb, CREST, ORCA and JSON (the "constraints only" exit)."""

import json
import logging
import os

import pytest

import racerts
from racerts import TransitionState
from racerts.cli import main
from racerts.restraints import DistanceRestraint, RestraintSet
from racerts.restraints.export import (
    ExportRestraints,
    export_restraints,
    xtb_force_constant,
)
from racerts.system import build_mol
from racerts.system.roles import containment

SMILES = ["CCl", "[Cl-]", "O"]
CONTACT = DistanceRestraint(2, 6, 2.9, 3.4)  # Cl2...O6 of the water: 3.15 A


@pytest.fixture
def ctx(sn2_ts_water):
    mol = build_mol(sn2_ts_water, -1, [0, 1, 2], input_smiles=SMILES)
    return racerts.Context.create(
        mol, TransitionState([0, 1, 2]), restraints=RestraintSet([CONTACT])
    )


def test_force_constants_in_xtb_units():
    # xtb: E = fc (d - d0)^2 in Eh and bohr (measured with xtb 6.6.1); racerts:
    # E = 1/2 k (d - bound)^2 in kcal/mol and A.
    assert xtb_force_constant(20.0) == pytest.approx(
        10.0 / 627.5094740629 * 0.529177210903**2
    )


def test_xtb(ctx):
    text = export_restraints(ctx, "xtb")
    assert "$constrain" in text and text.rstrip().endswith("$end")
    assert "atoms: 1-6" in text  # the frozen atoms, 1-based
    assert "distance: 3, 7, 3.150" in text  # the window centre
    assert f"force constant={xtb_force_constant(20.0):.6f}" in text
    assert "reference=" not in text and "$metadyn" not in text


def test_crest_takes_a_reference_and_samples_the_rest(ctx, tmp_path):
    reference = str(tmp_path / "coord.ref.xyz")
    text = export_restraints(ctx, "crest", reference=reference)
    assert f"reference={reference}" in text
    assert "$metadyn\n   atoms: 7-9" in text


def test_orca(ctx):
    text = export_restraints(ctx, "orca")
    assert text.startswith("%geom\n  Constraints\n")
    assert "{C 0 C}" in text and "{C 5 C}" in text  # the frozen atoms, 0-based
    assert "{B 2 6 3.150 C}" in text


def test_json_round_trip(ctx):
    data = json.loads(export_restraints(ctx, "json"))
    assert data["frozen"]["hard"] == [0, 1, 2, 3, 4, 5]
    assert RestraintSet.from_json(json.dumps(data["restraints"])) == ctx.restraints


def test_windows_without_a_lower_bound_are_left_out(ctx, caplog):
    contained = containment(ctx.mol, ctx.frozen, 6.0)
    ctx = racerts.Context.create(
        ctx.mol, ctx.task, restraints=ctx.restraints.merge(contained)
    )
    with caplog.at_level(logging.WARNING, logger="racerts"):
        text = export_restraints(ctx, "xtb")
    assert text.count("distance:") == 1 and "no lower bound" in caplog.text


def test_several_force_constants(ctx, caplog):
    stiff = DistanceRestraint(0, 6, 5.0, 6.0, force_constant=100.0)
    ctx = racerts.Context.create(
        ctx.mol, ctx.task, restraints=ctx.restraints.merge([stiff])
    )
    with caplog.at_level(logging.WARNING, logger="racerts"):
        text = export_restraints(ctx, "xtb")
    assert f"force constant={xtb_force_constant(100.0):.6f}" in text
    assert "one force constant" in caplog.text


def test_unknown_format(ctx):
    with pytest.raises(ValueError, match="format"):
        export_restraints(ctx, "gaussian")


def test_the_stage_writes_and_stops(ctx, tmp_path):
    path = str(tmp_path / "restraints.xcontrol")
    out = racerts.Pipeline([ExportRestraints("crest", path)]).run(ctx)
    assert len(out) == 0
    assert os.path.exists(path) and os.path.exists(path + ".ref.xyz")
    assert "reference=" in open(path).read()


def test_cli(sn2_ts_water, tmp_path):
    path = str(tmp_path / "c.xcontrol")
    main(
        [
            "ts", sn2_ts_water, "-r", "0", "1", "2", "-c", "-1",
            "--smiles", "CCl.[Cl-].O", "--restraint", "2", "6", "3.15",
            "--export-restraints", "crest", "--export-to", path,
            "-o", str(tmp_path / "ensemble.xyz"),
        ]
    )  # fmt: skip
    assert "distance: 3, 7, 3.150" in open(path).read()
    assert not os.path.exists(tmp_path / "ensemble.xyz")  # nothing embedded
