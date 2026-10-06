"""The scripts of examples/ run and write their files."""

import runpy
from pathlib import Path

import pytest

EXAMPLES = Path(__file__).resolve().parents[1] / "examples"
pytestmark = pytest.mark.skipif(
    not EXAMPLES.is_dir(), reason="no examples next to the tests"
)


@pytest.mark.parametrize(
    "script, written",
    [
        ("transition_state.py", ["ts_conformers.xyz"]),
        ("ground_state.py", ["proline.xyz", "settings.json"]),
        ("restraints.py", []),
        ("swap.py", ["butylbiphenyl.xyz"]),
        pytest.param(
            "other_levels.py",
            ["glycol_gfn2.xyz"],
            marks=[pytest.mark.ase, pytest.mark.xtb],
        ),
    ],
)
def test_an_example_runs(script, written, tmp_path, monkeypatch, capsys):
    if script == "other_levels.py":
        pytest.importorskip("tblite")
    monkeypatch.chdir(tmp_path)
    runpy.run_path(str(EXAMPLES / script), run_name="__main__")
    assert "conformers" in capsys.readouterr().out  # each prints a summary
    for name in written:
        assert (tmp_path / name).stat().st_size > 0
