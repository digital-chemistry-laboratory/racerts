"""The subcommands racerts ts, gs and constrained (the legacy form: tests/legacy)."""

import os

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from racerts import PipelineConfig
from racerts.cli import main, run_subcommand

from .conftest import EX


def _frames(path):
    lines = path.read_text().splitlines()
    n = int(lines[0])
    return [lines[i : i + n + 2] for i in range(0, len(lines), n + 2)]


def test_gs_writes_ground_state_conformers(tmp_path):
    out = tmp_path / "gs.xyz"
    main(["gs", "CCO", "-n", "10", "--seed", "3", "-o", str(out)])
    frames = _frames(out)

    assert len(frames) >= 1 and int(frames[0][0]) == 9
    assert "charge=0 spin=1 multiplicity=1 energy_method=MMFFOptimizer" in frames[0][1]


def test_ts_options_override_the_config_file(tmp_path):
    config = tmp_path / "config.json"
    PipelineConfig.from_dict({"embed": {"n_conformers": 7}, "seed": 99}).to_file(
        str(config)
    )
    out = tmp_path / "ts.xyz"
    ensemble = run_subcommand(
        ["ts", EX, "-r", "3", "4", "5", "-s", "CCCCCC=C", "--config", str(config)]
        + ["--seed", "12", "--refine", "uff", "--crest-energies", "-o", str(out)]
    )

    assert ensemble.record(ensemble.best()).provenance["seed"] == 12
    assert ensemble.mol.GetProp("energy_method") == "UFFOptimizer"
    assert len(_frames(out)) == len(ensemble) <= 7
    assert float(_frames(out)[0][1]) < 0.1  # Hartree only, as in CREST files


def test_ts_rejects_invalid_settings(tmp_path):
    config = tmp_path / "config.json"
    config.write_text('{"embed": {"mode": "dm"}}')

    with pytest.raises(ValueError, match="embed.mode"):
        main(["ts", EX, "-r", "3", "4", "5", "--config", str(config)])
    with pytest.raises(SystemExit):
        main(["ts", EX])  # the reacting atoms are required


def test_constrained_keeps_the_hard_atoms_at_the_geometry_of_the_file(tmp_path):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC(C)O"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    held = mol.GetConformer().GetPositions()[[3, 4, 5]]  # the C(C)O end
    options = ["--hard", "3", "4", "5", "-n", "10", "--seed", "3"]
    out = tmp_path / "constrained.xyz"

    # -- a file with bonds
    with_bonds = tmp_path / "pentan-2-ol.mol"
    Chem.MolToMolFile(mol, str(with_bonds))
    ensemble = run_subcommand(
        ["constrained", str(with_bonds), *options, "-o", str(out)]
    )

    assert len(_frames(out)) == len(ensemble) > 1
    for conf in ensemble.mol.GetConformers():
        assert np.abs(conf.GetPositions()[[3, 4, 5]] - held).max() < 1e-3

    # -- an xyz file takes its bonds from the SMILES, as racerts ts does
    xyz = tmp_path / "pentan-2-ol.xyz"
    Chem.MolToXYZFile(mol, str(xyz))
    from_xyz = run_subcommand(
        ["constrained", str(xyz), "-s", "CCCC(C)O", *options, "-o", str(out)]
    )

    assert from_xyz.mol.GetNumBonds() == mol.GetNumBonds() and len(from_xyz) > 1
    for conf in from_xyz.mol.GetConformers():
        assert np.abs(conf.GetPositions()[[3, 4, 5]] - held).max() < 1e-3

    # -- the atoms to hold are required, and checked against the molecule
    with pytest.raises(SystemExit):
        main(["constrained", str(with_bonds)])
    with pytest.raises(ValueError, match="frozen atoms"):
        main(["constrained", str(with_bonds), "--hard", "99", "-o", str(out)])


def test_legacy_help_points_to_the_subcommands(capsys):
    with pytest.raises(SystemExit):
        main(["-h"])

    assert "racerts ts -h" in capsys.readouterr().out


def test_the_console_script_exits_with_status_0(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "gs.xyz"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    command = "from racerts.cli import main; import sys; sys.exit(main())"
    result = subprocess.run(
        [sys.executable, "-c", command, "gs", "CO", "-n", "3", "-o", str(out)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout == "" and out.exists()


def test_python_m_racerts_cli(tmp_path):
    import subprocess
    import sys

    out = tmp_path / "gs.xyz"
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run(
        [sys.executable, "-m", "racerts.cli", "gs", "CO", "-n", "3", "-o", str(out)],
        env=env,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert out.exists()
