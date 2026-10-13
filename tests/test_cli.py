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


def _configuration(mol, conf_id):
    """CIP labels of the stereocentres and the double bonds' stereo, read from a conformer."""
    one = Chem.Mol(mol, confId=conf_id)
    Chem.AssignStereochemistryFrom3D(one)
    bonds = [str(b.GetStereo()) for b in one.GetBonds() if b.GetStereo()]
    return Chem.FindMolChiralCenters(one), bonds


def test_gs_takes_a_structure_file_and_keeps_its_configuration(tmp_path, capsys):
    mol = Chem.AddHs(Chem.MolFromSmiles(r"C[C@@H](O)/C=C\C"))  # (R,Z)-pent-3-en-2-ol
    AllChem.EmbedMolecule(mol, randomSeed=3)
    in_the_file = _configuration(mol, -1)
    assert in_the_file == ([(1, "R")], ["STEREOZ"])
    xyz, with_bonds, out = (tmp_path / n for n in ("m.xyz", "m.mol", "gs.xyz"))
    Chem.MolToXYZFile(mol, str(xyz))
    Chem.MolToMolFile(mol, str(with_bonds))
    options = ["-n", "10", "--seed", "3", "-o", str(out)]

    # -- an xyz file (bonds perceived), an xyz file with the SMILES for its graph, a
    # file with bonds: the molecule and the configuration are those of the file
    for source in ([str(xyz)], [str(xyz), "-s", "CC(O)C=CC"], [str(with_bonds)]):
        ensemble = run_subcommand(["gs", *source, *options])

        assert len(_frames(out)) == len(ensemble) > 1
        orders = sorted(b.GetBondTypeAsDouble() for b in ensemble.mol.GetBonds())
        assert orders == sorted(b.GetBondTypeAsDouble() for b in mol.GetBonds())
        for conf in ensemble.mol.GetConformers():
            assert _configuration(ensemble.mol, conf.GetId()) == in_the_file

    # -- no fallback to connectivity alone: without bond orders it is another molecule
    cation = Chem.AddHs(Chem.MolFromSmiles("C[NH3+]"))
    AllChem.EmbedMolecule(cation, randomSeed=3)
    charged = tmp_path / "methylammonium.xyz"
    Chem.MolToXYZFile(cation, str(charged))
    with pytest.raises(SystemExit):
        main(["gs", str(charged), *options])  # the charge is missing: no bonds
    assert "--graph connect" in capsys.readouterr().err
    assert len(run_subcommand(["gs", str(charged), "-c", "1", *options])) >= 1

    # -- connectivity alone on request
    single = run_subcommand(["gs", str(xyz), "--graph", "connect", *options])
    assert all(b.GetBondTypeAsDouble() == 1 for b in single.mol.GetBonds())


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
