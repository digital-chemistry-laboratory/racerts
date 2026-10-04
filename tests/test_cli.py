"""The subcommands racerts ts, gs and swap (the legacy form: tests/legacy)."""

import os

import pytest

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


def test_ts_rejects_invalid_settings(tmp_path, capsys):
    config = tmp_path / "config.json"
    config.write_text('{"embed": {"mode": "dm"}}')

    with pytest.raises(SystemExit):
        main(["ts", EX, "-r", "3", "4", "5", "--config", str(config)])
    assert "racerts: error: embed.mode must be one of" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        main(["ts", EX])  # the reacting atoms are required
    with pytest.raises(SystemExit):
        main(["ts", EX, "-r", "3", "--config", str(tmp_path / "missing.json")])
    assert "No such file" in capsys.readouterr().err
    with pytest.raises(ValueError, match="embed.mode"):  # -vv: the traceback
        main(["ts", EX, "-r", "3", "4", "5", "--config", str(config), "-vv"])


@pytest.mark.parametrize(
    "argv, message",
    [
        (["gs", "CCO", "--keep-hbonds"], "unrecognized arguments: --keep-hbonds"),
        (["gs", "CCO", "--contact", "0", "2"], "unrecognized arguments: --contact"),
        (["gs", "CCO", "--chirality-fallback", "legacy"], "unrecognized arguments"),
        (["ts", EX, "-r", "3", "--link-fragments"], "unrecognized arguments"),
        (["ts", EX, "-r", "3", "--neighbor-window", "0.5"], "need --active-window"),
        (["ts", EX, "-r", "300", "-n", "2"], "Invalid reacting atoms"),
        (["gs", "C1CC", "-n", "2"], "Invalid SMILES"),
    ],
)
def test_options_a_subcommand_cannot_use_are_errors(argv, message, capsys):
    with pytest.raises(SystemExit):
        main(argv)
    assert message in capsys.readouterr().err


def test_legacy_help_points_to_the_subcommands(capsys):
    with pytest.raises(SystemExit):
        main(["-h"])

    out = " ".join(capsys.readouterr().out.split())  # the help text is wrapped
    assert all(f"racerts {name} -h" in out for name in ("ts", "gs", "swap"))


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


def test_ts_options_for_the_new_settings(monkeypatch, capsys):
    import racerts.cli

    seen = {}

    def fake_generate_ts(*args, config, **kwargs):
        seen["config"] = config
        raise SystemExit(0)

    monkeypatch.setattr(racerts.cli, "generate_ts", fake_generate_ts)

    def config_of(*options):
        with pytest.raises(SystemExit):
            run_subcommand(["ts", EX, "-r", "3", "4", "5", *options])
        return seen["config"]

    config = config_of(
        "--count-policy", "fragments", "--sequential-seeds",
        "--chirality-fallback", "frozen_first", "--converge",
        "--anchor-free-energies", "--dielectric", "distance", "4", "--rmsd-hydrogens",
        "--check-stereo",
    )  # fmt: skip
    assert config.embed.count_policy == "fragments"
    assert config.embed.sequential_seeds is True
    assert config.embed.chirality_fallback == "frozen_first"
    assert config.refine.converge is True and config.refine.anchor_free_energies
    assert (config.refine.dielectric_model, config.refine.dielectric_constant) == (
        "distance",
        4.0,
    )
    assert config.prune.include_hs is True
    assert config.prune.check_stereo is True

    assert config_of("--legacy") == PipelineConfig.legacy()
    assert config_of("--legacy", "--converge").refine.converge is True
    with pytest.raises(SystemExit):
        run_subcommand(["ts", EX, "-r", "3", "--dielectric", "distance", "four"])
    assert "'four' is not a number" in capsys.readouterr().err


def test_ts_restraint_options(monkeypatch, capsys):
    import racerts.cli

    seen = {}

    def fake_generate_ts(*args, config, **kwargs):
        seen["config"] = config
        raise SystemExit(0)

    monkeypatch.setattr(racerts.cli, "generate_ts", fake_generate_ts)
    with pytest.raises(SystemExit):
        run_subcommand(
            ["ts", EX, "-r", "3", "4", "5", "--restraint", "0", "6", "4.6",
             "--restraint", "1", "6", "4.0", "--keep-hbonds", "--contact", "0", "9",
             "--keep-fragments", "--restraint-half-width", "0.3",
             "--restraint-force-constant", "50"]
        )  # fmt: skip
    restraints = seen["config"].restraints
    assert restraints.user == [[0, 6, 4.6], [1, 6, 4.0]]
    assert restraints.hbonds and restraints.keep_fragments
    assert restraints.contacts == [[0, 9]]
    assert (restraints.half_width, restraints.force_constant) == (0.3, 50.0)
    with pytest.raises(SystemExit):
        run_subcommand(["ts", EX, "-r", "3", "--restraint", "a", "6", "4.6"])
    assert "--restraint takes two atom indices" in capsys.readouterr().err


def test_gs_links_fragments(tmp_path):
    out = tmp_path / "gs.xyz"
    ensemble = run_subcommand(
        ["gs", "CC(=O)[O-].[NH4+]", "-n", "4", "--link-fragments", "-o", str(out)]
    )
    assert len(ensemble) > 0 and out.exists()


def test_swap_command(tmp_path, sn2_ts):
    # SN2 TS: H3 on the reacting carbon -> ethyl, the TS core held.
    out = tmp_path / "swapped.xyz"
    ensemble = run_subcommand(
        ["swap", sn2_ts, "-s", "CCl", "[Cl-]", "-c", "-1", "-r", "0", "1", "2",
         "--new", "[*]CC", "--remove", "3", "-n", "6", "-o", str(out)]
    )  # fmt: skip
    assert len(ensemble) >= 1 and out.exists()
    assert ensemble.mol.GetNumAtoms() == 6 - 1 + 7  # C2H5 for H
    with open(out) as handle:
        assert handle.readline().strip() == str(ensemble.mol.GetNumAtoms())


def test_swap_command_rejects_what_it_does_not_use(tmp_path, sn2_ts, capsys):
    with pytest.raises(SystemExit):
        run_subcommand(
            ["swap", sn2_ts, "-s", "CCl", "[Cl-]", "-c", "-1", "--new", "[*]C",
             "--remove", "3", "--keep-hbonds", "--embed", "bounds"]
        )  # fmt: skip
    assert "unrecognized arguments: --keep-hbonds --embed" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run_subcommand(["swap", sn2_ts, "--new", "[*]C"])  # no selector
    with pytest.raises(SystemExit):
        run_subcommand(
            ["swap", sn2_ts, "-s", "CCl", "[Cl-]", "-c", "-1", "--new", "[*]C",
             "--remove", "3", "--mode", "keep"]
        )  # fmt: skip


def test_swap_command_rejects_hard_with_sampling(sn2_ts, capsys):
    base = [
        "swap",
        sn2_ts,
        "-s",
        "CCl",
        "[Cl-]",
        "-c",
        "-1",
        "--new",
        "[*]C",
        "--remove",
        "3",
    ]
    with pytest.raises(SystemExit):
        run_subcommand(base + ["--conserve", "hard", "-n", "5"])
    assert "--conserve hard samples nothing" in capsys.readouterr().err


def test_swap_command_reports_config_errors(tmp_path, sn2_ts, capsys):
    config = tmp_path / "config.yaml"
    config.write_text("restraints:\n  hints: true\n")
    with pytest.raises(SystemExit):
        run_subcommand(
            ["swap", sn2_ts, "-s", "CCl", "[Cl-]", "-c", "-1", "--new", "[*]C",
             "--remove", "3", "--config", str(config)]
        )  # fmt: skip
    assert "config.restraints must be empty" in capsys.readouterr().err
