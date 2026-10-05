"""The subcommands racerts ts, gs and swap (the legacy form: tests/legacy)."""

import os

import pytest

import racerts.cli
from racerts import PipelineConfig, TransitionState
from racerts.cli import main, run_subcommand
from racerts.system.build import GRAPH_METHODS

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


def test_a_config_file_that_cannot_be_read_is_an_error_message(tmp_path, capsys):
    pytest.importorskip("yaml")
    broken = tmp_path / "config.yaml"
    broken.write_text("embed: {n_conformers: 5\nprune: [")
    with pytest.raises(SystemExit) as exit:
        main(["ts", EX, "-r", "3", "4", "5", "--config", str(broken)])
    assert exit.value.code == 2
    assert f"racerts: error: {broken}: " in capsys.readouterr().err


@pytest.mark.parametrize(
    "argv, message",
    [
        (["gs", "CCO", "--keep-hbonds"], "unrecognized arguments: --keep-hbonds"),
        (["gs", "CCO", "--contact", "0", "2"], "unrecognized arguments: --contact"),
        (["gs", "CCO", "--chirality-fallback", "legacy"], "unrecognized arguments"),
        (["gs", "CCO", "--restraint-fraction", "0.5"], "unrecognized arguments"),
        (["gs", "CCO", "--reference-bounds", "never"], "unrecognized arguments"),
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


@pytest.fixture
def call(monkeypatch):
    """
    Runs a subcommand up to its call of generate_ts, generate_gs or swap; returns the
    keyword arguments of that call, the positional ones as "args".
    """
    seen = {}

    def record(*args, **kwargs):
        seen.update(kwargs, args=args)
        raise SystemExit(0)

    for name in ("generate_ts", "generate_gs", "swap"):
        monkeypatch.setattr(racerts.cli, name, record)

    def run(*argv):
        seen.clear()
        with pytest.raises(SystemExit):
            run_subcommand(list(argv))
        return dict(seen)

    return run


TS = ("ts", EX, "-r", "3", "4", "5")


def _with(settings):
    """The default config as a dict, with the settings {"section.key": value}."""
    expected = PipelineConfig().to_dict()
    for name, value in settings.items():
        *sections, key = name.split(".")
        target = expected
        for section in sections:
            target = target[section]
        assert key in target
        target[key] = value
    return expected


def test_ts_without_options_runs_the_defaults(call):
    seen = call(*TS)

    assert seen.pop("args") == (EX, [3, 4, 5])
    assert seen.pop("config") == PipelineConfig()
    assert seen == dict(
        charge=0,
        smiles=None,
        multiplicity=None,
        frozen_atoms=None,
        mol_getter=None,
        auto_fallback=True,
        active_window=None,
        active_bonds=None,
        stratify=None,
        pipeline=None,
    )


@pytest.mark.parametrize(
    "options, settings",
    [
        (["-n", "7"], {"embed.n_conformers": 7}),
        (["--conf-factor", "40"], {"embed.conf_factor": 40}),
        (["--etkdg"], {"embed.etkdg": True}),
        (["--no-etkdg"], {"embed.etkdg": False}),
        (["--embed", "bounds"], {"embed.mode": "bounds"}),
        (["--count-policy", "fragments"], {"embed.count_policy": "fragments"}),
        (["--sequential-seeds"], {"embed.sequential_seeds": True}),
        (
            ["--chirality-fallback", "frozen_first"],
            {"embed.chirality_fallback": "frozen_first"},
        ),
        (["--reference-bounds", "always"], {"embed.reference_bounds": "always"}),
        (["--refine", "uff"], {"refine.backend": "uff"}),
        (["--converge"], {"refine.converge": True}),
        (["--energies-without-anchors"], {"refine.energies_without_anchors": True}),
        (
            ["--dielectric", "distance", "4"],
            {"refine.dielectric_model": "distance", "refine.dielectric_constant": 4.0},
        ),
        (["--no-fallback"], {"refine.fallback": False}),
        (["--energy-threshold", "5"], {"prune.energy_threshold": 5.0}),
        (["--rmsd-threshold", "0.3"], {"prune.rmsd_threshold": 0.3}),
        (["--rmsd-hydrogens"], {"prune.hydrogens": "all"}),
        (["--rmsd-hydrogens", "none"], {"prune.hydrogens": "none"}),
        (["--check-stereo"], {"prune.check_stereo": True}),
        (["--seed", "0"], {"seed": 0}),
        (["--num-threads", "2"], {"num_threads": 2}),
        (["--hints"], {"restraints.hints": True}),
        (["--keep-hbonds"], {"restraints.hbonds": True}),
        (["--keep-fragments"], {"restraints.keep_fragments": True}),
        (["--restraint-fraction", "0.5"], {"restraints.fraction": 0.5}),
        (["--contact", "0", "9"], {"restraints.contacts": [[0, 9]]}),
        (
            ["--restraint", "0", "6", "4.6", "--restraint", "1", "6", "4"],
            {"restraints.user": [[0, 6, 4.6], [1, 6, 4.0]]},
        ),
        (["--restraint-half-width", "0.3"], {"restraints.half_width": 0.3}),
        (["--restraint-force-constant", "50"], {"restraints.force_constant": 50.0}),
    ],
)
def test_ts_options_set_their_setting_and_nothing_else(call, options, settings):
    assert call(*TS, *options)["config"].to_dict() == _with(settings)


@pytest.mark.parametrize(
    "options, keyword, value",
    [
        (["-c", "-1"], "charge", -1),
        (["--multiplicity", "3"], "multiplicity", 3),
        (["-s", "CCCC", "C=C"], "smiles", ["CCCC", "C=C"]),
        (["--frozen-atoms", "3", "4", "5", "6"], "frozen_atoms", [3, 4, 5, 6]),
        (["--no-fallback"], "auto_fallback", False),
        (["--active-window", "0.2"], "active_window", 0.2),
        (["--active-window", "2.0", "2.9"], "active_window", (2.0, 2.9)),
        (
            ["--active-window", "0.2", "--active-bond", "3", "5", "--active-bond"]
            + ["4", "5"],
            "active_bonds",
            [[3, 5], [4, 5]],
        ),
        (["--active-window", "0.2", "--stratify", "4"], "stratify", 4),
        (
            ["--active-window", "0.2", "--neighbor-window", "0.05"],
            "neighbor_window",
            0.05,
        ),
    ],
)
def test_ts_options_reach_generate_ts(call, options, keyword, value):
    assert call(*TS, *options)[keyword] == value


@pytest.mark.parametrize("name", list(GRAPH_METHODS))
def test_ts_graph_option_chooses_the_first_graph_method(call, name):
    assert type(call(*TS, "--graph", name)["mol_getter"]) is GRAPH_METHODS[name]


def test_ts_legacy_option_and_wrong_values(call, capsys):
    assert call(*TS, "--legacy")["config"] == PipelineConfig.legacy()
    assert call(*TS, "--legacy", "--converge")["config"].refine.converge is True
    for options, message in [
        (["--dielectric", "distance", "four"], "'four' is not a number"),
        (["--restraint", "a", "6", "4.6"], "--restraint takes two atom indices"),
        (["--active-window", "1", "2", "3"], "--active-window takes DELTA or LO HI"),
        (["--stratify", "3"], "need --active-window"),
        (["--export-to", "x.inp"], "--export-to needs --export-restraints"),
    ]:
        with pytest.raises(SystemExit):
            run_subcommand([*TS, *options])
        assert message in capsys.readouterr().err


def test_export_options_replace_the_pipeline(call, tmp_path):
    out = str(tmp_path / "ensemble.xyz")
    (stage,) = call(*TS, "--export-restraints", "orca", "-o", out)["pipeline"].stages
    assert (stage.fmt, stage.path) == ("orca", str(tmp_path / "restraints.inp"))
    (stage,) = call("gs", "CCO", "--export-restraints", "json", "--export-to", out)[
        "pipeline"
    ].stages
    assert (stage.fmt, stage.path) == ("json", out)


def test_gs_options_reach_generate_gs(call):
    seen = call("gs", "CC(=O)[O-].[NH4+]", "--link-fragments", "--multiplicity", "1")

    assert seen.pop("args") == ("CC(=O)[O-].[NH4+]",)
    assert seen.pop("config").to_dict() == _with({"restraints.link_fragments": True})
    assert seen == dict(charge=None, multiplicity=1, pipeline=None)
    assert call("gs", "CCO", "-c", "0")["charge"] == 0


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


@pytest.mark.parametrize(
    "options, change",
    [
        (["--remove", "3"], dict(remove_atoms=[3])),
        (["--site", "2"], dict(site=2)),
        (["--center", "0", "2"], dict(center=0, substructure=2)),
        (["--old", "[C:1][H]"], dict(old_fragment="[C:1][H]")),
        (["--remove", "--attach", "1", "0"], dict(remove_atoms=[], attach_map={1: 0})),
        (
            ["--remove", "3", "--bond-type", "1", "double"],
            dict(remove_atoms=[3], bond_types={1: "double"}),
        ),
        (
            ["--remove", "3", "--mode", "renumber"],
            dict(remove_atoms=[3], mode="renumber"),
        ),
    ],
)
def test_swap_options_describe_the_swap(call, sn2_ts, options, change):
    seen = call(
        "swap", sn2_ts, "-s", "CCl", "[Cl-]", "-c", "-1", "--new", "[*]C", *options
    )

    mol, swap = seen["args"]
    assert mol.GetNumAtoms() == 6 and mol.GetIntProp("charge") == -1
    assert swap == racerts.Swap("[*]C", **change)
    assert seen["task"] is None and seen["config"] == PipelineConfig()
    assert (seen["conserve"], seen["routes"], seen["hard"]) == ("soft", None, [])


def test_swap_options_reach_swap(call, sn2_ts):
    seen = call(
        "swap", sn2_ts, "-s", "CCl", "[Cl-]", "-c", "-1", "--new", "[*]C", "--remove", "3",
        "-r", "0", "1", "2", "--conserve", "free", "--routes", "dg", "rigid",
        "--hard", "4", "5", "--multiplicity", "1", "-n", "6", "--seed", "3",
        "--reference-bounds", "never",
    )  # fmt: skip

    assert isinstance(seen["task"], TransitionState)
    assert list(seen["task"].reacting_atoms) == [0, 1, 2]
    assert (seen["conserve"], seen["routes"], seen["hard"]) == (
        "free",
        ["dg", "rigid"],
        [4, 5],
    )
    assert seen["multiplicity"] == 1
    assert seen["config"].to_dict() == _with(
        {"embed.n_conformers": 6, "seed": 3, "embed.reference_bounds": "never"}
    )


def test_swap_command_rejects_what_it_does_not_use(tmp_path, sn2_ts, capsys):
    with pytest.raises(SystemExit):
        run_subcommand(
            ["swap", sn2_ts, "-s", "CCl", "[Cl-]", "-c", "-1", "--new", "[*]C",
             "--remove", "3", "--keep-hbonds", "--embed", "bounds"]
        )  # fmt: skip
    assert "unrecognized arguments: --keep-hbonds --embed" in capsys.readouterr().err
    with pytest.raises(SystemExit):
        run_subcommand(
            ["swap", sn2_ts, "--new", "[*]C", "--remove", "3", "--export-restraints"]
            + ["xtb"]
        )
    assert "unrecognized arguments: --export-restraints" in capsys.readouterr().err
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
