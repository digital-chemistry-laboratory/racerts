import sys

import pytest


class FakeEnergyPruner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeRMSDPruner:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeMolGetter:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeEmbedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeOptimizer:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class FakeConfGen:
    def __init__(self, verbose=False, randomSeed=12, num_threads=1):
        self.init_args = dict(
            verbose=verbose, randomSeed=randomSeed, num_threads=num_threads
        )
        self.mol_getter = None
        self.embedder = None
        self.optimizer = None
        self._energy_pruner = None
        self._rmsd_pruner = None
        self.generated_args = None
        self.written = None

    # energy_pruner property
    @property
    def energy_pruner(self):
        return self._energy_pruner

    @energy_pruner.setter
    def energy_pruner(self, value):
        self._energy_pruner = value

    # rmsd_pruner property
    @property
    def rmsd_pruner(self):
        return self._rmsd_pruner

    @rmsd_pruner.setter
    def rmsd_pruner(self, value):
        self._rmsd_pruner = value

    def generate_conformers(
        self,
        filename,
        charge,
        reacting_atoms,
        *,
        frozen_atoms=None,
        input_smiles=None,
        number_of_conformers=-1,
        conf_factor=80,
        auto_fallback=True,
    ):
        self.generated_args = dict(
            filename=filename,
            charge=charge,
            reacting_atoms=list(reacting_atoms) if reacting_atoms is not None else None,
            frozen_atoms=list(frozen_atoms) if frozen_atoms is not None else None,
            input_smiles=list(input_smiles) if input_smiles is not None else None,
            number_of_conformers=number_of_conformers,
            conf_factor=conf_factor,
            auto_fallback=auto_fallback,
        )

    def write_xyz(self, path, use_energy=False, comment="0 1"):
        self.written = dict(path=path, use_energy=use_energy, comment=comment)


def _prepare_cli(monkeypatch):
    import racerts.cli as cli

    # Replace components with fakes
    monkeypatch.setattr(cli, "ConformerGenerator", FakeConfGen, raising=True)
    monkeypatch.setattr(cli, "EnergyPruner", FakeEnergyPruner, raising=True)
    monkeypatch.setattr(cli, "RMSDPruner", FakeRMSDPruner, raising=True)

    # Registries
    monkeypatch.setattr(
        cli,
        "mol_getters",
        {"bonds": FakeMolGetter, "connect": FakeMolGetter, "smiles": FakeMolGetter},
        raising=True,
    )
    monkeypatch.setattr(
        cli, "embedders", {"dm": FakeEmbedder, "cmap": FakeEmbedder}, raising=True
    )
    monkeypatch.setattr(
        cli, "optimizers", {"mmff": FakeOptimizer, "uff": FakeOptimizer}, raising=True
    )

    return cli


def test_cli_happy_path_defaults(monkeypatch, tmp_path):
    cli = _prepare_cli(monkeypatch)

    # Create fake input file to satisfy isfile check
    infile = tmp_path / "input.xyz"
    infile.write_text("2\ncomment\nH 0 0 0\nH 0 0 0\n")

    # Capture the created FakeConfGen instance by intercepting constructor
    created = {}

    def ctor(verbose=False, randomSeed=12, num_threads=1):
        inst = FakeConfGen(
            verbose=verbose, randomSeed=randomSeed, num_threads=num_threads
        )
        created["cg"] = inst
        return inst

    monkeypatch.setattr(cli, "ConformerGenerator", ctor, raising=True)

    argv = [
        "racerts",
        str(infile),
        "-c",
        "0",
        "--reacting_atoms",
        "1",
        "2",
        "3",
        "-o",
        "out.xyz",
    ]
    monkeypatch.setenv("PYTHONWARNINGS", "ignore")
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    cg = created["cg"]
    # Confirm generator created with defaults
    assert cg.init_args == dict(verbose=False, randomSeed=12, num_threads=1)

    # Energy pruner default threshold should be 20.0, verbose False
    assert isinstance(cg.energy_pruner, FakeEnergyPruner)
    assert cg.energy_pruner.kwargs["threshold"] == pytest.approx(20.0)
    assert cg.energy_pruner.kwargs["verbose"] is False

    # RMSD pruner defaults
    assert isinstance(cg.rmsd_pruner, FakeRMSDPruner)
    kw = cg.rmsd_pruner.kwargs
    assert kw["threshold"] == pytest.approx(0.125)
    assert kw["include_hs"] is False
    assert kw["filter_energies"] is True
    assert kw["filter_rotations"] is True
    assert kw["energy_threshold"] == pytest.approx(0.1)
    assert kw["rot_fraction_threshold"] == pytest.approx(0.03)
    assert kw["maxMatches"] == 10000
    assert kw["num_threads"] == 1

    # generate_conformers call wiring
    gen = cg.generated_args
    assert gen["filename"] == str(infile)
    assert gen["charge"] == 0
    assert gen["reacting_atoms"] == [1, 2, 3]
    assert gen["frozen_atoms"] == []
    assert gen["input_smiles"] == []
    assert gen["number_of_conformers"] == -1
    assert gen["conf_factor"] == 80
    assert gen["auto_fallback"] is True

    # write_xyz wiring
    assert cg.written["path"] == "out.xyz"
    assert cg.written["use_energy"] is False


def test_cli_select_components_and_flags(monkeypatch, tmp_path):
    cli = _prepare_cli(monkeypatch)

    infile = tmp_path / "input.xyz"
    infile.write_text("2\ncomment\nH 0 0 0\nH 0 0 0\n")

    created = {}

    def ctor(verbose=False, randomSeed=12, num_threads=1):
        inst = FakeConfGen(
            verbose=verbose, randomSeed=randomSeed, num_threads=num_threads
        )
        created["cg"] = inst
        return inst

    monkeypatch.setattr(cli, "ConformerGenerator", ctor, raising=True)

    argv = [
        "racerts",
        str(infile),
        "-c",
        "1",
        "--reacting_atoms",
        "0",
        "1",
        "--mol",
        "bonds",
        "--embed",
        "dm",
        "--ff",
        "uff",
        "--no_fallback",
        "--no_random_coords",
        "--rmsd_include_hs",
        "--no_filter_energies",
        "--no_filter_rotations",
        "--rmsd_thres",
        "0.2",
        "--rmsd_energy",
        "0.3",
        "--rmsd_rot_fraction",
        "0.05",
        "--rmsd_max_matches",
        "42",
        "-energy",
        "7.5",
        "--out_energies",
    ]
    monkeypatch.setattr(sys, "argv", argv)

    cli.main()

    cg = created["cg"]
    # Check component instances
    assert isinstance(cg.mol_getter, FakeMolGetter)
    assert isinstance(cg.embedder, FakeEmbedder)
    assert isinstance(cg.optimizer, FakeOptimizer)

    # Embedder received flags
    assert cg.embedder.kwargs["verbose"] is False
    assert cg.embedder.kwargs["num_threads"] == 1
    assert cg.embedder.kwargs["randomSeed"] == 12
    assert cg.embedder.kwargs["useRandomCoords"] is False

    # Pruner configurations
    assert cg.energy_pruner.kwargs["threshold"] == pytest.approx(7.5)
    rkw = cg.rmsd_pruner.kwargs
    assert rkw["threshold"] == pytest.approx(0.2)
    assert rkw["include_hs"] is True
    # store_false flags should flip to False when passed
    assert rkw["filter_energies"] is False
    assert rkw["filter_rotations"] is False
    assert rkw["energy_threshold"] == pytest.approx(0.3)
    assert rkw["rot_fraction_threshold"] == pytest.approx(0.05)
    assert rkw["maxMatches"] == 42

    # generate_conformers reflects args and fallback disabled
    gen = cg.generated_args
    assert gen["charge"] == 1
    assert gen["reacting_atoms"] == [0, 1]
    assert gen["auto_fallback"] is False
    assert cg.written["use_energy"] is True


def test_cli_missing_file_raises(monkeypatch, tmp_path):
    cli = _prepare_cli(monkeypatch)

    missing = tmp_path / "does_not_exist.xyz"

    argv = ["racerts", str(missing), "-c", "0"]
    monkeypatch.setattr(sys, "argv", argv)

    with pytest.raises(SystemExit):
        cli.main()
