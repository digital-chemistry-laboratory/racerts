"""PipelineConfig: plain data that builds the default pipeline."""

import json

import pytest

from racerts import (
    Constrained,
    EmbedConfig,
    GroundState,
    PipelineConfig,
    RefineConfig,
    TransitionState,
)
from racerts.embed import BoundsMatrixEmbedder, CmapEmbedder
from racerts.refine import UFFOptimizer


def test_defaults_are_the_legacy_settings():
    embed, refine, prune_energy, prune_rmsd = (
        PipelineConfig().build(TransitionState([0])).stages
    )
    embedder, optimizer = embed.embedder, refine.optimizer

    assert type(embedder) is CmapEmbedder and embedder.randomSeed == 12
    assert embedder.etkdg is False and embedder.useRandomCoords is True
    assert (embed.n_conformers, embed.conf_factor) == (-1, 80)
    assert type(optimizer).__name__ == "MMFFOptimizer" and refine.fallback is True
    assert optimizer.force_constant == 1e6
    assert prune_energy.pruner.threshold == 20.0
    assert prune_rmsd.pruner.threshold == 0.125
    assert prune_rmsd.pruner.energy_threshold == 0.1


def test_etkdg_by_default_only_without_frozen_atoms():
    def etkdg(task, **embed):
        config = PipelineConfig(embed=EmbedConfig(**embed))
        return config.build(task).stages[0].embedder.etkdg

    assert etkdg(GroundState()) is True
    assert etkdg(TransitionState([0])) is False
    assert etkdg(Constrained([0])) is False
    assert etkdg(TransitionState([0]), etkdg=True) is True


def test_settings_reach_the_components():
    config = PipelineConfig.from_dict(
        {
            "seed": 3,
            "num_threads": 2,
            "embed": {"mode": "bounds"},
            "refine": {"backend": "uff", "fallback": False},
            "prune": {"rmsd_threshold": 0.3, "include_hs": True},
        }
    )
    embed, refine, _, prune_rmsd = config.build(TransitionState([0])).stages

    assert type(embed.embedder) is BoundsMatrixEmbedder
    assert embed.embedder.randomSeed == 3 and embed.embedder.num_threads == 2
    assert type(refine.optimizer) is UFFOptimizer and refine.fallback is False
    assert prune_rmsd.pruner.threshold == 0.3 and prune_rmsd.pruner.include_hs is True


@pytest.mark.parametrize("suffix", [".json", ".yaml"])
def test_file_round_trip(tmp_path, suffix):
    if suffix == ".yaml":
        pytest.importorskip("yaml")
    config = PipelineConfig.from_dict({"seed": 7, "embed": {"n_conformers": 40}})
    path = str(tmp_path / f"config{suffix}")
    config.to_file(path)

    assert PipelineConfig.from_file(path) == config
    if suffix == ".json":
        assert json.load(open(path))["embed"]["n_conformers"] == 40


@pytest.mark.parametrize(
    "data, message",
    [
        ({"seeds": 1}, "Unknown keys in config"),
        ({"embed": {"n": 1}}, "Unknown keys in embed"),
        ({"embed": {"mode": "dm"}}, "embed.mode"),
        ({"refine": {"backend": "xtb"}}, "refine.backend"),
        ({"prune": 0.3}, "prune must be a mapping"),
    ],
)
def test_invalid_settings_raise(data, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig.from_dict(data)


def test_yaml_numbers_without_a_decimal_point(tmp_path):
    # YAML 1.1 (PyYAML's default) reads 1e6 as a string; racerts reads it as YAML 1.2.
    pytest.importorskip("yaml")
    path = tmp_path / "settings.yaml"
    path.write_text("refine:\n  force_constant: 1e6\nprune:\n  energy_threshold: 1e1\n")
    config = PipelineConfig.from_file(str(path))

    assert config.refine.force_constant == 1e6 and config.prune.energy_threshold == 10.0


@pytest.mark.parametrize(
    "data, message",
    [
        ({"prune": {"include_hs": 0}}, "prune.include_hs must be true or false"),
        ({"refine": {"fallback": 1}}, "refine.fallback must be true or false"),
        ({"embed": {"n_conformers": "50"}}, "embed.n_conformers must be an integer"),
        ({"embed": {"n_conformers": 0}}, "-1 .default count. or > 0"),
        ({"seed": True}, "seed must be an integer"),
        ({"refine": {"force_constant": "1e6"}}, "refine.force_constant must be a"),
        ({"prune": {"rmsd_threshold": -0.1}}, "must not be negative"),
    ],
)
def test_values_of_the_wrong_type_raise(data, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig.from_dict(data)


def test_sections_can_be_given_as_mappings():
    config = PipelineConfig(embed={"mode": "bounds"}, refine={"backend": "uff"})

    assert config.embed.mode == "bounds" and config.refine.backend == "uff"


def test_a_failed_yaml_write_leaves_the_file_alone(tmp_path, monkeypatch):
    import racerts.config

    def no_yaml(module, extra):
        raise ImportError(f"{module} is not installed")

    path = tmp_path / "settings.yaml"
    path.write_text("seed: 7\n")
    monkeypatch.setattr(racerts.config, "require", no_yaml)

    with pytest.raises(ImportError):
        PipelineConfig().to_file(str(path))
    assert path.read_text() == "seed: 7\n"


def test_ground_states_have_no_chirality_fallback():
    def fallback(task):
        return PipelineConfig().build(task).stages[0].embedder.chirality_fallback

    assert fallback(GroundState()) is False
    assert fallback(TransitionState([0])) == "legacy"
    assert fallback(Constrained([0])) == "legacy"


def test_the_legacy_preset(tmp_path):
    legacy = PipelineConfig.legacy()
    assert legacy.embed.count_policy == "legacy"
    assert legacy.embed.sequential_seeds is False
    assert legacy.embed.chirality_fallback == "legacy"
    assert legacy.refine.converge is False
    assert legacy.refine.anchor_free_energies is False
    assert legacy.prune.check_stereo is False

    # Settings given update the legacy ones, as dicts or as sections.
    config = PipelineConfig.legacy(seed=3, embed={"n_conformers": 5})
    assert (config.seed, config.embed.n_conformers) == (3, 5)
    assert config.embed.sequential_seeds is False
    # A section object is used as it is.
    config = PipelineConfig.legacy(refine=RefineConfig(backend="uff", converge=True))
    assert config.refine.backend == "uff" and config.refine.converge is True
    assert config.embed.sequential_seeds is False  # the others stay legacy

    # A config file with legacy=True: what the file leaves out is legacy.
    path = tmp_path / "config.json"
    path.write_text('{"embed": {"n_conformers": 5}, "refine": {"converge": true}}')
    config = PipelineConfig.from_file(str(path), legacy=True)
    assert config.embed.n_conformers == 5 and config.refine.converge is True
    assert config.embed.sequential_seeds is False


def test_new_settings_reach_the_components():
    config = PipelineConfig.from_dict(
        {
            "embed": {
                "count_policy": "fragments",
                "sequential_seeds": True,
                "chirality_fallback": "frozen_first",
            },
            "refine": {
                "converge": True,
                "anchor_free_energies": True,
                "dielectric_model": "distance",
                "dielectric_constant": 4,
            },
        }
    )
    assert PipelineConfig.from_dict(config.to_dict()) == config
    embed, refine = config.build(TransitionState([0])).stages[:2]
    assert embed.count_policy == "fragments"
    assert embed.embedder.sequential_seeds is True
    assert embed.embedder.chirality_fallback == "frozen_first"
    optimizer = refine.optimizer
    assert optimizer.converge is True and optimizer.anchor_free_energies is True
    assert (optimizer.dielectric_model, optimizer.dielectric_constant) == (
        "distance",
        4.0,
    )


@pytest.mark.parametrize(
    "section, settings, message",
    [
        ("embed", {"count_policy": "many"}, "embed.count_policy"),
        ("embed", {"chirality_fallback": True}, "embed.chirality_fallback"),
        ("refine", {"dielectric_model": "water"}, "refine.dielectric_model"),
        ("refine", {"dielectric_constant": 0}, "refine.dielectric_constant"),
        ("refine", {"backend": "uff", "dielectric_constant": 4}, "MMFF only"),
    ],
)
def test_new_settings_are_checked(section, settings, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig.from_dict({section: settings})


@pytest.mark.parametrize(
    "restraints, message",
    [
        ({"user": [[0, 6, "4.6"]]}, "restraints.user"),
        ({"user": [[0, 6.5, 4.6]]}, "restraints.user"),
        ({"contacts": [["a", 1]]}, "restraints.contacts"),
        ({"fragment_links": [[0, True]]}, "restraints.fragment_links"),
    ],
)
def test_restraint_items_are_checked_when_the_config_is_made(restraints, message):
    with pytest.raises(ValueError, match=message):
        PipelineConfig.from_dict({"restraints": restraints})


def test_a_config_file_holds_a_mapping(tmp_path):
    path = tmp_path / "config.json"
    for text in ("0", "[]"):
        path.write_text(text)
        with pytest.raises(ValueError, match="mapping"):
            PipelineConfig.from_file(str(path))
    empty = tmp_path / "empty.yaml"
    empty.write_text("")
    assert PipelineConfig.from_file(str(empty)) == PipelineConfig()


def test_the_eht_setting_warns_by_its_name_and_labels_the_energies(hept_1_ene_ts):
    import racerts

    with pytest.warns(FutureWarning, match="prune.eht_energies is deprecated"):
        config = PipelineConfig.from_dict(
            {"embed": {"n_conformers": 3}, "prune": {"eht_energies": True}}
        )
    ensemble = racerts.generate(
        hept_1_ene_ts, TransitionState([3, 4, 5]), config=config
    )
    assert ensemble.energy_method == "EHT"
    with pytest.raises(ValueError, match="eht_energies"):
        config.build(TransitionState([3, 4, 5], active_window=0.2))


def test_the_sections_and_stages_of_the_default_pipeline_are_exported():
    import racerts

    for name in ("EmbedConfig", "RefineConfig", "PruneConfig", "RestraintConfig"):
        assert name in racerts.__all__ and hasattr(racerts, name)
    for name in ("Embed", "Refine", "PruneEnergy", "PruneRMSD", "PruneCluster"):
        assert name in racerts.__all__ and hasattr(racerts, name)
    assert set(racerts.__all__) <= set(dir(racerts))


def test_cluster_pruning_setting():
    stages = (
        PipelineConfig.from_dict(
            {
                "prune": {
                    "method": "cluster",
                    "cluster_method": "leader",
                    "cluster_threshold": 0.8,
                }
            }
        )
        .build(TransitionState([0]))
        .stages
    )
    assert stages[-1].name == "prune_cluster"
    pruner = stages[-1].pruner
    assert (pruner.method, pruner.threshold) == ("leader", 0.8)
    assert PipelineConfig().build(TransitionState([0])).stages[-1].name == "prune_rmsd"
    with pytest.raises(ValueError, match="prune.method"):
        PipelineConfig.from_dict({"prune": {"method": "kmeans"}})
    with pytest.raises(ValueError, match="prune.cluster_method"):
        PipelineConfig.from_dict({"prune": {"cluster_method": "kmeans"}})
