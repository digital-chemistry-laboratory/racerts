"""PipelineConfig: plain data that builds the default pipeline."""

import json

import pytest

from racerts import (
    Constrained,
    EmbedConfig,
    GroundState,
    PipelineConfig,
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
    assert fallback(TransitionState([0])) is True
    assert fallback(Constrained([0])) is True
