"""PipelineConfig: the settings of the default pipeline, as plain data."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, Optional

import attrs
from attrs import define, field
from attrs.validators import optional

from racerts.embed import DEFAULT_CONF_FACTOR, EMBED_MODES, Embed, default_embedder
from racerts.pipeline import Pipeline
from racerts.prune import EnergyPruner, PruneEnergy, PruneRMSD, RMSDPruner
from racerts.refine import REFINE_BACKENDS, Refine
from racerts.task import Task
from racerts.utils.optional import require
from racerts.utils.validators import (
    choice,
    flag,
    integer,
    is_bool,
    non_negative,
    number,
    positive,
    positive_or,
)


@define
class EmbedConfig:
    """
    Attributes:
        mode: "cmap" (frozen atoms placed by a coordinate map) or "bounds" (their
            distances fixed in the bounds matrix).
        n_conformers: Conformers to embed; -1 for rotatable bonds * conf_factor + 30.
        conf_factor: Conformers per rotatable bond for the default count.
        etkdg: Use ETKDGv3 instead of plain distance geometry. None: plain distance
            geometry when atoms are frozen (as in legacy racerts), ETKDGv3 otherwise.
        use_random_coords: Start embedding from random coordinates.
    """

    _section = "embed"

    mode: str = choice("cmap", EMBED_MODES)
    n_conformers: int = integer(-1, positive_or(-1, "default count"))
    conf_factor: int = integer(DEFAULT_CONF_FACTOR, non_negative)
    etkdg: Optional[bool] = field(default=None, validator=optional(is_bool))
    use_random_coords: bool = flag(True)


@define
class RefineConfig:
    """
    Attributes:
        backend: "mmff" or "uff".
        fallback: Fall back to UFF when MMFF has no parameters.
        force_constant: Force constant (kcal/mol/A^2) that holds the frozen atoms.
    """

    _section = "refine"

    backend: str = choice("mmff", REFINE_BACKENDS)
    fallback: bool = flag(True)
    force_constant: float = number(1e6, positive)


@define
class PruneConfig:
    """
    Attributes:
        energy_threshold: Energy window (kcal/mol) above the lowest conformer.
        eht_energies: Rank by extended Hueckel (YAeHMOP) energies instead.
        rmsd_threshold: Heavy-atom RMSD (A) below which conformers are duplicates.
        include_hs: Include hydrogens in the RMSD.
        filter_energies: Conformers further apart in energy than
            rmsd_energy_threshold (kcal/mol) are not compared by RMSD.
        filter_rotations: Neither are conformers whose principal moments of inertia
            differ by more than rot_fraction_threshold.
        max_matches: Maximum number of symmetry-equivalent atom maps for the RMSD.
    """

    _section = "prune"

    energy_threshold: float = number(20.0, non_negative)
    eht_energies: bool = flag(False)
    rmsd_threshold: float = number(0.125, non_negative)
    include_hs: bool = flag(False)
    filter_energies: bool = flag(True)
    filter_rotations: bool = flag(True)
    rmsd_energy_threshold: float = number(0.1, non_negative)
    rot_fraction_threshold: float = number(0.03, non_negative)
    max_matches: int = integer(10000, positive)


def _section_field(cls):
    """A section setting: an instance of cls, or a mapping of its settings."""

    def convert(value):
        if isinstance(value, cls):
            return value
        _check_keys(value, cls, cls._section)
        return cls(**value)

    return field(factory=cls, converter=convert)


@define
class PipelineConfig:
    """
    Settings of the default pipeline (Embed, Refine, PruneEnergy, PruneRMSD). The
    defaults reproduce legacy racerts.

    Attributes:
        seed: Random seed (the RDKit embedding seed).
        num_threads: Threads for embedding, force-field refinement and RMSDs.
    """

    seed: int = integer(12)
    num_threads: int = integer(1)
    embed: EmbedConfig = _section_field(EmbedConfig)
    refine: RefineConfig = _section_field(RefineConfig)
    prune: PruneConfig = _section_field(PruneConfig)

    def to_dict(self) -> Dict[str, Any]:
        return attrs.asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> PipelineConfig:
        """
        Build from a (possibly partial) dict; unknown keys and values of the wrong
        type raise ValueError.
        """
        _check_keys(data, cls, "config")
        return cls(**data)

    def to_file(self, path: str) -> None:
        """Write as JSON, or as YAML for .yaml/.yml (needs PyYAML)."""
        yaml = require("yaml", "yaml") if _is_yaml(path) else None  # before open()
        with open(path, "w") as handle:
            if yaml is not None:
                yaml.safe_dump(self.to_dict(), handle, sort_keys=False)
            else:
                json.dump(self.to_dict(), handle, indent=2)
                handle.write("\n")

    @classmethod
    def from_file(cls, path: str) -> PipelineConfig:
        """Read JSON, or YAML for .yaml/.yml (needs PyYAML)."""
        with open(path) as handle:
            data = _load_yaml(handle) if _is_yaml(path) else json.load(handle)
        return cls.from_dict(data or {})

    def build(self, task: Task) -> Pipeline:
        """The default pipeline with these settings for the task."""
        embedder = default_embedder(
            task,
            self.seed,
            mode=self.embed.mode,
            etkdg=self.embed.etkdg,
            useRandomCoords=self.embed.use_random_coords,
            num_threads=self.num_threads,
        )
        optimizer = REFINE_BACKENDS[self.refine.backend](
            force_constant=self.refine.force_constant, num_threads=self.num_threads
        )
        prune = self.prune
        return Pipeline(
            [
                Embed(embedder, self.embed.n_conformers, self.embed.conf_factor),
                Refine(optimizer, fallback=self.refine.fallback),
                PruneEnergy(
                    EnergyPruner(
                        threshold=prune.energy_threshold,
                        YAeHMOP_energies=prune.eht_energies,
                    )
                ),
                PruneRMSD(
                    RMSDPruner(
                        threshold=prune.rmsd_threshold,
                        include_hs=prune.include_hs,
                        num_threads=self.num_threads,
                        filter_energies=prune.filter_energies,
                        filter_rotations=prune.filter_rotations,
                        energy_threshold=prune.rmsd_energy_threshold,
                        rot_fraction_threshold=prune.rot_fraction_threshold,
                        maxMatches=prune.max_matches,
                    )
                ),
            ]
        )


def _check_keys(data, cls, where: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be a mapping, not {type(data).__name__}.")
    unknown = set(data) - {a.name for a in attrs.fields(cls)}
    if unknown:
        raise ValueError(f"Unknown keys in {where}: {sorted(unknown)}.")


def _is_yaml(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in (".yaml", ".yml")


def _load_yaml(handle):
    yaml = require("yaml", "yaml")

    class Loader(yaml.SafeLoader):
        """Reads e.g. 1e6 as a number, as YAML 1.2 does (YAML 1.1: a string)."""

    Loader.add_implicit_resolver(
        "tag:yaml.org,2002:float",
        re.compile(r"^[-+]?(?:[0-9]+\.?[0-9]*|\.[0-9]+)[eE][-+]?[0-9]+$"),
        list("-+0123456789."),
    )
    return yaml.load(handle, Loader=Loader)
