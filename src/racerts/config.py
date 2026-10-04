"""PipelineConfig: the settings of the default pipeline, as plain data."""

import json
import os
import re
import typing
import warnings
from dataclasses import asdict, dataclass, field, fields
from numbers import Integral, Real
from typing import Any, Dict, Optional

from racerts.embed import (
    CHIRALITY_FALLBACK_MODES,
    COUNT_POLICIES,
    DEFAULT_CONF_FACTOR,
    DEFAULT_HINT_SHARE,
    EMBED_MODES,
    Embed,
    default_embedder,
)
from racerts.pipeline import Pipeline
from racerts.prune import (
    ClusterPruner,
    EnergyPruner,
    PruneCluster,
    PruneEnergy,
    PruneRMSD,
    RMSDPruner,
)
from racerts.prune.cluster import METHODS as CLUSTER_METHODS
from racerts.refine import REFINE_BACKENDS, Refine
from racerts.refine.forcefield import DIELECTRIC_MODELS
from racerts.restraints import build_restraints
from racerts.restraints.model import DEFAULT_FORCE_CONSTANT, DEFAULT_HALF_WIDTH
from racerts.restraints.sources import LINK_SEED, MAX_HINTS
from racerts.task import Task
from racerts.utils.optional import require
from racerts.validate import AttackFace, Connectivity, Validate


@dataclass
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
        count_policy: How n_conformers=-1 is counted: "legacy", "fragments" (adds the
            rigid-body freedom of fragments that move relative to the frozen core) or
            "per_bond" (max(7, 10 * rotatable bonds)).
        sequential_seeds: One seed stream for all conformers (a seed per conformer,
            from a start derived from seed); legacy racerts embeds its first 3
            conformers twice.
        chirality_fallback: When chirality checks fail with atoms held at a reference:
            "legacy" (drop all chiral tags, or stop enforcing chirality) or
            "frozen_first" (drop the tags of the frozen atoms first).
    """

    mode: str = "cmap"
    n_conformers: int = -1
    conf_factor: int = DEFAULT_CONF_FACTOR
    etkdg: Optional[bool] = None
    use_random_coords: bool = True
    count_policy: str = "legacy"
    sequential_seeds: bool = False
    chirality_fallback: str = "legacy"

    def __post_init__(self):
        _check_types(self, "embed")
        if self.mode not in EMBED_MODES:
            raise ValueError(f"embed.mode must be one of {sorted(EMBED_MODES)}.")
        if self.count_policy not in COUNT_POLICIES:
            raise ValueError(f"embed.count_policy must be one of {COUNT_POLICIES}.")
        if self.chirality_fallback not in CHIRALITY_FALLBACK_MODES:
            raise ValueError(
                f"embed.chirality_fallback must be one of {CHIRALITY_FALLBACK_MODES}."
            )
        if self.n_conformers != -1 and self.n_conformers < 1:
            raise ValueError("embed.n_conformers must be -1 (default count) or > 0.")
        if self.conf_factor < 0:
            raise ValueError("embed.conf_factor must not be negative.")


@dataclass
class RefineConfig:
    """
    Attributes:
        backend: "mmff" or "uff".
        fallback: Fall back to UFF when MMFF has no parameters.
        force_constant: Force constant (kcal/mol/A^2) that holds the frozen atoms.
        converge: Restart minimizations that stop early next to the frozen atoms;
            legacy racerts stops at the first converged call.
        anchor_free_energies: Report energies without the terms that hold the frozen
            atoms; legacy racerts includes them. With restraints, soft atoms or
            active-bond windows they are always left out.
        dielectric_model: MMFF electrostatics, "constant" or "distance"-dependent.
        dielectric_constant: MMFF dielectric constant.
    """

    backend: str = "mmff"
    fallback: bool = True
    force_constant: float = 1e6
    converge: bool = False
    anchor_free_energies: bool = False
    dielectric_model: str = "constant"
    dielectric_constant: float = 1.0

    def __post_init__(self):
        _check_types(self, "refine")
        if self.backend not in REFINE_BACKENDS:
            raise ValueError(
                f"refine.backend must be one of {sorted(REFINE_BACKENDS)}."
            )
        if self.force_constant <= 0:
            raise ValueError("refine.force_constant must be positive.")
        if self.dielectric_model not in DIELECTRIC_MODELS:
            raise ValueError(
                f"refine.dielectric_model must be one of {sorted(DIELECTRIC_MODELS)}."
            )
        if self.dielectric_constant <= 0:
            raise ValueError("refine.dielectric_constant must be positive.")
        if self.backend != "mmff" and (
            self.dielectric_model,
            self.dielectric_constant,
        ) != ("constant", 1.0):
            raise ValueError("The dielectric settings apply to MMFF only.")


@dataclass
class PruneConfig:
    """
    Attributes:
        energy_threshold: Energy window (kcal/mol) above the lowest conformer.
        eht_energies: Rank by extended Hueckel (YAeHMOP) energies instead (deprecated:
            rescore with an ASE calculator, the Rescore stage; not with active-bond
            windows).
        rmsd_threshold: Heavy-atom RMSD (A) below which conformers are duplicates.
        include_hs: Include hydrogens in the RMSD.
        filter_energies: Conformers further apart in energy than
            rmsd_energy_threshold (kcal/mol) are not compared by RMSD.
        filter_rotations: Neither are conformers whose principal moments of inertia
            differ by more than rot_fraction_threshold.
        max_matches: Maximum number of symmetry-equivalent atom maps for the RMSD.
            (These six RMSD settings act with method "rmsd" only.)
        check_stereo: After refinement, drop conformers whose specified stereo
            (outside the core atoms of the task, e.g. the reacting atoms) differs from
            the graph, e.g. after a chirality fallback of the embedding. After the
            legacy fallback, which drops the chiral tags, there is nothing left to
            check: use embed.chirality_fallback "frozen_first".
        method: "rmsd" (duplicates by RMSD, as legacy racerts) or "cluster" (one
            conformer per cluster, see ClusterPruner).
        cluster_method: "butina", "hierarchical" or "leader".
        cluster_threshold: Cluster distance (A, heavy-atom RMSD after superposition).
    """

    energy_threshold: float = 20.0
    eht_energies: bool = False
    rmsd_threshold: float = 0.125
    include_hs: bool = False
    filter_energies: bool = True
    filter_rotations: bool = True
    rmsd_energy_threshold: float = 0.1
    rot_fraction_threshold: float = 0.03
    max_matches: int = 10000
    check_stereo: bool = False
    method: str = "rmsd"
    cluster_method: str = "butina"
    cluster_threshold: float = 1.5

    def __post_init__(self):
        _check_types(self, "prune")
        if self.method not in ("rmsd", "cluster"):
            raise ValueError("prune.method must be 'rmsd' or 'cluster'.")
        if self.cluster_method not in CLUSTER_METHODS:
            raise ValueError(f"prune.cluster_method must be one of {CLUSTER_METHODS}.")
        for name in (
            "energy_threshold",
            "rmsd_threshold",
            "rmsd_energy_threshold",
            "rot_fraction_threshold",
            "cluster_threshold",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"prune.{name} must not be negative.")
        if self.max_matches < 1:
            raise ValueError("prune.max_matches must be positive.")
        if self.eht_energies:
            warnings.warn(
                "prune.eht_energies is deprecated; rescore the ensemble with an ASE "
                "calculator instead (the Rescore stage).",
                FutureWarning,  # shown by default, unlike DeprecationWarning
                stacklevel=3,
            )


@dataclass
class RestraintConfig:
    """
    Distance restraints (see racerts.restraints.build_restraints): windows in the
    embedding and flat-bottom terms in MMFF/UFF refinement; reported energies leave
    them out, and ASE refinements run without them.

    Attributes:
        user: [atom, atom, target distance (A)] triplets.
        half_width: Half width of the windows (A).
        force_constant: kcal/(mol A^2), as RDKit: 1/2 k (d - bound)^2.
        hbonds: Keep the hydrogen bonds of the reference geometry.
        contacts: [atom, atom] non-covalent contacts to keep as in the reference.
        keep_fragments: Keep fragments without core atoms (solvent) at the core.
        fragment_links: [atom, atom] links between fragments: windows
            [1.0, 1.3] x the vdW sum.
        link_fragments: Also choose links that join every fragment.
        hints: Candidate hydrogen bonds from the graph (at most max_hints) as
            embedding-only windows, used in a share hint_share of the conformers.
    """

    user: list = field(default_factory=list)
    half_width: float = DEFAULT_HALF_WIDTH
    force_constant: float = DEFAULT_FORCE_CONSTANT
    hbonds: bool = False
    contacts: list = field(default_factory=list)
    keep_fragments: bool = False
    fragment_links: list = field(default_factory=list)
    link_fragments: bool = False
    hints: bool = False
    max_hints: int = MAX_HINTS
    hint_share: float = DEFAULT_HINT_SHARE

    def __post_init__(self):
        _check_types(self, "restraints")
        if not 0 <= self.hint_share <= 1:
            raise ValueError("restraints.hint_share must be between 0 and 1.")
        if self.max_hints < 1:
            raise ValueError("restraints.max_hints must be positive.")
        for name, size in (("user", 3), ("contacts", 2), ("fragment_links", 2)):
            items = getattr(self, name)
            if not isinstance(items, (list, tuple)) or not all(
                _is_restraint_item(item, size) for item in items
            ):
                shape = "[atom, atom, distance]" if size == 3 else "[atom, atom]"
                raise ValueError(
                    f"restraints.{name} must be a list of {shape} lists (atom "
                    "indices as integers)."
                )
            setattr(self, name, [list(item) for item in items])
        for name in ("half_width", "force_constant"):
            if getattr(self, name) <= 0:
                raise ValueError(f"restraints.{name} must be positive.")

    def __bool__(self) -> bool:
        return bool(
            self.user
            or self.hbonds
            or self.contacts
            or self.keep_fragments
            or self.fragment_links
            or self.link_fragments
            or self.hints
        )

    def build(self, mol, frozen, seed: int = LINK_SEED):
        """The RestraintSet for mol (with the reference geometry) and its frozen set."""
        return build_restraints(
            mol,
            frozen,
            user=[tuple(item) for item in self.user],
            half_width=self.half_width,
            force_constant=self.force_constant,
            hbonds=self.hbonds,
            contacts=[tuple(item) for item in self.contacts],
            keep_fragments=self.keep_fragments,
            fragment_links=[tuple(item) for item in self.fragment_links] or None,
            link_fragments=self.link_fragments,
            seed=seed,
            hints=self.hints,
            max_hints=self.max_hints,
        )


def _is_restraint_item(item, size: int) -> bool:
    """[atom, atom] or [atom, atom, distance] with integer atoms and a number."""

    def is_index(value):
        return isinstance(value, Integral) and not isinstance(value, bool)

    if not isinstance(item, (list, tuple)) or len(item) != size:
        return False
    if not (is_index(item[0]) and is_index(item[1])):
        return False
    return size == 2 or (isinstance(item[2], Real) and not isinstance(item[2], bool))


_SECTIONS = {
    "embed": EmbedConfig,
    "refine": RefineConfig,
    "prune": PruneConfig,
    "restraints": RestraintConfig,
}


@dataclass
class PipelineConfig:
    """
    Settings of the default pipeline (Embed, Refine, PruneEnergy, PruneRMSD). The
    defaults reproduce legacy racerts.

    Attributes:
        seed: Random seed (the RDKit embedding seed).
        num_threads: Threads for embedding and force-field refinement (the latter
            gains little: RDKit's minimizer does not release Python's lock).
    """

    seed: int = 12
    num_threads: int = 1
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    refine: RefineConfig = field(default_factory=RefineConfig)
    prune: PruneConfig = field(default_factory=PruneConfig)
    restraints: RestraintConfig = field(default_factory=RestraintConfig)

    def __post_init__(self):
        _check_types(self, "")
        for key, section in _SECTIONS.items():
            value = getattr(self, key)
            if isinstance(value, dict):
                _check_keys(value, section, key)
                setattr(self, key, section(**value))
            elif not isinstance(value, section):
                raise ValueError(
                    f"{key} must be a mapping, not {type(value).__name__}."
                )

    @classmethod
    def legacy(cls, **settings) -> "PipelineConfig":
        """
        The settings of legacy racerts, whatever the defaults: the pipeline gives the
        results of ConformerGenerator().generate_conformers. settings as in
        PipelineConfig (e.g. seed); sections given as dicts update the legacy ones,
        sections given as EmbedConfig etc. are used as they are.
        """
        sections = {
            "embed": dict(
                count_policy="legacy",
                sequential_seeds=False,
                chirality_fallback="legacy",
            ),
            "refine": dict(converge=False, anchor_free_energies=False),
            "prune": dict(check_stereo=False),
        }
        for key, legacy in sections.items():
            given = settings.get(key, {})
            if isinstance(given, dict):
                settings[key] = {**legacy, **given}
        return cls(**settings)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PipelineConfig":
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
    def from_file(cls, path: str, legacy: bool = False) -> "PipelineConfig":
        """
        Read JSON, or YAML for .yaml/.yml (needs PyYAML). Settings missing from the
        file have their defaults, or with legacy those of PipelineConfig.legacy().
        """
        with open(path) as handle:
            data = _load_yaml(handle) if _is_yaml(path) else json.load(handle)
        if data is None:  # an empty YAML file
            data = {}
        if legacy:
            _check_keys(data, cls, "config")
            return cls.legacy(**data)
        return cls.from_dict(data)

    def build(self, task: Task) -> Pipeline:
        """
        The default pipeline with these settings for the task. The restraints section
        is not part of it: generate builds the restraints into the Context.
        """
        embed, refine = self.embed, self.refine
        embedder = default_embedder(
            task,
            self.seed,
            mode=embed.mode,
            etkdg=embed.etkdg,
            chirality_fallback=embed.chirality_fallback,
            useRandomCoords=embed.use_random_coords,
            sequential_seeds=embed.sequential_seeds,
            num_threads=self.num_threads,
        )
        options = dict(
            force_constant=refine.force_constant,
            num_threads=self.num_threads,
            converge=refine.converge,
            anchor_free_energies=refine.anchor_free_energies,
        )
        if refine.backend == "mmff":
            options.update(
                dielectric_model=refine.dielectric_model,
                dielectric_constant=refine.dielectric_constant,
            )
        optimizer = REFINE_BACKENDS[refine.backend](**options)
        prune = self.prune
        windowed = getattr(task, "windowed", False)
        if prune.eht_energies and windowed:  # windows are pruned per target, on copies
            raise ValueError(
                "prune.eht_energies does not combine with active-bond windows."
            )
        with warnings.catch_warnings():  # PruneConfig has warned about eht_energies
            warnings.simplefilter("ignore", FutureWarning)
            energy_pruner = EnergyPruner(
                threshold=prune.energy_threshold, YAeHMOP_energies=prune.eht_energies
            )
        faces = []
        if windowed and getattr(task, "stereo_filter", False):
            faces.append(Validate(AttackFace()))
        checks = []
        if prune.check_stereo:
            checks.append(Validate(Connectivity(bonds=False)))
        return Pipeline(
            [
                Embed(
                    embedder,
                    embed.n_conformers,
                    embed.conf_factor,
                    count_policy=embed.count_policy,
                    hint_share=self.restraints.hint_share,
                ),
                Refine(
                    optimizer,
                    fallback=self.refine.fallback,
                    stereo_anchors=embed.chirality_fallback == "frozen_first",
                ),
                *faces,
                *checks,
                PruneEnergy(energy_pruner),
                self._duplicates(),
            ]
        )

    def _duplicates(self):
        prune = self.prune
        if prune.method == "cluster":
            return PruneCluster(
                ClusterPruner(
                    threshold=prune.cluster_threshold, method=prune.cluster_method
                )
            )
        return PruneRMSD(
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
        )


_KINDS = {bool: "true or false", int: "an integer", float: "a number", str: "a string"}


def _check_types(obj, section: str) -> None:
    """Check the fields of a config dataclass against their annotations."""
    hints = typing.get_type_hints(type(obj))
    for f in fields(obj):
        kind = hints[f.name]
        if kind in _KINDS or typing.get_origin(kind) is typing.Union:
            name = f"{section}.{f.name}" if section else f.name
            setattr(obj, f.name, _convert(getattr(obj, f.name), kind, name))


def _convert(value, kind, name: str):
    if typing.get_origin(kind) is typing.Union:  # Optional[...]
        if value is None:
            return None
        kind = next(arg for arg in typing.get_args(kind) if arg is not type(None))
    if isinstance(value, bool) == (kind is bool):  # True/False are ints in Python
        if kind is float and isinstance(value, int):
            return float(value)
        if isinstance(value, kind):
            return value
    raise ValueError(f"{name} must be {_KINDS[kind]}, not {value!r}.")


def _check_keys(data, cls, where: str) -> None:
    if not isinstance(data, dict):
        raise ValueError(f"{where} must be a mapping, not {type(data).__name__}.")
    unknown = set(data) - {f.name for f in fields(cls)}
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
