"""PipelineConfig: the settings of the default pipeline, as plain data."""

import json
import os
import re
from numbers import Real
from typing import Any, Dict, Optional

import attrs
from attrs import define, field
from attrs.validators import optional

from racerts.embed import (
    CHIRALITY_FALLBACK_MODES,
    COUNT_POLICIES,
    DEFAULT_CONF_FACTOR,
    DEFAULT_HINT_SHARE,
    EMBED_MODES,
    HINT_ATTEMPTS,
    Embed,
    default_embedder,
)
from racerts.embed.bounds import REFERENCE_BOUNDS
from racerts.pipeline import Pipeline
from racerts.prune import (
    ClusterPruner,
    EnergyPruner,
    PruneCluster,
    PruneCount,
    PruneEnergy,
    PruneRMSD,
    RMSDPruner,
)
from racerts.prune.cluster import METHODS as CLUSTER_METHODS
from racerts.prune.rmsd import HYDROGENS
from racerts.refine import REFINE_BACKENDS, Refine
from racerts.refine.forcefield import DIELECTRIC_MODELS
from racerts.restraints import build_restraints
from racerts.restraints.model import DEFAULT_FORCE_CONSTANT, DEFAULT_HALF_WIDTH
from racerts.restraints.sources import HINT_WINDOW, LINK_SEED, MAX_HINTS
from racerts.system.roles import ROLE_OVERRIDES
from racerts.task import Task
from racerts.utils.checks import is_integer
from racerts.utils.optional import require
from racerts.utils.validators import (
    choice,
    current,
    flag,
    integer,
    is_bool,
    must,
    non_negative,
    number,
    positive,
    to_bool,
    to_float,
)
from racerts.validate import AttackFace, Connectivity, Validate


def _torsion_preferences_of_etkdg(instance, attribute, value) -> None:
    etkdg, preferences = current(
        instance, attribute, value, "etkdg", "torsion_preferences"
    )
    if etkdg is False and not preferences:
        raise ValueError("torsion_preferences applies to ETKDG only (etkdg is false).")


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
        torsion_preferences: ETKDG with its experimental torsion preferences
            (statistics of crystal structures, with the small-ring and macrocycle
            terms). False: its knowledge terms only (RDKit's KDG: planar sp2 atoms,
            linear sp atoms). A setting of ETKDG: plain distance geometry has neither.
        use_random_coords: Start embedding from random coordinates.
        count_policy: How n_conformers=-1 is counted: "fragments" (the default; adds
            the rigid-body freedom of fragments that move relative to the frozen core,
            so one fragment counts as "legacy"), "legacy" (legacy racerts) or
            "per_bond" (max(7, 10 * rotatable bonds)).
        sequential_seeds: One seed stream for all conformers, a seed per conformer from
            a start derived from seed (the default); legacy racerts (False) embeds its
            first 3 conformers twice.
        chirality_fallback: When chirality checks fail with atoms held at a reference:
            "frozen_first" (the default; drop the tags of the frozen atoms first, and
            remove conformers whose stereo is inverted) or "legacy" (drop all chiral
            tags, or stop enforcing chirality).
        reference_bounds: Distance bounds of the graph that exclude a distance of the
            reference geometry: widened to it when embedding fails without them
            ("fallback"), "always", or "never" (legacy racerts).
    """

    _section = "embed"

    mode: str = choice("cmap", EMBED_MODES)
    n_conformers: int = integer(
        -1, must(lambda n: n == -1 or n > 0, "be -1 (default count) or > 0")
    )
    conf_factor: int = integer(DEFAULT_CONF_FACTOR, non_negative)
    etkdg: Optional[bool] = field(
        default=None,
        converter=to_bool,
        validator=[optional(is_bool), _torsion_preferences_of_etkdg],
    )
    torsion_preferences: bool = flag(True, _torsion_preferences_of_etkdg)
    use_random_coords: bool = flag(True)
    count_policy: str = choice("fragments", COUNT_POLICIES)
    sequential_seeds: bool = flag(True)
    chirality_fallback: str = choice("frozen_first", CHIRALITY_FALLBACK_MODES)
    reference_bounds: str = choice("fallback", REFERENCE_BOUNDS)


def _dielectric_for_mmff(instance, attribute, value) -> None:
    backend, model, constant = current(
        instance, attribute, value, "backend", "dielectric_model", "dielectric_constant"
    )
    if backend != "mmff" and (model, constant) != ("constant", 1.0):
        raise ValueError("The dielectric settings apply to MMFF only.")


@define
class RefineConfig:
    """
    Attributes:
        backend: "mmff" or "uff".
        fallback: Fall back to UFF when MMFF has no parameters.
        force_constant: Force constant (kcal/mol/A^2) that holds the frozen atoms.
        converge: Restart minimizations that stop early next to the frozen atoms;
            legacy racerts stops at the first converged call.
        energies_without_anchors: Report energies without the terms that hold the frozen
            atoms (the default); legacy racerts includes them. With restraints, soft
            atoms or active-bond windows they are always left out.
        dielectric_model: MMFF electrostatics, "constant" or "distance"-dependent.
        dielectric_constant: MMFF dielectric constant.
        num_workers: Worker processes of the force-field refinement (1: none). They
            give the results of one process; worth it for hundreds of conformers of a
            large molecule.
    """

    _section = "refine"

    backend: str = choice("mmff", REFINE_BACKENDS, _dielectric_for_mmff)
    fallback: bool = flag(True)
    force_constant: float = number(1e6, positive)
    converge: bool = flag(False)
    energies_without_anchors: bool = flag(True)
    dielectric_model: str = choice("constant", DIELECTRIC_MODELS, _dielectric_for_mmff)
    dielectric_constant: float = number(1.0, positive, _dielectric_for_mmff)
    num_workers: int = integer(1, positive)


def _polar_without_prefilters(instance, attribute, value) -> None:
    hydrogens, energies, rotations = current(
        instance, attribute, value, "hydrogens", "filter_energies", "filter_rotations"
    )
    if hydrogens == "polar" and (energies or rotations):
        raise ValueError(
            "prune.hydrogens 'polar' needs prune.filter_energies and "
            "prune.filter_rotations off."
        )


@define
class PruneConfig:
    """
    Attributes:
        energy_threshold: Energy window (kcal/mol) above the lowest conformer.
        rmsd_threshold: RMSD (A) below which conformers are duplicates.
        hydrogens: The hydrogens in that RMSD: "none" (the heavy atoms, as legacy
            racerts), "polar" (the default: also the hydrogens on N, O, P and S, so
            that the rotamers of a hydrogen bond stay apart; needs both prefilters
            off) or "all".
        filter_energies: Conformers further apart in energy than
            rmsd_energy_threshold (kcal/mol) are not compared by RMSD.
        filter_rotations: Neither are conformers whose principal moments of inertia
            differ by more than rot_fraction_threshold. Both are prefilters of legacy
            racerts and off by default: they are no bounds of the RMSD, so duplicates
            that they keep apart stay in the ensemble.
        max_matches: Maximum number of symmetry-equivalent atom maps for the RMSD.
            (These six RMSD settings act with method "rmsd" only.) With both
            prefilters off every pair is decided by its RMSD, and max_matches is the
            most atom maps that are listed before local symmetry is assigned instead
            (racerts.rmsd).
        check_stereo: After refinement, drop conformers whose specified stereo
            (outside the core atoms of the task, e.g. the reacting atoms) differs from
            the graph, e.g. after a chirality fallback of the embedding. After the
            legacy fallback, which drops the chiral tags, there is nothing left to
            check: use embed.chirality_fallback "frozen_first". On by default.
        method: "rmsd" (duplicates by RMSD, as legacy racerts) or "cluster" (one
            conformer per cluster, see [ClusterPruner][racerts.prune.ClusterPruner]).
        cluster_method: "butina", "hierarchical" or "leader".
        cluster_threshold: Cluster distance (A, heavy-atom RMSD after superposition).
        max_conformers: Keep at most this many conformers, the lowest, after the
            duplicates are pruned (PruneCount): a bound on what a slower level after the
            pipeline gets, whose cost grows with the conformers that survive the
            pruning; 0 (the default): all.
    """

    _section = "prune"

    energy_threshold: float = number(20.0, non_negative)
    rmsd_threshold: float = number(0.125, non_negative)
    hydrogens: str = choice("polar", HYDROGENS, _polar_without_prefilters)
    filter_energies: bool = flag(False, _polar_without_prefilters)
    filter_rotations: bool = flag(False, _polar_without_prefilters)
    rmsd_energy_threshold: float = number(0.1, non_negative)
    rot_fraction_threshold: float = number(0.03, non_negative)
    max_matches: int = integer(100, positive)
    check_stereo: bool = flag(True)
    method: str = choice("rmsd", ("rmsd", "cluster"))
    cluster_method: str = choice("butina", CLUSTER_METHODS)
    cluster_threshold: float = number(1.5, non_negative)
    max_conformers: int = integer(0, non_negative)


def _is_restraint_item(item, size: int) -> bool:
    """[atom, atom] or [atom, atom, distance] with integer atoms and a number."""

    if not isinstance(item, (list, tuple)) or len(item) != size:
        return False
    if not (is_integer(item[0]) and is_integer(item[1])):
        return False
    return size == 2 or (isinstance(item[2], Real) and not isinstance(item[2], bool))


def _as_lists(items):
    """Restraint items as lists (e.g. tuples from Python); the rest is left to the check."""
    if isinstance(items, (list, tuple)) and all(
        isinstance(item, (list, tuple)) for item in items
    ):
        return [list(item) for item in items]
    return items


def _restraint_items(size: int):
    """A list of [atom, atom] (size 2) or [atom, atom, distance] (size 3) items."""
    shape = "[atom, atom, distance]" if size == 3 else "[atom, atom]"
    check = must(
        lambda items: (
            isinstance(items, list)
            and all(_is_restraint_item(item, size) for item in items)
        ),
        f"be a list of {shape} lists (atom indices as integers)",
    )
    return field(factory=list, converter=_as_lists, validator=check)


def _window(value):
    """A window [lower, upper] (A) as two floats; other values are left to its check."""
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return [to_float(x) for x in value]
    return value


def _is_window(window) -> bool:
    return (
        isinstance(window, list)
        and len(window) == 2
        and all(isinstance(x, float) for x in window)
        and 0 < window[0] < window[1]
    )


def _roles(roles) -> dict:
    """{atom: role} with integer atoms (JSON and YAML keys can be strings)."""
    if not isinstance(roles, dict):
        raise ValueError("restraints.roles must be a mapping {atom: role}.")
    checked = {}
    for atom, role in roles.items():
        try:
            index = int(atom)
        except (TypeError, ValueError):
            raise ValueError(
                f"restraints.roles: {atom!r} is not an atom index."
            ) from None
        if role not in ROLE_OVERRIDES:
            raise ValueError(
                f"restraints.roles: unknown role {role!r} (use {ROLE_OVERRIDES})."
            )
        checked[index] = role
    return checked


@define
class RestraintConfig:
    """
    Distance restraints (see [racerts.restraints.build_restraints][]): windows in the
    embedding and flat-bottom terms in refinement (MMFF/UFF and ASE calculators);
    reported energies leave them out.

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
            embedding-only windows, used in a share hint_share of the conformers. A
            hint that gives no conformer within hint_attempts RDKit attempts per
            conformer is dropped (see [Embed][racerts.embed.Embed]); 0: RDKit's own
            limit. hint_window: the window of a hint's H...acceptor distance (A).
        contain: Radius (A) of a containment restraint for every fragment that no
            restraint anchors (e.g. a counterion near the core); 0: none.
        fraction: The probability with which an embedding batch takes each restraint
            of the reference geometry (hbonds, contacts, keep_fragments); refinement
            follows the batch. 1: every conformer takes them all.
        roles: {atom: role} overrides of the fragment roles ("anchored",
            "contained", "free"; see racerts.system.roles).
    """

    _section = "restraints"

    user: list = _restraint_items(3)
    half_width: float = number(DEFAULT_HALF_WIDTH, positive)
    force_constant: float = number(DEFAULT_FORCE_CONSTANT, positive)
    hbonds: bool = flag(False)
    contacts: list = _restraint_items(2)
    keep_fragments: bool = flag(False)
    fragment_links: list = _restraint_items(2)
    link_fragments: bool = flag(False)
    hints: bool = flag(False)
    max_hints: int = integer(MAX_HINTS, positive)
    hint_share: float = number(
        DEFAULT_HINT_SHARE, must(lambda share: 0 <= share <= 1, "be between 0 and 1")
    )
    hint_attempts: int = integer(HINT_ATTEMPTS, non_negative)
    hint_window: list = field(
        factory=lambda: list(HINT_WINDOW),
        converter=_window,
        validator=must(_is_window, "be a window [lower, upper] with 0 < lower < upper"),
    )
    contain: float = number(
        0.0, must(lambda radius: radius >= 0, "be a radius >= 0 (0: none)")
    )
    fraction: float = number(1.0, must(lambda share: 0 < share <= 1, "be in (0, 1]"))
    roles: dict = field(factory=dict, converter=_roles)

    def __bool__(self) -> bool:
        return bool(
            self.user
            or self.hbonds
            or self.contacts
            or self.keep_fragments
            or self.fragment_links
            or self.link_fragments
            or self.hints
            or self.contain
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
            hint_window=tuple(self.hint_window),
            contain=self.contain,
        )


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
    Settings of the default pipeline (Embed, Refine, a stereo check, PruneEnergy,
    PruneRMSD). PipelineConfig.legacy() gives those of legacy racerts; the defaults
    differ in embed.sequential_seeds, embed.chirality_fallback, embed.count_policy,
    refine.energies_without_anchors, prune.check_stereo, prune.filter_energies,
    prune.filter_rotations, prune.hydrogens and prune.max_matches.

    Attributes:
        seed: The seed of a run; the seeds of its batches and conformers are derived
            from it (racerts.utils.seeds).
        num_threads: Threads for embedding and force-field refinement (the latter
            gains little: RDKit's minimizer does not release Python's lock).
    """

    seed: int = integer(12)
    num_threads: int = integer(1)
    embed: EmbedConfig = _section_field(EmbedConfig)
    refine: RefineConfig = _section_field(RefineConfig)
    prune: PruneConfig = _section_field(PruneConfig)
    restraints: RestraintConfig = _section_field(RestraintConfig)

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
            "refine": dict(converge=False, energies_without_anchors=False),
            "prune": dict(
                check_stereo=False,
                filter_energies=True,
                filter_rotations=True,
                hydrogens="none",
                max_matches=10000,
            ),
        }
        for key, legacy in sections.items():
            given = settings.get(key, {})
            if isinstance(given, dict):
                settings[key] = {**legacy, **given}
        return cls(**settings)

    def to_dict(self) -> Dict[str, Any]:
        return attrs.asdict(self)

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
            try:
                data = _load_yaml(handle) if _is_yaml(path) else json.load(handle)
            except ValueError as error:  # not JSON, or not YAML (see _load_yaml)
                raise ValueError(f"{path}: {error}") from None
        if data is None:  # an empty YAML file
            data = {}
        if legacy:
            _check_keys(data, cls, "config")
            return cls.legacy(**data)
        return cls.from_dict(data)

    def embed_stage(self, task: Task) -> Embed:
        """The Embed stage of the default pipeline for the task."""
        embed = self.embed
        embedder = default_embedder(
            task,
            self.seed,
            mode=embed.mode,
            etkdg=embed.etkdg,
            torsion_preferences=embed.torsion_preferences,
            chirality_fallback=embed.chirality_fallback,
            reference_bounds=embed.reference_bounds,
            useRandomCoords=embed.use_random_coords,
            sequential_seeds=embed.sequential_seeds,
            num_threads=self.num_threads,
        )
        return Embed(
            embedder,
            embed.n_conformers,
            embed.conf_factor,
            count_policy=embed.count_policy,
            hint_share=self.restraints.hint_share,
            hint_attempts=self.restraints.hint_attempts,
            restraint_fraction=self.restraints.fraction,
        )

    def refine_stage(self) -> Refine:
        """The Refine stage of the default pipeline: the force field of the settings."""
        refine = self.refine
        options = dict(
            force_constant=refine.force_constant,
            num_threads=self.num_threads,
            converge=refine.converge,
            energies_without_anchors=refine.energies_without_anchors,
            num_workers=refine.num_workers,
        )
        if refine.backend == "mmff":
            options.update(
                dielectric_model=refine.dielectric_model,
                dielectric_constant=refine.dielectric_constant,
            )
        return Refine(
            REFINE_BACKENDS[refine.backend](**options),
            fallback=refine.fallback,
            stereo_anchors=self.embed.chirality_fallback == "frozen_first",
        )

    def build(self, task: Task) -> Pipeline:
        """
        The default pipeline with these settings for the task. The restraints section
        is not part of it: generate builds the restraints into the Context.
        """
        prune = self.prune
        windowed = getattr(task, "windowed", False)
        energy_pruner = EnergyPruner(threshold=prune.energy_threshold)
        faces = []
        if windowed and getattr(task, "stereo_filter", False):
            faces.append(Validate(AttackFace()))
        checks = []
        if prune.check_stereo:
            checks.append(Validate(Connectivity(bonds=False)))
        count = [PruneCount(prune.max_conformers)] if prune.max_conformers else []
        return Pipeline(
            [
                self.embed_stage(task),
                self.refine_stage(),
                *faces,
                *checks,
                PruneEnergy(energy_pruner),
                self._duplicates(),
                *count,
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
        if not (prune.filter_energies or prune.filter_rotations):
            return PruneRMSD(
                RMSDPruner(
                    prune.rmsd_threshold,
                    hydrogens=prune.hydrogens,
                    max_maps=prune.max_matches,
                )
            )
        # with a prefilter: the pruner of legacy racerts (imported here: racerts.compat
        # imports this module)
        from racerts.compat.pruner import RMSDPruner as LegacyRMSDPruner

        return PruneRMSD(
            LegacyRMSDPruner(
                threshold=prune.rmsd_threshold,
                hydrogens=prune.hydrogens,
                num_threads=self.num_threads,
                filter_energies=prune.filter_energies,
                filter_rotations=prune.filter_rotations,
                energy_threshold=prune.rmsd_energy_threshold,
                rot_fraction_threshold=prune.rot_fraction_threshold,
                maxMatches=prune.max_matches,
            )
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
    try:
        return yaml.load(handle, Loader=Loader)
    except yaml.YAMLError as error:  # on one line, as a ValueError like json's
        raise ValueError(" ".join(str(error).split())) from None
