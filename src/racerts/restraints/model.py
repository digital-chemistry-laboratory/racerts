"""Distance restraints: flat-bottom windows between atom pairs."""

import inspect
import json
import math
from dataclasses import asdict, dataclass, replace
from numbers import Real
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from racerts.utils.checks import is_integer

DEFAULT_HALF_WIDTH = 0.25  # A
DEFAULT_FORCE_CONSTANT = 20.0  # kcal/(mol A^2)
STAGES = ("embed", "refine", "both")


def accepts_restraints(method) -> bool:
    """Whether a method of a component (embed, _refine) takes a restraints argument."""
    try:
        return "restraints" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True)
class DistanceRestraint:
    """
    A window [lower, upper] (A) for the distance between two atoms.

    Embedding uses the window as the bounds of the pair. Refinement adds the term
    1/2 k (d - bound)^2 outside it, with k = force_constant in kcal/(mol A^2);
    reported energies leave this term out.

    Attributes:
        first, second: The atoms (stored in ascending order).
        stage: Where it applies: "embed", "refine" or "both".
        source: Where it comes from ("user", "hbond", "contact", "fragment", ...).
        label: A name for the provenance (default "source:first-second").
    """

    first: int
    second: int
    lower: float
    upper: float
    force_constant: float = DEFAULT_FORCE_CONSTANT
    stage: str = "both"
    source: str = "user"
    label: str = ""

    def __post_init__(self):
        if not (is_integer(self.first) and is_integer(self.second)):
            raise TypeError("Restraint atom indices must be integers.")
        if min(self.first, self.second) < 0 or self.first == self.second:
            raise ValueError("A restraint needs two different, non-negative atoms.")
        for name in ("lower", "upper", "force_constant"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Real):
                raise TypeError(f"Restraint {name} must be a number.")
        if not (math.isfinite(self.lower) and math.isfinite(self.upper)) or not (
            0 <= self.lower < self.upper
        ):
            raise ValueError(
                f"Restraint bounds must satisfy 0 <= lower < upper, not "
                f"{self.lower}, {self.upper}."
            )
        if not math.isfinite(self.force_constant) or self.force_constant <= 0:
            raise ValueError("Restraint force constant must be positive.")
        if self.stage not in STAGES:
            raise ValueError(f"Restraint stage must be one of {STAGES}.")
        # RDKit needs plain Python numbers, e.g. not numpy integers.
        first, second = sorted((int(self.first), int(self.second)))
        object.__setattr__(self, "first", first)
        object.__setattr__(self, "second", second)
        for name in ("lower", "upper", "force_constant"):
            object.__setattr__(self, name, float(getattr(self, name)))
        if not self.label:
            object.__setattr__(self, "label", f"{self.source}:{first}-{second}")

    @property
    def pair(self) -> Tuple[int, int]:
        return self.first, self.second

    @classmethod
    def around(
        cls,
        first: int,
        second: int,
        target: float,
        half_width: float = DEFAULT_HALF_WIDTH,
        **settings,
    ) -> "DistanceRestraint":
        """The window target +/- half_width (the lower bound at least 0)."""
        if isinstance(target, bool) or not isinstance(target, Real):
            raise TypeError(f"The target distance must be a number, not {target!r}.")
        if not math.isfinite(target) or target <= 0:
            raise ValueError(f"The target distance must be positive, not {target}.")
        if not math.isfinite(half_width) or half_width <= 0:
            raise ValueError("The half width must be positive.")
        return cls(
            first,
            second,
            max(0.0, float(target) - half_width),
            float(target) + half_width,
            **settings,
        )

    def violation(self, positions) -> float:
        """How far (A) the distance in positions lies outside the window (0 inside)."""
        positions = np.asarray(positions, dtype=float)
        d = float(np.linalg.norm(positions[self.first] - positions[self.second]))
        return max(0.0, self.lower - d, d - self.upper)


# The restraints of the reference geometry that Embed(restraint_fraction=...) takes in
# some batches only; user restraints, links, containment and task windows always apply.
OPTIONAL_SOURCES = ("hbond", "contact", "fragment")
# The sources of the restraints that place the reacting atoms of a windowed TS: they
# are part of the task, not restraints that a refinement can release.
WINDOW_SOURCES = ("active", "neighbor", "core", "target")


def applying(restraints, provenance) -> list:
    """
    The restraints that apply to a conformer: all of them, but of the optional ones
    (OPTIONAL_SOURCES) only those of its embedding batch, if its provenance lists
    them ("restraint_subset", see Embed restraint_fraction), and only the windows of
    the task (WINDOW_SOURCES) if its last refinement released the restraints
    ("restraints_released", see Refine restraints).
    """
    if provenance.get("restraints_released"):
        restraints = [
            r for r in restraints if getattr(r, "source", None) in WINDOW_SOURCES
        ]
    subset = provenance.get("restraint_subset")
    if subset is None:
        return list(restraints)
    subset = set(subset)
    return [
        r
        for r in restraints
        if getattr(r, "source", None) not in OPTIONAL_SOURCES or r.label in subset
    ]


SOFT_TOLERANCE = 0.3  # A, the flat bottom of the position restraints of soft atoms
SOFT_FORCE_CONSTANT = 5.0  # kcal/(mol A^2)


@dataclass(frozen=True)
class PositionRestraint:
    """
    Holds an atom near a point (A): in MMFF/UFF refinement a flat-bottom term
    E = 1/2 k (d - tolerance)^2 beyond tolerance, d the distance from the point (in
    the frame of the reference, on which the conformers are aligned). Used for the
    soft atoms of a task (FrozenSet.soft); reported energies leave it out.
    """

    atom: int
    point: Tuple[float, float, float]
    tolerance: float = SOFT_TOLERANCE
    force_constant: float = SOFT_FORCE_CONSTANT
    stage: str = "refine"
    source: str = "soft"

    def __post_init__(self):
        if not is_integer(self.atom):
            raise TypeError("A position restraint needs an integer atom index.")
        if self.atom < 0:
            raise ValueError("A position restraint needs a non-negative atom index.")
        point = tuple(float(x) for x in self.point)
        if len(point) != 3 or not all(math.isfinite(x) for x in point):
            raise ValueError(f"Invalid point {self.point}.")
        if not self.tolerance >= 0 or not self.force_constant > 0:
            raise ValueError("tolerance must be >= 0 and force_constant positive.")
        object.__setattr__(self, "atom", int(self.atom))
        object.__setattr__(self, "point", point)
        object.__setattr__(self, "tolerance", float(self.tolerance))
        object.__setattr__(self, "force_constant", float(self.force_constant))

    @property
    def label(self) -> str:
        return f"{self.source}:{self.atom}"

    def violation(self, positions) -> float:
        """How far (A) the atom lies beyond the tolerance (0 within)."""
        d = float(np.linalg.norm(np.asarray(positions, float)[self.atom] - self.point))
        return max(0.0, d - self.tolerance)


def position_restraints(reference, atoms, conf_id: int = -1) -> List[PositionRestraint]:
    """The soft atoms held at their positions in conformer conf_id of reference."""
    positions = reference.GetConformer(conf_id).GetPositions()
    return [PositionRestraint(int(i), tuple(positions[i])) for i in atoms]


class RestraintSet:
    """
    Restraints with one per atom pair. Adding a restraint for a pair that has one
    raises if they differ, unless override=True (then the new one wins, e.g. user
    restraints over generated ones).
    """

    def __init__(self, restraints: Iterable[DistanceRestraint] = ()):
        self._by_pair: Dict[Tuple[int, int], DistanceRestraint] = {}
        for restraint in restraints:
            self.add(restraint)

    def add(self, restraint: DistanceRestraint, override: bool = False) -> None:
        if not isinstance(restraint, DistanceRestraint):
            raise TypeError(f"Not a DistanceRestraint: {restraint!r}")
        previous = self._by_pair.get(restraint.pair)
        if previous is not None and previous != restraint and not override:
            raise ValueError(
                f"Conflicting restraints for atom pair {restraint.pair}: "
                f"{previous} and {restraint}."
            )
        self._by_pair[restraint.pair] = restraint

    def merge(self, other: Iterable[DistanceRestraint]) -> "RestraintSet":
        """A new set with the restraints of both; other wins for shared pairs."""
        merged = RestraintSet(self)
        for restraint in other:
            merged.add(restraint, override=True)
        return merged

    def __iter__(self):
        return iter(self._by_pair.values())

    def __len__(self) -> int:
        return len(self._by_pair)

    def __bool__(self) -> bool:
        return bool(self._by_pair)

    def __eq__(self, other) -> bool:
        return isinstance(other, RestraintSet) and list(self) == list(other)

    def __repr__(self) -> str:
        return f"RestraintSet({list(self)!r})"

    def for_stage(self, stage: str) -> List[DistanceRestraint]:
        """The restraints that apply in "embed" or "refine"."""
        if stage not in ("embed", "refine"):
            raise ValueError("stage must be 'embed' or 'refine'.")
        return [r for r in self if r.stage in (stage, "both")]

    def by_source(self, *sources: str) -> "RestraintSet":
        return RestraintSet(r for r in self if r.source in sources)

    def without_pairs_within(self, atoms: Sequence[int]) -> "RestraintSet":
        """Without restraints between two of the atoms (e.g. both frozen)."""
        atoms = set(atoms)
        return RestraintSet(
            r for r in self if not (r.first in atoms and r.second in atoms)
        )

    def check_atoms(self, n_atoms: int) -> None:
        wrong = [r.pair for r in self if r.second >= n_atoms]
        if wrong:
            raise ValueError(
                f"Restraints {wrong} refer to atoms outside the {n_atoms}-atom molecule."
            )

    def violations(self, positions) -> np.ndarray:
        return np.array([r.violation(positions) for r in self])

    def sample(self, rng, fraction: float) -> "RestraintSet":
        """A random subset: each restraint is kept with probability fraction. rng: the
        random numbers (racerts.utils.seeds.Stream)."""
        if not 0 <= fraction <= 1:
            raise ValueError("fraction must be between 0 and 1.")
        return RestraintSet(r for r in self if rng.random() < fraction)

    def remap(self, index_map: Dict[int, int]) -> "RestraintSet":
        """The restraints whose atoms index_map maps, with the new indices."""
        remapped = []
        for r in self:
            if r.first in index_map and r.second in index_map:
                default = r.label == f"{r.source}:{r.first}-{r.second}"
                remapped.append(
                    replace(
                        r,
                        first=index_map[r.first],
                        second=index_map[r.second],
                        label="" if default else r.label,
                    )
                )
        return RestraintSet(remapped)

    def to_json(self, path: Optional[str] = None) -> str:
        text = json.dumps([asdict(r) for r in self], indent=1)
        if path is not None:
            with open(path, "w") as handle:
                handle.write(text + "\n")
        return text

    @classmethod
    def from_json(cls, text_or_path: str) -> "RestraintSet":
        text = text_or_path
        if not text_or_path.lstrip().startswith("["):
            with open(text_or_path) as handle:
                text = handle.read()
        return cls(DistanceRestraint(**item) for item in json.loads(text))
