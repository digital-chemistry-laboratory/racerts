"""Distance restraints: flat-bottom windows between atom pairs."""

import json
import math
from dataclasses import asdict, dataclass, replace
from numbers import Integral, Real
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

DEFAULT_HALF_WIDTH = 0.25  # A
DEFAULT_FORCE_CONSTANT = 20.0  # kcal/(mol A^2)
STAGES = ("embed", "refine", "both")


def _is_index(value) -> bool:
    return isinstance(value, Integral) and not isinstance(value, bool)


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
        if not (_is_index(self.first) and _is_index(self.second)):
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

    def sample(self, rng: np.random.Generator, fraction: float) -> "RestraintSet":
        """A random subset: each restraint is kept with probability fraction."""
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
