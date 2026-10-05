"""Independent runs of a search: their union, and whether they agree."""

import itertools
import logging
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from racerts.geometry import rmsd_within, symmetry_maps

from .ensemble import ConformerEnsemble

logger = logging.getLogger(__name__)

KB = 0.0019872041  # kcal/(mol K)
# kcal/mol: copies of one minimum agree to this (as the RMSD pruner's energy prefilter):
# converged force-field energies, and all others (optimizations to a force threshold).
FORCE_FIELD_TOLERANCE = 0.1
TOLERANCE = 1.0
FORCE_FIELD_METHODS = ("MMFFOptimizer", "UFFOptimizer")
MAX_SUBSETS = 200  # of k runs, for the merged free energies


@dataclass
class ConvergenceReport:
    """
    How well independent runs agree. Energies in kcal/mol.

    Attributes:
        labels: The runs (e.g. their seeds).
        conformers: The number of conformers of each run.
        lowest: The lowest energy of each run, above the lowest of all.
        free_energy: The ensemble free energy -RT ln sum exp(-E/RT) of each run, above
            that of the union of all runs (duplicates counted once).
        spread: The largest difference of free_energy between two runs.
        found_again: (a, b) -> the share of the population of run a, within window of
            its own minimum, that run b has too (the same conformer: energy and RMSD).
        merged: k -> (mean, worst) free energy of k runs together, above that of all,
            over the choices of k runs.
        leave_one_out: The worst free energy of all runs but one: what the last run
            still changed.
        window, temperature, energy_tolerance, rmsd: The settings of the comparison.
    """

    labels: List
    conformers: List[int]
    lowest: List[float]
    free_energy: List[float]
    spread: float
    found_again: Dict[Tuple, float]
    merged: Dict[int, Tuple[float, float]]
    leave_one_out: float
    window: float
    temperature: float
    energy_tolerance: float
    rmsd: float
    energy_method: Optional[str] = None
    counts: Dict[Tuple, Tuple[int, int]] = field(default_factory=dict, repr=False)

    def converged(self, tolerance: float = 0.3) -> bool:
        """Whether the last run changed the free energy of the union by no more than
        tolerance (kcal/mol): leave_one_out <= tolerance."""
        return self.leave_one_out <= tolerance

    def to_dict(self) -> dict:
        """The report with JSON keys ("a -> b" for the pairs of found_again)."""
        data = asdict(self)
        for name in ("found_again", "counts"):
            data[name] = {f"{a} -> {b}": v for (a, b), v in getattr(self, name).items()}
        return data

    def __str__(self) -> str:
        method = f", energies {self.energy_method}" if self.energy_method else ""
        lines = [
            f"{len(self.labels)} runs{method} (kcal/mol, {self.temperature:.2f} K)",
            "run         conformers   lowest   free energy   (above the union of all)",
        ]
        for label, n, low, g in zip(
            self.labels, self.conformers, self.lowest, self.free_energy
        ):
            lines.append(f"{label!s:10s} {n:11d} {low:8.2f} {g:13.2f}")
        lines.append(f"spread of the free energy between runs: {self.spread:.2f}")
        lines.append(
            f"found again: the population of a run within {self.window:g} of its "
            "minimum that another run has too"
        )
        for (a, b), share in self.found_again.items():
            hits, total = self.counts.get((a, b), (0, 0))
            lines.append(f"  {a} -> {b}: {share:4.0%} ({hits} of {total} conformers)")
        shares = list(self.found_again.values())
        lines.append(f"  mean {np.mean(shares):.0%}, lowest {min(shares):.0%}")
        lines.append(
            "merged runs: the free energy above that of all runs (mean, worst)"
        )
        for k, (mean, worst) in self.merged.items():
            lines.append(f"  {k} run{'s' if k > 1 else ''}: {mean:.2f}, {worst:.2f}")
        lines.append(
            f"leaving one run out changes the free energy by up to "
            f"{self.leave_one_out:.2f}"
        )
        return "\n".join(lines)


@dataclass
class Runs:
    """
    The result of generate_runs: the ensemble of each seed, their union without
    duplicates (merge_runs) and the comparison of the runs (compare_runs).
    """

    seeds: List[int]
    ensembles: List[ConformerEnsemble]
    merged: ConformerEnsemble
    report: ConvergenceReport


def merge_runs(
    ensembles: Sequence[ConformerEnsemble],
    labels: Optional[Sequence] = None,
    energy_tolerance: Optional[float] = None,
    rmsd: float = 0.25,
) -> ConformerEnsemble:
    """
    The union of the runs with every conformer once, ordered by energy. Two conformers
    are the same if their energies differ by at most energy_tolerance (kcal/mol;
    default: 0.1 for MMFF and UFF energies, 1.0 for others) and their symmetry-aware
    heavy-atom RMSD after superposition is at most rmsd (A). The lowest copy is kept,
    with its run ("run") and all runs that found it ("found_by") in its provenance.
    """
    rows, matcher, labels = _rows(ensembles, labels, energy_tolerance, rmsd)
    kept = _distinct(rows, matcher)
    merged = ensembles[0].filter([])
    method = _energy_method(ensembles)
    if method:
        merged.mol.SetProp("energy_method", method)
    for row in kept:
        source = ensembles[row.run]
        conf_id = merged.mol.AddConformer(
            source.mol.GetConformer(row.conf_id), assignId=True
        )
        found_by = sorted({r.run for r in row.copies})
        merged.add_provenance(
            conf_id, run=labels[row.run], found_by=[labels[i] for i in found_by]
        )
    return merged


def compare_runs(
    ensembles: Sequence[ConformerEnsemble],
    labels: Optional[Sequence] = None,
    window: float = 2.0,
    temperature: float = 298.15,
    energy_tolerance: Optional[float] = None,
    rmsd: float = 0.25,
) -> ConvergenceReport:
    """
    How well independent runs of one search agree (see ConvergenceReport), e.g. the
    ensembles of several seeds. The energies must compare: one method, every conformer
    with an energy, no active bonds held at targets (compare after the free refinement).

    Args:
        ensembles: One ensemble per run, of the same molecule.
        labels: Names of the runs (default 0, 1, ...).
        window: found_again counts the conformers within this of a run's minimum.
        temperature: Of the populations and free energies (K).
        energy_tolerance, rmsd: When two conformers are the same (see merge_runs).
    """
    rows, matcher, labels = _rows(ensembles, labels, energy_tolerance, rmsd)
    rt = KB * temperature
    lowest = min(row.energy for row in rows)
    kept = _distinct(rows, matcher)
    n = len(ensembles)

    def free_energy(runs) -> float:
        # the union of these runs: each distinct conformer at its lowest copy there
        energies = [
            min(c.energy for c in row.copies if c.run in runs)
            for row in kept
            if any(c.run in runs for c in row.copies)
        ]
        energies = np.asarray(energies) - lowest
        return float(-rt * np.log(np.exp(-energies / rt).sum()))

    union = free_energy(set(range(n)))
    per_run = [free_energy({i}) - union for i in range(n)]
    found_again, counts = {}, {}
    for a, b in itertools.permutations(range(n), 2):
        mine = [row for row in rows if row.run == a]
        low = min(row.energy for row in mine)
        mine = [row for row in mine if row.energy - low <= window]
        weights = np.exp(-(np.array([row.energy for row in mine]) - low) / rt)
        hits = np.array([any(c.run == b for c in row.keeper.copies) for row in mine])
        found_again[(labels[a], labels[b])] = float(weights[hits].sum() / weights.sum())
        counts[(labels[a], labels[b])] = (int(hits.sum()), len(mine))
    merged = {}
    for k in range(1, n + 1):
        subsets = list(
            itertools.islice(itertools.combinations(range(n), k), MAX_SUBSETS)
        )
        values = [free_energy(set(subset)) - union for subset in subsets]
        merged[k] = (float(np.mean(values)), float(max(values)))
    return ConvergenceReport(
        labels=list(labels),
        conformers=[len(e) for e in ensembles],
        lowest=[min(r.energy for r in rows if r.run == i) - lowest for i in range(n)],
        free_energy=per_run,
        spread=max(per_run) - min(per_run),
        found_again=found_again,
        merged=merged,
        leave_one_out=merged[n - 1][1],
        window=window,
        temperature=temperature,
        energy_tolerance=matcher.energy_tolerance,
        rmsd=rmsd,
        energy_method=_energy_method(ensembles),
        counts=counts,
    )


@dataclass(eq=False)
class _Row:
    """A conformer of a run; keeper: the lowest copy of it in all runs, which lists
    the copies."""

    run: int
    conf_id: int
    energy: float
    positions: np.ndarray
    keeper: Optional["_Row"] = None
    copies: List["_Row"] = field(default_factory=list)


class _Matcher:
    def __init__(self, mol, energy_tolerance: float, rmsd: float):
        self.energy_tolerance, self.rmsd = energy_tolerance, rmsd
        self.symmetry = symmetry_maps(mol)

    def same(self, a: _Row, b: _Row) -> bool:
        return abs(a.energy - b.energy) <= self.energy_tolerance and rmsd_within(
            a.positions, b.positions, self.rmsd, self.symmetry.atoms, self.symmetry.maps
        )


def _energy_method(ensembles) -> Optional[str]:
    methods = {e.energy_method for e in ensembles} - {None}
    if len(methods) > 1:
        raise ValueError(
            f"The energies come from different methods ({', '.join(sorted(methods))}) "
            "and cannot be compared."
        )
    return methods.pop() if methods else None


def _rows(ensembles, labels, energy_tolerance, rmsd):
    ensembles = list(ensembles)
    if len(ensembles) < 2:
        raise ValueError("Comparing or merging needs at least two runs.")
    labels = list(range(len(ensembles))) if labels is None else list(labels)
    if len(labels) != len(ensembles) or len(set(labels)) != len(labels):
        raise ValueError("labels: one name per run, each different.")
    method = _energy_method(ensembles)
    if energy_tolerance is None:
        force_field = method in FORCE_FIELD_METHODS
        energy_tolerance = FORCE_FIELD_TOLERANCE if force_field else TOLERANCE
    rows = []
    for run, ensemble in enumerate(ensembles):
        if not len(ensemble):
            raise ValueError(f"Run {labels[run]} has no conformers.")
        for conf_id, energy in zip(ensemble.conf_ids, ensemble.energies()):
            if not np.isfinite(energy):
                raise ValueError(
                    f"Conformer {conf_id} of run {labels[run]} has no energy."
                )
            if ensemble.provenance(conf_id).get("active_bond_targets"):
                raise ValueError(
                    "The active bonds are held at targets, where energies do not "
                    "compare: compare the runs after the free refinement."
                )
            positions = ensemble.mol.GetConformer(conf_id).GetPositions()
            rows.append(_Row(run, conf_id, float(energy), positions))
    matcher = _Matcher(ensembles[0].mol, energy_tolerance, rmsd)
    return rows, matcher, labels


def _distinct(rows: List[_Row], matcher: _Matcher) -> List[_Row]:
    """The distinct conformers (the lowest copy of each), by energy; every row gets
    its keeper, every keeper its copies."""
    kept: List[_Row] = []
    for row in sorted(rows, key=lambda r: r.energy):
        row.copies, row.keeper = [], row
        # keepers within the tolerance below this energy (kept is sorted by energy)
        for other in reversed(kept):
            if row.energy - other.energy > matcher.energy_tolerance:
                break
            if matcher.same(row, other):
                row.keeper = other
                break
        if row.keeper is row:
            kept.append(row)
        row.keeper.copies.append(row)
    return kept
