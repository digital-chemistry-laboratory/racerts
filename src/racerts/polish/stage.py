"""Polish: the lowest families of an ensemble completed one change at a time."""

import logging
from typing import List, Optional, Sequence, Union

import numpy as np
from rdkit import Chem

from racerts.exploit.moves import ring_moves, rotate, torsion_moves
from racerts.exploit.population import (
    Population,
    add_refined,
    discovery,
    duplicate_atoms,
    failed_checks,
    held_atoms,
    lowest_by_bin,
    reference_positions,
    refine_candidates,
    refine_stage,
    require_refined,
)
from racerts.pipeline import ConformerEnsemble
from racerts.prune.targets import window_bins
from racerts.recombine.torsions import Torsion, torsion_angles
from racerts.refine.forcefield import MMFFOptimizer, UFFOptimizer
from racerts.refine.stage import energy_label
from racerts.system.rings import closest_form, mean_plane, pucker_forms
from racerts.utils.checks import check_count, check_positive
from racerts.validate.checks import (
    CLASH_FACTOR,
    OVERLAP_FACTOR,
    clash_limits,
    first_clash,
    gate,
)

logger = logging.getLogger(__name__)

KINDS = ("pucker", "torsion")
MODES = ("descent", "lowest")


class Polish:
    """
    Completes the lowest families of the conformers it gets, one change at a time, and
    returns them with the minima it finds. A family is a combination of torsion states
    (each rotatable bond of Exploit's torsion moves: gauche+, anti or gauche- between
    two sp3 atoms, otherwise one of two states); conformers of one family still differ
    in their ring forms and in the exact torsions, and a stochastic search can stall one
    change away from a lower minimum.

    A pass refines every child of the family's current best that differs from it in
    one change: a torsion turned to each of its other states (by 120 or 240 degrees
    between sp3 atoms, otherwise by 180), or a ring in each of its other canonical forms
    (every non-aromatic five- or six-membered ring of Exploit's ring moves, one at a
    time: chairs and twist-boats, envelopes and twists). If the lowest child lies more
    than tolerance below the current best, the next pass starts from it; otherwise the
    family is complete. The families are the lowest at the start, completed one after
    the other; a family that reaches a conformer that a pass has started from already
    is complete too (mode "descent"). With mode "lowest" the families are chosen again
    after every round: each of the lowest whose best member no pass has started from
    gets a pass, until there is none (tolerance plays no part); a lower child thus
    starts a family of its own. No random numbers: the result does not depend on the
    seed.

    The atoms that the task holds never move (the bonds and rings of the moves are those
    without them). Children with overlapping atoms are dropped; the others are refined,
    checked by the gate and compared with the conformers as in Exploit and Recombine: a
    duplicate is a re-find, any other joins them.

    Provenance: route "polish", parents (the conformer changed), polish_change (e.g.
    "torsion 1-2 +120" or "ring 0-1-2-3-4-5 chair"), polish_family and polish_pass (both
    counted from 0) for the new conformers; polish_hits for every member. stats holds
    the numbers of the last call: families, passes, children, overlapping,
    optimizations, observations, accepted, refound, rejected, missing_mass and
    completeness (as for Exploit), and stop ("converged", "budget", or why nothing was
    changed).

    Args:
        refine: A Refine stage, or an optimizer for one (as for Recombine).
        kinds: The changes: "torsion", "pucker" or both.
        families: How many of the lowest families to complete.
        tolerance: A child lower by more than this (kcal/mol) is the next start
            (mode "descent").
        mode: "descent" or "lowest" (see above).
        max_optimizations: The most refinements, then it stops ("budget"). None: until
            every family is complete.
        overlap_factor, rmsd_threshold, include_hs, clash_factor: As in Recombine.
    """

    name = "polish"

    def __init__(
        self,
        refine,
        kinds: Sequence[str] = KINDS,
        families: int = 3,
        tolerance: float = 0.1,
        mode: str = "descent",
        max_optimizations: Optional[int] = None,
        overlap_factor: float = OVERLAP_FACTOR,
        rmsd_threshold: float = 0.25,
        include_hs: Union[bool, str] = False,
        clash_factor: float = CLASH_FACTOR,
    ):
        kinds = tuple(kinds)
        if not kinds or set(kinds) - set(KINDS):
            raise ValueError(f"kinds must be some of {KINDS}, not {kinds!r}.")
        if tolerance < 0:
            raise ValueError("tolerance must not be negative.")
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, not {mode!r}.")
        if max_optimizations is not None:
            max_optimizations = check_count("max_optimizations", max_optimizations)
        check_positive(overlap_factor=overlap_factor, rmsd_threshold=rmsd_threshold)
        self.refine = refine_stage(refine)
        self.kinds = kinds
        self.families = check_count("families", families)
        self.tolerance = tolerance
        self.mode = mode
        self.max_optimizations = max_optimizations
        self.overlap_factor = overlap_factor
        self.rmsd_threshold = rmsd_threshold
        duplicate_atoms(include_hs)  # checks it
        self.include_hs = include_hs
        self.clash_factor = clash_factor
        self.stats: dict = {}

    def run(self, ctx, ensemble: ConformerEnsemble) -> ConformerEnsemble:
        counts = dict(families=0, passes=0, children=0, overlapping=0)
        counts.update(optimizations=0, observations=0, accepted=0, refound=0)
        counts.update(rejected=0)
        if not len(ensemble):
            self.stats = {**counts, "stop": "no conformers"}
            return ensemble
        require_refined(ensemble, self._labels(), "Polish")
        held = held_atoms(ctx)
        mol = ensemble.mol
        turns = torsion_moves(mol, held) if "torsion" in self.kinds else []
        rings = ring_moves(mol, held) if "pucker" in self.kinds else []
        if not turns and not rings:
            logger.warning(
                "Polish: nothing to change (no torsion or ring outside the atoms that "
                "the task holds); the ensemble passes unchanged."
            )
            self.stats = {**counts, "stop": "nothing to change"}
            return ensemble

        population = Population(ctx, ensemble, self.include_hs)
        checks = gate(self.clash_factor).validators
        states = _states(mol, torsion_moves(mol, held))
        pairs = {}  # reference id -> the pairs of atoms that may overlap
        started = set()  # the conformers that a pass has started from

        def one_pass(current, family, number):
            """Every child of current with one change, refined: the member that the
            lowest child is (or found again), and whether the budget ended it."""
            started.add(current)
            made, overlapping = self._children(
                ctx, population, current, turns, rings, pairs
            )
            counts["overlapping"] += overlapping
            budget = False
            if self.max_optimizations is not None:
                left = self.max_optimizations - counts["optimizations"]
                if len(made) >= left:
                    made, budget = made[:left], True
            for _, _, record in made:
                record.update(polish_family=family, polish_pass=number)
            if not made:
                return None, budget
            refined = refine_candidates(ctx, ensemble, made, self.refine, "Polish")
            counts["passes"] += 1
            counts["children"] += len(made)
            counts["optimizations"] += len(refined)
            add_refined(
                ctx, population, refined, made, checks, self.rmsd_threshold, counts
            )
            return self._lowest(ctx, population, refined, checks), budget

        stop = "converged"
        if self.mode == "descent":
            for family, (_, current) in enumerate(self._starts(population, states)):
                counts["families"] += 1
                number = 0
                while current not in started:
                    lowest, budget = one_pass(current, family, number)
                    number += 1
                    if budget:
                        stop = "budget"
                        break
                    if lowest is None or (
                        population.energies[lowest]
                        >= population.energies[current] - self.tolerance
                    ):
                        break
                    current = lowest
                if stop == "budget":
                    break
        else:  # the lowest families after every round, each best member once
            families, number = {}, 0
            while stop != "budget":
                fresh = [
                    (signature, i)
                    for signature, i in self._starts(population, states)
                    if i not in started
                ]
                if not fresh:
                    break
                for signature, current in fresh:
                    family = families.setdefault(signature, len(families))
                    if one_pass(current, family, number)[1]:
                        stop = "budget"
                        break
                number += 1
            counts["families"] = len(families)

        for conf_id in population.ids:
            ensemble.add_provenance(conf_id, polish_hits=population.hits[conf_id])
        missing, completeness = discovery(population, counts)
        self.stats = {**counts, "stop": stop, "missing_mass": round(missing, 4)}
        self.stats.update(completeness=round(completeness, 3))
        logger.info(
            "Polish: %d families in %d passes; %d children (%d more with overlapping "
            "atoms), %d refined, %d new conformers, %d found again, %d rejected; "
            "stopped (%s).",
            counts["families"],
            counts["passes"],
            counts["children"],
            counts["overlapping"],
            counts["optimizations"],
            counts["accepted"],
            counts["refound"],
            counts["rejected"],
            stop,
        )
        return ensemble

    def _labels(self) -> list:
        """The names that the energies of the refiner can have (as for Recombine)."""
        optimizer = self.refine.optimizer
        if optimizer is None:
            optimizer = MMFFOptimizer()
        labels = [energy_label(optimizer)]
        if self.refine.fallback and isinstance(optimizer, MMFFOptimizer):
            labels.append(energy_label(UFFOptimizer()))
        return labels

    def _starts(self, population, states) -> List[tuple]:
        """(signature, member): the lowest member of each of the lowest families (by
        the energy above the lowest of its window bin), lowest first."""
        lowest = lowest_by_bin(population)
        above = {
            i: population.energies[i] - lowest[population.bins[i]]
            for i in population.ids
        }
        angles = torsion_angles(
            [population.positions(i) for i in population.ids], states.torsions
        )
        signatures = {
            i: states.of(row)
            for i, row in zip(population.ids, angles.reshape(len(population.ids), -1))
        }
        starts = {}
        for i in sorted(population.ids, key=lambda i: (above[i], i)):
            starts.setdefault(signatures[i], i)
        return list(starts.items())[: self.families]

    def _children(self, ctx, population, parent, turns, rings, pairs) -> tuple:
        """
        Every child of parent with one change, as candidates ((parent, positions,
        record) each), and the number dropped for overlapping atoms.
        """
        positions = population.positions(parent)
        made, overlapping = [], 0
        changes = []
        for move in turns:
            a, b = move.bond
            sp3 = all(
                population.ensemble.mol.GetAtomWithIdx(i).GetHybridization()
                == Chem.HybridizationType.SP3
                for i in move.bond
            )
            for angle in (120.0, -120.0) if sp3 else (180.0,):
                child = positions.copy()
                rotate(
                    child,
                    list(move.moving),
                    child[b].copy(),
                    child[b] - child[a],
                    angle,
                )
                changes.append((child, f"torsion {min(a, b)}-{max(a, b)} {angle:+.0f}"))
        for move in rings:
            ring = list(move.ring)
            center, normal = mean_plane(positions[ring])
            forms = pucker_forms(len(ring))
            current = closest_form((positions[ring] - center) @ normal, forms)
            for k, (form, _) in enumerate(forms):
                if k == current:
                    continue
                child = positions.copy()
                move.apply(child, None, form=form)
                changes.append((child, f"ring {'-'.join(map(str, ring))} {form}"))
        ref_id = population.provenance(parent).get("reference")
        if ref_id not in pairs:
            pairs[ref_id] = clash_limits(
                population.ensemble.mol,
                held_atoms(ctx),
                reference_positions(ctx, ref_id),
                self.overlap_factor,
                hydrogen_factor=self.overlap_factor,
            )
        for child, change in changes:
            if first_clash(child, pairs[ref_id]) is not None:
                overlapping += 1
                continue
            record = dict(route="polish", parents=[parent], polish_change=change)
            made.append((parent, child, record))
        return made, overlapping

    def _lowest(self, ctx, population, refined, checks) -> Optional[int]:
        """The member that the lowest refined child that passed the checks is (or
        found again), if any."""
        failed = failed_checks(ctx, refined, checks) if len(refined) else {}
        bins = window_bins(ctx, refined)
        valid = [k for k in refined.conf_ids if k in bins and k not in failed]
        if not valid:
            return None
        k = min(valid, key=refined.energy)
        positions = refined.mol.GetConformer(k).GetPositions()
        return population.duplicate(
            positions, refined.energy(k), bins[k], self.rmsd_threshold
        )

    def __repr__(self) -> str:
        return (
            f"Polish({self.refine.optimizer!r}, kinds={self.kinds}, families="
            f"{self.families}, tolerance={self.tolerance}, mode={self.mode!r}, "
            f"max_optimizations="
            f"{self.max_optimizations})"
        )


class _States:
    """The torsion states of the rotatable bonds: three between sp3 atoms, else two."""

    def __init__(self, torsions: List[Torsion], folds: List[int]):
        self.torsions = torsions
        self.folds = np.array(folds)

    def of(self, angles) -> tuple:
        """The state of each torsion at angles (degrees): for three states 0 (0 to
        120), 1 (anti) or 2; for two, 0 (|angle| < 90) or 1."""
        angles = np.asarray(angles, dtype=float)
        three = np.floor((angles % 360.0) / 120.0)
        two = np.floor(((angles + 90.0) % 360.0) / 180.0)
        return tuple(np.where(self.folds == 3, three, two).astype(int).tolist())


def _states(mol: Chem.Mol, moves) -> _States:
    """The states of the bonds of the torsion moves, measured with a heavy neighbour at
    each end where there is one."""
    sp3 = Chem.HybridizationType.SP3
    torsions, folds = [], []
    for move in moves:
        a, b = move.bond

        def neighbour(atom, other):
            around = [
                n
                for n in mol.GetAtomWithIdx(atom).GetNeighbors()
                if n.GetIdx() != other
            ]
            return min(
                around, key=lambda n: (n.GetAtomicNum() == 1, n.GetIdx())
            ).GetIdx()

        torsions.append(Torsion((neighbour(a, b), a, b, neighbour(b, a)), move.moving))
        both = all(mol.GetAtomWithIdx(i).GetHybridization() == sp3 for i in (a, b))
        folds.append(3 if both else 2)
    return _States(torsions, folds)
