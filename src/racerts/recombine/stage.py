"""Recombine: the torsions of the low conformers in combinations that none of them
has."""

import logging
from typing import Optional, Union

import numpy as np

from racerts.exploit.population import (
    Population,
    add_refined,
    discovery,
    duplicate_atoms,
    found_lower,
    held_atoms,
    lowest_by_bin,
    reference_positions,
    refine_candidates,
    refine_stage,
    require_refined,
)
from racerts.pipeline import ConformerEnsemble
from racerts.refine.forcefield import MMFFOptimizer, UFFOptimizer
from racerts.refine.stage import energy_label
from racerts.utils import seeds
from racerts.utils.checks import check_count, check_positive
from racerts.validate.checks import (
    CLASH_FACTOR,
    OVERLAP_FACTOR,
    clash_limits,
    first_clash,
    gate,
)

from .torsions import set_torsions, torsion_angles, torsions

logger = logging.getLogger(__name__)

PER_TORSION = 50  # children of a round for every torsion, by default
MOST_CHILDREN = 2400  # ... and no more than these
DRAWS = 30  # draws for every child of a round, before the round has fewer


class Recombine:
    """
    New conformers from the torsions of the low conformers it gets, which it returns
    with the minima that the new combinations lead to. A large flexible molecule can
    have every torsion of a lower conformer among its low conformers and never all of
    them in one: an embedding draws the torsions almost independently, and neither
    local moves (Exploit) nor dynamics from the low conformers (Metadynamics) bring
    together what different conformers have. This is the crossing of CREST's genetic
    step, in torsions.

    Parents: the conformers within window of the lowest, the lowest parents of them.
    A child is a first parent, drawn with Boltzmann weights at temperature, in which
    every torsion that differs by more than different degrees from that of a second
    parent, drawn evenly, takes the second one's value with the probability share (at
    least one does). The torsions are those of racerts.recombine.torsions: the planar
    ones of amides and esters too, since only values that a parent has are passed on.
    One side of the bond turns rigidly; the atoms that the task holds (the hard, core
    and soft atoms of its FrozenSet) never move.

    Children with overlapping atoms (closer than overlap_factor times the sum of
    their vdW radii, hydrogens included) are dropped. The others are refined in one
    call, checked by the gate (frozen atoms, bonds and stereo, clashes, restraints)
    and compared with the conformers as in Exploit: a duplicate is a re-find, any
    other joins them. There is no energy window: a pruner after the stage sets it.

    Rounds: with rounds > 1, a round that found a conformer more than 0.1 kcal/mol
    below the lowest before it is followed by another, whose parents include the new
    conformers, until a round finds nothing lower or rounds are done.

    Provenance: route "cross", parents (the first and the second), cross_torsions
    (how many were turned) and cross_round for the new conformers, which keep the
    reference, the active-bond targets and the batch of their first parent;
    cross_hits (the children that ended in it) for every member. stats holds the
    numbers of the last call: rounds, parents and torsions (of the last round),
    children, overlapping, optimizations (the refined children), observations,
    accepted, refound, lowered, merged, rejected, missing_mass and completeness (as for Exploit, of the
    minima within 3 kcal/mol) and stop ("rounds", "nothing lower", or why nothing was
    made).

    Args:
        refine: A Refine stage, or an optimizer for one (Refine(optimizer,
            fallback=False)): any optimizer, a force field too. The ensemble's
            energies must come from it (energy_method).
        window: The parents lie within this of the lowest conformer (kcal/mol; for a
            TS with active-bond windows: of the lowest of their window bin).
        parents: The most parents.
        temperature: Of the weights exp(-E / temperature) with which the first
            parent is drawn, as an energy (kcal/mol).
        different: Torsions of two parents that differ by more than this are passed
            on (degrees).
        share: The probability that a child takes such a torsion from its second
            parent.
        children: Of a round. Default: PER_TORSION for every torsion, MOST_CHILDREN
            at most. A round makes no child twice (on one first parent, the same
            torsions taken at values within `different`, whichever the second parent)
            and has fewer if the parents allow no more.
        rounds: The most rounds (see above); default: one.
        overlap_factor: See above; the default is that of the clash filter of the
            staged workflow, below the clash_factor of the gate: the refinement
            relaxes close contacts.
        rmsd_threshold, include_hs, clash_factor: As in Exploit.
    """

    name = "recombine"
    route = "cross"  # of the children's provenance, and the prefix of its keys
    fewest_parents = 2

    def __init__(
        self,
        refine,
        window: float = 6.0,
        parents: int = 120,
        temperature: float = 2.0,
        different: float = 40.0,
        share: float = 0.5,
        children: Optional[int] = None,
        rounds: int = 1,
        overlap_factor: float = OVERLAP_FACTOR,
        rmsd_threshold: float = 0.25,
        include_hs: Union[bool, str] = "polar",
        clash_factor: float = CLASH_FACTOR,
    ):
        positive = dict(window=window, temperature=temperature)
        positive.update(overlap_factor=overlap_factor, rmsd_threshold=rmsd_threshold)
        check_positive(**positive)
        parents = check_count("parents", parents, 2)
        rounds = check_count("rounds", rounds)
        if children is not None:
            children = check_count("children", children)
        if not 0 < different < 180:
            raise ValueError("different must lie between 0 and 180 degrees.")
        if not 0 < share <= 1:
            raise ValueError("share must be a probability above 0.")
        self.refine = refine_stage(refine)
        self.window = window
        self.parents = parents
        self.temperature = temperature
        self.different = different
        self.share = share
        self.children = children
        self.rounds = rounds
        self.overlap_factor = overlap_factor
        self.rmsd_threshold = rmsd_threshold
        duplicate_atoms(include_hs)  # checks it
        self.include_hs = include_hs
        self.clash_factor = clash_factor
        self.stats: dict = {}

    def run(self, ctx, ensemble: ConformerEnsemble) -> ConformerEnsemble:
        counts = dict(rounds=0, parents=0, torsions=0, children=0, overlapping=0)
        counts.update(optimizations=0, observations=0, accepted=0, refound=0)
        counts.update(rejected=0, lowered=0, merged=0)
        if not len(ensemble):
            self.stats = {**counts, "stop": "no conformers"}
            return ensemble
        stage = type(self).__name__
        require_refined(ensemble, self._labels(), stage)
        found = self._torsions(ensemble.mol, held_atoms(ctx))
        counts["torsions"] = len(found)
        if not found:
            logger.warning(
                "%s: nothing to %s (no torsion outside the atoms that the task holds); "
                "the ensemble passes unchanged.",
                stage,
                self.name,
            )
            self.stats = {**counts, "stop": "no torsions"}
            return ensemble

        population = Population(ctx, ensemble, self.include_hs)
        checks = gate(self.clash_factor).validators
        n = self.children or min(MOST_CHILDREN, PER_TORSION * len(found))
        pairs = {}  # reference id -> the pairs of atoms that may overlap
        stop = "rounds"
        for number in range(self.rounds):
            lowest = lowest_by_bin(population)
            parents = self._parents(population, lowest)
            counts["parents"] = len(parents)
            if len(parents) < self.fewest_parents:
                stop = "no parents"
                break
            rng = seeds.Stream(ctx.seed, self.name, number)
            made, overlapping = self._children(
                ctx, population, parents, lowest, found, n, rng, pairs
            )
            counts["overlapping"] += overlapping
            for _, _, record in made:
                record[f"{self.route}_round"] = number
            if not made:
                stop = "no children"
                break
            refined = refine_candidates(ctx, ensemble, made, self.refine, stage)
            counts["rounds"] += 1
            counts["children"] += len(made)
            counts["optimizations"] += len(refined)
            add_refined(
                ctx, population, refined, made, checks, self.rmsd_threshold, counts
            )
            if not found_lower(population, lowest):
                stop = "nothing lower"
                break

        for conf_id in population.ids:
            hits = {f"{self.route}_hits": population.hits[conf_id]}
            ensemble.add_provenance(conf_id, **hits)
        missing, completeness = discovery(population, counts)
        self.stats = {**counts, "stop": stop, "missing_mass": round(missing, 4)}
        self.stats.update(completeness=round(completeness, 3))
        logger.info(
            "%s: %d rounds, %d torsions, %d parents in the last round; %d "
            "children (%d more with overlapping atoms), %d refined, %d new "
            "conformers, %d found again, %d rejected; stopped (%s).",
            stage,
            counts["rounds"],
            counts["torsions"],
            counts["parents"],
            counts["children"],
            counts["overlapping"],
            counts["optimizations"],
            counts["accepted"],
            counts["refound"],
            counts["rejected"],
            stop,
        )
        return ensemble

    def _torsions(self, mol, held) -> list:
        return torsions(mol, held)

    def _labels(self) -> list:
        """The names that the energies of the refiner can have: its own, and that
        of UFF where Refine falls back to it."""
        optimizer = self.refine.optimizer
        if optimizer is None:
            optimizer = MMFFOptimizer()
        labels = [energy_label(optimizer)]
        if self.refine.fallback and isinstance(optimizer, MMFFOptimizer):
            labels.append(energy_label(UFFOptimizer()))
        return labels

    def _parents(self, population, lowest: dict) -> list:
        """The members within window of the lowest of their window bin: the lowest
        parents of them, lowest first."""

        def above(conf_id):
            return population.energies[conf_id] - lowest[population.bins[conf_id]]

        within = [i for i in population.ids if above(i) <= self.window]
        return sorted(within, key=lambda i: (above(i), i))[: self.parents]

    def _children(
        self, ctx, population, parents, lowest, found, n, rng, pairs
    ) -> tuple:
        """
        Up to n children of the parents, each another combination, as candidates
        ((first parent, positions, record) each), and the number of children that
        were dropped for overlapping atoms. lowest: the lowest energy of every
        window bin; found: the torsions.
        """
        positions = np.array([population.positions(i) for i in parents])
        angles = torsion_angles(positions, found)
        above = np.array(
            [population.energies[i] - lowest[population.bins[i]] for i in parents]
        )
        weights = np.cumsum(np.exp(-above / self.temperature))
        made, seen, overlapping = [], {}, 0
        for _ in range(DRAWS * n):
            if len(made) == n:
                break
            first = int(np.searchsorted(weights, rng.random() * weights[-1], "right"))
            second = rng.integers(len(parents))
            apart = np.abs((angles[first] - angles[second] + 180.0) % 360.0 - 180.0)
            differing = np.nonzero(apart > self.different)[0].tolist()
            if first == second or not differing:
                continue
            take = [k for k in differing if rng.random() < self.share]
            take = take or [rng.choice(differing)]
            # the same child: the same first parent, the same torsions taken, values
            # within `different` of those of a child made before (any second parent)
            values = angles[second, take]
            earlier = seen.setdefault((first, tuple(take)), [])
            if any(
                (np.abs((values - v + 180.0) % 360.0 - 180.0) <= self.different).all()
                for v in earlier
            ):
                continue
            earlier.append(values)
            child = set_torsions(
                positions[first], found, {k: angles[second, k] for k in take}
            )
            if self._overlaps(ctx, population, parents[first], child, pairs):
                overlapping += 1
                continue
            record = dict(route=self.route, parents=[parents[first], parents[second]])
            record["cross_torsions"] = len(take)
            made.append((parents[first], child, record))
        return made, overlapping

    def _overlaps(self, ctx, population, parent, child, pairs) -> bool:
        """Whether atoms of the child of parent overlap (overlap_factor); pairs: the
        pairs of atoms that may, by reference id, filled as needed."""
        ref_id = population.provenance(parent).get("reference")
        if ref_id not in pairs:
            pairs[ref_id] = clash_limits(
                population.ensemble.mol,
                held_atoms(ctx),
                reference_positions(ctx, ref_id),
                self.overlap_factor,
                hydrogen_factor=self.overlap_factor,
            )
        return first_clash(child, pairs[ref_id]) is not None

    def __repr__(self) -> str:
        return (
            f"Recombine({self.refine.optimizer!r}, window={self.window}, parents="
            f"{self.parents}, children={self.children}, rounds={self.rounds})"
        )
