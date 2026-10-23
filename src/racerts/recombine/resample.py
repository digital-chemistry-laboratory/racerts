"""Resample: each torsion of a new conformer drawn from the conformers of the
ensemble, one by one."""

from typing import Optional, Union

import numpy as np

from racerts.exploit.moves import planar_bonds
from racerts.exploit.population import duplicate_atoms, refine_stage
from racerts.utils.checks import check_count, check_positive
from racerts.validate.checks import CLASH_FACTOR, OVERLAP_FACTOR

from .stage import DRAWS, Recombine
from .torsions import set_torsions, torsion_angles, torsions


class Resample(Recombine):
    """
    New conformers from the distribution of every torsion over the conformers it gets,
    which it returns with the minima that they lead to. A child is a member, drawn
    evenly, in which each torsion takes the value of a member drawn for it alone:
    evenly (each member counts once, so a torsion state is drawn as often as the
    minima found have it), or with the weights exp(-E / temperature). The torsions are
    thus drawn independently from their marginal distributions, and the child keeps the
    rest of its parent (rings, the reference of a held core, its active-bond targets).
    Recombine passes on what two parents have; Resample what the whole ensemble has.

    explore: the probability that a torsion takes a random value instead, so that a
    state that no member has can appear; never the planar bonds of amides and esters,
    whose other value is an isomer.

    polar_hydrogens: the rotors of OH, NH2 and SH groups are torsions too (by default;
    their hydrogens decide hydrogen bonds, and a value is still only one that a member
    has, or a random one with explore).

    Everything else is as in Recombine: the torsions of racerts.recombine.torsions, the
    atoms that the task holds never move, children with overlapping atoms are dropped,
    the others refined, checked by the gate and compared with the conformers (a
    duplicate is a re-find); with rounds > 1 a round that found a conformer more than
    0.1 kcal/mol below the lowest is followed by another, which draws from the members
    it found too. A child differs from its parent in one torsion at least, by more than
    different degrees, and no child is made twice (the same parent, every torsion
    within different of an earlier child's).

    Provenance: route "resample", parents (the one parent) and resample_round for the
    new conformers; resample_hits for every member. stats as for Recombine, with
    parents: the members drawn from (of the last round).

    Args:
        refine: A Refine stage, or an optimizer for one, whose energies the ensemble
            has (as for Recombine).
        window: The members drawn from lie within this of the lowest (kcal/mol; of
            their window bin for a TS with active-bond windows). None: all.
        parents: At most this many of them, the lowest. None: all.
        temperature: Of the weights of the members whose values a torsion takes, as an
            energy (kcal/mol). None: even.
        explore, polar_hydrogens: See above.
        different, children, rounds, overlap_factor, rmsd_threshold, include_hs,
        clash_factor: As in Recombine.
    """

    name = "resample"
    route = "resample"
    fewest_parents = 1  # with explore, one member is enough

    def __init__(
        self,
        refine,
        window: Optional[float] = None,
        parents: Optional[int] = None,
        temperature: Optional[float] = None,
        explore: float = 0.0,
        polar_hydrogens: bool = True,
        different: float = 40.0,
        children: Optional[int] = None,
        rounds: int = 1,
        overlap_factor: float = OVERLAP_FACTOR,
        rmsd_threshold: float = 0.25,
        include_hs: Union[bool, str] = False,
        clash_factor: float = CLASH_FACTOR,
    ):
        # Recombine's settings of its pairs (share) are not Resample's: the same
        # attributes, set here.
        positive = dict(overlap_factor=overlap_factor, rmsd_threshold=rmsd_threshold)
        if window is not None:
            positive["window"] = window
        if temperature is not None:
            positive["temperature"] = temperature
        check_positive(**positive)
        if parents is not None:
            parents = check_count("parents", parents)
        rounds = check_count("rounds", rounds)
        if children is not None:
            children = check_count("children", children)
        if not 0 < different < 180:
            raise ValueError("different must lie between 0 and 180 degrees.")
        if not 0 <= explore <= 1:
            raise ValueError("explore must be a probability.")
        self.refine = refine_stage(refine)
        self.window = np.inf if window is None else window  # Recombine's _parents
        self.parents = parents  # None: [:None], all
        self.temperature = temperature
        self.explore = explore
        self.polar_hydrogens = polar_hydrogens
        self.different = different
        self.children = children
        self.rounds = rounds
        self.overlap_factor = overlap_factor
        self.rmsd_threshold = rmsd_threshold
        duplicate_atoms(include_hs)  # checks it
        self.include_hs = include_hs
        self.clash_factor = clash_factor
        self.stats: dict = {}

    def _torsions(self, mol, held) -> list:
        return torsions(mol, held, polar_hydrogens=self.polar_hydrogens)

    def _children(
        self, ctx, population, parents, lowest, found, n, rng, pairs
    ) -> tuple:
        """
        Up to n children of the members parents, as candidates ((parent, positions,
        record) each), and the number dropped for overlapping atoms. lowest: the lowest
        energy of every window bin; found: the torsions.
        """
        positions = np.array([population.positions(i) for i in parents])
        angles = torsion_angles(positions, found)
        if self.temperature is None:
            weights = np.ones(len(parents))
        else:
            above = [
                population.energies[i] - lowest[population.bins[i]] for i in parents
            ]
            weights = np.exp(-np.array(above) / self.temperature)
        weights = np.cumsum(weights)
        planar = planar_bonds(population.ensemble.mol)
        free = np.array([frozenset(t.atoms[1:3]) not in planar for t in found])
        columns = np.arange(len(found))
        made, seen, overlapping = [], {}, 0
        for _ in range(DRAWS * n):
            if len(made) == n:
                break
            parent = int(rng.integers(len(parents)))
            draws = np.array([rng.random() for _ in found]) * weights[-1]
            values = angles[np.searchsorted(weights, draws, "right"), columns]
            for k in np.nonzero(free)[0] if self.explore else ():
                if rng.random() < self.explore:
                    values[k] = rng.uniform(-180.0, 180.0)
            if not (self._apart(values, angles[parent]) > self.different).any():
                continue  # the parent again
            earlier = seen.setdefault(parent, [])
            if any((self._apart(values, v) <= self.different).all() for v in earlier):
                continue
            earlier.append(values)
            child = set_torsions(positions[parent], found, dict(enumerate(values)))
            if self._overlaps(ctx, population, parents[parent], child, pairs):
                overlapping += 1
                continue
            made.append(
                (
                    parents[parent],
                    child,
                    dict(route=self.route, parents=[parents[parent]]),
                )
            )
        return made, overlapping

    @staticmethod
    def _apart(angles, others) -> np.ndarray:
        return np.abs((np.asarray(angles) - np.asarray(others) + 180.0) % 360.0 - 180.0)

    def __repr__(self) -> str:
        return (
            f"Resample({self.refine.optimizer!r}, window={self.window}, parents="
            f"{self.parents}, temperature={self.temperature}, explore={self.explore}, "
            f"children={self.children}, rounds={self.rounds})"
        )
