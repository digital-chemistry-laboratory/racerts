"""Metadynamics: molecular dynamics that a bias pushes away from where it has been."""

import logging
from typing import Iterable, List, Optional, Sequence, Tuple, Union

from rdkit import Chem

from racerts.exploit.moves import _PLANAR, torsion_moves
from racerts.exploit.stage import _Population, add_refined, refine_candidates
from racerts.io import ase as ase_io
from racerts.pipeline import ConformerEnsemble
from racerts.refine.ase import ASEOptimizer
from racerts.refine.stage import Refine, energy_label
from racerts.rmsd import heavy_atoms
from racerts.system.spec import infer_charge_and_multiplicity
from racerts.task.base import check_atom_indices
from racerts.utils.checks import is_integer
from racerts.validate.checks import CLASH_FACTOR, gate

logger = logging.getLogger(__name__)

# The strengths and the widths of the bias in the runs of CREST (Pracht, Bohle, Grimme,
# Phys. Chem. Chem. Phys. 2020, 22, 7169): a run for every strength with every width.
K_PER_ATOM = (3.0, 1.5, 0.75)  # mEh per atom of the molecule
ALPHA = (1.3, 0.78, 0.47, 0.28)  # 1/bohr^2
MIN_LENGTH, MAX_LENGTH = 5.0, 200.0  # ps: the default length of a run lies between
MACROCYCLE = 8  # atoms: the single bonds of rings of this size or larger turn
LOWER = 0.1  # kcal/mol: a round that lowers the minimum by less found nothing lower


def flexibility(mol: Chem.Mol, held: Iterable[int] = ()) -> float:
    """
    The share of the bonds between heavy atoms that turn: the rotatable bonds that
    Exploit turns (torsion_moves: single, not in a ring, not planar as in an amide,
    with the atoms that the task holds on one side at most), and the single bonds of
    rings of MACROCYCLE or more atoms that are not planar either.
    """
    held = set(held)
    planar = {frozenset((m[0], m[2])) for m in mol.GetSubstructMatches(_PLANAR)}
    rings = mol.GetRingInfo()
    turning, bonds = len(torsion_moves(mol, held)), 0
    for bond in mol.GetBonds():
        pair = frozenset((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 1 for i in pair):
            continue
        bonds += 1
        turning += (
            bond.GetBondType() == Chem.BondType.SINGLE
            and rings.MinBondRingSize(bond.GetIdx()) >= MACROCYCLE
            and pair not in planar
            and not pair <= held
        )
    return turning / bonds if bonds else 0.0


def default_length(mol: Chem.Mol, held: Iterable[int] = ()) -> float:
    """
    The length of a run (ps) for a molecule, by its size and its flexibility: 0.1 (n
    + 0.1 n^2) for n = flexibility times the number of atoms that the task does not
    hold, but at least MIN_LENGTH and at most MAX_LENGTH. Modelled on the rule of
    CREST, with the flexibility of the graph (flexibility) for that of its bond
    orders.
    """
    held = set(held)
    n = flexibility(mol, held) * (mol.GetNumAtoms() - len(held))
    return min(MAX_LENGTH, max(MIN_LENGTH, 0.1 * (n + 0.1 * n * n)))


class Metadynamics:
    """
    A search by metadynamics from the conformers it gets, which it returns with the
    new minima it finds: molecular dynamics with a history-dependent bias on the RMSD,
    as in CREST (Grimme, J. Chem. Theory Comput. 2019, 15, 2847). Exploration: unlike
    Exploit it leaves the neighbourhood of its conformers, and it needs no moves to be
    defined (rings, macrocycles, the folding of a chain).

    One run: Langevin dynamics at constant temperature from one conformer. Every
    deposit ps the structure it is at becomes a reference, and each reference i adds
    k exp(-alpha RMSD_i^2) to the potential (RMSD over the biased atoms after the
    best superposition; k = k_per_atom times the number of atoms of the molecule),
    which pushes the run on. The runs differ in strength (runs) and take the
    conformers in turn, lowest in energy first: with one conformer all start from
    it; conformers beyond the number of runs start none.

    Rounds: with rounds > 1, a round that found a conformer more than LOWER below
    the lowest before it is followed by another, whose runs start from the lowest
    found so far (with random numbers of their own), until a round finds nothing
    lower or rounds are done: the iteration of CREST. stats["stop"] says which
    ("nothing lower" or "rounds"), also for a single round.

    The atoms that the task holds (the hard, core and soft atoms of its FrozenSet:
    what Exploit does not move either) are fixed in the dynamics and not in the
    bias, which therefore is on the shape of the other atoms, not on where they are
    relative to the fixed ones. The distance restraints of the context act in the
    refinement only.

    The snapshots of a round (one every snapshot ps) are refined in one call. A
    refined snapshot is accepted if it passes the gate checks (frozen atoms, bonds and
    stereo, clashes, restraints) and is further than rmsd_threshold (symmetry-aware,
    heavy atoms, or all atoms with include_hs) from every member of similar energy;
    a duplicate counts as a re-find of that member. There is no energy window: a
    pruner after the stage sets it.

    Provenance: route "mtd", parent (the conformer the run started from),
    mtd_round, mtd_run, mtd_k, mtd_alpha and mtd_time (ps) for accepted conformers,
    which keep their parent's reference, active-bond targets and batch; mtd_hits
    (the snapshots that ended in it) for every member. stats holds the numbers of
    the last call: rounds, runs, steps, snapshots, refined, accepted, refound,
    rejected, length (of a run, ps) and stop.

    Args:
        refine: A Refine stage, or an optimizer for one (Refine(optimizer,
            fallback=False)), as for Exploit: the optimizer must be an ASEOptimizer
            (with another one the stage warns and returns the ensemble unchanged),
            and the ensemble's energies must come from it (energy_method). Give it
            the steps to reach a minimum from a structure at temperature (max_steps).
        md_calculator: The calculator of the dynamics: an ASE calculator, or a
            callable that returns one (a new one for every run). Default: that of
            the refinement, prepared by its prepare hook. Usual: a cheap potential
            here and the level of the pipeline in refine.
        runs: The strengths of the runs: (k_per_atom, alpha) pairs in mEh and
            1/bohr^2; a number n for the first n of the default, the twelve of
            CREST (every K_PER_ATOM with every ALPHA; beyond twelve they are taken
            again from the first, with other random numbers).
        length: Of every run (ps). Default: by the size and the flexibility of the
            molecule (default_length): between 5 and 200 ps, e.g. 9 ps for
            heptane-1,7-diol (25 atoms) and 36 ps for icosane (62 atoms).
        rounds: The most rounds (see above); default: one.
        temperature: Of the thermostat (K).
        deposit: The time from one reference of the bias to the next (ps).
        snapshot: The time from one snapshot to the next (ps).
        time_step: fs; 2 fs go with hydrogen_mass.
        hydrogen_mass: The mass of the hydrogens in the dynamics (u).
        friction: Of the thermostat (1/fs).
        atoms: The atoms in the RMSD of the bias; default: the heavy atoms. Those
            that the task holds are left out.
        rmsd_threshold, include_hs, clash_factor: As in Exploit.
    """

    name = "metadynamics"

    def __init__(
        self,
        refine,
        md_calculator=None,
        runs: Union[None, int, Sequence[Tuple[float, float]]] = None,
        length: Optional[float] = None,
        rounds: int = 1,
        temperature: float = 300.0,
        deposit: float = 1.0,
        snapshot: float = 0.1,
        time_step: float = 2.0,
        hydrogen_mass: float = 4.0,
        friction: float = 0.02,
        atoms: Optional[Sequence[int]] = None,
        rmsd_threshold: float = 0.25,
        include_hs: bool = False,
        clash_factor: float = CLASH_FACTOR,
    ):
        positive = dict(temperature=temperature, deposit=deposit, snapshot=snapshot)
        positive.update(time_step=time_step, hydrogen_mass=hydrogen_mass)
        positive.update(friction=friction, rmsd_threshold=rmsd_threshold)
        if length is not None:
            positive["length"] = length
        for name, value in positive.items():
            if not value > 0:
                raise ValueError(f"{name} must be positive, not {value!r}.")
        if length is not None and snapshot > length:
            raise ValueError("A run is shorter than the time to its first snapshot.")
        if isinstance(rounds, bool) or not is_integer(rounds) or rounds < 1:
            raise ValueError(f"rounds must be a positive number, not {rounds!r}.")
        self.refine = refine if isinstance(refine, Refine) else Refine(refine, False)
        self.md_calculator = md_calculator
        self._md_calculator_is_factory = (
            md_calculator is not None and ase_io.check_calculator(md_calculator)
        )
        self.runs = _strengths(runs)
        self.length = length
        self.rounds = int(rounds)
        self.temperature = temperature
        self.deposit = deposit
        self.snapshot = snapshot
        self.time_step = time_step
        self.hydrogen_mass = hydrogen_mass
        self.friction = friction
        self.atoms = None if atoms is None else [int(i) for i in atoms]
        self.rmsd_threshold = rmsd_threshold
        self.include_hs = include_hs
        self.clash_factor = clash_factor
        self.stats: dict = {}

    def run(self, ctx, ensemble: ConformerEnsemble) -> ConformerEnsemble:
        counts = dict(rounds=0, runs=0, steps=0, snapshots=0, refined=0)
        counts.update(accepted=0, refound=0, rejected=0)
        optimizer = self.refine.optimizer
        if not isinstance(optimizer, ASEOptimizer):
            logger.warning(
                "Metadynamics needs an ASE refiner (xTB or an MLIP), as Exploit: with "
                "%s the ensemble passes unchanged.",
                type(optimizer).__name__,
            )
            self.stats = {**counts, "stop": "no ASE refiner"}
            return ensemble
        if not len(ensemble):
            self.stats = {**counts, "stop": "no conformers"}
            return ensemble
        label = energy_label(optimizer)
        if ensemble.energy_method != label:
            raise ValueError(
                f"Metadynamics needs conformers refined by its optimizer "
                f"(energy_method {label!r}), not {ensemble.energy_method!r}: refine "
                "them first."
            )
        missing = [i for i in ensemble.conf_ids if ensemble.energy(i) is None]
        if missing:
            raise ValueError(
                f"Conformers {missing} have no energy: Metadynamics needs refined "
                "conformers."
            )
        if len(self.biased_atoms(ctx, ensemble.mol)) < 2:
            logger.warning(
                "Metadynamics: nothing to bias (the bias is on the shape of at least "
                "two atoms that the task does not hold); the ensemble passes unchanged."
            )
            self.stats = {**counts, "stop": "nothing to bias"}
            return ensemble

        length = self.length
        if length is None:
            length = max(default_length(ensemble.mol, _held(ctx)), self.snapshot)
        population = _Population(ctx, ensemble, self.include_hs)
        checks = gate(self.clash_factor).validators
        stop = "rounds"
        for number in range(self.rounds):
            bins = set(population.bins.values())
            lowest = {key: population.minimum(key) for key in bins}
            snapshots = self._dynamics(ctx, population, length, number, counts)
            refined = refine_candidates(
                ctx, ensemble, snapshots, self.refine, "Metadynamics"
            )
            counts["rounds"] += 1
            counts["snapshots"] += len(snapshots)
            counts["refined"] += len(refined)
            add_refined(
                ctx, population, refined, snapshots, checks, self.rmsd_threshold, counts
            )
            if not any(
                population.minimum(key) < energy - LOWER
                for key, energy in lowest.items()
            ):
                stop = "nothing lower"
                break

        for conf_id in population.ids:
            ensemble.add_provenance(conf_id, mtd_hits=population.hits[conf_id])
        self.stats = {**counts, "length": round(length, 3), "stop": stop}
        logger.info(
            "Metadynamics: %d rounds of %d runs of %.1f ps (%d steps), %d snapshots, "
            "%d refined, %d new conformers, %d found again, %d rejected; stopped (%s).",
            counts["rounds"],
            len(self.runs),
            length,
            counts["steps"],
            counts["snapshots"],
            counts["refined"],
            counts["accepted"],
            counts["refound"],
            counts["rejected"],
            stop,
        )
        return ensemble

    def biased_atoms(self, ctx, mol) -> List[int]:
        """The atoms in the RMSD of the bias: atoms (default: the heavy atoms)
        without those that the task holds."""
        held = _held(ctx)
        chosen = heavy_atoms(mol) if self.atoms is None else self.atoms
        check_atom_indices(mol, chosen, "atoms of the bias")
        return [i for i in chosen if i not in held]

    def _dynamics(self, ctx, population, length, round_number, counts) -> list:
        """
        The snapshots of the runs of a round, each length ps long: (parent,
        positions, record) each, the record with the round, the run, its strength
        and the time of the snapshot.
        """
        from .dynamics import DynamicsConfig, Run, run_all

        mol = population.ensemble.mol
        per_ps = 1000.0 / self.time_step
        config = DynamicsConfig(
            steps=max(1, round(length * per_ps)),
            deposit=max(1, round(self.deposit * per_ps)),
            snapshot=max(1, round(self.snapshot * per_ps)),
            biased=tuple(self.biased_atoms(ctx, mol)),
            fixed=tuple(sorted(_held(ctx))),
            temperature=self.temperature,
            time_step=self.time_step,
            friction=self.friction,
            hydrogen_mass=self.hydrogen_mass,
        )
        optimizer = self.refine.optimizer
        calculator, is_factory = self.md_calculator, self._md_calculator_is_factory
        prepare = None
        if calculator is None:  # that of the refinement, prepared as it prepares it
            calculator, prepare = optimizer.calculator, optimizer.prepare
            is_factory = optimizer._calculator_is_factory
        starts = sorted(population.ids, key=lambda i: (population.energies[i], i))
        parents = [starts[number % len(starts)] for number in range(len(self.runs))]
        runs = []
        for number, (k_per_atom, alpha) in enumerate(self.runs):
            parent = parents[number]
            atoms = self._atoms(mol, parent)
            reference = None
            if prepare is not None:
                reference = self._reference(ctx, population.provenance(parent), atoms)
            runs.append(
                Run(
                    round=round_number,
                    number=number,
                    atoms=atoms,
                    calculator=calculator,
                    is_factory=is_factory,
                    k_per_atom=k_per_atom,
                    alpha=alpha,
                    seed=ctx.seed,
                    prepare=prepare,
                    reference=reference,
                    expected_errors=optimizer.expected_errors,
                )
            )
        snapshots = []
        results = run_all(runs, config, optimizer.num_workers)
        for run, (found, error) in zip(runs, results):
            steps = config.steps
            if error is not None:  # as a refinement that failed: it costs this run
                steps = len(found) * config.snapshot
                logger.warning(
                    "Metadynamics: run %d ended after %d of %d steps: %s",
                    run.number,
                    steps,
                    config.steps,
                    error,
                )
            counts["runs"] += 1
            counts["steps"] += steps
            for k, positions in enumerate(found, start=1):
                record = dict(
                    route="mtd",
                    parent=parents[run.number],
                    mtd_round=round_number,
                    mtd_run=run.number,
                    mtd_k=run.k_per_atom,
                    mtd_alpha=run.alpha,
                    mtd_time=round(k * config.snapshot / per_ps, 6),
                )
                snapshots.append((parents[run.number], positions, record))
        return snapshots

    def _atoms(self, mol, conf_id: int):
        """A conformer as the Atoms that the refinement makes of it."""
        optimizer = self.refine.optimizer
        state = infer_charge_and_multiplicity(
            mol, optimizer.charge, optimizer.multiplicity
        )
        return optimizer._to_atoms(mol, conf_id, state)

    def _reference(self, ctx, provenance: dict, atoms):
        """
        The Atoms that the refinement prepares a calculator on, for a conformer:
        its reference (which its provenance names if there are several), or
        without one atoms.
        """
        reference = ctx.reference
        if reference is None:
            return atoms
        ref_id = provenance.get("reference")
        if ref_id is None or reference.GetNumConformers() <= 1:
            ref_id = reference.GetConformer(self.refine.optimizer.conf_id_ref).GetId()
        return self._atoms(reference, int(ref_id))

    def __repr__(self) -> str:
        return (
            f"Metadynamics({self.refine.optimizer!r}, runs={len(self.runs)}, "
            f"length={self.length}, rounds={self.rounds})"
        )


def _held(ctx) -> set:
    """The atoms that the task holds: fixed in the dynamics."""
    return {*ctx.frozen.hard, *ctx.frozen.core, *ctx.frozen.soft}


def _strengths(runs) -> List[Tuple[float, float]]:
    """The (k_per_atom, alpha) of the runs (see Metadynamics)."""
    combinations = [(k, alpha) for k in K_PER_ATOM for alpha in ALPHA]
    if runs is None:
        return combinations
    wrong = ValueError(
        "runs must be a positive number of runs or (k_per_atom, alpha) pairs with "
        f"k_per_atom >= 0 and alpha > 0, not {runs!r}."
    )
    if isinstance(runs, (bool, str)):
        raise wrong
    if is_integer(runs):
        if runs < 1:
            raise wrong
        return [combinations[n % len(combinations)] for n in range(int(runs))]
    try:
        pairs = [(float(k), float(alpha)) for k, alpha in runs]
    except (TypeError, ValueError):
        raise wrong from None
    if not pairs or any(k < 0 or not alpha > 0 for k, alpha in pairs):
        raise wrong
    return pairs
