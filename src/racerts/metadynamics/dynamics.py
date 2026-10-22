"""The dynamics of Metadynamics: Langevin dynamics with a bias on the RMSD to the
structures that a run has visited (needs ASE)."""

import multiprocessing
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Tuple

import numpy as np
from ase import units
from ase.calculators.calculator import Calculator, all_changes
from ase.constraints import FixAtoms
from ase.md.langevin import Langevin

from racerts.refine.parallel import (
    _calculator_payload,
    _resolve_workers,
    pin_to_single_thread,
)
from racerts.utils import seeds

try:
    from ase.md.velocitydistribution import thermalize_momenta
except ImportError:  # ASE < 3.29: the same draws under the older name
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution

    def thermalize_momenta(atoms, temperature_K, *, rng=None):
        MaxwellBoltzmannDistribution(atoms, temperature_K=temperature_K, rng=rng)


def bias_terms(positions, references, k: float, alpha: float):
    """
    Energy and forces of the bias k * sum_i exp(-alpha * RMSD_i^2) at positions (n
    atoms, 3): RMSD_i is that to reference i (references: (m, n, 3), each centred)
    after the best superposition (Kabsch, proper rotations). The units are those of
    k and of 1 / alpha (e.g. eV and A^2).

    The force: with x and the superposed reference r both centred, RMSD^2 = sum_j
    |x_j - r_j|^2 / n. Its derivative with respect to x_j is 2 (x_j - r_j) / n: the
    best rotation is stationary, and a shift of the centroid cancels (the
    differences sum to zero). Atom j therefore feels
    (2 alpha k / n) exp(-alpha RMSD^2) (x_j - r_j), away from the reference.
    """
    x = np.asarray(positions, dtype=float)
    x = x - x.mean(axis=0)
    references = np.asarray(references, dtype=float)
    u, _, vt = np.linalg.svd(np.swapaxes(references, 1, 2) @ x)
    u[:, :, 2] *= np.where(np.linalg.det(u @ vt) < 0, -1.0, 1.0)[:, None]
    delta = x - references @ (u @ vt)
    gauss = k * np.exp(-alpha * (delta**2).sum(axis=(1, 2)) / len(x))
    forces = 2.0 * alpha / len(x) * np.einsum("m,mnj->nj", gauss, delta)
    return float(gauss.sum()), forces


class BiasedCalculator(Calculator):
    """
    Another ASE calculator plus the bias of a metadynamics run (bias_terms) on the
    atoms given: every structure passed to deposit becomes a reference that pushes
    the atoms away from it. k in eV, alpha in 1/A^2.

    results["energy"] and results["forces"] include the bias; results["bias_energy"]
    is its part of the energy.
    """

    implemented_properties = ["energy", "free_energy", "forces"]

    def __init__(self, calculator, atoms: Sequence[int], k: float, alpha: float, **kw):
        super().__init__(**kw)
        self.calculator = calculator
        self.biased = [int(i) for i in atoms]
        self.k = k
        self.alpha = alpha
        self.references: List[np.ndarray] = []

    def deposit(self, positions) -> None:
        """A new reference: the biased atoms at positions (all atoms, A)."""
        chosen = np.asarray(positions, dtype=float)[self.biased]
        self.references.append(chosen - chosen.mean(axis=0))
        self.reset()  # the results so far are those of the bias before

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        energy = self.calculator.get_potential_energy(self.atoms)
        forces = np.array(self.calculator.get_forces(self.atoms), dtype=float)
        bias = 0.0
        if self.references:
            bias, pushes = bias_terms(
                self.atoms.positions[self.biased], self.references, self.k, self.alpha
            )
            forces[self.biased] += pushes
        self.results = {
            "energy": energy + bias,
            "free_energy": energy + bias,
            "forces": forces,
            "bias_energy": bias,
        }


@dataclass(frozen=True)
class DynamicsConfig:
    """
    The settings that the runs of a search share.

    Attributes:
        steps: The length of a run.
        deposit: The steps from one reference of the bias to the next; the first is
            the start.
        snapshot: The steps from one snapshot to the next.
        biased: The atoms in the RMSD of the bias.
        fixed: The atoms that do not move (FixAtoms).
        temperature: Of the thermostat (K).
        time_step: fs.
        friction: Of the thermostat (1/fs).
        hydrogen_mass: The mass of the hydrogens (u), which allows the time step.
    """

    steps: int
    deposit: int
    snapshot: int
    biased: Tuple[int, ...]
    fixed: Tuple[int, ...] = ()
    temperature: float = 300.0
    time_step: float = 2.0
    friction: float = 0.02
    hydrogen_mass: float = 4.0


class _Normal:
    """The draws of a seeds.Stream with the interface ASE asks of a generator."""

    def __init__(self, stream):
        self.stream = stream

    def standard_normal(self, size):
        return self.stream.normal(int(np.prod(size))).reshape(size)


def run_dynamics(
    atoms,
    calculator,
    config: DynamicsConfig,
    k_per_atom: float,
    alpha: float,
    stream,
    snapshots=None,
) -> List[np.ndarray]:
    """
    One run from atoms (which stay as they are) on calculator: Langevin dynamics at
    constant temperature, in which the structure of every config.deposit steps
    becomes a reference of the bias, with the strength k_per_atom (mEh) times the
    number of atoms and the width alpha (1/bohr^2), as in CREST. The velocities and
    the thermostat draw from stream (racerts.utils.seeds.Stream).

    Returns the positions after every config.snapshot steps; they are also appended
    to snapshots, if given, so that those of a run that fails are not lost.
    """
    started = _start(atoms, calculator, config, k_per_atom, alpha, stream)
    return _advance(started, config, [] if snapshots is None else snapshots)


def _start(atoms, calculator, config, k_per_atom, alpha, stream):
    """The run set up: (atoms, bias, dynamics); no calculation yet."""
    atoms = atoms.copy()
    masses = atoms.get_masses()
    masses[atoms.numbers == 1] = config.hydrogen_mass
    atoms.set_masses(masses)
    if config.fixed:
        atoms.set_constraint(FixAtoms(indices=list(config.fixed)))
    bias = BiasedCalculator(
        calculator,
        config.biased,
        k_per_atom * 1e-3 * units.Hartree * len(atoms),
        alpha / units.Bohr**2,
    )
    atoms.calc = bias
    rng = _Normal(stream)
    thermalize_momenta(atoms, config.temperature, rng=rng)
    dynamics = Langevin(
        atoms,
        config.time_step * units.fs,
        temperature_K=config.temperature,
        friction=config.friction / units.fs,
        fixcm=False,
        rng=rng,
    )
    return atoms, bias, dynamics


def _advance(started, config, snapshots) -> List[np.ndarray]:
    """The steps of a run set up by _start, its snapshots appended to snapshots."""
    atoms, bias, dynamics = started
    done = 0
    while done < config.steps:
        if done % config.deposit == 0:
            bias.deposit(atoms.get_positions())
        ahead = min(
            config.deposit - done % config.deposit,
            config.snapshot - done % config.snapshot,
            config.steps - done,
        )
        dynamics.run(ahead)
        done += ahead
        if done % config.snapshot == 0:
            positions = atoms.get_positions()
            if not np.isfinite(positions).all():
                raise FloatingPointError(f"the positions diverged within {done} steps")
            snapshots.append(positions)
    return snapshots


@dataclass(frozen=True)
class Run:
    """
    One run of a search, as a worker process gets it.

    Attributes:
        round, number: The round and the number of the run in it, which name its
            random numbers.
        atoms: The start (ASE Atoms).
        calculator: An ASE calculator, or with is_factory a callable that returns
            one.
        k_per_atom, alpha: The strength (mEh) and the width (1/bohr^2) of the bias.
        seed: The seed of the search.
        prepare, reference: prepare(calculator, reference) is called before the run
            (see ASEOptimizer).
        expected_errors: The exception types of a calculation that end this run
            only; any other error is raised.
    """

    round: int
    number: int
    atoms: Any
    calculator: Any
    is_factory: bool
    k_per_atom: float
    alpha: float
    seed: int
    prepare: Optional[Callable[[Any, Any], None]] = None
    reference: Any = None
    expected_errors: tuple = (Exception,)


def _run(job) -> Tuple[List[np.ndarray], Optional[str]]:
    """The snapshots of a run, and the error that ended it early, if one did."""
    config, run = job
    calculator = run.calculator() if run.is_factory else run.calculator
    if run.prepare is not None:
        run.prepare(calculator, run.reference.copy())
    stream = seeds.Stream(run.seed, "metadynamics", run.round, run.number)
    found: List[np.ndarray] = []
    # an error in setting the run up is no failed calculation: it is raised as it is
    started = _start(run.atoms, calculator, config, run.k_per_atom, run.alpha, stream)
    try:
        _advance(started, config, found)
    except run.expected_errors as error:
        return found, f"{type(error).__name__}: {error}"
    return found, None


def run_all(
    runs: Sequence[Run], config: DynamicsConfig, num_workers: Optional[int] = 1
) -> List[Tuple[List[np.ndarray], Optional[str]]]:
    """
    The runs, one after the other, or for num_workers None or > 1 in spawned
    single-thread worker processes (as the refinement of ASEOptimizer): (snapshots,
    error) for each, the error that of a calculation that ended the run early.
    """
    jobs = [(config, run) for run in runs]
    if (num_workers is not None and num_workers <= 1) or len(jobs) < 2:
        return [_run(job) for job in jobs]
    # The runs share their calculator: a clear message if it cannot be sent.
    _calculator_payload(runs[0].calculator, runs[0].is_factory)
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(
        max_workers=_resolve_workers(num_workers, len(jobs)),
        mp_context=context,
        initializer=pin_to_single_thread,
    ) as pool:
        return list(pool.map(_run, jobs))
