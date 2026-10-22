"""Metadynamics: molecular dynamics pushed away from where it has been."""

import logging

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

import racerts
from racerts import Metadynamics
from racerts.metadynamics import ALPHA, K_PER_ATOM, default_length, flexibility
from racerts.rmsd import rmsd, superpose
from racerts.utils import seeds

pytestmark = pytest.mark.ase
ase = pytest.importorskip("ase")
from ase.calculators.lj import LennardJones  # noqa: E402
from ase.optimize import LBFGS  # noqa: E402

from racerts.metadynamics.dynamics import (  # noqa: E402
    BiasedCalculator,
    DynamicsConfig,
    Run,
    bias_terms,
    run_all,
    run_dynamics,
)
from racerts.refine import ASEOptimizer, MMFFOptimizer  # noqa: E402

from .mmff_calculator import MMFFCalculator, numerical_forces  # noqa: E402

OXYGENS, HYDROXYL_HS = (0, 8), (9, 24)  # of the diol below


def _diol():
    """Heptane-1,7-diol, stretched out: all its torsions anti, the oxygens 9.9 A
    apart."""
    mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCCCCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    conf = mol.GetConformer()
    for first in range(6):
        rdMolTransforms.SetDihedralDeg(
            conf, first, first + 1, first + 2, first + 3, 180.0
        )
    AllChem.MMFFOptimizeMolecule(mol, maxIters=2000)
    return mol


def _mmff(mol, **settings):
    """MMFF94 through ASE as the level of a refinement (a stand-in for xTB or an
    MLIP)."""
    settings = {"fmax": 0.05, "max_steps": 500, **settings}
    return ASEOptimizer(
        MMFFCalculator(mol), optimizer_cls=LBFGS, method="MMFF94", **settings
    )


def _start(mol, optimizer, task=None, seed=12):
    """A context and the conformer of mol, refined by optimizer."""
    ctx = racerts.Context.create(mol, task or racerts.GroundState(), seed=seed)
    ensemble = racerts.ConformerEnsemble(Chem.Mol(ctx.mol))
    return ctx, racerts.Refine(optimizer).run(ctx, ensemble)


def _hydrogen_bond(positions):
    """The shorter of the two O-H...O distances of the diol (A)."""
    return min(
        np.linalg.norm(positions[h] - positions[o])
        for h, o in zip(HYDROXYL_HS, reversed(OXYGENS))
    )


def _positions(ensemble):
    return [conf.GetPositions() for conf in ensemble.mol.GetConformers()]


def test_the_bias():
    rng = np.random.default_rng(0)
    n, k, alpha = 7, 0.4, 0.9
    x = rng.normal(size=(n, 3))
    structures = rng.normal(size=(3, n, 3))
    references = structures - structures.mean(axis=1, keepdims=True)

    # -- each reference adds k exp(-alpha RMSD^2), after superposition
    energy, forces = bias_terms(x, references, k, alpha)
    expected = sum(k * np.exp(-alpha * rmsd(x, y) ** 2) for y in structures)
    assert energy == pytest.approx(expected, abs=1e-12)

    # -- the force: (2 alpha k / n) exp(-alpha RMSD^2) (x_j - reference_j)
    # with x and the superposed reference both centred, away from the reference
    one, force = bias_terms(x, references[:1], k, alpha)
    delta = (x - x.mean(axis=0)) - superpose(references[0], x)
    assert one == pytest.approx(k * np.exp(-alpha * (delta**2).sum() / n))
    np.testing.assert_allclose(force, 2 * alpha * one / n * delta, atol=1e-12)
    # ... is the gradient of the energy, for any number of references
    numerical = np.zeros((n, 3))
    step = 1e-5
    for j in range(n):
        for axis in range(3):
            ends = []
            for sign in (1, -1):
                moved = x.copy()
                moved[j, axis] += sign * step
                ends.append(bias_terms(moved, references, k, alpha)[0])
            numerical[j, axis] = -(ends[0] - ends[1]) / (2 * step)
    np.testing.assert_allclose(forces, numerical, atol=1e-8)
    assert np.abs(forces).max() > 1e-3

    # -- it turns and moves with the structure, and vanishes on a reference
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    turned = bias_terms(x @ turn.T + [1.0, -2.0, 0.5], references, k, alpha)
    assert turned[0] == pytest.approx(energy, abs=1e-12)
    np.testing.assert_allclose(turned[1], forces @ turn.T, atol=1e-12)
    on_top = bias_terms(structures[0] @ turn.T, references[:1], k, alpha)
    assert on_top[0] == pytest.approx(k) and np.abs(on_top[1]).max() < 1e-12

    # -- on a calculator: the chosen atoms carry the bias, the others do not
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    atoms = racerts.ConformerEnsemble(mol).to_ase(0)
    plain = MMFFCalculator(mol)
    carbons = [0, 1, 2, 3]
    atoms.calc = BiasedCalculator(plain, carbons, k, alpha)
    assert atoms.get_potential_energy() == plain.get_potential_energy(atoms)
    start = atoms.get_positions()
    atoms.calc.deposit(start)  # the structure it is at: the bias has its maximum
    assert atoms.get_potential_energy() == pytest.approx(
        plain.get_potential_energy(atoms) + k
    )
    np.testing.assert_allclose(atoms.get_forces(), plain.get_forces(atoms), atol=1e-12)
    atoms.set_dihedral(0, 1, 2, 3, 60.0, indices=[3, 11, 12, 13])
    bias, on_carbons = bias_terms(
        atoms.positions[carbons], [start[carbons] - start[carbons].mean(0)], k, alpha
    )
    assert atoms.get_potential_energy() == pytest.approx(
        plain.get_potential_energy(atoms) + bias
    )
    assert atoms.calc.results["bias_energy"] == pytest.approx(bias) and bias < k
    pushed = atoms.get_forces() - plain.get_forces(atoms)
    np.testing.assert_allclose(pushed[carbons], on_carbons, atol=1e-12)
    assert np.abs(pushed[4:]).max() == 0
    # forces of 3 eV/A on the turned butane: central differences are right to a relative 1e-6
    np.testing.assert_allclose(
        atoms.get_forces(), numerical_forces(atoms), rtol=1e-5, atol=1e-6
    )


def test_the_search_on_a_diol():
    mol = _diol()
    assert _hydrogen_bond(mol.GetConformer().GetPositions()) > 9

    # -- from one stretched conformer to several minima, a hydrogen-bonded one too
    # Three runs with CREST's strongest and widest bias, deposited five times as
    # often as by default, so that 3 ps are enough to fold the chain.
    optimizer = _mmff(mol)
    ctx, ensemble = _start(mol, optimizer)
    search = Metadynamics(
        optimizer, runs=[(3.0, 0.28)] * 3, length=3.0, deposit=0.2, snapshot=0.2
    )
    out = search.run(ctx, ensemble)
    assert out.energy_method == "MMFF94" and len(out) > 5
    assert min(map(_hydrogen_bond, _positions(out))) < 2.5
    # No two alike, as for Exploit: within 0.25 A and 1 kcal/mol of each other.
    kernel = racerts.rmsd.SymmetricRMSD(out.mol, "heavy")
    structures = [kernel.prepare(x) for x in _positions(out)]
    energies = out.energies()
    for k, first in enumerate(structures):
        for j in np.nonzero(np.abs(energies[:k] - energies[k]) <= 1.0)[0]:
            assert not kernel.within(first, structures[j], 0.25)

    # -- the provenance and the statistics
    assert out.conf_ids[0] == 0 and "route" not in out.provenance(0)
    for conf_id in out.conf_ids[1:]:
        provenance = out.provenance(conf_id)
        assert provenance["route"] == "mtd" and provenance["parent"] == 0
        assert provenance["mtd_round"] == 0 and provenance["mtd_run"] in (0, 1, 2)
        assert (provenance["mtd_k"], provenance["mtd_alpha"]) == (3.0, 0.28)
        assert 0 < provenance["mtd_time"] <= 3.0
        assert provenance["mtd_hits"] >= 1
    stats = search.stats
    assert (stats["runs"], stats["steps"], stats["snapshots"]) == (3, 4500, 45)
    assert stats["refined"] == 45 and stats["rounds"] == 1 and stats["length"] == 3.0
    assert stats["accepted"] == len(out) - 1
    assert stats["accepted"] + stats["refound"] + stats["rejected"] == 45
    hits = sum(out.provenance(i)["mtd_hits"] for i in out.conf_ids)
    assert hits == stats["accepted"] + stats["refound"]

    # -- the same seed gives the same conformers, another seed others
    def short(seed):
        ctx, ensemble = _start(mol, optimizer, seed=seed)
        search = Metadynamics(
            optimizer, runs=[(3.0, 0.28)], length=1.0, deposit=0.2, snapshot=0.25
        )
        return search.run(ctx, ensemble)

    first, again, other = short(12), short(12), short(13)
    assert len(first) > 1
    np.testing.assert_array_equal(_positions(first), _positions(again))
    for conf_id in first.conf_ids:  # but for the time that the refinement took
        same = {**again.provenance(conf_id), "wall_time": None}
        assert {**first.provenance(conf_id), "wall_time": None} == same
    assert len(other) != len(first) or not np.allclose(
        _positions(first), _positions(other), atol=1e-3
    )


def test_what_the_task_holds_stays_where_it_is():
    mol = _diol()
    held = [0, 1, 2, 3]  # one end of the chain
    reference = mol.GetConformer().GetPositions()

    # -- in the dynamics: fixed, and not in the bias
    atoms = racerts.ConformerEnsemble(mol).to_ase(0)
    config = DynamicsConfig(
        steps=200, deposit=50, snapshot=100, biased=(4, 5, 6, 7, 8), fixed=tuple(held)
    )
    stream = seeds.Stream(12, "metadynamics", 0)
    snapshots = run_dynamics(atoms, MMFFCalculator(mol), config, 3.0, 0.28, stream)
    assert len(snapshots) == 2
    for snapshot in snapshots:
        np.testing.assert_array_equal(snapshot[held], reference[held])
        assert np.abs(snapshot - reference).max() > 0.3  # the rest moved
    np.testing.assert_array_equal(atoms.get_positions(), reference)  # the start stays

    # -- in the stage: the hard atoms of the task, in every conformer it adds
    optimizer = _mmff(mol)
    ctx, ensemble = _start(mol, optimizer, racerts.Constrained(hard=held))
    search = Metadynamics(optimizer, runs=[(3.0, 0.28)], length=1.0, deposit=0.2)
    assert search.biased_atoms(ctx, ensemble.mol) == [4, 5, 6, 7, 8]
    out = search.run(ctx, ensemble)
    assert len(out) > 1
    assert racerts.validate.FrozenCore(1e-9).validate(ctx, out) == {}
    for positions in _positions(out):
        assert np.abs(positions[held] - reference[held]).max() < 1e-9
    moved = [rmsd(x, reference, [4, 5, 6, 7, 8], align=False) for x in _positions(out)]
    assert max(moved) > 0.5

    # -- atoms of one's own choice, without those that the task holds
    chosen = Metadynamics(optimizer, atoms=[2, 3, 4, 8, 9])
    assert chosen.biased_atoms(ctx, ensemble.mol) == [4, 8, 9]

    # -- the targets of a windowed TS take turns as starts: energies compare only within
    # a window bin (here the shorter targets lie higher, as their restraints make them)
    from types import SimpleNamespace

    from racerts.metadynamics.stage import run_starts

    targets = [2.0, 2.0, 2.1, 2.1, 2.2, 2.2]
    population = SimpleNamespace(
        ids=list(range(6)),
        energies={
            i: -100.0 - 30.0 * (t - 2.0) + 0.1 * (i % 2) for i, t in enumerate(targets)
        },
        bins={i: (("0-8", t),) for i, t in enumerate(targets)},
    )
    assert run_starts(population) == [
        4,
        2,
        0,
        5,
        3,
        1,
    ]  # lowest of each bin, then the second
    population.bins = {i: () for i in range(6)}  # no windows: by energy alone
    assert run_starts(population) == [4, 5, 2, 3, 0, 1]

    # -- nothing left to bias: every heavy atom held
    ctx, ensemble = _start(mol, optimizer, racerts.Constrained(hard=range(9)))
    search = Metadynamics(optimizer, runs=1, length=0.2)
    assert search.run(ctx, ensemble).conf_ids == [0]
    assert search.stats["stop"] == "nothing to bias" and search.stats["rounds"] == 0


def _hydrogens(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


# methanol and one water, hydrogen-bonded (O...O 2.93 A): the lowest MMFF complex that
# racerts.generate_gs gives for "CO.O" with linked fragments
METHANOL_WATER = [
    (0.213, 0.675, 0.056),
    (-0.213, -0.675, -0.056),
    (1.955, -2.650, -0.088),
    (0.149, 0.991, 1.100),
    (1.249, 0.748, -0.283),
    (-0.420, 1.309, -0.569),
    (-1.138, -0.698, 0.252),
    (1.248, -1.984, 0.042),
    (1.844, -2.878, -1.023),
]
METHANOL, WATER = [0, 1, 3, 4, 5, 6], [2, 7, 8]


def _methanol_water():
    mol = _hydrogens("CO.O")
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, position in enumerate(METHANOL_WATER):
        conf.SetAtomPosition(atom, position)
    mol.AddConformer(conf)
    return mol


def test_a_free_molecule_around_held_atoms():
    from racerts.metadynamics.dynamics import frame_is_fixed
    from racerts.restraints import DistanceRestraint

    mol = _methanol_water()
    start = mol.GetConformer().GetPositions()
    atoms = racerts.ConformerEnsemble(mol).to_ase(0)

    # -- held atoms fix the frame: the bias is on where the water is, not only on its
    # shape, which does not change when it moves as a whole
    shifted = start.copy()
    shifted[WATER] += (1.0, 0.0, 0.0)
    for superposed, expected in ((True, 1.0), (False, np.exp(-1.0))):
        bias = BiasedCalculator(MMFFCalculator(mol), WATER, 1.0, 1.0, superposed)
        bias.deposit(start)
        atoms.calc = bias
        atoms.set_positions(shifted)
        atoms.get_potential_energy()
        assert bias.results["bias_energy"] == pytest.approx(expected)
    assert frame_is_fixed(start[METHANOL])
    assert not frame_is_fixed(start[[0, 1]])  # the others can turn about two atoms
    assert not frame_is_fixed(np.array([[0.0, 0, 0], [1.2, 0, 0], [2.6, 0, 0]]))

    # -- the distance restraints that refinement holds act in the dynamics too
    config = DynamicsConfig(
        steps=500,
        deposit=50,
        snapshot=50,
        biased=tuple(WATER),
        fixed=tuple(METHANOL),
        superpose=False,
    )
    held = DistanceRestraint(1, 2, 2.6, 3.0, force_constant=100.0)
    longest = []
    for restraints in ([], [held]):  # without it, the bias drives the water away
        stream = seeds.Stream(12, "metadynamics", 0)
        calculator = MMFFCalculator(mol)
        snapshots = run_dynamics(
            atoms, calculator, config, 3.0, 0.28, stream, restraints=restraints
        )
        longest.append(max(np.linalg.norm(x[1] - x[2]) for x in snapshots))
    assert longest[0] > 4.5 and longest[1] < 3.6

    # -- in the stage: one free heavy atom is enough to bias, the water is free, so the
    # dynamics hold it with a containment restraint, and it ends up elsewhere
    optimizer = _mmff(mol)
    ctx, ensemble = _start(mol, optimizer, racerts.Constrained(hard=METHANOL))
    search = Metadynamics(optimizer, runs=2, length=5.0)
    assert search.biased_atoms(ctx, ensemble.mol) == [2]
    refined = ensemble.mol.GetConformer().GetPositions()  # where the run starts
    out = search.run(ctx, ensemble)
    radius = search.stats["containment"]  # between the central atoms: C and O
    assert radius == pytest.approx(np.linalg.norm(refined[0] - refined[2]) + 2.0)
    positions = _positions(out)
    assert len(positions) > 1
    assert max(np.linalg.norm(x[2] - start[2]) for x in positions) > 1.5

    # -- no containment when asked, or when a restraint already holds the water
    search = Metadynamics(optimizer, runs=1, length=0.2, containment=None)
    search.run(ctx, ensemble)
    assert search.stats["containment"] is None
    linked = racerts.Context.create(
        mol,
        racerts.Constrained(hard=METHANOL),
        restraints=[DistanceRestraint(1, 2, 2.6, 3.4)],
    )
    search = Metadynamics(optimizer, runs=1, length=0.2)
    start = racerts.ConformerEnsemble(Chem.Mol(linked.mol))
    search.run(linked, racerts.Refine(optimizer).run(linked, start))
    assert search.stats["containment"] is None


def test_the_default_length_of_a_run():
    # -- the share of the bonds that turn
    assert flexibility(_hydrogens("c1ccccc1")) == 0
    assert flexibility(_hydrogens("C1CCCCC1")) == 0
    assert flexibility(_hydrogens("CCCC")) == pytest.approx(1 / 3)  # not the ends
    assert flexibility(_hydrogens("OCCCCCCCO")) == 1  # the OH rotors too
    assert flexibility(_hydrogens("CC(=O)NC")) == 0  # the amide is planar
    assert flexibility(_hydrogens("C1CCCCCCCCCCC1")) == 1  # a macrocycle
    lactam = _hydrogens("O=C1CCCCCCCCCCN1")
    assert flexibility(lactam) == pytest.approx(11 / 13)  # but for C=O and C-N

    # -- 0.1 (n + 0.1 n^2) ps for n flexible atoms, between 5 and 200 ps
    diol = _hydrogens("OCCCCCCCO")  # 25 atoms, all flexible
    assert default_length(diol) == pytest.approx(0.1 * (25 + 0.1 * 25**2))  # 8.75
    assert default_length(_hydrogens("c1ccccc1")) == 5.0
    assert default_length(_hydrogens("CCCC")) == 5.0
    icosane = _hydrogens("C" * 20)  # 62 atoms, 17 of 19 bonds turn
    n = 62 * 17 / 19
    assert default_length(icosane) == pytest.approx(0.1 * (n + 0.1 * n**2))  # 36 ps
    assert default_length(_hydrogens("C" * 60)) == 200.0
    # What the task holds counts neither as flexible nor as size.
    held = [0, 1, 2, 3]
    # only 1-2 is stuck: 0-1 turns the hydroxyl H, 2-3 the atoms 4 to 8
    assert flexibility(diol, held) == pytest.approx(7 / 8)
    n = 21 * 7 / 8
    assert default_length(diol, held) == pytest.approx(0.1 * (n + 0.1 * n**2))

    # -- a stage without a length takes it
    mol = _hydrogens("CCCC")
    AllChem.EmbedMolecule(mol, randomSeed=3)
    optimizer = _mmff(mol)
    ctx, ensemble = _start(mol, optimizer)
    search = Metadynamics(optimizer, runs=1, snapshot=2.5)
    search.run(ctx, ensemble)
    stats = search.stats
    assert (stats["length"], stats["steps"], stats["snapshots"]) == (5.0, 2500, 2)


class Scripted(ASEOptimizer):
    """Leaves the structures as they are and gives the conformers of its nth call
    the energies of its nth script, each 2 kcal/mol above the one before."""

    def __init__(self, *scripts):
        super().__init__(LennardJones(), method="scripted")
        self.scripts = list(scripts)

    def _refine(self, mol, reference, anchors, restraints=()):
        first = self.scripts.pop(0)
        for k, conf in enumerate(mol.GetConformers()):
            conf.SetDoubleProp("energy", first + 2.0 * k)
        return 0


def test_rounds():
    mol = _hydrogens("CCCC")
    AllChem.EmbedMolecule(mol, randomSeed=3)
    few = dict(runs=2, length=0.02, snapshot=0.01)  # two snapshots of each run
    md_calculator = MMFFCalculator(mol)

    def search(rounds, *scripts, **settings):
        optimizer = Scripted(0.0, *scripts)  # the start at 0 kcal/mol
        ctx, ensemble = _start(mol, optimizer)
        settings = {**few, **settings}
        stage = Metadynamics(optimizer, md_calculator, rounds=rounds, **settings)
        return stage.run(ctx, ensemble), stage.stats

    # -- a round that found a lower conformer is followed by another, from there
    # The first round ends at -20, -18, -16 and -14 kcal/mol, the second above the
    # start: it found nothing lower, and the third round is not needed.
    out, stats = search(3, -20.0, 10.0)
    assert (stats["rounds"], stats["stop"]) == (2, "nothing lower")
    assert (stats["runs"], stats["snapshots"], stats["accepted"]) == (4, 8, 8)
    rounds = [out.provenance(i).get("mtd_round") for i in out.conf_ids]
    assert rounds == [None, 0, 0, 0, 0, 1, 1, 1, 1]
    parents = [out.provenance(i).get("parent") for i in out.conf_ids]
    assert parents[1:5] == [0, 0, 0, 0]  # of the one conformer there was
    assert parents[5:] == [1, 1, 2, 2]  # of the two lowest of the first round

    # -- every run of every round has random numbers of its own
    atoms = racerts.ConformerEnsemble(mol).to_ase(0)
    config = DynamicsConfig(steps=5, deposit=5, snapshot=5, biased=(0, 1, 2, 3))

    def run(round_number, number, seed=12):
        settings = dict(calculator=md_calculator, is_factory=False, seed=seed)
        return Run(round_number, number, atoms, k_per_atom=3.0, alpha=1.3, **settings)

    runs = [run(0, 0), run(0, 0), run(1, 0), run(0, 1), run(0, 0, seed=13)]
    same, again, *others = [found[0] for found, _ in run_all(runs, config)]
    np.testing.assert_array_equal(same, again)
    for other in others:
        assert np.abs(other - same).max() > 1e-3

    # -- the rounds end where they are told to, or when nothing lower is found
    _, stats = search(1, -20.0)
    assert (stats["rounds"], stats["stop"]) == (1, "rounds")  # another could follow
    _, stats = search(2, -20.0, -40.0)
    assert (stats["rounds"], stats["stop"]) == (2, "rounds")
    _, stats = search(3, 10.0)
    assert (stats["rounds"], stats["stop"]) == (1, "nothing lower")
    # A new conformer that is lower by less than 0.1 kcal/mol is not lower.
    _, stats = search(3, -0.05, rmsd_threshold=1e-6)
    assert (stats["rounds"], stats["stop"]) == (1, "nothing lower")
    assert stats["accepted"] == 4
    for wrong in (0, 1.5, True):
        with pytest.raises(ValueError, match="rounds"):
            Metadynamics(_mmff(mol), rounds=wrong)


class Counted(MMFFCalculator):
    """MMFF94 that counts its calculations."""

    calls = 0

    def calculate(self, *args, **kwargs):
        Counted.calls += 1
        super().calculate(*args, **kwargs)


class Prepared(MMFFCalculator):
    """MMFF94 that fails unless _prepare has seen it."""

    prepared_on = None

    def calculate(self, *args, **kwargs):
        if self.prepared_on is None:
            raise RuntimeError("not prepared")
        super().calculate(*args, **kwargs)


def _prepare(calculator, reference_atoms):
    calculator.prepared_on = reference_atoms.get_positions()


MADE_HERE = []


def counted_force_field():
    """A calculator factory that counts its calls in the process it runs in."""
    MADE_HERE.append(1)
    return MMFFCalculator(_diol())


def test_the_calculators_and_what_the_stage_needs(caplog):
    mol = _diol()
    few = dict(runs=1, length=0.02, snapshot=0.01)  # ten steps, two snapshots

    # -- the dynamics run on the calculator of the refinement
    optimizer = ASEOptimizer(Counted(mol), method="MMFF94", max_steps=2)
    ctx, ensemble = _start(mol, optimizer)
    Counted.calls = 0
    search = Metadynamics(optimizer, **few)
    search.run(ctx, ensemble)
    assert search.stats["steps"] == 10 and search.stats["snapshots"] == 2
    assert Counted.calls > 10 + 2  # the steps, and the two refinements

    # -- or on a calculator of their own: an instance, or a factory (one per run)
    for md_calculator in (MMFFCalculator(mol), lambda: MMFFCalculator(mol)):
        ctx, ensemble = _start(mol, optimizer)
        Counted.calls = 0
        search = Metadynamics(optimizer, md_calculator=md_calculator, **few)
        out = search.run(ctx, ensemble)
        assert search.stats["snapshots"] == 2 == search.stats["refined"]
        assert 2 <= Counted.calls < 10  # the refinements only
        assert out.energy_method == "MMFF94"

    # -- the refinement's prepare hook also prepares the calculator of the dynamics
    for calculator in (Prepared(mol), lambda: Prepared(mol)):
        optimizer = ASEOptimizer(
            calculator, method="MMFF94", max_steps=2, prepare=_prepare
        )
        ctx, ensemble = _start(mol, optimizer)
        search = Metadynamics(optimizer, **few)
        search.run(ctx, ensemble)
        assert search.stats["steps"] == 10

    # -- a run whose calculation fails ends there, and the others go on
    class Fails(LennardJones):
        def calculate(self, *args, **kwargs):
            raise RuntimeError("the calculation failed")

    optimizer = _mmff(mol)
    ctx, ensemble = _start(mol, optimizer)
    calculators = iter([Fails(), MMFFCalculator(mol)])
    search = Metadynamics(
        optimizer, md_calculator=lambda: next(calculators), **{**few, "runs": 2}
    )
    with caplog.at_level(logging.WARNING, logger="racerts"):
        search.run(ctx, ensemble)
    assert "run 0 ended after 0 of 10 steps: RuntimeError" in caplog.text
    assert search.stats["snapshots"] == 2 and search.stats["steps"] == 10
    # ... but a round whose runs all fail is no search that found nothing lower
    search = Metadynamics(optimizer, md_calculator=Fails(), **{**few, "runs": 2})
    with pytest.raises(racerts.errors.NoConformersError, match="every run"):
        search.run(ctx, ensemble)
    # and an error in setting up the dynamics (not of a calculation) is raised as it is
    import racerts.metadynamics.dynamics as dynamics

    class Refuses:
        def __init__(self, *args, **kwargs):
            raise TypeError("an argument this ASE does not take")

    original, dynamics.Langevin = dynamics.Langevin, Refuses
    try:
        with pytest.raises(TypeError, match="this ASE"):
            Metadynamics(optimizer, md_calculator=MMFFCalculator(mol), **few).run(
                ctx, ensemble
            )
    finally:
        dynamics.Langevin = original

    # -- with workers of the refinement, the runs go to processes and give the same
    ensembles = []
    for workers in (1, 2):  # serial_below: the few refinements stay in this process
        optimizer = _mmff(mol, num_workers=workers, serial_below=10)
        ctx, ensemble = _start(mol, _mmff(mol))
        search = Metadynamics(
            optimizer, counted_force_field, runs=2, length=0.2, snapshot=0.1
        )
        del MADE_HERE[:]
        ensembles.append(search.run(ctx, ensemble))
        assert len(MADE_HERE) == (2 if workers == 1 else 0)  # one for every run
        assert search.stats["snapshots"] == 4
    np.testing.assert_array_equal(_positions(ensembles[1]), _positions(ensembles[0]))

    # -- the refinement must be an ASE one, and the conformers must come from it
    ctx, ensemble = _start(mol, MMFFOptimizer())
    before = ensemble.mol.GetConformer(0).GetPositions()
    caplog.clear()
    search = Metadynamics(MMFFOptimizer(), md_calculator=MMFFCalculator(mol))
    with caplog.at_level(logging.WARNING, logger="racerts"):
        out = search.run(ctx, ensemble)
    assert "needs an ASE refiner" in caplog.text
    assert out.conf_ids == [0] and search.stats["stop"] == "no ASE refiner"
    np.testing.assert_array_equal(out.mol.GetConformer(0).GetPositions(), before)
    with pytest.raises(ValueError, match="energy_method"):
        Metadynamics(_mmff(mol)).run(ctx, ensemble)

    # -- the runs: CREST's twelve, the first of them, or strengths of one's own
    optimizer = _mmff(mol)
    combinations = [(k, alpha) for k in K_PER_ATOM for alpha in ALPHA]
    assert K_PER_ATOM == (3.0, 1.5, 0.75) and ALPHA == (1.3, 0.78, 0.47, 0.28)
    assert Metadynamics(optimizer).runs == combinations
    assert Metadynamics(optimizer, runs=3).runs == combinations[:3]
    assert Metadynamics(optimizer, runs=14).runs == combinations + combinations[:2]
    assert Metadynamics(optimizer, runs=[(2, 0.5)]).runs == [(2.0, 0.5)]
    for wrong in (0, [], [(1.0, 0.0)], [(-1.0, 1.0)], "all", True):
        with pytest.raises(ValueError, match="runs"):
            Metadynamics(optimizer, runs=wrong)
    for wrong in (
        dict(length=0),
        dict(length=-1.0),
        dict(temperature=-1.0),
        dict(snapshot=0.0),
        dict(deposit=0.0),
        dict(length=0.05, snapshot=0.1),
        dict(time_step=0),
        dict(rmsd_threshold=0),
    ):
        with pytest.raises(ValueError):
            Metadynamics(optimizer, **wrong)
    with pytest.raises(ValueError, match="calculator"):
        Metadynamics(optimizer, md_calculator="mmff")
