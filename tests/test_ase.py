"""ASE refinement and rescoring: hooks, failures, convergence and provenance."""

import numpy as np
import pytest
from rdkit import Chem

import racerts
from racerts import Rescore, TransitionState
from racerts.utils.units import EV_TO_KCAL_MOL

pytestmark = pytest.mark.ase
ase = pytest.importorskip("ase")
from ase.calculators.lj import LennardJones  # noqa: E402

from racerts.refine import ASEOptimizer, Outcome, optimize_one  # noqa: E402


class FailOnMark(LennardJones):
    """Lennard-Jones that fails for structures marked with atoms.info["fail"]."""

    def calculate(self, atoms=None, *args, **kwargs):
        if atoms is not None and atoms.info.get("fail"):
            raise RuntimeError("SCF not converged")
        super().calculate(atoms, *args, **kwargs)


class NeedsPreparation(LennardJones):
    """Lennard-Jones that fails unless prepare_lj has run on it."""

    prepared_on = None

    def calculate(self, atoms=None, *args, **kwargs):
        if self.prepared_on is None:
            raise RuntimeError("not prepared")
        super().calculate(atoms, *args, **kwargs)


def prepare_lj(calculator, reference_atoms):
    """Module level, so that worker processes can unpickle it."""
    calculator.prepared_on = reference_atoms.get_positions()


class MarkingOptimizer(ASEOptimizer):
    """Marks the Atoms of the conformers in fail_ids for FailOnMark."""

    fail_ids = ()

    def _to_atoms(self, mol, conf_id, state):
        atoms = super()._to_atoms(mol, conf_id, state)
        atoms.info["fail"] = conf_id in self.fail_ids
        return atoms


@pytest.fixture
def refined(hept_1_ene_ts):
    """Four MMFF conformers of the TS of ex.xyz, with their context."""
    ctx = racerts.Context.create(hept_1_ene_ts, TransitionState([3, 4, 5]))
    ensemble = racerts.Pipeline([racerts.Embed(n_conformers=4), racerts.Refine()]).run(
        ctx
    )
    return ensemble, ctx


def _lj_energy(positions, symbols):
    atoms = ase.Atoms(symbols=symbols, positions=positions)
    atoms.calc = LennardJones()
    return atoms.get_potential_energy() * EV_TO_KCAL_MOL


def test_rescore_replaces_the_energies(refined):
    ensemble, ctx = refined
    before = ensemble.copy()
    Rescore(LennardJones()).run(ctx, ensemble)

    assert ensemble.energy_method == "LennardJones"
    symbols = [a.GetSymbol() for a in ensemble.mol.GetAtoms()]
    for conf_id in ensemble.conf_ids:
        positions = ensemble.mol.GetConformer(conf_id).GetPositions()
        # Single points: the geometry stays.
        assert np.array_equal(
            positions, before.mol.GetConformer(conf_id).GetPositions()
        )
        assert ensemble.energy(conf_id) == pytest.approx(_lj_energy(positions, symbols))
        provenance = ensemble.provenance(conf_id)
        assert provenance["previous_energy"] == before.energy(conf_id)
        assert provenance["previous_energy_method"] == "MMFFOptimizer"
        assert provenance["embedder"] == "CmapEmbedder"  # earlier entries are kept


class AlwaysFails(LennardJones):
    def calculate(self, *args, **kwargs):
        raise RuntimeError("SCF not converged")


@pytest.mark.parametrize("on_fail", ["clear", "drop"])
def test_rescore_failures(refined, on_fail, caplog):
    ensemble, ctx = refined
    first, second = ensemble.conf_ids[:2]
    # A factory: one calculator per conformer without workers; the second fails.
    calculators = iter([LennardJones(), AlwaysFails(), LennardJones(), LennardJones()])

    Rescore(lambda: next(calculators), method="LJ", on_fail=on_fail).run(ctx, ensemble)
    assert "SCF not converged" in caplog.text
    assert ensemble.energy(first) is not None
    if on_fail == "clear":
        assert ensemble.energy(second) is None
    else:
        assert second not in ensemble.conf_ids


def test_rescore_raises_if_every_conformer_fails(refined):
    ensemble, ctx = refined
    before = ensemble.copy()
    with pytest.raises(
        racerts.NoConformersError, match="all 4 conformers: RuntimeError: SCF"
    ):
        Rescore(AlwaysFails).run(ctx, ensemble)

    # The ensemble is as before: its energies still belong to its energy method.
    assert ensemble.energy_method == "MMFFOptimizer"
    assert ensemble.energies().tolist() == before.energies().tolist()
    assert "previous_energy" not in ensemble.provenance(ensemble.conf_ids[0])


def test_rescore_can_add_a_correction(refined):
    # e.g. a solvation term from a cheaper method, on top of the energies of the search
    ensemble, ctx = refined
    before = ensemble.energies()
    shifts = [0.01 * k for k in range(len(ensemble))]  # eV
    Rescore(batch=lambda structures: shifts, method="shift", add=True).run(
        ctx, ensemble
    )
    assert ensemble.energy_method == "MMFFOptimizer+shift"
    expected = before + np.array(shifts) * EV_TO_KCAL_MOL
    assert ensemble.energies() == pytest.approx(expected)
    provenance = ensemble.provenance(ensemble.conf_ids[1])
    assert provenance["previous_energy"] == pytest.approx(before[1])
    assert provenance["previous_energy_method"] == "MMFFOptimizer"
    assert provenance["energy_correction"] == pytest.approx(0.01 * EV_TO_KCAL_MOL)

    # A correction needs an energy to add to.
    bare = ensemble.copy()
    for conf in bare.mol.GetConformers():
        conf.ClearProp("energy")
    with pytest.raises(ValueError, match="no energy to add"):
        Rescore(batch=lambda structures: shifts, method="shift", add=True).run(
            ctx, bare
        )


def test_rescore_with_a_batch_function(refined):
    ensemble, ctx = refined
    seen = []

    def batch(structures):
        seen.append(len(structures))
        return [1.0, None, float("nan"), 2.0][: len(structures)]

    Rescore(batch=batch).run(ctx, ensemble)
    assert seen == [4]
    assert ensemble.energy_method == "batch"
    energies = ensemble.energies()
    assert energies[0] == pytest.approx(EV_TO_KCAL_MOL)
    assert np.isnan(energies[1]) and np.isnan(energies[2])

    with pytest.raises(ValueError, match="returned 1 energies for 4"):
        Rescore(batch=lambda structures: [0.0]).run(ctx, ensemble)
    with pytest.raises(ValueError, match="either"):
        Rescore()
    with pytest.raises(ValueError, match="on_fail"):
        Rescore(batch=batch, on_fail="ignore")


def test_the_prepare_hook(refined):
    # -- prepare gets the reference
    ensemble, ctx = refined
    reference = ctx.reference.GetConformer().GetPositions()

    calculator = NeedsPreparation()
    optimizer = ASEOptimizer(calculator, max_steps=2, prepare=prepare_lj)
    optimizer.refine(ensemble.mol, ctx.reference, ctx.frozen.hard)
    assert np.array_equal(calculator.prepared_on, reference)

    with pytest.raises(RuntimeError, match="not prepared"):
        ASEOptimizer(NeedsPreparation(), max_steps=2).refine(ensemble.mol)

    # -- prepare in worker processes
    ensemble, ctx = refined
    optimizer = ASEOptimizer(
        NeedsPreparation, max_steps=2, num_workers=2, prepare=prepare_lj
    )
    optimizer.refine(ensemble.mol, ctx.reference, ctx.frozen.hard)
    assert not np.isnan(ensemble.energies()).any()

    # -- prepare without reference gets the unrelaxed first conformer
    ensemble, _ = refined
    first = ensemble.mol.GetConformer(ensemble.conf_ids[0]).GetPositions()
    RecordingPrepare.seen = []
    ASEOptimizer(LennardJones, max_steps=3, prepare=RecordingPrepare.prepare).refine(
        ensemble.mol
    )
    assert len(RecordingPrepare.seen) == 4  # a factory: one calculator per conformer
    assert all(np.allclose(seen, first) for seen in RecordingPrepare.seen)


def test_convergence_is_recorded_and_unconverged_conformers_can_be_dropped(
    refined, caplog
):
    ensemble, ctx = refined
    kept = ensemble.copy()
    ASEOptimizer(LennardJones(), max_steps=1).refine(kept.mol)
    for conf_id in kept.conf_ids:
        provenance = kept.provenance(conf_id)
        assert provenance["converged"] is False and provenance["n_steps"] == 1
        assert provenance["wall_time"] >= 0

    with pytest.raises(
        racerts.NoConformersError, match="No conformer converged within 1 steps"
    ):
        ASEOptimizer(LennardJones(), max_steps=1, drop_unconverged=True).refine(
            ensemble.copy().mol
        )

    # Some converge: the others are dropped, with a warning.
    class ConvergesButTheSecond:
        runs, nsteps = 0, 1

        def __init__(self, atoms, **kwargs):
            pass

        def run(self, fmax, steps):
            type(self).runs += 1
            type(self).limits = (fmax, steps)
            return type(self).runs != 2

    partly = ensemble.copy()
    ASEOptimizer(
        LennardJones(), optimizer_cls=ConvergesButTheSecond, drop_unconverged=True
    ).refine(partly.mol)
    assert ConvergesButTheSecond.limits == (0.05, 100)  # eV/A and steps, the defaults
    assert partly.conf_ids == [i for k, i in enumerate(ensemble.conf_ids) if k != 1]
    assert "Dropping 1 conformers that did not converge" in caplog.text

    # Converged ones stay: a large fmax converges at once.
    converged = ensemble.copy()
    ASEOptimizer(LennardJones(), fmax=1e6, drop_unconverged=True).refine(converged.mol)
    assert converged.conf_ids == ensemble.conf_ids
    assert all(converged.provenance(i)["converged"] for i in converged.conf_ids)


def test_ase_refinement_ids_single_points_optimizers_and_failures(refined, caplog):
    from ase.optimize import FIRE

    # -- ase refinement keeps the ids
    ensemble, ctx = refined
    ensemble = ensemble.filter(ensemble.conf_ids[1:])
    ids = ensemble.conf_ids
    ASEOptimizer(LennardJones, max_steps=2, num_workers=2).refine(
        ensemble.mol, ctx.reference, ctx.frozen.hard
    )
    assert ensemble.conf_ids == ids

    # -- single points skip the optimizer
    ensemble, _ = refined

    class NoOptimizer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("no optimizer for single points")

    ASEOptimizer(LennardJones(), optimizer_cls=NoOptimizer, max_steps=0).optimize(
        ensemble.mol
    )
    assert not np.isnan(ensemble.energies()).any()

    # -- any ase style optimizer class
    # e.g. Sella; here FIRE with its own keyword arguments.

    ensemble, ctx = refined
    ASEOptimizer(
        LennardJones(), optimizer_cls=FIRE, optimizer_kwargs={"maxstep": 0.05}
    ).refine(ensemble.mol, ctx.reference, ctx.frozen.hard)
    assert not np.isnan(ensemble.energies()).any()
    mol = Chem.Mol(ensemble.mol)
    assert mol.GetNumConformers() == 4

    # -- a failing conformer in the process pool
    caplog.clear()
    ensemble, ctx = refined
    optimizer = MarkingOptimizer(FailOnMark, max_steps=2, num_workers=2)
    optimizer.fail_ids = (ensemble.conf_ids[1],)
    optimizer.refine(ensemble.mol, ctx.reference, ctx.frozen.hard)

    has_energy = [e is not None for e in map(ensemble.energy, ensemble.conf_ids)]
    assert has_energy == [True, False, True, True]
    assert "SCF not converged" in caplog.text


class CannotBeSent(RuntimeError):
    """An error that pickle cannot take to another process."""

    def __reduce__(self):
        raise TypeError("holds a handle")


class FailsUnsendably(LennardJones):
    def calculate(self, atoms=None, *args, **kwargs):
        if atoms is not None and atoms.info.get("fail"):
            raise CannotBeSent("the device is gone")
        super().calculate(atoms, *args, **kwargs)


CALLS_HERE = []


def counted_lj():
    """A calculator factory that counts its calls in the process it runs in."""
    CALLS_HERE.append(1)
    return LennardJones()


def test_which_errors_cost_a_conformer_and_when_workers_start(refined):
    start, ctx = refined
    ids = start.conf_ids

    def run(calculator=FailOnMark, **settings):
        ensemble = start.copy()
        optimizer = MarkingOptimizer(calculator, max_steps=2, **settings)
        optimizer.fail_ids = (ids[1],)
        optimizer.refine(ensemble.mol, ctx.reference, ctx.frozen.hard)
        return ensemble

    # -- the error of a failed conformer is in its provenance
    ensemble = run()
    assert ensemble.provenance(ids[1])["error"] == "RuntimeError: SCF not converged"
    assert "error" not in ensemble.provenance(ids[0])

    # -- expected_errors: only these cost a conformer; any other error ends the run,
    # also from a worker process
    for workers in (1, 2):
        ensemble = run(num_workers=workers, expected_errors=RuntimeError)
        assert ensemble.energy(ids[1]) is None and ensemble.energy(ids[0]) is not None
        with pytest.raises(RuntimeError, match="SCF not converged"):
            run(num_workers=workers, expected_errors=(ArithmeticError, OSError))
    # An error that cannot be sent from a worker arrives with its type and text.
    with pytest.raises(RuntimeError, match="CannotBeSent: the device is gone"):
        run(FailsUnsendably, num_workers=2, expected_errors=())
    with pytest.raises(CannotBeSent):
        run(FailsUnsendably, expected_errors=())

    # If every conformer fails with an expected error, nothing is left: that has a
    # type of its own, so that a caller can tell it from an error of the backend.
    def all_fail(ensemble):
        optimizer = MarkingOptimizer(FailOnMark, max_steps=2)
        optimizer.fail_ids = tuple(ids)
        optimizer.refine(ensemble.mol, ctx.reference, ctx.frozen.hard)

    with pytest.raises(racerts.NoConformersError, match="failed for all 4"):
        all_fail(start.copy())
    assert issubclass(racerts.NoConformersError, RuntimeError)  # as it was raised
    for wrong in ("RuntimeError", (RuntimeError, "x"), [RuntimeError], int):
        with pytest.raises(TypeError, match="expected_errors"):
            ASEOptimizer(LennardJones(), expected_errors=wrong)

    # -- serial_below: no worker processes for fewer conformers than this
    del CALLS_HERE[:]
    run(counted_lj, num_workers=2)
    assert not CALLS_HERE  # the calculators were made in the workers
    run(counted_lj, num_workers=2, serial_below=len(ids) + 1)
    assert len(CALLS_HERE) == len(ids)  # one per conformer, in this process
    del CALLS_HERE[:]
    run(counted_lj, num_workers=2, serial_below=len(ids))
    assert not CALLS_HERE
    with pytest.raises(ValueError, match="serial_below"):
        ASEOptimizer(LennardJones(), serial_below=-1)


class OneCall(MarkingOptimizer):
    """
    Stands for the relaxation of another package, which takes all structures in one
    call: _relax is the one step it replaces. Its "relaxation" moves every atom
    max_steps times a little along its Lennard-Jones force and never converges; it
    reports a structure marked to fail as failed.
    """

    def __init__(self, **settings):
        super().__init__(LennardJones(), method="one call", **settings)
        self.calls = []

    def _relax(self, tasks, config, reference=None, num_workers=1):
        self.calls.append((len(tasks), config.max_steps, len(config.restraints)))
        outcomes = []
        for conf_id, atoms in tasks:
            if atoms.info["fail"]:
                outcomes.append(Outcome(conf_id, error="RuntimeError: lost"))
                continue
            work = atoms.copy()  # with the frozen atoms as constraints
            work.calc = LennardJones()
            for _ in range(config.max_steps):
                work.set_positions(work.get_positions() + 1e-3 * work.get_forces())
            outcomes.append(
                Outcome(
                    conf_id,
                    work.get_positions(),
                    work.get_potential_energy(),
                    converged=config.max_steps == 0,
                    n_steps=config.max_steps,
                )
            )
        return outcomes


def test_a_subclass_can_replace_the_relaxation_step(refined):
    from racerts.restraints import DistanceRestraint

    start, ctx = refined
    ids = start.conf_ids
    hard = list(ctx.frozen.hard)
    symbols = [a.GetSymbol() for a in start.mol.GetAtoms()]

    # -- one call with all structures; what is around the step is the optimizer's:
    # the energies in kcal/mol, the provenance, the frozen atoms
    optimizer = OneCall(max_steps=3)
    ensemble = start.copy()
    optimizer.refine(ensemble.mol, ctx.reference, hard)
    assert optimizer.calls == [(4, 3, 0)]
    for conf_id in ids:
        before = start.mol.GetConformer(conf_id).GetPositions()
        after = ensemble.mol.GetConformer(conf_id).GetPositions()
        assert np.abs(after - before).max() > 1e-4
        np.testing.assert_allclose(after[hard], before[hard], atol=1e-6)
        assert ensemble.energy(conf_id) == pytest.approx(_lj_energy(after, symbols))
        provenance = ensemble.provenance(conf_id)
        assert provenance["converged"] is False and provenance["n_steps"] == 3

    # -- the restraints arrive with the settings, and single points as max_steps 0
    window = DistanceRestraint(0, 6, 2.0, 9.0)
    optimizer.refine(start.copy().mol, ctx.reference, hard, restraints=[window])
    points = start.copy()
    OneCall(max_steps=0).optimize(points.mol)
    assert optimizer.calls[-1] == (4, 3, 1)
    assert points.energies() == pytest.approx(
        [_lj_energy(start.mol.GetConformer(i).GetPositions(), symbols) for i in ids]
    )

    # -- a structure reported as failed costs its conformer; all of them, the run
    optimizer = OneCall(max_steps=3)
    optimizer.fail_ids = (ids[1],)
    ensemble = start.copy()
    optimizer.refine(ensemble.mol, ctx.reference, hard)
    has_energy = [e is not None for e in map(ensemble.energy, ids)]
    assert has_energy == [True, False, True, True]
    assert ensemble.provenance(ids[1])["error"] == "RuntimeError: lost"
    optimizer.fail_ids = tuple(ids)
    with pytest.raises(racerts.NoConformersError, match="failed for all 4"):
        optimizer.refine(start.copy().mol, ctx.reference, hard)

    # -- as a stage, and one Outcome for each structure, in their order
    staged = racerts.Refine(OneCall(max_steps=3)).run(ctx, start.copy())
    assert staged.energy_method == "one call" and staged.conf_ids == ids

    class LosesOne(OneCall):
        def _relax(self, tasks, config, reference=None, num_workers=1):
            return super()._relax(tasks, config)[1:]

    with pytest.raises(ValueError, match="one Outcome for each structure"):
        LosesOne(max_steps=1).refine(start.copy().mol, ctx.reference, hard)

    # -- a package that only runs the structures its own way, with calculators of
    # its own, takes the relaxation of one structure from here: optimize_one. The
    # restraints, the frozen atoms and a failed calculation are then as without it.
    class OwnCalculators(MarkingOptimizer):
        def _relax(self, tasks, config, reference=None, num_workers=1):
            done = {
                task[0]: optimize_one(FailOnMark(), config, task)
                for task in reversed(tasks)  # its own order
            }
            return [done[conf_id] for conf_id, _ in tasks]

    def relaxed(optimizer_cls):
        optimizer = optimizer_cls(FailOnMark(), max_steps=3)
        optimizer.fail_ids = (ids[2],)
        ensemble = start.copy()
        optimizer.refine(ensemble.mol, ctx.reference, hard, restraints=[window])
        return ensemble

    own, plain = relaxed(OwnCalculators), relaxed(MarkingOptimizer)
    assert own.energy(ids[2]) is None and plain.energy(ids[2]) is None
    assert own.provenance(ids[2])["error"] == "RuntimeError: SCF not converged"
    for conf_id in (ids[0], ids[1], ids[3]):
        assert own.energy(conf_id) == pytest.approx(plain.energy(conf_id), abs=1e-9)
        np.testing.assert_allclose(
            own.mol.GetConformer(conf_id).GetPositions(),
            plain.mol.GetConformer(conf_id).GetPositions(),
            atol=1e-9,
        )


class RecordingPrepare:
    """Records the positions prepare gets (serial runs only)."""

    seen = []

    @classmethod
    def prepare(cls, calculator, atoms):
        cls.seen.append(atoms.get_positions().copy())


def test_an_ensemble_gives_ase_atoms():
    # ConformerEnsemble.to_ase: the conformer with its charge and multiplicity where
    # ASE calculators read them.
    config = racerts.PipelineConfig(embed={"n_conformers": 2})
    ensemble = racerts.generate_gs("C[NH3+]", config=config)
    conf_id = ensemble.best()
    atoms = ensemble.to_ase(conf_id)
    assert atoms.get_chemical_symbols() == [
        a.GetSymbol() for a in ensemble.mol.GetAtoms()
    ]
    np.testing.assert_allclose(
        atoms.get_positions(), ensemble.mol.GetConformer(conf_id).GetPositions()
    )
    assert (atoms.info["charge"], atoms.info["multiplicity"]) == (1, 1)
    assert atoms.get_initial_charges().sum() == pytest.approx(1.0)
