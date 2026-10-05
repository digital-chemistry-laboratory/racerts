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

from racerts.refine import ASEOptimizer  # noqa: E402


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
    with pytest.raises(RuntimeError, match="all 4 conformers: RuntimeError: SCF"):
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


def test_prepare_gets_the_reference(refined):
    ensemble, ctx = refined
    reference = ctx.reference.GetConformer().GetPositions()

    calculator = NeedsPreparation()
    optimizer = ASEOptimizer(calculator, max_steps=2, prepare=prepare_lj)
    optimizer.refine(ensemble.mol, ctx.reference, ctx.frozen.hard)
    assert np.array_equal(calculator.prepared_on, reference)

    with pytest.raises(RuntimeError, match="not prepared"):
        ASEOptimizer(NeedsPreparation(), max_steps=2).refine(ensemble.mol)


def test_prepare_in_worker_processes(refined):
    ensemble, ctx = refined
    optimizer = ASEOptimizer(
        NeedsPreparation, max_steps=2, num_workers=2, prepare=prepare_lj
    )
    optimizer.refine(ensemble.mol, ctx.reference, ctx.frozen.hard)
    assert not np.isnan(ensemble.energies()).any()


def test_a_failing_conformer_in_the_process_pool(refined, caplog):
    ensemble, ctx = refined
    optimizer = MarkingOptimizer(FailOnMark, max_steps=2, num_workers=2)
    optimizer.fail_ids = (ensemble.conf_ids[1],)
    optimizer.refine(ensemble.mol, ctx.reference, ctx.frozen.hard)

    has_energy = [e is not None for e in map(ensemble.energy, ensemble.conf_ids)]
    assert has_energy == [True, False, True, True]
    assert "SCF not converged" in caplog.text


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

    with pytest.raises(RuntimeError, match="No conformer converged within 1 steps"):
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


def test_ase_refinement_keeps_the_ids(refined):
    ensemble, ctx = refined
    ensemble = ensemble.filter(ensemble.conf_ids[1:])
    ids = ensemble.conf_ids
    ASEOptimizer(LennardJones, max_steps=2, num_workers=2).refine(
        ensemble.mol, ctx.reference, ctx.frozen.hard
    )
    assert ensemble.conf_ids == ids


def test_single_points_skip_the_optimizer(refined):
    ensemble, _ = refined

    class NoOptimizer:
        def __init__(self, *args, **kwargs):
            raise AssertionError("no optimizer for single points")

    ASEOptimizer(LennardJones(), optimizer_cls=NoOptimizer, max_steps=0).optimize(
        ensemble.mol
    )
    assert not np.isnan(ensemble.energies()).any()


def test_any_ase_style_optimizer_class(refined):
    # e.g. Sella; here FIRE with its own keyword arguments.
    from ase.optimize import FIRE

    ensemble, ctx = refined
    ASEOptimizer(
        LennardJones(), optimizer_cls=FIRE, optimizer_kwargs={"maxstep": 0.05}
    ).refine(ensemble.mol, ctx.reference, ctx.frozen.hard)
    assert not np.isnan(ensemble.energies()).any()
    mol = Chem.Mol(ensemble.mol)
    assert mol.GetNumConformers() == 4


class RecordingPrepare:
    """Records the positions prepare gets (serial runs only)."""

    seen = []

    @classmethod
    def prepare(cls, calculator, atoms):
        cls.seen.append(atoms.get_positions().copy())


def test_prepare_without_reference_gets_the_unrelaxed_first_conformer(refined):
    ensemble, _ = refined
    first = ensemble.mol.GetConformer(ensemble.conf_ids[0]).GetPositions()
    RecordingPrepare.seen = []
    ASEOptimizer(LennardJones, max_steps=3, prepare=RecordingPrepare.prepare).refine(
        ensemble.mol
    )
    assert len(RecordingPrepare.seen) == 4  # a factory: one calculator per conformer
    assert all(np.allclose(seen, first) for seen in RecordingPrepare.seen)
