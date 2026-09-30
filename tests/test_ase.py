"""ASE refinement: hooks, failures, convergence and provenance."""

import numpy as np
import pytest
from rdkit import Chem

import racerts
from racerts import TransitionState

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
