"""Restraints in ASE refinement: the flat-bottom calculator wrapper and the refiner."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import racerts
from racerts import TransitionState
from racerts.restraints import DistanceRestraint, PositionRestraint
from racerts.system import build_mol
from racerts.utils.units import EV_TO_KCAL_MOL

pytestmark = pytest.mark.ase
pytest.importorskip("ase")
from ase import Atoms  # noqa: E402
from ase.calculators.calculator import Calculator, all_changes  # noqa: E402
from ase.calculators.lj import LennardJones  # noqa: E402

from racerts.refine import ASEOptimizer  # noqa: E402
from racerts.refine.restrained import RestrainedCalculator  # noqa: E402


class Zero(Calculator):
    """No energy and no forces: only the restraint terms remain."""

    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {"energy": 0.0, "forces": np.zeros((len(self.atoms), 3))}


def _atoms(positions):
    return Atoms("H" * len(positions), positions=positions)


def _energy(positions, restraints, base=Zero):
    atoms = _atoms(positions)
    atoms.calc = RestrainedCalculator(base(), restraints)
    return atoms.get_potential_energy()


@pytest.mark.parametrize("d, excess", [(1.0, 0.5), (2.0, 0.0), (3.5, 0.5)])
def test_a_window_is_flat_bottom(d, excess):
    # [1.5, 3.0] with k = 20 kcal/(mol A^2): 1/2 k (d - bound)^2 outside, 0 inside
    window = DistanceRestraint(0, 1, 1.5, 3.0, force_constant=20.0)
    energy = _energy([[0, 0, 0], [d, 0, 0]], [window])
    assert energy == pytest.approx(0.5 * 20.0 * excess**2 / EV_TO_KCAL_MOL)


def test_the_restraint_terms():
    # -- a position restraint holds beyond its tolerance
    held = PositionRestraint(0, (0.0, 0.0, 0.0), tolerance=0.3, force_constant=5.0)
    assert _energy([[0.2, 0, 0]], [held]) == 0.0
    assert _energy([[0.5, 0, 0]], [held]) == pytest.approx(
        0.5 * 5.0 * 0.2**2 / EV_TO_KCAL_MOL
    )

    # -- forces are the negative gradient
    rng = np.random.default_rng(0)
    positions = rng.normal(size=(4, 3)) * 1.5
    restraints = [
        DistanceRestraint(0, 1, 0.2, 0.4, force_constant=30.0),  # stretched
        DistanceRestraint(1, 2, 6.0, 8.0, force_constant=10.0),  # compressed
        PositionRestraint(3, (0.5, -0.2, 0.1), tolerance=0.1, force_constant=5.0),
    ]
    atoms = _atoms(positions)
    atoms.calc = RestrainedCalculator(Zero(), restraints)
    forces = atoms.get_forces()
    assert np.abs(forces).max() > 0.1  # the terms are active
    numeric = np.zeros_like(forces)
    h = 1e-5
    for i in range(len(positions)):
        for k in range(3):
            shifted = positions.copy()
            shifted[i, k] += h
            up = _energy(shifted, restraints)
            shifted[i, k] -= 2 * h
            numeric[i, k] = -(up - _energy(shifted, restraints)) / (2 * h)
    np.testing.assert_allclose(forces, numeric, atol=1e-6)

    # -- the wrapped calculator adds its terms
    positions = [[0, 0, 0], [1.1, 0, 0], [0, 1.2, 0]]
    window = DistanceRestraint(0, 1, 2.0, 3.0, force_constant=20.0)
    plain = _atoms(positions)
    plain.calc = LennardJones()
    wrapped = _atoms(positions)
    wrapped.calc = RestrainedCalculator(LennardJones(), [window])
    extra = 0.5 * 20.0 * 0.9**2 / EV_TO_KCAL_MOL
    assert wrapped.get_potential_energy() == pytest.approx(
        plain.get_potential_energy() + extra
    )
    assert wrapped.calc.results["restraint_energy"] == pytest.approx(extra)
    difference = wrapped.get_forces() - plain.get_forces()
    assert np.allclose(difference[2], 0) and np.allclose(difference.sum(axis=0), 0)

    # -- other restraint kinds raise
    with pytest.raises(TypeError, match="ASE"):
        RestrainedCalculator(Zero(), ["not a restraint"])


# ---- ASEOptimizer with restraints


def _diol(n=3, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles("OCCCCO"))  # O0 ... O5
    AllChem.EmbedMultipleConfs(mol, n, randomSeed=seed)
    return mol


def _o_o(mol, conf_id):
    p = mol.GetConformer(conf_id).GetPositions()
    return float(np.linalg.norm(p[0] - p[5]))


def test_refinement_with_restraints_in_worker_processes():
    # The restraints reach the workers; results equal the serial run.
    window = DistanceRestraint(0, 5, 2.6, 3.0, force_constant=20.0)
    results = []
    for workers in (1, 2):
        mol = _diol()
        ASEOptimizer(LennardJones(), max_steps=20, num_workers=workers).refine(
            mol, restraints=[window]
        )
        results.append([c.GetPositions() for c in mol.GetConformers()])
    np.testing.assert_allclose(results[0], results[1], atol=1e-8)


@pytest.mark.xtb
def test_refinement_holds_a_window_with_restraint_free_energies():
    # k = 200: at the default 20, GFN2's torsions hold a conformer of another basin
    # up to ~0.4 A outside (a soft preference, as with MMFF).
    TBLite = pytest.importorskip("tblite.ase").TBLite
    mol = _diol()
    window = DistanceRestraint(0, 5, 2.6, 3.0, force_constant=200.0)
    ASEOptimizer(TBLite(method="GFN2-xTB", verbosity=0), max_steps=300).refine(
        mol, restraints=[window]
    )
    single_point = TBLite(method="GFN2-xTB", verbosity=0)
    ensemble = racerts.ConformerEnsemble(mol)
    for conf in mol.GetConformers():
        assert ensemble.provenance(conf.GetId())["converged"]
        assert 2.6 - 0.1 < _o_o(mol, conf.GetId()) < 3.0 + 0.1
        atoms = Atoms(
            [a.GetSymbol() for a in mol.GetAtoms()], positions=conf.GetPositions()
        )
        atoms.calc = single_point
        expected = atoms.get_potential_energy() * EV_TO_KCAL_MOL
        assert conf.GetDoubleProp("energy") == pytest.approx(expected, abs=1e-6)


@pytest.mark.xtb
def test_soft_atoms_without_anchors_stay_near_the_reference():
    # Without anchors the conformers are aligned on the soft atoms, as for MMFF.
    TBLite = pytest.importorskip("tblite.ase").TBLite
    mol = _diol()
    reference = Chem.Mol(mol, confId=0)
    points = reference.GetConformer().GetPositions()
    soft = [PositionRestraint(i, tuple(points[i]), tolerance=0.1) for i in (0, 1, 2)]
    optimizer = ASEOptimizer(TBLite(method="GFN2-xTB", verbosity=0), max_steps=300)
    optimizer.refine(mol, reference, restraints=soft)
    for conf in mol.GetConformers():
        p = conf.GetPositions()
        assert max(np.linalg.norm(p[r.atom] - r.point) for r in soft) < 0.1 + 0.1


def test_window_mode_takes_ase_refiners(sn2_ts):
    mol = build_mol(sn2_ts, -1, [0, 1, 2], input_smiles=["CCl", "[Cl-]"])
    task = TransitionState([0, 1, 2], active_bonds=[(0, 1), (0, 2)], active_window=0.2)
    ctx = racerts.Context.create(mol, task)
    ensemble = racerts.Embed(n_conformers=2).run(ctx)
    racerts.Refine(ASEOptimizer(LennardJones(), max_steps=2)).run(ctx, ensemble)


@pytest.mark.xtb
def test_window_holds_with_gfn2(sn2_ts):
    TBLite = pytest.importorskip("tblite.ase").TBLite
    mol = build_mol(sn2_ts, -1, [0, 1, 2], input_smiles=["CCl", "[Cl-]"])
    task = TransitionState([0, 1, 2], active_bonds=[(0, 1), (0, 2)], active_window=0.2)
    ctx = racerts.Context.create(mol, task)
    ensemble = racerts.Embed(n_conformers=3).run(ctx)
    gfn2 = ASEOptimizer(TBLite(method="GFN2-xTB", verbosity=0), max_steps=300)
    ensemble = racerts.Refine(gfn2).run(ctx, ensemble)
    assert len(ensemble) > 0
    for conf_id in ensemble.conf_ids:
        provenance = ensemble.provenance(conf_id)
        assert provenance["converged"]
        targets = provenance["active_bond_targets"]
        lengths = provenance["active_bond_lengths"]
        for pair, target in targets.items():
            assert abs(lengths[pair] - target) < 0.03
