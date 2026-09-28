import importlib
import logging

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from racerts import ConformerGenerator
from racerts.optimizer import ASEOptimizer, optimizers
from racerts.optimizer.ase import (
    infer_charge_and_multiplicity,
    rdkit_conformer_to_ase_atoms,
    write_ase_positions_to_rdkit,
)

try:
    LennardJones = importlib.import_module("ase.calculators.lj").LennardJones
except Exception:
    LennardJones = None

pytestmark = [
    pytest.mark.ase,
    pytest.mark.skipif(LennardJones is None, reason="ASE is not importable."),
]


def _build_test_mol(num_confs=3):
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    params = AllChem.ETKDGv3()
    params.randomSeed = 12
    AllChem.EmbedMultipleConfs(mol, numConfs=num_confs, params=params)
    return mol


def _reference_from_conf(mol: Chem.Mol, conf_id: int = 0) -> Chem.Mol:
    reference = Chem.Mol(mol)
    reference.RemoveAllConformers()
    reference.AddConformer(Chem.Conformer(mol.GetConformer(conf_id)), assignId=True)
    return reference


def test_rdkit_ase_roundtrip_positions():
    mol = _build_test_mol(num_confs=1)
    conf_id = mol.GetConformer().GetId()
    atoms = rdkit_conformer_to_ase_atoms(mol, conf_id=conf_id)

    positions = atoms.get_positions()
    positions[0] = positions[0] + np.array([0.1, -0.2, 0.3])
    atoms.set_positions(positions)

    write_ase_positions_to_rdkit(atoms, mol=mol, conf_id=conf_id)
    updated_positions = mol.GetConformer(conf_id).GetPositions()

    assert np.allclose(updated_positions, positions)


def test_ase_optimizer_with_calculator_instance_sets_energies():
    mol = _build_test_mol(num_confs=3)
    reference = _reference_from_conf(mol, conf_id=0)

    optimizer = ASEOptimizer(
        calculator=LennardJones(),
        num_workers=1,
        fmax=0.1,
        max_steps=10,
    )
    optimizer.tune_ts_conformers(mol=mol, reference=reference, align_indices=[0, 1])

    for conf in mol.GetConformers():
        assert conf.HasProp("energy")
        assert np.isfinite(conf.GetDoubleProp("energy"))


def test_ase_optimizer_with_calculator_class_sets_energies():
    mol = _build_test_mol(num_confs=3)
    reference = _reference_from_conf(mol, conf_id=0)

    optimizer = ASEOptimizer(
        calculator=LennardJones,
        num_workers=1,
        fmax=0.1,
        max_steps=10,
    )
    optimizer.tune_ts_conformers(mol=mol, reference=reference, align_indices=[0, 1])

    for conf in mol.GetConformers():
        assert conf.HasProp("energy")
        assert np.isfinite(conf.GetDoubleProp("energy"))


def test_ase_optimizer_with_calculator_callable_sets_energies():
    mol = _build_test_mol(num_confs=4)
    reference = _reference_from_conf(mol, conf_id=0)

    optimizer = ASEOptimizer(
        calculator=lambda: LennardJones(),
        num_workers=1,
        fmax=0.1,
        max_steps=10,
    )
    optimizer.tune_ts_conformers(mol=mol, reference=reference, align_indices=[0, 1])

    for conf in mol.GetConformers():
        assert conf.HasProp("energy")
        assert np.isfinite(conf.GetDoubleProp("energy"))


def _aligned(mol: Chem.Mol, reference: Chem.Mol, align_indices) -> Chem.Mol:
    """
    The conformers after the alignment of tune_ts_conformers alone. The aligned atoms
    cannot all match the reference, since each embedded conformer has its own C-C
    distance (up to 0.03 A apart, depending on the RDKit version).
    """
    aligned = Chem.Mol(mol)
    ASEOptimizer(calculator=LennardJones()).align_mols(
        aligned, reference, align_indices
    )
    return aligned


def _assert_aligned_atoms_did_not_move(mol, aligned, align_indices):
    for conf, start in zip(mol.GetConformers(), aligned.GetConformers()):
        assert np.allclose(
            conf.GetPositions()[align_indices],
            start.GetPositions()[align_indices],
            atol=1e-6,
        )


def test_ase_optimizer_keeps_align_indices_fixed():
    mol = _build_test_mol(num_confs=2)
    reference = _reference_from_conf(mol, conf_id=0)
    align_indices = [0, 1]
    aligned = _aligned(mol, reference, align_indices)

    optimizer = ASEOptimizer(
        calculator=LennardJones(),
        num_workers=1,
        fmax=0.1,
        max_steps=10,
    )
    optimizer.tune_ts_conformers(
        mol=mol,
        reference=reference,
        align_indices=align_indices,
    )

    _assert_aligned_atoms_did_not_move(mol, aligned, align_indices)


def test_ase_optimizer_external_align_and_optimize():
    fix_atoms = importlib.import_module("ase.constraints").FixAtoms

    mol = _build_test_mol(num_confs=2)
    reference = _reference_from_conf(mol, conf_id=0)
    optimizer = ASEOptimizer(
        calculator=LennardJones(),
        num_workers=1,
        fmax=0.1,
        max_steps=10,
    )

    optimizer.conf_id_ref = reference.GetConformer().GetId()
    optimizer.align_mols(mol=mol, reference=reference, align_indices=[0, 1])

    failures = optimizer.optimize(mol=mol, constraints=[fix_atoms(indices=[0, 1])])
    assert isinstance(failures, int)

    for conf in mol.GetConformers():
        assert conf.HasProp("energy")


def test_ase_optimizer_constructor_validation():
    with pytest.raises(ValueError, match="must be provided"):
        ASEOptimizer()

    with pytest.raises(ValueError, match="instance or a callable"):
        ASEOptimizer(calculator=object())


def test_optimizer_registry_includes_ase():
    assert "ase" in optimizers
    assert optimizers["ase"] is ASEOptimizer


def test_neutral_singlet_charge_and_multiplicity_on_atoms():
    mol = _build_test_mol(num_confs=1)
    conf_id = mol.GetConformer().GetId()
    atoms = rdkit_conformer_to_ase_atoms(mol, conf_id=conf_id)

    assert atoms.info["charge"] == 0
    assert atoms.info["spin"] == 1
    assert atoms.info["multiplicity"] == 1


def test_charged_radical_charge_and_multiplicity_on_atoms():
    # Methylammonium radical cation: +1 formal charge, 17 electrons -> doublet.
    mol = Chem.AddHs(Chem.MolFromSmiles("[CH2][NH3+]"))
    AllChem.EmbedMolecule(mol, randomSeed=12)

    inferred = infer_charge_and_multiplicity(mol)
    assert inferred == {"charge": 1, "multiplicity": 2}

    atoms = rdkit_conformer_to_ase_atoms(mol, conf_id=mol.GetConformer().GetId())
    assert atoms.info["charge"] == 1
    assert atoms.info["spin"] == 2
    assert atoms.info["multiplicity"] == 2
    # Calculators such as tblite read the totals from initial charges and magmoms.
    assert atoms.get_initial_charges().sum() == pytest.approx(1)
    assert atoms.get_initial_magnetic_moments().sum() == pytest.approx(1)


def test_given_charge_is_used_without_warnings(caplog):
    mol = _build_test_mol(num_confs=3)  # neutral closed-shell ethanol

    with caplog.at_level(logging.WARNING):
        ASEOptimizer(calculator=LennardJones(), charge=-1, max_steps=1).optimize(mol)

    assert not caplog.records  # checked once, in generate_conformers


def test_charge_and_multiplicity_overrides():
    mol = _build_test_mol(num_confs=1)
    atoms = rdkit_conformer_to_ase_atoms(
        mol, conf_id=mol.GetConformer().GetId(), charge=-1, multiplicity=3
    )
    assert atoms.info["charge"] == -1
    assert atoms.info["spin"] == 3
    assert atoms.info["multiplicity"] == 3
    assert atoms.get_initial_charges().sum() == pytest.approx(-1)
    assert atoms.get_initial_magnetic_moments().sum() == pytest.approx(2)


def test_ase_optimizer_charge_and_multiplicity_overrides():
    seen = []

    class RecordingCalculator(LennardJones):
        def calculate(self, atoms=None, *args, **kwargs):
            seen.append((atoms.info["charge"], atoms.info["multiplicity"]))
            super().calculate(atoms, *args, **kwargs)

    optimizer = ASEOptimizer(
        calculator=RecordingCalculator(), charge=-1, multiplicity=3, max_steps=1
    )
    optimizer.optimize(_build_test_mol(num_confs=1))

    assert seen and set(seen) == {(-1, 3)}


def _failing_calculator():
    class FailingCalculator(LennardJones):
        def calculate(self, *args, **kwargs):
            raise RuntimeError("SCF not converged")

    return FailingCalculator()


def test_ase_optimizer_failed_conformer_keeps_no_energy(caplog):
    mol = _build_test_mol(num_confs=3)
    for conf in mol.GetConformers():
        conf.SetDoubleProp("energy", 0.0)  # stale energy from an earlier step
    calculators = iter([LennardJones(), _failing_calculator(), LennardJones()])
    optimizer = ASEOptimizer(
        calculator=lambda: next(calculators), num_workers=1, fmax=0.1, max_steps=5
    )

    with caplog.at_level(logging.WARNING):
        optimizer.optimize(mol)

    assert [conf.HasProp("energy") for conf in mol.GetConformers()] == [
        True,
        False,
        True,
    ]
    assert "SCF not converged" in caplog.text


def test_ase_optimizer_raises_if_all_conformers_fail():
    optimizer = ASEOptimizer(calculator=_failing_calculator, num_workers=1)

    with pytest.raises(RuntimeError, match="SCF not converged"):
        optimizer.optimize(_build_test_mol(num_confs=2))


def test_written_ensemble_is_read_by_ase(tmp_path):
    read = importlib.import_module("ase.io").read
    mol = _build_test_mol(num_confs=2)
    mol.GetConformer(0).SetDoubleProp("energy", 23.06054783061903)  # 1 eV
    mol.SetIntProp("charge", -1)
    mol.SetIntProp("multiplicity", 2)
    cg = ConformerGenerator()
    cg.mol = mol

    cg.write_xyz(str(tmp_path / "out.xyz"))
    frames = read(str(tmp_path / "out.xyz"), index=":")

    assert len(frames) == 2
    assert all(f.info["charge"] == -1 and f.info["spin"] == 2 for f in frames)
    assert frames[0].get_potential_energy() == pytest.approx(1.0)
    assert np.allclose(
        frames[1].get_positions(), mol.GetConformer(1).GetPositions(), atol=1e-5
    )


def test_ase_optimizer_num_workers_sets_energies():
    mol = _build_test_mol(num_confs=4)
    reference = _reference_from_conf(mol, conf_id=0)

    # Use the calculator class as a (picklable) factory so the spawn pool can build a
    # fresh calculator per worker.
    optimizer = ASEOptimizer(
        calculator=LennardJones,
        num_workers=2,
        fmax=0.1,
        max_steps=10,
    )
    optimizer.tune_ts_conformers(mol=mol, reference=reference, align_indices=[0, 1])

    for conf in mol.GetConformers():
        assert conf.HasProp("energy")
        assert np.isfinite(conf.GetDoubleProp("energy"))


def test_ase_optimizer_num_workers_none_sets_energies():
    # num_workers=None enables automatic worker selection.
    mol = _build_test_mol(num_confs=4)
    reference = _reference_from_conf(mol, conf_id=0)

    optimizer = ASEOptimizer(
        calculator=LennardJones,
        num_workers=None,
        fmax=0.1,
        max_steps=10,
    )
    optimizer.tune_ts_conformers(mol=mol, reference=reference, align_indices=[0, 1])

    for conf in mol.GetConformers():
        assert conf.HasProp("energy")
        assert np.isfinite(conf.GetDoubleProp("energy"))


def test_ase_optimizer_num_workers_keeps_align_indices_fixed():
    mol = _build_test_mol(num_confs=4)
    reference = _reference_from_conf(mol, conf_id=0)
    align_indices = [0, 1]
    aligned = _aligned(mol, reference, align_indices)

    optimizer = ASEOptimizer(
        calculator=LennardJones,
        num_workers=2,
        fmax=0.1,
        max_steps=10,
    )
    optimizer.tune_ts_conformers(
        mol=mol, reference=reference, align_indices=align_indices
    )

    # The constraint also holds in the worker processes.
    _assert_aligned_atoms_did_not_move(mol, aligned, align_indices)


def test_ase_optimizer_num_workers_matches_serial():
    reference = _reference_from_conf(_build_test_mol(num_confs=1), conf_id=0)

    serial_mol = _build_test_mol(num_confs=4)
    parallel_mol = Chem.Mol(serial_mol)

    serial = ASEOptimizer(
        calculator=LennardJones(), num_workers=1, fmax=0.1, max_steps=10
    )
    serial.tune_ts_conformers(mol=serial_mol, reference=reference, align_indices=[0, 1])

    parallel = ASEOptimizer(
        calculator=LennardJones, num_workers=2, fmax=0.1, max_steps=10
    )
    parallel.tune_ts_conformers(
        mol=parallel_mol, reference=reference, align_indices=[0, 1]
    )

    for serial_conf, parallel_conf in zip(
        serial_mol.GetConformers(), parallel_mol.GetConformers()
    ):
        assert np.allclose(
            serial_conf.GetPositions(), parallel_conf.GetPositions(), atol=1e-3
        )
