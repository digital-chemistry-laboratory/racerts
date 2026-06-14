import importlib

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

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
        num_threads=1,
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
        num_threads=1,
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
        num_threads=2,
        fmax=0.1,
        max_steps=10,
    )
    optimizer.tune_ts_conformers(mol=mol, reference=reference, align_indices=[0, 1])

    for conf in mol.GetConformers():
        assert conf.HasProp("energy")
        assert np.isfinite(conf.GetDoubleProp("energy"))


def test_ase_optimizer_keeps_align_indices_fixed():
    mol = _build_test_mol(num_confs=2)
    reference = _reference_from_conf(mol, conf_id=0)
    align_indices = [0, 1]

    optimizer = ASEOptimizer(
        calculator=LennardJones(),
        num_threads=1,
        fmax=0.1,
        max_steps=10,
    )
    optimizer.tune_ts_conformers(
        mol=mol,
        reference=reference,
        align_indices=align_indices,
    )

    reference_positions = reference.GetConformer().GetPositions()
    for conf in mol.GetConformers():
        positions = conf.GetPositions()
        assert np.allclose(
            positions[align_indices],
            reference_positions[align_indices],
            atol=1e-2,
        )


def test_ase_optimizer_external_align_and_optimize():
    fix_atoms = importlib.import_module("ase.constraints").FixAtoms

    mol = _build_test_mol(num_confs=2)
    reference = _reference_from_conf(mol, conf_id=0)
    optimizer = ASEOptimizer(
        calculator=LennardJones(),
        num_threads=1,
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
    # Methylammonium radical cation: +1 formal charge, one radical electron -> doublet.
    mol = Chem.AddHs(Chem.MolFromSmiles("[CH2][NH3+]"))
    AllChem.EmbedMolecule(mol, randomSeed=12)

    inferred = infer_charge_and_multiplicity(mol)
    assert inferred == {"charge": 1, "multiplicity": 2}

    atoms = rdkit_conformer_to_ase_atoms(mol, conf_id=mol.GetConformer().GetId())
    assert atoms.info["charge"] == 1
    assert atoms.info["spin"] == 2
    assert atoms.info["multiplicity"] == 2


def test_charge_and_multiplicity_overrides():
    mol = _build_test_mol(num_confs=1)
    atoms = rdkit_conformer_to_ase_atoms(
        mol, conf_id=mol.GetConformer().GetId(), charge=-1, multiplicity=3
    )
    assert atoms.info["charge"] == -1
    assert atoms.info["spin"] == 3
    assert atoms.info["multiplicity"] == 3


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


def test_ase_optimizer_num_workers_none_uses_all_cores():
    # num_workers=None means "auto-size to all available cores" (the relax_conformers
    # convention); it must take the process path and produce finite energies.
    mol = _build_test_mol(num_confs=4)
    reference = _reference_from_conf(mol, conf_id=0)

    optimizer = ASEOptimizer(
        calculator=LennardJones,
        num_workers=None,
        fmax=0.1,
        max_steps=10,
    )
    assert optimizer._use_processes() is True
    optimizer.tune_ts_conformers(mol=mol, reference=reference, align_indices=[0, 1])

    for conf in mol.GetConformers():
        assert conf.HasProp("energy")
        assert np.isfinite(conf.GetDoubleProp("energy"))


def test_ase_optimizer_num_workers_keeps_align_indices_fixed():
    mol = _build_test_mol(num_confs=4)
    reference = _reference_from_conf(mol, conf_id=0)
    align_indices = [0, 1]

    optimizer = ASEOptimizer(
        calculator=LennardJones,
        num_workers=2,
        fmax=0.1,
        max_steps=10,
    )
    optimizer.tune_ts_conformers(
        mol=mol, reference=reference, align_indices=align_indices
    )

    reference_positions = reference.GetConformer().GetPositions()
    for conf in mol.GetConformers():
        positions = conf.GetPositions()
        assert np.allclose(
            positions[align_indices],
            reference_positions[align_indices],
            atol=1e-2,
        )


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
