import logging
import os
import shlex

import numpy as np
import pytest  # noqa
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, Descriptors

from racerts import ConformerGenerator
from racerts.embedder import BoundsMatrixEmbedder
from racerts.mol_getter import MolGetterBonds, MolGetterConnectivity, MolGetterSMILES
from racerts.optimizer import UFFOptimizer
from racerts.utils import (
    atom_idx_input_validation,
    get_frozen_atoms,
    infer_charge_and_multiplicity,
)

DATA = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
filenames = [
    os.path.join(DATA, "ex.xyz"),
    os.path.join(DATA, "ex.mol"),
]
for filename in filenames:
    if not os.path.isfile(filename):
        raise FileNotFoundError(
            f"File {filename} not found. Please make sure the test data is available."
        )

charge = 0
reacting_atoms = [3, 4, 5]
input_smiles = ["CCCCCC=C"]


def _test_default_getter(cg):
    for filename in filenames:
        # determine bonds
        mol = cg.get_mol(filename, charge, reacting_atoms, input_smiles=input_smiles)
        assert isinstance(mol, Chem.rdchem.Mol)
        assert isinstance(cg.mol_getter, MolGetterSMILES)

        # determine bonds
        mol1 = cg.get_mol(filename, charge, reacting_atoms)
        assert isinstance(mol1, Chem.rdchem.Mol)
        assert isinstance(
            cg.mol_getter, MolGetterSMILES
        )  # no change in default setting

        # determine connectivity
        if filename.endswith(".xyz"):
            mol2 = cg.get_mol(filename, charge + 5, reacting_atoms)
            assert isinstance(mol2, Chem.rdchem.Mol)
            assert False not in [
                b.GetBondType() == Chem.rdchem.BondType.SINGLE
                for b in list(mol2.GetBonds())
            ]  # only single bonds
            assert isinstance(
                cg.mol_getter, MolGetterSMILES
            )  # no change in default setting
        if filename.endswith(".mol") or filename.endswith(".sdf"):
            with pytest.raises(ValueError):
                cg.get_mol(
                    filename, charge + 5, reacting_atoms, input_smiles=input_smiles
                )
            assert isinstance(cg.mol_getter, MolGetterSMILES)
    return True


def _test_conformer_generator(cg):
    for filename in filenames:
        # determine bonds
        mol = cg.get_mol(filename, charge, reacting_atoms, input_smiles=input_smiles)
        assert isinstance(mol, Chem.rdchem.Mol)

        # checks if the idx are valid
        assert atom_idx_input_validation(mol, [mol.GetNumAtoms() - 1]) is True
        assert atom_idx_input_validation(mol, [mol.GetNumAtoms()]) is False
        try:
            cg.generate_conformers(
                filename, charge, reacting_atoms=[100]
            )  # should throw value error
            assert False
        except Exception as e:
            assert isinstance(e, ValueError)

        # finds neighbors to be frozen if not provided
        assert get_frozen_atoms(mol, reacting_atoms=[]) == []
        assert len(get_frozen_atoms(mol, reacting_atoms=[1])) != 0
        frozen = [1, 100, 1000]
        assert frozen == get_frozen_atoms(
            mol, reacting_atoms=[1], frozen_atoms=frozen
        )  # provided frozen atoms are not overwritten!

        # Embed
        n = 5
        m = mol
        new_m = Chem.Mol(m)
        new_m.RemoveAllConformers()

        valid = cg.embed_TS(
            mol_ts=m,
            new_mol=new_m,
            reacting_atoms=reacting_atoms,
            frozen_atoms=get_frozen_atoms(mol, reacting_atoms),
            number_of_conformers=n,
            conf_factor=1,
        )
        assert isinstance(valid, Chem.rdchem.Mol)
        assert m.GetNumConformers() == 1
        assert valid.GetNumConformers() == n
        assert valid == new_m

        new_m = Chem.Mol(m)
        new_m.RemoveAllConformers()

        all_constrained = cg.embed_TS(
            mol_ts=m,
            new_mol=new_m,
            reacting_atoms=[a.GetIdx() for a in m.GetAtoms()],
            frozen_atoms=[a.GetIdx() for a in m.GetAtoms()],
            number_of_conformers=n,
        )
        assert isinstance(valid, Chem.rdchem.Mol)
        assert m.GetNumConformers() == 1
        assert all_constrained.GetNumConformers() == n
        assert all_constrained == new_m

        # FF
        new_mol = cg.optimize(
            new_mol=valid, mol_ts=m, frozen_atoms=get_frozen_atoms(mol, reacting_atoms)
        )
        assert new_mol.GetNumConformers() == valid.GetNumConformers()

        all_constrained_new_mol = cg.optimize(
            new_mol=all_constrained,
            mol_ts=m,
            frozen_atoms=get_frozen_atoms(mol, reacting_atoms),
        )
        assert (
            all_constrained_new_mol.GetNumConformers()
            == all_constrained.GetNumConformers()
        )

        # Prune
        pruned_new_mol = cg.prune(new_mol)
        assert pruned_new_mol.GetNumConformers() <= new_mol.GetNumConformers()

        pruned_all_constrained_new_mol = cg.prune(all_constrained_new_mol)
        assert (
            pruned_all_constrained_new_mol.GetNumConformers()
            <= all_constrained_new_mol.GetNumConformers()
        )

    return True


def test():
    random_seed = 12

    cg = ConformerGenerator(randomSeed=random_seed)
    assert _test_default_getter(cg)
    assert _test_conformer_generator(cg)

    cg = ConformerGenerator(randomSeed=random_seed)
    cg.embedder = BoundsMatrixEmbedder(randomSeed=random_seed)
    assert _test_conformer_generator(cg)

    cg = ConformerGenerator(randomSeed=random_seed)
    cg.mol_getter = MolGetterBonds()
    assert _test_conformer_generator(cg)

    cg = ConformerGenerator(randomSeed=random_seed)
    cg.mol_getter = MolGetterConnectivity()
    assert _test_conformer_generator(cg)

    cg = ConformerGenerator(randomSeed=random_seed)
    cg.optimizer = UFFOptimizer()
    assert _test_conformer_generator(cg)


def test_default_conf_factor_is_80():
    class SpyEmbedder:
        n = None

        def embed_TS(self, mol_ts, mol, reacting_atoms, frozen_atoms, n, verbose):
            SpyEmbedder.n = n
            mol.AddConformer(Chem.Conformer(mol_ts.GetConformer()), assignId=True)
            return [0], []

    cg = ConformerGenerator()
    cg.embedder = SpyEmbedder()
    mol = cg.generate_conformers(
        filenames[0], charge, reacting_atoms, input_smiles=input_smiles
    )

    assert SpyEmbedder.n == Descriptors.NumRotatableBonds(mol) * 80 + 30


def test_embedding_without_conformers_raises():
    class EmptyEmbedder:
        def embed_TS(self, mol_ts, mol, reacting_atoms, frozen_atoms, n, verbose):
            return [], []

    cg = ConformerGenerator()
    cg.embedder = EmptyEmbedder()

    with pytest.raises(RuntimeError, match="no conformers"):
        cg.generate_conformers(
            filenames[0], charge, reacting_atoms, input_smiles=input_smiles
        )


@pytest.mark.parametrize("auto_fallback", [True, False])
def test_optimizer_errors_are_not_swallowed(auto_fallback):
    # Only MMFF falls back to UFF; other optimizer errors must reach the caller.
    class FailingOptimizer:
        def tune_ts_conformers(self, mol, reference, align_indices):
            raise RuntimeError("boom")

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMultipleConfs(mol, 2, randomSeed=1)
    cg = ConformerGenerator()
    cg.optimizer = FailingOptimizer()

    with pytest.raises(RuntimeError, match="boom"):
        cg.optimize(mol, Chem.Mol(mol), [0, 1], auto_fallback=auto_fallback)


def test_uff_fallback_is_logged_every_time(boronic_acid, caplog):
    with caplog.at_level(logging.WARNING):
        for _ in range(2):
            mol = ConformerGenerator(randomSeed=12).generate_conformers(
                boronic_acid,
                0,
                [0, 1, 2],
                input_smiles=["C=CCB(O)O"],
                number_of_conformers=5,
            )

    assert caplog.text.count("falling back to UFF") == 2
    assert mol.GetProp("energy_method") == "UFFOptimizer"


def test_charge_is_passed_on_for_graphs_without_formal_charges(sn2_ts, caplog):
    cg = ConformerGenerator(randomSeed=12)
    cg.mol_getter = MolGetterConnectivity()  # perceives no formal charges

    with caplog.at_level(logging.WARNING):
        mol = cg.generate_conformers(sn2_ts, -1, [0, 1, 2], number_of_conformers=2)

    assert infer_charge_and_multiplicity(mol) == {"charge": -1, "multiplicity": 1}
    assert "charge" not in caplog.text.lower()


def test_charge_and_multiplicity_that_do_not_fit_the_electrons_are_flagged_once(
    sn2_ts, tmp_path, caplog
):
    cg = ConformerGenerator(randomSeed=12)
    cg.mol_getter = MolGetterConnectivity()

    def run(**kwargs):
        cg.generate_conformers(sn2_ts, reacting_atoms=[0, 1, 2], **kwargs)
        cg.write_xyz(str(tmp_path / "out.xyz"))

    with caplog.at_level(logging.WARNING):
        run(charge=0, number_of_conformers=2)  # charge -1 forgotten: 43 electrons
    assert caplog.text.count("odd number of electrons") == 1
    assert infer_charge_and_multiplicity(cg.mol) == {"charge": 0, "multiplicity": 2}

    caplog.clear()
    with caplog.at_level(logging.WARNING):
        run(charge=-1, multiplicity=2, number_of_conformers=2)
        run(charge=-1, multiplicity=3, number_of_conformers=2)  # a triplet fits
    assert caplog.text.count("does not fit") == 1


def test_radical_electrons_of_the_ts_graph_do_not_set_the_multiplicity(
    sn2_ts_symmetric, caplog
):
    # The symmetric TS drops the template C-Cl bond, leaving two radical centres.
    kwargs = dict(input_smiles=["CCl", "[Cl-]"], number_of_conformers=2)

    with caplog.at_level(logging.WARNING):
        mol = ConformerGenerator(randomSeed=12).generate_conformers(
            sn2_ts_symmetric, -1, [0, 1, 2], **kwargs
        )
    assert sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms()) == 2
    assert infer_charge_and_multiplicity(mol)["multiplicity"] == 1
    assert "electrons" not in caplog.text

    mol = ConformerGenerator(randomSeed=12).generate_conformers(
        sn2_ts_symmetric, -1, [0, 1, 2], multiplicity=3, **kwargs
    )
    assert infer_charge_and_multiplicity(mol)["multiplicity"] == 3


def _ensemble_with_one_missing_energy():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMultipleConfs(mol, 2, randomSeed=1)
    mol.GetConformer(0).SetDoubleProp("energy", 23.06054783061903)  # 1 eV
    mol.SetIntProp("charge", -1)
    mol.SetIntProp("multiplicity", 2)
    mol.SetProp("energy_method", "MMFFOptimizer")
    return mol


def _comment_lines(path, n_atoms):
    return path.read_text().splitlines()[1 :: n_atoms + 2]


def test_write_xyz_writes_extended_xyz_by_default(tmp_path, caplog):
    cg = ConformerGenerator()
    cg.mol = _ensemble_with_one_missing_energy()

    with caplog.at_level(logging.WARNING):
        cg.write_xyz(str(tmp_path / "out.xyz"))

    first, second = [
        dict(field.split("=", 1) for field in shlex.split(line))
        for line in _comment_lines(tmp_path / "out.xyz", cg.mol.GetNumAtoms())
    ]
    assert first == {
        "Properties": "species:S:1:pos:R:3",
        "racerts_energy": "1.00000000",
        "charge": "-1",
        "spin": "2",
        "multiplicity": "2",
        "energy_method": "MMFFOptimizer",
        "pbc": "F F F",
    }
    assert "racerts_energy" not in second
    assert "no energy" in caplog.text


def test_write_xyz_crest_energies_and_custom_comment(tmp_path, caplog):
    cg = ConformerGenerator()
    cg.mol = _ensemble_with_one_missing_energy()
    n_atoms = cg.mol.GetNumAtoms()

    with caplog.at_level(logging.WARNING):
        cg.write_xyz(str(tmp_path / "crest.xyz"), use_energy=True)
    # Strict float parsers of CREST files would fail on nan: leave the conformer out.
    assert _comment_lines(tmp_path / "crest.xyz", n_atoms) == ["0.036749"]
    assert "left out" in caplog.text

    cg.write_xyz(str(tmp_path / "custom.xyz"), comment="0 1")
    assert _comment_lines(tmp_path / "custom.xyz", n_atoms) == ["0 1", "0 1"]


# Embedding results differ between RDKit versions (also between 2025.03 and 2025.09).
REFERENCE_RDKIT = "2025.03.2"
REFERENCE_0_1_7 = os.path.join(DATA, f"ex_0.1.7_rdkit{REFERENCE_RDKIT}.xyz")


def _read_reference(path):
    lines = open(path).read().splitlines()
    frames, i = [], 0
    while i < len(lines):
        n = int(lines[i])
        fields = dict(field.split("=") for field in lines[i + 1].split())
        xyz = [
            [float(v) for v in line.split()[1:]] for line in lines[i + 2 : i + 2 + n]
        ]
        frames.append((int(fields["conf_id"]), float(fields["energy"]), np.array(xyz)))
        i += n + 2
    return frames


@pytest.mark.skipif(
    rdBase.rdkitVersion != REFERENCE_RDKIT,
    reason=f"the 0.1.7 reference was made with RDKit {REFERENCE_RDKIT}",
)
def test_ensemble_matches_the_0_1_7_release():
    mol = ConformerGenerator().generate_conformers(
        filenames[0], 0, [3, 4, 5], input_smiles=["CCCCCC=C"], number_of_conformers=50
    )
    reference = _read_reference(REFERENCE_0_1_7)

    assert [conf.GetId() for conf in mol.GetConformers()] == [r[0] for r in reference]
    for conf, (_, energy, xyz) in zip(mol.GetConformers(), reference):
        assert conf.GetDoubleProp("energy") == pytest.approx(energy, abs=1e-6)
        assert np.allclose(conf.GetPositions(), xyz, atol=1e-6)
