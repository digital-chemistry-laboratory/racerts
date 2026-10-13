"""ConformerEnsemble reads and writes the conformer properties of its molecule."""

import math

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from racerts import ConformerEnsemble
from racerts.io import write_xyz


@pytest.fixture
def ethanol():
    """Three conformers; energies 2.0, none, 1.0 kcal/mol."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMultipleConfs(mol, 3, randomSeed=7)
    mol.GetConformer(0).SetDoubleProp("energy", 2.0)
    mol.GetConformer(2).SetDoubleProp("energy", 1.0)
    mol.SetProp("energy_method", "MMFFOptimizer")
    return ConformerEnsemble(mol)


def test_energies_and_best(ethanol):
    energies = ethanol.energies()

    assert energies[0] == 2.0 and math.isnan(energies[1]) and energies[2] == 1.0
    assert ethanol.best() == 2


def test_best_needs_an_energy():
    mol = Chem.AddHs(Chem.MolFromSmiles("C"))
    AllChem.EmbedMolecule(mol, randomSeed=1)

    with pytest.raises(ValueError, match="No conformer has an energy"):
        ConformerEnsemble(mol).best()


def test_records_view_the_conformer_properties(ethanol):
    ethanol.add_provenance(embedder="CmapEmbedder", seed=12)  # all conformers
    ethanol.add_provenance(2, batch=0)
    records = ethanol.records

    assert records[2] == ethanol.record(2)
    assert records[2].provenance == {"embedder": "CmapEmbedder", "seed": 12, "batch": 0}
    assert records[2].energy == 1.0 and records[2].energy_method == "MMFFOptimizer"
    assert records[1].energy is None and records[1].provenance["seed"] == 12
    # One store: the provenance is an RDKit property of the conformer.
    assert ethanol.mol.GetConformer(2).HasProp("provenance")


def test_filter_keeps_ids_and_data(ethanol):
    ethanol.add_provenance(2, seed=12)
    subset = ethanol.filter([0, 2])

    assert subset.conf_ids == [0, 2] and len(ethanol) == 3
    assert subset.energy(2) == 1.0 and subset.provenance(2) == {"seed": 12}
    with pytest.raises(ValueError, match="No conformers with ids"):
        ethanol.filter([5])


def test_merge_renumbers_the_added_conformers(ethanol):
    merged = ethanol.merge(ethanol.filter([2]))

    assert merged.conf_ids == [0, 1, 2, 3]
    assert merged.energy(3) == 1.0
    assert np.allclose(
        merged.mol.GetConformer(3).GetPositions(),
        ethanol.mol.GetConformer(2).GetPositions(),
    )


def test_merge_needs_the_same_graph(ethanol):
    other = Chem.AddHs(Chem.MolFromSmiles("OCC"))  # same atoms, other order
    AllChem.EmbedMolecule(other, randomSeed=1)

    with pytest.raises(ValueError, match="same molecular graph"):
        ethanol.merge(ConformerEnsemble(other))


def test_write_xyz_is_the_io_writer(ethanol, tmp_path):
    ethanol.write_xyz(str(tmp_path / "a.xyz"))
    write_xyz(ethanol.mol, str(tmp_path / "b.xyz"))

    assert (tmp_path / "a.xyz").read_text() == (tmp_path / "b.xyz").read_text()


def test_summary(ethanol):
    assert ethanol.summary() == (
        "3 conformers; energies (MMFFOptimizer): lowest 1.0000 kcal/mol "
        "(conformer 2), window 1.00 kcal/mol"
    )


def test_pickle_keeps_the_conformer_data(ethanol):
    import pickle

    ethanol.add_provenance(2, seed=12)
    ethanol.mol.SetIntProp("charge", -1)
    copy = pickle.loads(pickle.dumps(ethanol))

    assert copy.records == ethanol.records
    assert copy.mol.GetIntProp("charge") == -1


def test_merge_keeps_one_energy_method(ethanol):
    embedded = ethanol.filter([1])  # no energy
    embedded.mol.ClearProp("energy_method")

    assert embedded.merge(ethanol).mol.GetProp("energy_method") == "MMFFOptimizer"
    other = ethanol.copy()
    other.mol.SetProp("energy_method", "UFFOptimizer")
    with pytest.raises(ValueError, match="different methods"):
        ethanol.merge(other)


def test_copy_is_independent(ethanol):
    copy = ethanol.copy()
    copy.mol.GetConformer(0).SetDoubleProp("energy", 5.0)

    assert ethanol.energy(0) == 2.0
