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


@pytest.mark.parametrize("renumber", [False, True])
def test_filter_keeps_the_requested_order(ethanol, renumber):
    ethanol.add_provenance(2, seed=12)
    before = ethanol.mol.ToBinary()
    subset = ethanol.filter([2, 0], renumber=renumber)

    assert subset.conf_ids == ([0, 1] if renumber else [2, 0])
    assert subset.energies().tolist() == [1.0, 2.0]
    assert subset.provenance(subset.conf_ids[0]) == {"seed": 12}
    assert np.array_equal(
        subset.mol.GetConformer(subset.conf_ids[0]).GetPositions(),
        ethanol.mol.GetConformer(2).GetPositions(),
    )
    assert ethanol.mol.ToBinary() == before  # unchanged


@pytest.mark.parametrize(
    "ids, error", [([0, 0], ValueError), ([7], ValueError), ([True], TypeError)]
)
def test_filter_rejects_invalid_ids(ethanol, ids, error):
    with pytest.raises(error):
        ethanol.filter(ids)


def _with_conformer(smiles, x=0.0, energy=None):
    mol = Chem.MolFromSmiles(smiles)
    conf = Chem.Conformer(mol.GetNumAtoms())
    conf.SetAtomPosition(0, (x, 2, 3))
    if energy is not None:
        conf.SetDoubleProp("energy", energy)
    mol.AddConformer(conf)
    return ConformerEnsemble(mol)


@pytest.mark.parametrize("first, second", [("C", "CC"), ("CO", "CN"), ("CO", "OC"),
                                           ("CO", "[13CH3]O")])  # fmt: skip
def test_merge_rejects_other_atoms(first, second):
    for identity in ("graph", "elements"):
        with pytest.raises(ValueError, match="can be merged"):
            _with_conformer(first).merge(_with_conformer(second), identity=identity)


@pytest.mark.parametrize(
    "first, second",
    [
        ("CC", "[CH2][CH2]"),  # radicals
        ("[NH3]->[Cu+2]", "[NH3+]-[Cu+]"),  # bond type, charges
        ("F[C@](Cl)(Br)I", "F[C@@](Cl)(Br)I"),  # stereo
    ],
)
def test_merge_by_elements_ignores_the_representation(first, second):
    a, b = _with_conformer(first, 0, -2.0), _with_conformer(second, 1, -1.0)
    before = [a.mol.ToBinary(), b.mol.ToBinary()]
    merged = a.merge(b, identity="elements")

    assert merged.conf_ids == [0, 1]
    assert merged.energies().tolist() == [-2.0, -1.0]
    assert merged.mol.GetConformer(1).GetPositions()[0].tolist() == [1, 2, 3]
    assert Chem.MolToSmiles(merged.mol) == Chem.MolToSmiles(a.mol)
    assert [a.mol.ToBinary(), b.mol.ToBinary()] == before
    if "@" not in first:  # the graph identity sees radicals, bonds and charges
        with pytest.raises(ValueError, match="same molecular graph"):
            a.merge(b)
    with pytest.raises(ValueError, match="identity"):
        a.merge(b, identity="smiles")


def test_energies_in_other_units(ethanol):
    energies = ethanol.energies(unit="eV")
    assert energies[0] == pytest.approx(2.0 / 23.06054783061903)
    assert math.isnan(energies[1])
    assert ethanol.energies("hartree")[2] == pytest.approx(1.0 / 627.5094740629)
    assert ethanol.energies("kJ/mol")[2] == pytest.approx(4.184)
    with pytest.raises(ValueError, match="unit"):
        ethanol.energies("kcal")


def test_from_frames_takes_positions_not_results():
    template = Chem.AddHs(Chem.MolFromSmiles("[2H]O"))  # D-O-H
    template.GetAtomWithIdx(0).SetAtomMapNum(7)
    template.SetProp("_thermo", "stale")
    template.SetIntProp("charge", 0)
    old = Chem.Conformer(3)
    old.SetDoubleProp("energy", -100)
    template.AddConformer(old)
    positions = np.array([[0, 1, 0], [0, 0, 0], [1, 0, 0]], dtype=float)
    frames = [positions[[1, 2, 0]], positions[[1, 2, 0]].tolist()]

    ensemble = ConformerEnsemble.from_frames(template, frames, atom_order=[2, 0, 1])
    assert ensemble.conf_ids == [0, 1]
    for conf_id in ensemble.conf_ids:
        assert np.array_equal(
            ensemble.mol.GetConformer(conf_id).GetPositions(), positions
        )
        assert ensemble.energy(conf_id) is None
        assert ensemble.provenance(conf_id) == {"source": "external", "frame": conf_id}
    atom = ensemble.mol.GetAtomWithIdx(0)
    assert (atom.GetIsotope(), atom.GetAtomMapNum()) == (2, 7)
    assert not ensemble.mol.HasProp("_thermo") and ensemble.mol.HasProp("charge")
    assert template.GetNumConformers() == 1  # unchanged


def test_from_frames_with_ase_atoms():
    ase = pytest.importorskip("ase")
    from ase.constraints import FixAtoms

    template = Chem.MolFromSmiles("[H][H]")
    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    assert len(ConformerEnsemble.from_frames(template, [atoms, atoms])) == 2
    invalid = [
        ([], "empty"),
        ([ase.Atoms("H")], "count or order"),
        ([ase.Atoms("He2")], "count or order"),
        ([ase.Atoms("H2", positions=[[np.nan, 0, 0], [0, 0, 0]])], "nonfinite"),
        ([ase.Atoms("H2", pbc=True)], "isolated"),
        ([ase.Atoms("H2", constraint=FixAtoms(indices=[0]))], "unconstrained"),
        ([np.zeros((3, 3))], "shape"),
    ]
    for frames, message in invalid:
        with pytest.raises(ValueError, match=message):
            ConformerEnsemble.from_frames(template, frames)
    with pytest.raises(TypeError, match="ASE Atoms or arrays"):
        ConformerEnsemble.from_frames(template, [object()])
    with pytest.raises(ValueError, match="permutation"):
        ConformerEnsemble.from_frames(template, [atoms], atom_order=[0, 0])
    for smiles in ("", "*"):
        with pytest.raises(ValueError, match="real atoms"):
            ConformerEnsemble.from_frames(Chem.MolFromSmiles(smiles), [])


def test_imported_mirror_images_are_caught_by_validation():
    import racerts
    from racerts.validate import IdentityFilter

    template = Chem.MolFromSmiles("F[C@](Cl)(Br)I")
    xyz = np.array([[1, 1, 1], [0, 0, 0], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]], float)
    xyz *= np.array([1.35, 0, 1.77, 1.94, 2.14])[:, None] / np.sqrt(3)
    # A structure and its mirror image: import takes both, validation keeps one.
    ensemble = ConformerEnsemble.from_frames(template, [xyz, -xyz])
    ctx = racerts.Context.create(ensemble.mol, racerts.GroundState())
    kept = IdentityFilter().run(ctx, ensemble.copy())
    assert len(kept) == 1
    mirror = 1 - kept.conf_ids[0]
    with pytest.raises(RuntimeError, match="No conformer passed"):
        IdentityFilter().run(ctx, ensemble.filter([mirror]))


def test_write_sdf(ethanol, tmp_path):
    ethanol.add_provenance(0, seed=12)
    path = str(tmp_path / "ensemble.sdf")
    ethanol.write_sdf(path)
    records = list(Chem.SDMolSupplier(path, removeHs=False))
    assert [r.GetIntProp("conf_id") for r in records] == ethanol.conf_ids
    assert records[0].GetDoubleProp("energy") == 2.0 and not records[1].HasProp(
        "energy"
    )
    assert records[0].GetProp("energy_method") == "MMFFOptimizer"
    assert records[0].GetProp("provenance") == '{"seed": 12}'
    assert np.allclose(
        records[2].GetConformer().GetPositions(),
        ethanol.mol.GetConformer(2).GetPositions(),
        atol=1e-4,
    )
