"""racerts.geometry: superposition and (symmetry-aware) RMSD."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolAlign

import racerts.geometry
from racerts.geometry import heavy_atoms, rmsd, rmsd_within, symmetry_maps


def _rotation(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def _embedded(smiles, n=1, seed=3):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(mol, n, randomSeed=seed)
    return mol


def _swapped(positions, i, j):
    """positions with the coordinates of atoms i and j exchanged."""
    swapped = positions.copy()
    swapped[[i, j]] = swapped[[j, i]]
    return swapped


def test_the_rmsd_superposition_selection_and_checks():
    # -- rotations and mirror images
    rng = np.random.default_rng(3)
    a = rng.normal(size=(6, 3))
    moved = a @ _rotation(0.7).T + [1, 2, 3]
    assert rmsd(a, moved) == pytest.approx(0, abs=1e-10)
    mirror = a * [-1, 1, 1]
    assert rmsd(a, mirror) > 0.1  # proper rotations only

    # -- without superposition the frame counts
    a = np.random.default_rng(4).normal(size=(5, 3))
    assert rmsd(a, a + [1, 0, 0], align=False) == pytest.approx(1.0)
    assert rmsd(a, a + [1, 0, 0]) == pytest.approx(0, abs=1e-10)

    # -- atom selection controls alignment and measurement
    a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [5, 5, 5]], float)
    b = a.copy()
    b[3] = [-5, -5, -5]  # only the last atom differs
    assert rmsd(a, b, [0, 1, 2]) == pytest.approx(0, abs=1e-10)
    assert rmsd(a, b) > 0.5

    # -- the coordinates and maps are checked
    a = np.zeros((3, 3))
    b = a.copy()
    b[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        rmsd(a, b)
    with pytest.raises(ValueError, match="shapes"):
        rmsd(a, np.zeros((2, 3)))
    with pytest.raises(ValueError, match="rows of 3"):
        rmsd(a, a, maps=[[0, 1]])
    with pytest.raises(IndexError):
        rmsd(a, a, maps=[[0, 1, 3]])
    with pytest.raises(TypeError):
        rmsd(a, a, maps=[[0.0, 1.0, 2.0]])

    # -- heavy atoms
    assert heavy_atoms(Chem.AddHs(Chem.MolFromSmiles("CO"))) == [0, 1]
    assert heavy_atoms(Chem.MolFromSmiles("[H][H]")) == [0, 1]


@pytest.mark.parametrize(
    "indices, error",
    [([], ValueError), ([0, 0], ValueError), ([0, 9], IndexError),
     ([0.0, 1.0], TypeError), ([True, 1], TypeError)],
)  # fmt: skip
def test_the_selection_is_checked(indices, error):
    a = np.zeros((3, 3))
    with pytest.raises(error):
        rmsd(a, a, indices)


@pytest.mark.parametrize(
    "smiles, i, j, n_maps",
    [
        ("CC(C)C", 0, 2, 6),  # isobutane: the methyl carbons
        ("CC(=O)[O-]", 2, 3, 2),  # acetate: the oxygens (terminal groups)
    ],
)
def test_the_symmetry_maps_find_equivalent_atoms(smiles, i, j, n_maps):
    mol = _embedded(smiles)
    symmetry = symmetry_maps(mol)
    assert symmetry.atoms == heavy_atoms(mol)
    assert len(symmetry.maps) == n_maps
    assert (symmetry.maps[0] == np.arange(len(symmetry.atoms))).all()  # identity
    a = mol.GetConformer().GetPositions()
    b = _swapped(a, i, j)
    assert rmsd(a, b, symmetry.atoms, align=False) > 0.5
    assert rmsd(a, b, symmetry.atoms, symmetry.maps) == pytest.approx(0, abs=1e-10)


def test_symmetry_maps_and_within(monkeypatch):
    # -- the symmetry maps keep the atoms that remove hs keeps
    # As legacy racerts: Chem.RemoveHs keeps e.g. deuterium (and hydrogens bonded to
    # two atoms), so the RMSD compares them too.
    mol = _embedded("[2H]C(C)O")
    assert symmetry_maps(mol).atoms == [0, 1, 2, 3]
    assert symmetry_maps(mol, include_hs=True).atoms == list(range(mol.GetNumAtoms()))

    # -- the rmsd is rdkits best rms
    mol = _embedded("CC(C)(C)CC(=O)[O-]", n=6)
    symmetry = symmetry_maps(mol)
    heavy = Chem.RemoveHs(mol)
    for i, j in [(0, 1), (2, 5), (3, 4)]:
        a = mol.GetConformer(i).GetPositions()
        b = mol.GetConformer(j).GetPositions()
        best = rdMolAlign.GetBestRMS(Chem.Mol(heavy), heavy, prbId=i, refId=j)
        assert rmsd(a, b, symmetry.atoms, symmetry.maps) == pytest.approx(
            best, abs=1e-6
        )

    # -- within agrees with the rmsd
    mol = _embedded("CC(C)(C)CC(=O)[O-]")
    symmetry = symmetry_maps(mol)
    a = mol.GetConformer().GetPositions()
    rng = np.random.default_rng(5)
    for noise in (0.01, 0.05, 0.1, 0.2):
        for b in (a, _swapped(a, 0, 2)):  # two methyl carbons
            b = b + rng.normal(0, noise, a.shape)
            for align in (True, False):
                value = rmsd(a, b, symmetry.atoms, symmetry.maps, align=align)
                assert rmsd_within(
                    a, b, 0.125, symmetry.atoms, symmetry.maps, align=align
                ) == (value <= 0.125)

    # -- within tries the identity first
    mol = _embedded("CC(C)(C)CC(=O)[O-]")  # 12 maps
    symmetry = symmetry_maps(mol)
    a = mol.GetConformer().GetPositions()
    evaluated = []
    msd = racerts.geometry._msd

    def counting(a, b, align):
        evaluated.append(len(b))
        return msd(a, b, align)

    monkeypatch.setattr(racerts.geometry, "_msd", counting)
    assert rmsd_within(a, a + 0.01, 0.125, symmetry.atoms, symmetry.maps)
    assert evaluated == [1]  # the identity map is within: the others are not needed
    evaluated.clear()
    assert rmsd_within(a, _swapped(a, 0, 2), 0.125, symmetry.atoms, symmetry.maps)
    assert evaluated == [1, 11]  # a duplicate by symmetry only: the other maps
