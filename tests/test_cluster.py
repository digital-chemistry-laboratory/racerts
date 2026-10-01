"""Clustering of conformers and family selection."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from racerts import ConformerEnsemble
from racerts.prune import ClusterPruner, FamilySelector, PruneCluster

RNG = np.random.default_rng(11)


def _mol_with(geometries, energies=None, smiles="CCCCCCCC"):
    """A molecule (its atoms only matter) with one conformer per geometry."""
    mol = Chem.MolFromSmiles(smiles)
    for k, positions in enumerate(geometries):
        conf = Chem.Conformer(mol.GetNumAtoms())
        for i, p in enumerate(positions):
            conf.SetAtomPosition(i, p.tolist())
        if energies is not None:
            conf.SetDoubleProp("energy", float(energies[k]))
        mol.AddConformer(conf, assignId=True)
    return mol


def _three_groups(size=4):
    """Three tight groups (0.1 A noise) around three distinct geometries (> 2 A)."""
    bases = [RNG.normal(scale=3.0, size=(8, 3)) for _ in range(3)]
    geometries, groups = [], []
    for g, base in enumerate(bases):
        for _ in range(size):
            geometries.append(base + RNG.normal(scale=0.1 / np.sqrt(3), size=(8, 3)))
            groups.append(g)
    energies = RNG.uniform(0, 10, size=len(geometries))
    return geometries, groups, energies


@pytest.mark.parametrize(
    "settings",
    [
        dict(method="butina", threshold=1.0),
        dict(method="hierarchical", threshold=0.75, linkage="average"),
        dict(method="leader", threshold=1.0),
    ],
)
def test_three_groups_give_three_lowest_representatives(settings):
    geometries, groups, energies = _three_groups()
    mol = _mol_with(geometries, energies)
    pruner = ClusterPruner(**settings)

    clusters = pruner.clusters(mol)
    assert sorted(sorted(groups[i] for i in c) for c in clusters) == [
        [0] * 4,
        [1] * 4,
        [2] * 4,
    ]
    kept = [c.GetId() for c in pruner.prune(Chem.Mol(mol)).GetConformers()]
    lowest = [
        min((i for i in range(12) if groups[i] == g), key=lambda i: energies[i])
        for g in range(3)
    ]
    assert sorted(kept) == sorted(lowest)


def test_rotations_cluster_but_mirror_images_stay_apart():
    base = RNG.normal(scale=2.0, size=(8, 3))
    c, s = np.cos(0.9), np.sin(0.9)
    rotated = base @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + 4.0
    mirror = base * [-1, 1, 1]
    mol = _mol_with([base, rotated, mirror])
    assert ClusterPruner(threshold=0.3).clusters(mol) == [[0, 1], [2]]


def test_the_atom_selection_controls_alignment_and_distance():
    base = RNG.normal(scale=2.0, size=(8, 3))
    other = base.copy()
    other[7] += 5.0  # only the last atom differs
    mol = _mol_with([base, other])
    assert len(ClusterPruner(threshold=0.3).clusters(mol)) == 2
    assert ClusterPruner(threshold=0.3, atom_indices=range(7)).clusters(mol) == [[0, 1]]


def test_leader_with_a_metric():
    mol = _mol_with([RNG.normal(size=(8, 3)) for _ in range(4)], energies=[3, 0, 2, 1])

    def by_parity(mol, a, b):  # same parity of the conformer id: the same family
        return 0.0 if a % 2 == b % 2 else 10.0

    clusters = ClusterPruner(method="leader", threshold=1.0, metric=by_parity).clusters(
        mol
    )
    assert clusters == [[1, 3], [2, 0]]  # by energy, families by their best member


def test_the_symmetric_kernel_ignores_hydrogens_and_symmetry():
    # Two conformers of 2-methylpropane that differ by swapping methyl groups.
    mol = Chem.AddHs(Chem.MolFromSmiles("CC(C)C"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    swapped = Chem.Conformer(mol.GetConformer())
    positions = mol.GetConformer().GetPositions()
    for a, b in ((0, 2),):
        swapped.SetAtomPosition(a, positions[b].tolist())
        swapped.SetAtomPosition(b, positions[a].tolist())
    mol.AddConformer(swapped, assignId=True)
    assert len(ClusterPruner(threshold=0.1, kernel="symmetric").clusters(mol)) == 1
    assert len(ClusterPruner(threshold=0.1).clusters(mol)) == 2  # fixed atoms


def test_centroid_representative():
    base = np.zeros((8, 3))
    base[:, 0] = np.arange(8)
    shifts = [0.0, 0.2, 0.4]  # the middle one is the centroid
    geometries = [base + np.outer(np.arange(8) % 2, [0, s, 0]) for s in shifts]
    mol = _mol_with(geometries, energies=[0, 1, 2])
    kept = ClusterPruner(threshold=5.0, representative="centroid").prune(Chem.Mol(mol))
    assert [c.GetId() for c in kept.GetConformers()] == [1]
    kept = ClusterPruner(threshold=5.0).prune(Chem.Mol(mol))
    assert [c.GetId() for c in kept.GetConformers()] == [0]


def test_family_selector_fills_round_robin():
    # Families A (energies 0, 3, 4), B (1, 5), C (2); four places: A0, B0, C0, A1.
    a, b, c = (RNG.normal(scale=3.0, size=(8, 3)) for _ in range(3))
    noise = lambda: RNG.normal(scale=0.02, size=(8, 3))  # noqa: E731
    geometries = [a + noise(), a + noise(), a + noise(), b + noise(), b + noise(), c]
    energies = [0, 3, 4, 1, 5, 2]
    ensemble = ConformerEnsemble(_mol_with(geometries, energies))
    selected = FamilySelector(4, ClusterPruner(threshold=0.5)).run(None, ensemble)
    assert selected.energies().tolist() == [0, 1, 2, 3]
    assert FamilySelector(4, renumber=True).run(None, ensemble).conf_ids == [0, 1, 2, 3]
    ensemble.mol.GetConformer(2).ClearProp("energy")
    with pytest.raises(ValueError, match="finite energy"):
        FamilySelector(2).run(None, ensemble)


def test_prune_cluster_stage_and_edge_cases():
    geometries, _, energies = _three_groups(size=2)
    ensemble = ConformerEnsemble(_mol_with(geometries, energies))
    assert len(PruneCluster(ClusterPruner(threshold=1.0)).run(None, ensemble)) == 3
    assert ClusterPruner().clusters(Chem.MolFromSmiles("CC")) == []
    hydrogen = _mol_with([np.array([[0, 0, 0], [0, 0, 0.74]])] * 2, smiles="[H][H]")
    assert ClusterPruner(threshold=0.1).clusters(hydrogen) == [[0, 1]]


@pytest.mark.parametrize(
    "settings, error",
    [
        (dict(threshold=-1), ValueError),
        (dict(threshold="1"), TypeError),
        (dict(method="kmeans"), ValueError),
        (dict(kernel="euclid"), ValueError),
        (dict(representative="first"), ValueError),
    ],
)
def test_settings_are_checked(settings, error):
    with pytest.raises(error):
        ClusterPruner(**settings)
