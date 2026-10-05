"""
The symmetry-minimised RMSD without listing every equivalent atom mapping
(racerts.symmetry), against the listing of all maps (racerts.geometry). The one error
that must not happen is a value below the true RMSD: it would merge distinct conformers.
"""

import itertools

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from racerts.geometry import rmsd, symmetry_maps
from racerts.symmetry import (
    SymmetricRMSD,
    bound_descriptors,
    matching_graph,
    symmetry_classes,
)

MOLECULES = {  # each a kind of local symmetry
    "methyls on one carbon": "CCC(C)(C)O",
    "a ring flip, an isopropyl group and a carboxylic acid": "CC(C)c1ccc(cc1)CC(=O)O",
    "a CF3 group": "Cc1ccc(cc1)C(F)(F)F",
    "a carboxylate": "CC(C)CC(=O)[O-]",
    "two stereocentres, meso": "C[C@@H](O)[C@H](C)O",
    "identical fragments": "CC(N)=O.O.O.O",
}
THRESHOLD = 0.125
_made = {}


def _conformers(smiles, n=3):
    if smiles not in _made:
        mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
        params = AllChem.ETKDGv3()
        params.randomSeed = 11
        ids = list(AllChem.EmbedMultipleConfs(mol, n, params))
        AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=200)
        _made[smiles] = mol, [mol.GetConformer(i).GetPositions() for i in ids]
    return _made[smiles]


def _all_maps(mol, hydrogens):
    """(atoms, maps) of every automorphism: the listing that the factorised method must
    agree with. For "polar" from the graph of the module, else as racerts.geometry."""
    if hydrogens != "polar":
        found = symmetry_maps(mol, hydrogens == "all", 200000)
        return np.asarray(found.atoms), np.asarray(found.maps)
    graph, index, _ = matching_graph(mol, "polar")
    matches = graph.GetSubstructMatches(
        graph, uniquify=False, useChirality=True, maxMatches=200000
    )
    return index, index[np.array(matches)]


def _true_rmsd(a, b, atoms, maps):
    return min(rmsd(a[atoms], b[m]) for m in maps)


def _cases(smiles, hydrogens):
    """(a, b, whether b is a planted duplicate of a, their true RMSD), and the number
    of maps: every pair of conformers, and each conformer with equivalent atoms
    exchanged."""
    key = (smiles, hydrogens)
    if key not in _made:
        mol, conformers = _conformers(smiles)
        atoms, maps = _all_maps(mol, hydrogens)
        rng = np.random.default_rng(len(smiles))
        pairs = [(a, b, False) for a, b in itertools.combinations(conformers, 2)]
        pairs += [
            (a, _planted(a, _random_automorphism(mol, rng), rng), True)
            for a in conformers
        ]
        cases = [(a, b, twin, _true_rmsd(a, b, atoms, maps)) for a, b, twin in pairs]
        _made[key] = cases, len(maps)
    return _made[key]


def _random_automorphism(mol, rng):
    """An automorphism of all atoms found without listing them: the first maps that
    RDKit returns for random renumberings, multiplied."""
    n = mol.GetNumAtoms()
    result = np.arange(n)
    for _ in range(4):
        order = rng.permutation(n)
        renumbered = Chem.RenumberAtoms(mol, order.tolist())
        found = renumbered.GetSubstructMatches(
            renumbered, uniquify=False, useChirality=True, maxMatches=40
        )
        match = np.array(found[int(rng.integers(len(found)))])
        factor = np.empty(n, dtype=np.intp)
        factor[order] = order[match]
        result = factor[result]
    return result


def _planted(positions, automorphism, rng, noise=0.02):
    """The conformer with equivalent atoms exchanged, a little noise, turned and moved."""
    out = np.array(positions)
    out[automorphism] = positions
    out += rng.normal(scale=noise, size=out.shape)
    q, r = np.linalg.qr(rng.normal(size=(3, 3)))
    q *= np.sign(np.diag(r))
    if np.linalg.det(q) < 0:
        q[:, 0] = -q[:, 0]
    return out @ q + rng.normal(scale=3.0, size=3)


@pytest.mark.parametrize("smiles", MOLECULES.values(), ids=list(MOLECULES))
@pytest.mark.parametrize("hydrogens", ["none", "polar", "all"])
@pytest.mark.parametrize(
    "max_maps", [100, 1], ids=["default", "as factorised as can be"]
)
def test_never_below_the_rmsd_over_all_maps(smiles, hydrogens, max_maps):
    mol, _ = _conformers(smiles)
    cases, n_maps = _cases(smiles, hydrogens)
    kernel = SymmetricRMSD(mol, hydrogens, max_maps=max_maps)
    assert kernel.order == n_maps and not kernel.truncated
    for a, b, duplicate, true in cases:
        value = kernel.rmsd(a, b)
        assert value >= true - 1e-9
        assert kernel.lower_bound(a, b) <= true + 1e-7
        if duplicate or kernel.level == 0:  # exact: a duplicate, or plain enumeration
            assert value == pytest.approx(true, abs=1e-7)
            assert kernel.within(a, b, THRESHOLD) == (true <= THRESHOLD)
        assert not (kernel.within(a, b, THRESHOLD) and true > THRESHOLD + 1e-6)


@pytest.mark.parametrize("smiles", MOLECULES.values(), ids=list(MOLECULES))
@pytest.mark.parametrize("hydrogens", ["none", "polar", "all"])
def test_the_bounds_never_exceed_the_rmsd(smiles, hydrogens):
    # Pairs whose descriptors are further apart than the threshold need no RMSD.
    mol, _ = _conformers(smiles)
    graph, index, weight = matching_graph(mol, hydrogens)
    classes = symmetry_classes(graph, weight)
    for a, b, _, true in _cases(smiles, hydrogens)[0]:
        da, db = (bound_descriptors(x, index, weight, classes) for x in (a, b))
        for first, second in zip(da, db):
            assert float(np.linalg.norm(first - second)) <= true + 1e-9


def test_the_bounds_tell_most_different_conformers_apart():
    # 30 conformers of a flexible chain: most pairs are further apart by their
    # descriptors alone than the threshold of a duplicate.
    mol, conformers = _conformers("OCCCCCCN", n=30)
    graph, index, weight = matching_graph(mol, "polar")
    classes = symmetry_classes(graph, weight)
    described = [bound_descriptors(x, index, weight, classes) for x in conformers]
    apart = [
        max(float(np.linalg.norm(u - v)) for u, v in zip(da, db)) > THRESHOLD
        for da, db in itertools.combinations(described, 2)
    ]
    assert np.mean(apart) > 0.9


def test_hydrogens_beyond_any_listing():
    # P(tBu)3 with all atoms: 1296 maps of the heavy atoms, 1.3e10 with the hydrogens.
    mol, conformers = _conformers("CC(C)(C)P(C(C)(C)C)C(C)(C)C", n=2)
    kernel = SymmetricRMSD(mol, "all")
    assert kernel.order > 10**10 and kernel.level > 0 and not kernel.truncated
    rng = np.random.default_rng(3)
    twin = _planted(conformers[0], _random_automorphism(mol, rng), rng)
    assert kernel.rmsd(conformers[0], twin) < 0.05
    assert kernel.within(conformers[0], twin, THRESHOLD)
    # A second embedding of this rigid molecule is the same conformer with its
    # equivalent atoms in other places; one hydrogen moved by 1 A is another structure.
    assert kernel.within(conformers[0], conformers[1], THRESHOLD)
    moved = conformers[0].copy()
    moved[mol.GetNumAtoms() - 1] += (1.0, 0.0, 0.0)
    assert kernel.rmsd(conformers[0], moved) > 0.1
    assert not kernel.within(conformers[0], moved, 0.1)


def test_polar_hydrogens_tell_rotamers_apart():
    # Two glycerol conformers that differ in an O-H rotor only: one conformer for the
    # heavy atoms, two for a hydrogen bond.
    mol = Chem.AddHs(Chem.MolFromSmiles("OCC(O)CO"))
    AllChem.EmbedMolecule(mol, randomSeed=5)
    AllChem.MMFFOptimizeMolecule(mol)
    turned = Chem.Conformer(mol.GetConformer())
    hydroxyl = mol.GetSubstructMatch(Chem.MolFromSmarts("[#6][#6][OX2][H]"))
    Chem.rdMolTransforms.SetDihedralDeg(
        turned, *hydroxyl, Chem.rdMolTransforms.GetDihedralDeg(turned, *hydroxyl) + 120
    )
    a, b = mol.GetConformer().GetPositions(), turned.GetPositions()
    assert SymmetricRMSD(mol, "none").rmsd(a, b) < 1e-6
    assert SymmetricRMSD(mol, "polar").rmsd(a, b) > THRESHOLD
    assert SymmetricRMSD(mol, "all").rmsd(a, b) > THRESHOLD
    with pytest.raises(ValueError, match="'heavy', 'none', 'polar', 'all'"):
        SymmetricRMSD(mol, "some")


def test_chirality_limits_the_maps():
    meso = Chem.AddHs(Chem.MolFromSmiles("C[C@@H](O)[C@H](C)O"))
    chiral = Chem.AddHs(Chem.MolFromSmiles("C[C@H](O)[C@H](C)O"))
    flat = Chem.AddHs(Chem.MolFromSmiles("CC(O)C(C)O"))
    orders = [SymmetricRMSD(mol, "none").order for mol in (meso, chiral, flat)]
    assert orders == [1, 2, 2]  # the mirror map of the meso form inverts both centres
