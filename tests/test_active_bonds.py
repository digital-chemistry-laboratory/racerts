"""Active-bond windows of TSs and the attack-face filter, on the aldol TS."""

import collections
import os

import numpy as np
import pytest

from racerts import TransitionState
from racerts.system import build_mol

from .conftest import DATA

ALDOL = os.path.join(DATA, "aldol_ts.xyz")  # tests/data/make_aldol_ts.py
REACTING = [0, 10, 11, 12, 19]
SMILES = ["OC(=O)[C@@H]1CCCN1C(C)=C", "O=Cc1ccccc1"]
CC = (10, 12)  # the forming C-C bond, 2.2 A in the seed


@pytest.fixture(scope="module")
def aldol():
    return build_mol(ALDOL, 0, REACTING, input_smiles=SMILES)


def test_the_legacy_ts_has_no_windows(aldol):
    legacy = TransitionState(REACTING)
    assert not legacy.windowed and len(legacy.restraints(aldol)) == 0
    assert legacy.frozen_atoms(aldol).core == tuple(REACTING)


def test_active_bonds_are_the_forming_bonds(aldol):
    task = TransitionState(REACTING, active_window=0.25)
    assert task.active_pairs(aldol) == [CC, (11, 19)]  # C-C, and H...O of the transfer
    frozen = task.frozen_atoms(aldol)
    assert not set(frozen.hard) & set(REACTING)  # the reacting atoms are free
    sources = collections.Counter(r.source for r in task.restraints(aldol))
    assert sources["active"] == 2 and sources["neighbor"] == 8 and sources["core"] > 0


# ---- regression tests ----


@pytest.mark.parametrize(
    "bonds, error",
    [([(10, 99)], ValueError), ([(1, 8)], ValueError), ([(10, 10)], ValueError)],
)
def test_active_bonds_are_checked(aldol, bonds, error):
    with pytest.raises(error):
        TransitionState(REACTING, active_bonds=bonds, active_window=0.2).active_pairs(
            aldol
        )


def test_ring_13_pairs_are_not_forming_bonds():
    # Benchmark SNAr and cyclization TSs: reacting ring atoms 1,3 apart (2.4 A) were
    # taken for forming bonds, and their in-plane "faces" flipped at random.
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("C1CCC1"))  # 1,3 C...C about 2.2 A
    AllChem.EmbedMolecule(mol, randomSeed=1)
    assert TransitionState([0, 1, 2], active_window=0.3).active_pairs(mol) == []
    explicit = TransitionState([0, 1, 2], active_bonds=[(0, 2)], active_window=0.3)
    assert explicit.active_pairs(mol) == [(0, 2)]


def test_three_membered_forming_bonds_stay_active():
    # A reductive-elimination TS C-Pd-C: C...C 1.88 A at a 56 degree angle forms a
    # bond although the two carbons share Pd.
    from rdkit import Chem
    from rdkit.Geometry import Point3D

    mol = Chem.RWMol(Chem.MolFromSmiles("C[Pd]C"))
    conf = Chem.Conformer(3)
    half = np.radians(28)
    for i, (x, y) in enumerate(
        [(np.cos(half), np.sin(half)), (0, 0), (np.cos(half), -np.sin(half))]
    ):
        conf.SetAtomPosition(i, Point3D(2.0 * x, 2.0 * y, 0.0))
    mol.AddConformer(conf)
    assert TransitionState([0, 1, 2], active_window=0.3).active_pairs(mol) == [(0, 2)]
