import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

# Cl- + CH3Cl SN2 transition state (charge -1). Early TS: the C-Cl distance of 2.15 A is
# perceived as a bond, the C..Cl distance of 2.50 A to the nucleophile is not.
SN2_TS = """6
Cl- + CH3Cl SN2 TS (early), charge -1
C      0.000000     0.000000     0.000000
Cl     0.000000     0.000000     2.150000
Cl     0.000000     0.000000    -2.500000
H      1.070000     0.000000     0.100000
H     -0.535000     0.926647     0.100000
H     -0.535000    -0.926647     0.100000
"""

# Symmetric (D3h) SN2 transition state: at 2.32 A neither C-Cl bond is perceived, so the
# C-Cl bond of a "CCl.[Cl-]" template is dropped and C and Cl carry radical electrons.
SN2_TS_SYMMETRIC = """6
Cl- + CH3Cl SN2 TS (symmetric), charge -1
C      0.000000     0.000000     0.000000
Cl     0.000000     0.000000     2.320000
Cl     0.000000     0.000000    -2.320000
H      1.070000     0.000000     0.000000
H     -0.535000     0.926647     0.000000
H     -0.535000    -0.926647     0.000000
"""


@pytest.fixture
def sn2_ts(tmp_path):
    path = tmp_path / "sn2_ts.xyz"
    path.write_text(SN2_TS)
    return str(path)


@pytest.fixture
def sn2_ts_symmetric(tmp_path):
    path = tmp_path / "sn2_ts_symmetric.xyz"
    path.write_text(SN2_TS_SYMMETRIC)
    return str(path)


@pytest.fixture
def boronic_acid(tmp_path):
    """Allylboronic acid: boron has no MMFF parameters, so refinement falls back to UFF."""
    mol = Chem.AddHs(Chem.MolFromSmiles("C=CCB(O)O"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    path = tmp_path / "boronic_acid.xyz"
    Chem.MolToXYZFile(mol, str(path))
    return str(path)
