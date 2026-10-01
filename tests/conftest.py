import os

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
EX = os.path.join(DATA, "ex.xyz")  # hept-1-ene TS, reacting atoms 3, 4, 5

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


# The early SN2 TS with a water hydrogen-bonded to the nucleophile:
# Cl2...H7 = 2.20 A and Cl2...O6 = 3.16 A.
SN2_TS_WATER = SN2_TS.replace("6\n", "9\n", 1).replace("(early)", "(early) + water") + (
    "O      0.000000     0.000000    -5.660000\n"
    "H      0.000000     0.000000    -4.700000\n"
    "H      0.929422     0.000000    -5.900365\n"
)

# A second water donates a hydrogen bond to the first one (O6...H10 = 1.95 A), which is
# closer to it (relative to van der Waals radii) than to the nucleophile.
SN2_TS_TWO_WATERS = SN2_TS_WATER.replace("9\n", "12\n", 1).replace(
    "+ water", "+ two waters"
) + (
    "O     -1.746000     0.000000    -7.988000\n"
    "H     -1.170000     0.000000    -7.220000\n"
    "H     -2.634000     0.000000    -7.623000\n"
)


@pytest.fixture
def sn2_ts_water(tmp_path):
    path = tmp_path / "sn2_ts_water.xyz"
    path.write_text(SN2_TS_WATER)
    return str(path)


@pytest.fixture
def sn2_ts_two_waters(tmp_path):
    path = tmp_path / "sn2_ts_two_waters.xyz"
    path.write_text(SN2_TS_TWO_WATERS)
    return str(path)


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


@pytest.fixture(scope="session")
def hept_1_ene_ts():
    """The TS of ex.xyz: the graph of its SMILES with the TS geometry (do not modify)."""
    from racerts.system import build_mol

    return build_mol(EX, 0, [3, 4, 5], input_smiles=["CCCCCC=C"])
