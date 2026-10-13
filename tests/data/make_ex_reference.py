"""Write the reference ensemble of test_ensemble_matches_the_0_1_7_release.

Run with racerts 0.1.7 importable (e.g. a checkout of 528834e) and RDKit 2025.03.2, the
version the test expects (embedding results differ between RDKit versions):

    PYTHONPATH=<racerts 0.1.7>/src python tests/data/make_ex_reference.py

It writes tests/data/ex_0.1.7_rdkit<version>.xyz: the ensemble of ex.xyz (hept-1-ene TS,
reacting atoms 3, 4, 5, SMILES CCCCCC=C, 50 embedded conformers, default seed), with the
conformer id and the MMFF energy in kcal/mol on each comment line.
"""

import os

from rdkit import rdBase

from racerts import ConformerGenerator

HERE = os.path.dirname(os.path.abspath(__file__))

mol = ConformerGenerator().generate_conformers(
    os.path.join(HERE, "ex.xyz"),
    0,
    [3, 4, 5],
    input_smiles=["CCCCCC=C"],
    number_of_conformers=50,
)
path = os.path.join(HERE, f"ex_0.1.7_rdkit{rdBase.rdkitVersion}.xyz")
with open(path, "w") as handle:
    for conf in mol.GetConformers():
        energy = conf.GetDoubleProp("energy")
        handle.write(f"{mol.GetNumAtoms()}\nconf_id={conf.GetId()} energy={energy!r}\n")
        for atom, (x, y, z) in zip(mol.GetAtoms(), conf.GetPositions()):
            handle.write(f"{atom.GetSymbol()} {x:.8f} {y:.8f} {z:.8f}\n")
print(f"wrote {path}")
