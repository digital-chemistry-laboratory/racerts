from rdkit import Chem
from rdkit.Chem import AllChem

m = Chem.MolFromSmiles("CCCCCC=C")
m = Chem.AddHs(m)
AllChem.EmbedMolecule(m)

Chem.MolToXYZFile(m, "ex.xyz")