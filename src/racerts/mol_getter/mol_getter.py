from abc import abstractmethod
from typing import List, Union

from rdkit import Chem
from rdkit.Chem.AllChem import SanitizeMol  # type: ignore
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import rdFMCS
from racerts.utils import suppress_std


class BaseMolGetter:
    @abstractmethod
    def __init__(self, assignBonds: bool, allowChargedFragments: bool, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_mol(self, file_name: str, **kwargs) -> Union[Chem.Mol, None]:
        """
        Create a molecular graph from an XYZ file.
        """
        raise NotImplementedError


class MolGetterBonds(BaseMolGetter):
    def __init__(
        self, assignBonds: bool = True, allowChargedFragments: bool = True, **kwargs
    ):
        self.assignBonds = assignBonds
        self.allowChargedFragments = allowChargedFragments

    def get_mol(self, file_name: str, **kwargs) -> Union[Chem.Mol, None]:
        """
        Create a molecular graph from an XYZ file.

        Args:
            file_name (str): The path to the XYZ file.
            charge (int): The molecular charge.
        Returns:
            Chem.Mol: The molecule object.
        """
        charge = kwargs.get("charge", 0)
        if "charge" not in kwargs:
            print("No charge provided, defaulting to 0")

        mol_ts = Chem.MolFromXYZFile(file_name)

        if mol_ts is None:
            return None

        if self.assignBonds:
            with suppress_std():
                rdDetermineBonds.DetermineBonds(
                    mol_ts,
                    charge=charge,
                    allowChargedFragments=self.allowChargedFragments,
                )

        if mol_ts is None:
            raise ValueError(
                f"Failed to create molecule from {file_name}. Check the file format and content."
            )

        return mol_ts


class MolGetterConnectivity(BaseMolGetter):

    def __init__(self, **kwargs):
        pass

    def get_mol(self, file_name: str, **kwargs) -> Union[Chem.Mol, None]:
        """
        Create a molecular graph from an XYZ file.

        Args:
            file_name (str): The path to the XYZ file.
            charge (int): The molecular charge.
        Returns:
            Chem.Mol: The molecule object.
        """

        mol_ts = Chem.MolFromXYZFile(file_name)

        if mol_ts is None:
            return None

        # if self.assignBonds:
        rdDetermineBonds.DetermineConnectivity(mol_ts)
        SanitizeMol(
            mol_ts,
            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        )
        Chem.AssignStereochemistryFrom3D(mol_ts)

        if mol_ts is None:
            raise ValueError(
                f"Failed to create molecule from {file_name}. Check the file format and content."
            )
        return mol_ts


class MolGetterSMILES(BaseMolGetter):

    def __init__(self, **kwargs):
        pass

    def combine_mols(self, smiles_list) -> Chem.Mol:
        if not isinstance(smiles_list, list):
            raise ValueError("Input SMILES must be provided as a list.")

        combined_mol = Chem.AddHs(Chem.MolFromSmiles(smiles_list[0]))

        if len(smiles_list) > 1:
            for smiles in smiles_list[1:]:
                mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
                combined_mol = Chem.CombineMols(combined_mol, mol)

        return combined_mol

    def match_AtomMapNum(self, ref_mol: Chem.Mol, mol: Chem.Mol) -> List[Chem.Mol]:
        """
        Match the atom map numbers for the reference molecule and the molecule of interest.
        The atom map is arbitrarly chosen for the reference molecule if not available, and the
        one for the molecule of interest is matched by iteratively searching for
        the MCS.

        Developer info: MCS works fine until two or more possible MCS
        are possible for the structure (Try GetSubstructureMatches to get this info)...
        this usually happens for reacting atoms, which are anyway fixed later.

        Args:
            ref_mol (Chem.Mol): The molecule of reference
            mol (Chem.Mol): The molecule of interest

        Returns:
            List[Chem.Mol]: The atom-maped molecule objects that are returned.
        """
        if ref_mol.GetAtomWithIdx(0).GetAtomMapNum() == 0:
            for atom in ref_mol.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx() + 1)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(-atom.GetIdx() - 1)

        truncate_ref = Chem.RWMol(ref_mol)
        truncate_mol = Chem.RWMol(mol)
        while truncate_ref.GetNumAtoms() > 0 and truncate_mol.GetNumAtoms() > 0:
            MCresult = rdFMCS.FindMCS(
                [truncate_ref, truncate_mol], bondCompare=rdFMCS.BondCompare.CompareAny
            )
            mc_mol = MCresult.queryMol
            highlight_mcs_r = truncate_ref.GetSubstructMatch(mc_mol, useChirality=False)
            highlight_mcs_p = truncate_mol.GetSubstructMatch(mc_mol, useChirality=False)
            for id in range(len(highlight_mcs_r)):
                id_map_r = truncate_ref.GetAtomWithIdx(
                    highlight_mcs_r[id]
                ).GetAtomMapNum()
                id_map_p = truncate_mol.GetAtomWithIdx(
                    highlight_mcs_p[id]
                ).GetAtomMapNum()
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() == id_map_p:
                        atom.SetAtomMapNum(id_map_r)
            truncate_ref.BeginBatchEdit()
            for id in highlight_mcs_r:
                truncate_ref.RemoveAtom(id)
            truncate_ref.CommitBatchEdit()
            truncate_mol.BeginBatchEdit()
            for id in highlight_mcs_p:
                truncate_mol.RemoveAtom(id)
            truncate_mol.CommitBatchEdit()

        return [ref_mol, mol]

    def set_coords(
        self, pmol: Chem.Mol, mol_ts: Chem.Mol, reacting_atoms: List[int]
    ) -> Chem.Mol:
        """
        Assign the topoly (connectivity and bond orders) from the SMILES structure to the xyz file of the transition state.
        Only bonds present between reacting atoms presents in pmol and in mol_ts after determining the connectivity will be copied.

        Args:
            pmol (Chem.Mol): Mol object from which the topology (connectivity and bond orders) is taken
            mol_ts (Chem.Mol): Mol object with 3D coordinates but no topology
            reacting_atoms (List[int]): List of the reacting atoms based on the indexes of mol_ts

        Returns:
            Chem.Mol: The final molecule with coordinates and bond order information
        """
        for atom in pmol.GetAtoms():
            for atom2 in mol_ts.GetAtoms():
                if atom.GetAtomMapNum() == atom2.GetAtomMapNum():
                    atom2.SetFormalCharge(atom.GetFormalCharge())
        emol = Chem.EditableMol(mol_ts)
        for bond in mol_ts.GetBonds():
            emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in pmol.GetBonds():
            for atom in mol_ts.GetAtoms():
                if atom.GetAtomMapNum() == bond.GetBeginAtom().GetAtomMapNum():
                    id1 = atom.GetIdx()
                elif atom.GetAtomMapNum() == bond.GetEndAtom().GetAtomMapNum():
                    id2 = atom.GetIdx()
            if id1 not in reacting_atoms or id2 not in reacting_atoms:
                emol.AddBond(id1, id2, order=bond.GetBondType())
            elif mol_ts.GetBondBetweenAtoms(id1, id2) is not None:
                emol.AddBond(id1, id2, order=bond.GetBondType())
        new_mol = emol.GetMol()
        Chem.SanitizeMol(new_mol)
        Chem.SetDoubleBondNeighborDirections(new_mol, new_mol.GetConformer())
        Chem.DetectBondStereochemistry(new_mol)
        for (
            atom
        ) in (
            new_mol.GetAtoms()
        ):  # <- We need to remove the atom map numbers to avoid the artifacts
            atom.SetAtomMapNum(0)
        Chem.AssignStereochemistryFrom3D(
            new_mol
        )  # <- Now we use the 3D conformer to set the stereochemistry
        Chem.AssignCIPLabels(new_mol)
        # Chem.AssignStereochemistry(new_mol) <- This is deprecated in favor of Chem.AssignCIPLabels(new_mol) above
        return new_mol

    def get_mol(self, file_name: str, **kwargs) -> Chem.Mol:

        if "input_smiles" not in kwargs:
            raise ValueError("No input SMILES provided in the SMILES method.")
        if "reacting_atoms" not in kwargs:
            raise ValueError("No reacting atoms provided.")

        input_smiles = kwargs.get("input_smiles", None)
        reacting_atoms = kwargs.get("reacting_atoms", [])

        if type(input_smiles) is str:
            input_smiles = [input_smiles]
        if not type(input_smiles) is list and type(reacting_atoms) is not list:
            raise ValueError(
                "Input SMILES and reacting atoms must be provided as a list."
            )
        input_mol = self.combine_mols(input_smiles)

        mol_ts = Chem.MolFromXYZFile(file_name)
        new_mol = self.setup_mol(mol_ts, reacting_atoms, input_mol)
        if new_mol is None:
            raise ValueError(
                f"Failed to create molecule from {file_name}. Check the file format and content."
            )
        return new_mol

    def setup_mol(self, mol_ts, reacting_atoms, input_mol):
        rdDetermineBonds.DetermineConnectivity(mol_ts)
        # Chem.AssignStereochemistryFrom3D(mol_ts) <- This is not needed at this point as chirality is not used in atom mapping, we will do it later.
        [input_mol, mol_ts] = self.match_AtomMapNum(input_mol, mol_ts)

        new_mol = self.set_coords(input_mol, mol_ts, reacting_atoms)

        return new_mol
