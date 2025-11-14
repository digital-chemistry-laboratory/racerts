from abc import abstractmethod

from rdkit import Chem
from rdkit.Chem import rdEHTTools
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem import rdMolAlign

import numpy as np


class BasePruner:
    @abstractmethod
    def __init__(self, threshold: float, verbose: bool = False, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def prune(self, mol: Chem.Mol) -> Chem.Mol:
        raise NotImplementedError


class EnergyPruner(BasePruner):

    def __init__(self, threshold: float = 20.0, verbose: bool = False, **kwargs):
        self.YAeHMOP_energies = kwargs.get("YAeHMOP_energies", False)
        self.threshold = threshold
        self.verbose = verbose

    def set_QM_energies(self, mol, verbose=False):
        """
        Sets quantum mechanical (QM) energies for all conformers of a molecule using RDKit's
        EHT tools. QM energies are set as a property on each conformer.

        Args:
            mol (RDKit Mol): The molecule whose conformers will have QM energies set.
            verbose (bool, optional): If True, prints failure messages for conformers where
                                    QM energy calculation fails. Defaults to False.
        """
        idx = [conf.GetId() for conf in mol.GetConformers()]

        for id in idx:
            passed, res = rdEHTTools.RunMol(mol, confId=id)

            if passed is True:
                e = res.totalEnergy
                mol.GetConformer().SetDoubleProp("energy", e)
            else:
                if verbose is True:
                    print("failure of rdEHTTools")

    def get_minimal_energy(self, mol, verbose=False):
        """
        Finds and returns the minimal energy among all conformers of a molecule. The minimal
        energy is also set as a property on the molecule.

        Args:
            mol (RDKit Mol): The molecule to evaluate.
            verbose (bool, optional): If True, prints the minimal energy found. Defaults to False.

        Returns:
            float: The minimal energy found among all conformers.
        """
        if "minimal_energy" not in mol.GetPropNames():

            min_energy = np.min(
                [np.inf]
                + [
                    conf.GetDoubleProp("energy")
                    for conf in mol.GetConformers()
                    if conf.HasProp("energy")
                ]
            )
            mol.SetDoubleProp("minimal_energy", min_energy)

        if verbose is True:
            print(f"Minimal energy conformer: {min_energy} kcal/mol")

        return mol.GetDoubleProp("minimal_energy")

    def prune(self, mol: Chem.Mol):

        if self.YAeHMOP_energies:
            self.set_QM_energies(mol, self.verbose)

        min_energy = self.get_minimal_energy(mol, self.verbose)
        conformers_to_remove = []

        for conf in mol.GetConformers():
            if conf.GetDoubleProp("energy") - min_energy > self.threshold:
                conformers_to_remove.append(conf.GetId())

        for conf_id in conformers_to_remove:
            mol.RemoveConformer(conf_id)

        return mol


class RMSDPruner(BasePruner):
    def __init__(self, threshold=0.125, verbose=False, **kwargs):
        self.include_hs = kwargs.get("include_hs", False)
        self.threshold = threshold
        self.verbose = verbose
        self.num_threads = kwargs.get("num_threads", 1)
        self.filter_energies = kwargs.get("filter_energies", True)
        self.filter_rotations = kwargs.get("filter_rotations", True)
        self.energy_threshold = kwargs.get("energy_threshold", 0.1)
        self.rot_fraction_threshold = kwargs.get("rot_fraction_threshold", 0.03)
        self.maxMatches = kwargs.get("maxMatches", 10000)

    def get_sorted_conf_energy(self, mol):
        """
        Returns a list of molecule conformers sorted by their energy, if energy properties
        are set.

        Args:
            mol (RDKit Mol): The molecule whose conformers to sort.

        Returns:
            list of RDKit Conformers: The sorted list of conformers, or None if any conformer
                                    does not have an energy property set.
        """
        for conf in mol.GetConformers():
            if not conf.HasProp("energy"):
                return mol.GetConformers()

        sorted_list = sorted(
            mol.GetConformers(), key=lambda x: x.GetDoubleProp("energy")
        )

        return sorted_list

    def prune(self, mol):

        conf_idx = [conf.GetId() for conf in self.get_sorted_conf_energy(mol)]

        candidates = np.array(conf_idx)
        keep_list = []

        while len(candidates) > 0:
            keeper = candidates[0]
            keep_list.append(keeper)

            similarity = self.check_similarity(
                mol=mol,
                id=keeper,
                j_s=candidates,
                filter_energies=self.filter_energies,
                filter_rotations=self.filter_rotations,
                energy_threshold=self.energy_threshold,
                rot_fraction_threshold=self.rot_fraction_threshold,
                maxMatches=self.maxMatches,
            )

            candidates = candidates[similarity]

        conformers_to_remove = [
            conf.GetId()
            for conf in mol.GetConformers()
            if conf.GetId() not in keep_list
        ]

        for id in conformers_to_remove:
            mol.RemoveConformer(id)

        return mol

    def calc_rotations(self, m, id):
        return (
            rdMolDescriptors.CalcPMI1(m, confId=id),
            rdMolDescriptors.CalcPMI2(m, confId=id),
            rdMolDescriptors.CalcPMI3(m, confId=id),
        )

    def check_similarity(
        self,
        mol,
        id,
        j_s,
        filter_energies=True,
        filter_rotations=True,
        energy_threshold=0.05,
        rot_fraction_threshold=0.03,
        maxMatches=100000,
    ):

        ref_mol = Chem.Mol(mol)  # only the current candidate
        ref_conformer = mol.GetConformer(int(id))
        ref_mol.RemoveAllConformers()
        id = ref_mol.AddConformer(ref_conformer, assignId=True)

        filter_energies = filter_energies and ref_conformer.HasProp("energy")
        if filter_energies:
            if ref_conformer.HasProp("energy"):
                ref_energy = ref_conformer.GetDoubleProp("energy")
            else:
                print("No energy property provided!")

        if filter_rotations:
            ref_rotations = self.calc_rotations(ref_mol, id=id)

        checked = []

        for j in j_s:
            conf = mol.GetConformer(int(j))

            # check energy similarity: conformers with significant energy difference unlikely structurally very similar
            if filter_energies:
                if conf.HasProp("energy"):
                    delta_e = abs(ref_energy - conf.GetDoubleProp("energy"))
                    if delta_e > energy_threshold:
                        checked.append(True)
                        continue
                else:
                    print("No energy property provided!")

            # check rotational similarity: conformers with signigicant rotational constance difference unlikely structurally very similar
            if filter_rotations:
                rot = self.calc_rotations(mol, id=int(j))
                check = False
                for i in range(3):
                    f_rot = abs(ref_rotations[i] - rot[i]) / ref_rotations[i]
                    if f_rot > rot_fraction_threshold:
                        check = True
                        break
                if check:
                    checked.append(True)
                    continue

            # check rmsd similarity
            ref_align_mol = Chem.Mol(ref_mol)
            ref_align_mol.AddConformer(Chem.Conformer(conf))

            if self.include_hs is False:
                try:
                    ref_align_mol = Chem.RemoveHs(ref_align_mol, sanitize=True)
                except Exception:
                    ref_align_mol = Chem.RemoveHs(ref_align_mol, sanitize=False)

            rmsd = self.calc_rmsd(
                ref_align_mol, ref_align_mol, -1, int(j), maxMatches=maxMatches
            )
            checked.append(rmsd > self.threshold)

        return checked

    def calc_rmsd(self, mol1, mol2, id_1, id_2, maxMatches=10000, maps=None):
        if maps is None:
            maps = self.get_atom_maps(mol1, mol2, maxMatches)
        rmsd = rdMolAlign.GetBestRMS(
            mol1,
            mol2,
            prbId=id_1,
            refId=id_2,
            numThreads=self.num_threads,
            map=maps,
            symmetrizeConjugatedTerminalGroups=True,
        )
        return rmsd

    def get_atom_maps(self, mol1, mol2, maxMatches, symmetrize=True):

        if symmetrize:
            mol1 = self.symmetrize_terminal_atoms(mol1)
            mol2 = self.symmetrize_terminal_atoms(mol2)
        maps = mol1.GetSubstructMatches(
            mol2,
            maxMatches=maxMatches,
            uniquify=False,
            useChirality=True,
            useQueryQueryMatches=False,
        )
        maps = [[(i, j) for i, j in enumerate(list(matches))] for matches in maps]
        return maps

    def symmetrize_terminal_atoms(self, mol):
        """
        Symmetrize terminal O or N atoms (degree 1) in specific bonding patterns:
        - Sets formal charge to 0
        - Replaces their bond with an unspecified bond (to generalize single/double)

        Args:
            mol (Chem.Mol or Chem.RWMol): Input molecule

        Returns:
            Chem.RWMol: Modified molecule
        """
        # Ensure mol is editable
        rw_mol = Chem.RWMol(mol)

        atom_pattern = "O,N;D1"
        qsmarts = f"[{atom_pattern};$([{atom_pattern}]-[*]=[{atom_pattern}]),$([{atom_pattern}]=[*]-[{atom_pattern}])]~[*]"
        qry = Chem.MolFromSmarts(qsmarts)

        matches = rw_mol.GetSubstructMatches(qry)
        if not matches:
            return rw_mol  # return unchanged

        for match in matches:
            atom_idx, nbr_idx = match[0], match[1]
            atom = rw_mol.GetAtomWithIdx(atom_idx)
            atom.SetFormalCharge(0)
            bond = rw_mol.GetBondBetweenAtoms(atom_idx, nbr_idx)
            if bond is None:
                raise RuntimeError("could not find expected bond")
            rw_mol.RemoveBond(atom_idx, nbr_idx)
            rw_mol.AddBond(atom_idx, nbr_idx, Chem.BondType.UNSPECIFIED)

        return rw_mol
