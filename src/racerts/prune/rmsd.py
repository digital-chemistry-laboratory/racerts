"""Pruning of duplicates by RMSD (after energy and rotational-constant filters)."""

import inspect
import logging

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolAlign, rdMolDescriptors

from .base import BasePruner, check_threshold, drop_conformers_without_energy

logger = logging.getLogger(__name__)

# Terminal O or N (degree 1) in X-*=X or X=*-X, e.g. carboxylate or nitro oxygens.
_TERMINAL = "O,N;D1"
_TERMINAL_O_N = Chem.MolFromSmarts(
    f"[{_TERMINAL};$([{_TERMINAL}]-[*]=[{_TERMINAL}]),$([{_TERMINAL}]=[*]-[{_TERMINAL}])]"
    "~[*]"
)


class RMSDPruner(BasePruner):
    """
    Drops duplicates: conformers within threshold (A, symmetry-aware heavy-atom RMSD,
    or all atoms with include_hs) of a lower one. Pairs whose energies differ by more
    than energy_threshold (kcal/mol; catmlp's default 0.1 is in eV) or whose principal
    moments of inertia differ by more than rot_fraction_threshold are not compared.
    """

    def __init__(self, threshold=0.125, verbose=False, **kwargs):
        check_threshold(threshold, "threshold")
        for name in ("energy_threshold", "rot_fraction_threshold"):
            if name in kwargs:
                check_threshold(kwargs[name], name)
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

        if any(conf.HasProp("energy") for conf in mol.GetConformers()):
            drop_conformers_without_energy(mol)

        conf_idx = [conf.GetId() for conf in self.get_sorted_conf_energy(mol)]
        # The atom maps depend on the graph only: computed once, not for every pair
        # (not for legacy subclasses whose check_similarity takes no maps).
        options = {}
        if len(conf_idx) > 1 and _takes_maps(self.check_similarity):
            options["maps"] = self.symmetry_maps(mol)

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
                **options,
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
        maps=None,
    ):

        ref_mol = Chem.Mol(mol)  # only the current candidate
        ref_conformer = mol.GetConformer(int(id))
        ref_mol.RemoveAllConformers()
        id = ref_mol.AddConformer(ref_conformer, assignId=True)

        filter_energies = filter_energies and ref_conformer.HasProp("energy")
        if filter_energies:
            ref_energy = ref_conformer.GetDoubleProp("energy")

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
                    logger.warning("Conformer %d has no energy property.", int(j))

            # check rotational similarity: conformers with signigicant rotational constance difference unlikely structurally very similar
            if filter_rotations:
                rot = self.calc_rotations(mol, id=int(j))
                check = False
                for i in range(3):
                    difference = abs(ref_rotations[i] - rot[i])
                    if ref_rotations[i] == 0.0:
                        f_rot = 0.0 if rot[i] == 0.0 else np.inf
                    else:
                        f_rot = difference / ref_rotations[i]
                    if f_rot > rot_fraction_threshold:
                        check = True
                        break
                if check:
                    checked.append(True)
                    continue

            # check rmsd similarity
            ref_align_mol = Chem.Mol(ref_mol)
            # A new id: keeping id j would clash with the reference copy when j == 0.
            candidate_id = ref_align_mol.AddConformer(
                Chem.Conformer(conf), assignId=True
            )

            if self.include_hs is False:
                try:
                    ref_align_mol = Chem.RemoveHs(ref_align_mol, sanitize=True)
                except Exception:
                    ref_align_mol = Chem.RemoveHs(ref_align_mol, sanitize=False)

            rmsd = self.calc_rmsd(
                ref_align_mol,
                ref_align_mol,
                -1,
                candidate_id,
                maxMatches=maxMatches,
                maps=maps,
            )
            checked.append(rmsd > self.threshold)

        return checked

    def symmetry_maps(self, mol):
        """
        The atom maps of the symmetry-aware RMSD of mol (without hydrogens unless
        include_hs), as check_similarity uses them; the same for every conformer pair.
        """
        graph = Chem.Mol(mol)
        graph.RemoveAllConformers()
        if self.include_hs is False:  # as in check_similarity
            try:
                graph = Chem.RemoveHs(graph, sanitize=True)
            except Exception:
                graph = Chem.RemoveHs(graph, sanitize=False)
        maps = self.get_atom_maps(graph, graph, self.maxMatches)
        if len(maps) >= self.maxMatches:
            logger.warning(
                "The symmetry matches reach maxMatches=%d, so the RMSD may miss "
                "equivalent atom mappings, and duplicates of symmetric structures (e.g. "
                "identical solvent molecules) can remain; increase maxMatches.",
                self.maxMatches,
            )
        return maps

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
            same = mol2 is mol1
            mol1 = self.symmetrize_terminal_atoms(mol1)
            mol2 = mol1 if same else self.symmetrize_terminal_atoms(mol2)
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

        matches = rw_mol.GetSubstructMatches(_TERMINAL_O_N)
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


def _takes_maps(method) -> bool:
    try:
        return "maps" in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
