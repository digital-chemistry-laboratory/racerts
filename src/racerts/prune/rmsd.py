"""Pruning of duplicates by RMSD (after energy and rotational-constant filters)."""

import inspect
import logging

import numpy as np
from rdkit.Chem import rdMolDescriptors

from racerts.geometry import (
    atom_matches,
    rmsd,
    rmsd_within,
    symmetrize_terminal_atoms,
    symmetry_maps,
)

from .base import BasePruner, check_threshold, drop_conformers_without_energy

logger = logging.getLogger(__name__)


class RMSDPruner(BasePruner):
    """
    Drops duplicates: conformers within threshold (A) of a lower one by the RMSD of
    racerts.geometry: over the heavy atoms (all atoms with include_hs), the smallest
    over the symmetry maps of the graph, after superposition (align=False: in the frame
    that the conformers share, e.g. of a frozen core). The RMSD is computed only for
    pairs whose energies differ by at most energy_threshold (kcal/mol; catmlp's default
    0.1 is in eV) and whose principal moments of inertia differ by at most
    rot_fraction_threshold.
    """

    def __init__(self, threshold=0.125, verbose=False, **kwargs):
        check_threshold(threshold, "threshold")
        for name in ("energy_threshold", "rot_fraction_threshold"):
            if name in kwargs:
                check_threshold(kwargs[name], name)
        self.include_hs = kwargs.get("include_hs", False)
        self.align = kwargs.get("align", True)
        self.threshold = threshold
        self.verbose = verbose
        self.num_threads = kwargs.get("num_threads", 1)  # unused (legacy)
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
        # The atoms and symmetry maps depend on the graph only: computed once, not for
        # every pair (not for legacy subclasses whose check_similarity takes none).
        options = {}
        if len(conf_idx) > 1 and _takes(self.check_similarity, "symmetry"):
            options["symmetry"] = symmetry_maps(mol, self.include_hs, self.maxMatches)

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
        symmetry=None,
    ):
        """
        For each conformer j of j_s, whether it differs from conformer id: in energy,
        in principal moments of inertia (filters), or else by an RMSD above threshold.
        symmetry: the atoms and maps of the RMSD (racerts.geometry.symmetry_maps).
        """
        ref_conformer = mol.GetConformer(int(id))
        filter_energies = filter_energies and ref_conformer.HasProp("energy")
        if filter_energies:
            ref_energy = ref_conformer.GetDoubleProp("energy")
        if filter_rotations:
            ref_rotations = self.calc_rotations(mol, id=int(id))
        if symmetry is None:
            symmetry = symmetry_maps(mol, self.include_hs, maxMatches)
        ref_positions = ref_conformer.GetPositions()

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
            duplicate = rmsd_within(
                ref_positions,
                conf.GetPositions(),
                self.threshold,
                symmetry.atoms,
                symmetry.maps,
                align=self.align,
            )
            checked.append(not duplicate)

        return checked

    def calc_rmsd(self, mol1, mol2, id_1, id_2, maxMatches=10000, maps=None):
        """
        The RMSD between conformer id_1 of mol1 and id_2 of mol2 (graphs of the same
        atoms) after superposition, the smallest over maps: lists of (index in mol1,
        index in mol2) pairs, by default get_atom_maps(mol1, mol2). The pruner uses
        racerts.geometry directly.
        """
        if maps is None or len(maps) == 0:
            maps = self.get_atom_maps(mol1, mol2, maxMatches)
        if not maps:
            raise ValueError("The graphs of mol1 and mol2 do not match.")
        pairs = np.asarray(maps, dtype=np.intp)
        a = mol1.GetConformer(int(id_1)).GetPositions()
        b = mol2.GetConformer(int(id_2)).GetPositions()
        return min(rmsd(a[p[:, 0]], b[p[:, 1]]) for p in pairs)

    def get_atom_maps(self, mol1, mol2, maxMatches, symmetrize=True):
        """The matches of mol2 in mol1 as lists of (index, index) pairs."""
        matches = atom_matches(mol1, mol2, maxMatches, symmetrize)
        return [list(enumerate(match)) for match in matches]

    def symmetrize_terminal_atoms(self, mol):
        """racerts.geometry.symmetrize_terminal_atoms."""
        return symmetrize_terminal_atoms(mol)


def _takes(method, name: str) -> bool:
    try:
        return name in inspect.signature(method).parameters
    except (TypeError, ValueError):
        return False
