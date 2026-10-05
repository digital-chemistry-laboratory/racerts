"""Pruning of duplicates by RMSD."""

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
from racerts.symmetry import SymmetricRMSD, bound_descriptors, symmetry_classes

from .base import BasePruner, check_threshold, drop_conformers_without_energy

logger = logging.getLogger(__name__)

# amu A^2: a principal moment of inertia below this is zero (the axis of a linear
# molecule, a single atom). Computed moments of such axes are rounding noise (1e-16 to
# 1e-10); the smallest real moment, of H2, is 0.28.
ZERO_MOMENT = 1e-6
HYDROGENS = ("none", "polar", "all")


class RMSDPruner(BasePruner):
    """
    Drops duplicates: conformers within threshold (A) of a lower one by the RMSD over
    the symmetry of the graph, after superposition (align=False: in the frame that the
    conformers share, e.g. of a frozen core).

    hydrogens: which hydrogens count, "none" (the heavy atoms), "polar" (also the
    hydrogens on N, O, P and S, so that the rotamers of a hydrogen bond stay apart) or
    "all"; include_hs=True is "all".

    Without the two prefilters (filter_energies=False, filter_rotations=False) every
    pair is decided by its RMSD: a pair is skipped only if a lower bound of the RMSD
    (racerts.symmetry.bound_descriptors) is above the threshold, and the RMSD does not
    list more than maxMatches equivalent atom mappings (racerts.symmetry.SymmetricRMSD).

    With a prefilter (the defaults of legacy racerts) the RMSD is computed only for
    pairs whose energies differ by at most energy_threshold (kcal/mol) and whose
    principal moments of inertia differ by at most rot_fraction_threshold, over a list
    of at most maxMatches atom mappings. Conformers that are duplicates by their RMSD
    can pass these two filters apart and both stay.
    """

    def __init__(self, threshold=0.125, verbose=False, **kwargs):
        check_threshold(threshold, "threshold")
        for name in ("energy_threshold", "rot_fraction_threshold"):
            if name in kwargs:
                check_threshold(kwargs[name], name)
        hydrogens = kwargs.get("hydrogens")
        if hydrogens is None:
            hydrogens = "all" if kwargs.get("include_hs", False) else "none"
        if hydrogens not in HYDROGENS:
            raise ValueError(
                f"hydrogens must be one of {HYDROGENS}, not {hydrogens!r}."
            )
        self.hydrogens = hydrogens
        self.include_hs = hydrogens == "all"
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
        The conformers of mol sorted by energy, or in their own order if any of them
        has no energy.
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
        prefiltered = self.filter_energies or self.filter_rotations
        overridden = type(self).check_similarity is not RMSDPruner.check_similarity
        if not (prefiltered or overridden):
            keep = self._by_rmsd_alone(mol, conf_idx)
            for conf_id in [i for i in conf_idx if i not in keep]:
                mol.RemoveConformer(conf_id)
            return mol
        if self.hydrogens == "polar":
            raise ValueError(
                "hydrogens='polar' needs filter_energies=False and "
                "filter_rotations=False: the prefilters of legacy racerts work with "
                "no or all hydrogens."
            )
        # The atoms and symmetry maps depend on the graph only: computed once, not for
        # every pair (not for legacy subclasses whose check_similarity takes none).
        options = {}
        if len(conf_idx) > 1 and _takes(self.check_similarity, "symmetry"):
            options["symmetry"] = symmetry_maps(mol, self.include_hs, self.maxMatches)

        candidates = np.array(conf_idx)
        keep_list = []

        while len(candidates) > 0:
            # The keeper is not compared with itself: its RMSD to itself is a
            # rounding error above zero, which threshold 0 would not remove.
            keeper, candidates = candidates[0], candidates[1:]
            keep_list.append(keeper)
            if not len(candidates):
                break

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

    def _by_rmsd_alone(self, mol, conf_ids) -> set:
        """
        The conformers to keep, of conf_ids in their order: each is compared with the
        kept ones that a lower bound of the RMSD does not already tell apart, the
        closest by that bound first, until one is within the threshold.
        """
        if len(conf_ids) < 2:
            return set(conf_ids)
        kernel = SymmetricRMSD(
            mol,
            self.hydrogens,
            max_maps=self.maxMatches,
            hard_max_maps=max(self.maxMatches, 10000),
        )
        classes = symmetry_classes(kernel.graph, kernel.weight)
        kept, prepared, described = [], [], ([], [])
        for conf_id in conf_ids:
            positions = mol.GetConformer(int(conf_id)).GetPositions()
            own = bound_descriptors(positions, kernel.index, kernel.weight, classes)
            candidate = kernel.prepare(positions, align=self.align)
            duplicate = False
            if kept:
                bounds = np.maximum(
                    *(
                        np.linalg.norm(np.array(others) - vector, axis=1)
                        for others, vector in zip(described, own)
                    )
                )
                for k in np.argsort(bounds):
                    if bounds[k] > self.threshold + 1e-9:
                        break
                    if kernel.within(
                        prepared[k], candidate, self.threshold, align=self.align
                    ):
                        duplicate = True
                        break
            if not duplicate:
                kept.append(conf_id)
                prepared.append(candidate)
                for others, vector in zip(described, own):
                    others.append(vector)
        return set(kept)

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
                    if ref_rotations[i] <= ZERO_MOMENT:
                        f_rot = 0.0 if rot[i] <= ZERO_MOMENT else np.inf
                    else:
                        f_rot = abs(ref_rotations[i] - rot[i]) / ref_rotations[i]
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
        The RMSD between conformer id_1 of mol1 and id_2 of mol2 after superposition,
        the smallest over maps: lists of index pairs as get_atom_maps gives them (the
        default). Meant for mol1 and mol2 with the same atom order, as in legacy
        racerts; the pruner uses racerts.geometry directly.
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
        """
        The matches of mol2 in mol1 as lists of (mol2 index, mol1 index) pairs.
        calc_rmsd reads the pairs the other way round, as legacy racerts does, which
        is the same for two conformers of one molecule.
        """
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
