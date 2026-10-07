"""Pruning of duplicates by RMSD."""

import logging

import numpy as np

from racerts.symmetry import LISTED_MAPS, bound_descriptors, symmetry_classes
from racerts.utils.checks import is_integer

from .base import (
    BasePruner,
    check_graph,
    check_threshold,
    drop_conformers_without_energy,
    symmetry_kernel,
)

logger = logging.getLogger(__name__)

HYDROGENS = ("none", "polar", "all")
MAX_MAPS = 100  # equivalent atom mappings that are listed; see racerts.symmetry


class RMSDPruner(BasePruner):
    """
    Drops duplicates: conformers within threshold (A) of a lower one by the RMSD over
    the symmetry of the graph, after superposition (align=False: in the frame that the
    conformers share, e.g. of a frozen core). The conformers are taken by energy, so
    the lowest of a set of duplicates stays (in their own order if one has no energy).

    Every pair is decided by its RMSD; a pair is skipped only where a lower bound of
    the RMSD (racerts.symmetry.bound_descriptors) is above the threshold, or where
    the energies are further apart than energy_tolerance.

    Args:
        hydrogens: Which hydrogens count: "none" (the heavy atoms), "polar" (also
            those on N, O, P and S, so that the rotamers of a hydrogen bond stay
            apart) or "all".
        max_maps: The most equivalent atom mappings that are listed; above it local
            symmetry is assigned without a list (racerts.symmetry.SymmetricRMSD).
        graph: For conformers that are stored without bonds (e.g. read from
            coordinates): a molecule of the same atoms in the same order with the
            bonds. The equivalent atoms and the hydrogens that count are read from
            it. Without bonds, all atoms of an element count as equivalent.
        energy_tolerance: Conformers whose energies differ by more than this
            (kcal/mol) are not duplicates, however close they are: two refined
            structures with different energies are two stationary points. For
            ensembles that are refined at one level (all conformers need an energy);
            None (default): the RMSD alone decides.

    The pruner of legacy racerts, with its energy and inertia prefilters and its
    keywords, is racerts.compat.pruner.RMSDPruner.
    """

    def __init__(
        self,
        threshold=0.125,
        *,
        hydrogens="polar",
        align=True,
        max_maps=MAX_MAPS,
        graph=None,
        energy_tolerance=None,
        verbose=False,
        **legacy,
    ):
        if legacy:
            raise TypeError(
                f"RMSDPruner takes no {', '.join(sorted(legacy))}: the prefilters and "
                "keywords of legacy racerts are those of "
                "racerts.compat.pruner.RMSDPruner (also racerts.pruner.RMSDPruner)."
            )
        check_threshold(threshold, "threshold")
        if hydrogens not in HYDROGENS:
            raise ValueError(
                f"hydrogens must be one of {HYDROGENS}, not {hydrogens!r}."
            )
        if not is_integer(max_maps) or max_maps < 1:
            raise ValueError("max_maps must be a positive integer.")
        check_graph(graph)
        if energy_tolerance is not None:
            check_threshold(energy_tolerance, "energy_tolerance")
        self.threshold = threshold
        self.hydrogens = hydrogens
        self.align = align
        self.max_maps = max_maps
        self.graph = graph
        self.energy_tolerance = energy_tolerance
        self.verbose = verbose

    def prune(self, mol):
        if any(conf.HasProp("energy") for conf in mol.GetConformers()):
            drop_conformers_without_energy(mol)
        conf_ids = [conf.GetId() for conf in self._by_energy(mol)]
        keep = self._distinct(mol, conf_ids)
        for conf_id in [i for i in conf_ids if i not in keep]:
            mol.RemoveConformer(conf_id)
        return mol

    def _by_energy(self, mol):
        """The conformers by energy, or in their own order if one of them has none."""
        conformers = mol.GetConformers()
        if not all(conf.HasProp("energy") for conf in conformers):
            return conformers
        return sorted(conformers, key=lambda conf: conf.GetDoubleProp("energy"))

    def _distinct(self, mol, conf_ids) -> set:
        """
        The conformers to keep, of conf_ids in their order: each is compared with the
        kept ones that a lower bound of the RMSD does not already tell apart, the
        closest by that bound first, until one is within the threshold.
        """
        energies = None
        if self.energy_tolerance is not None:
            conformers = [mol.GetConformer(int(i)) for i in conf_ids]
            if not all(conf.HasProp("energy") for conf in conformers):
                raise ValueError(
                    "energy_tolerance needs the energies of the conformers: refine "
                    "them first, or leave it out."
                )
            energies = [conf.GetDoubleProp("energy") for conf in conformers]
        if len(conf_ids) < 2:
            return set(conf_ids)
        kernel = symmetry_kernel(
            mol,
            self.graph,
            conf_ids,
            self.hydrogens,
            max_maps=self.max_maps,
            hard_max_maps=max(self.max_maps, LISTED_MAPS),
        )
        classes = symmetry_classes(kernel.graph, kernel.weight)
        kept, prepared, described, kept_energies = [], [], ([], []), []
        for n, conf_id in enumerate(conf_ids):
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
                if energies is not None:  # another energy: another stationary point
                    apart = np.abs(np.array(kept_energies) - energies[n])
                    bounds = np.where(apart > self.energy_tolerance, np.inf, bounds)
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
                if energies is not None:
                    kept_energies.append(energies[n])
                for others, vector in zip(described, own):
                    others.append(vector)
        return set(kept)
