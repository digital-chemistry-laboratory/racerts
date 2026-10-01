"""Clustering of conformers (Butina, hierarchical, leader) and family selection."""

import logging
from typing import Callable, List, Optional, Sequence

import numpy as np
from rdkit import Chem

from racerts.geometry import heavy_atoms, rmsd, symmetry_maps
from racerts.pipeline import ConformerEnsemble

from .base import BasePruner, check_threshold

logger = logging.getLogger(__name__)

METHODS = ("butina", "hierarchical", "leader")
KERNELS = ("aligned", "symmetric")
REPRESENTATIVES = ("lowest_energy", "centroid")

Metric = Callable[[Chem.Mol, int, int], float]


class ClusterPruner(BasePruner):
    """
    Groups conformers whose distance is below threshold and keeps one per group.

    Args:
        threshold: The distance (A) below which conformers are grouped: for "butina"
            the neighbourhood radius, for "hierarchical" the cut height of the tree,
            for "leader" the radius around each leader.
        method: "butina" (RDKit's Butina clustering),
            "hierarchical" (scipy linkage), or "leader" (in order of energy,
            a conformer joins the first leader within threshold or becomes a leader).
        kernel: The distance, an RMSD of racerts.geometry after superposition:
            "aligned" (a fixed atom correspondence; for bondless TS carriers) or
            "symmetric" (the smallest over the symmetry maps of the graph, as
            RMSDPruner).
        atom_indices: Atoms of the distance; default: heavy atoms (all atoms if there
            are none). The "symmetric" kernel always uses the atoms of symmetry_maps
            (the heavy atoms).
        representative: The conformer kept per cluster: "lowest_energy" (the first by
            conformer order if an energy is missing) or "centroid" (the smallest sum
            of distances to the other members).
        linkage: The linkage of "hierarchical" (scipy method, e.g. "average").
        metric: Instead of the kernel: metric(mol, conf_id_a, conf_id_b) -> distance,
            e.g. a distance over paired reactant and product conformers.
    """

    def __init__(
        self,
        threshold: float = 1.5,
        method: str = "butina",
        kernel: str = "aligned",
        atom_indices: Optional[Sequence[int]] = None,
        representative: str = "lowest_energy",
        linkage: str = "average",
        metric: Optional[Metric] = None,
        verbose: bool = False,
        **kwargs,
    ):
        check_threshold(threshold, "threshold")
        for value, allowed, name in (
            (method, METHODS, "method"),
            (kernel, KERNELS, "kernel"),
            (representative, REPRESENTATIVES, "representative"),
        ):
            if value not in allowed:
                raise ValueError(f"{name} must be one of {allowed}, not {value!r}.")
        self.threshold = threshold
        self.method = method
        self.kernel = kernel
        self.atom_indices = None if atom_indices is None else list(atom_indices)
        self.representative = representative
        self.linkage = linkage
        self.metric = metric
        self.verbose = verbose
        self.max_matches = kwargs.get("maxMatches", 10000)

    def distances(self, mol: Chem.Mol, conf_ids: Sequence[int]) -> np.ndarray:
        """The symmetric matrix of distances between the conformers conf_ids."""
        n = len(conf_ids)
        matrix = np.zeros((n, n))
        if n < 2:
            return matrix
        if self.metric is not None:
            pairs = self.metric
        else:
            if self.kernel == "aligned":
                atoms, maps = self.atom_indices or heavy_atoms(mol), None
            else:  # the maps depend on the graph only: once for all pairs
                atoms, maps = symmetry_maps(mol, max_matches=self.max_matches)
            positions = {i: mol.GetConformer(i).GetPositions() for i in conf_ids}

            def pairs(_, a, b):
                return rmsd(positions[a], positions[b], atoms, maps)

        for i in range(1, n):
            for j in range(i):
                matrix[i, j] = matrix[j, i] = pairs(mol, conf_ids[i], conf_ids[j])
        return matrix

    def clusters(self, mol: Chem.Mol) -> List[List[int]]:
        """
        The clusters as lists of conformer ids: each sorted by energy (conformer order
        where an energy is missing), the clusters by their first member.
        """
        return self._cluster(mol)[0]

    def _cluster(self, mol: Chem.Mol):
        """The clusters, the distance matrix, and its conformer ids."""
        conf_ids = _by_energy(mol)
        if not conf_ids:
            return [], np.zeros((0, 0)), []
        matrix = self.distances(mol, conf_ids)
        n = len(conf_ids)
        if self.method == "leader":
            leaders: List[List[int]] = []
            for i in range(n):
                for group in leaders:
                    if matrix[i, group[0]] < self.threshold:
                        group.append(i)
                        break
                else:
                    leaders.append([i])
            groups = leaders
        elif self.method == "butina":
            from rdkit.ML.Cluster import Butina

            condensed = [matrix[i, j] for i in range(1, n) for j in range(i)]
            groups = [
                list(c)
                for c in Butina.ClusterData(
                    condensed, n, self.threshold, isDistData=True
                )
            ]
        else:
            from scipy.cluster import hierarchy
            from scipy.spatial.distance import squareform

            if n == 1:
                labels = [1]
            else:
                tree = hierarchy.linkage(
                    squareform(matrix, checks=False), method=self.linkage
                )
                labels = hierarchy.fcluster(
                    tree, t=self.threshold, criterion="distance"
                )
            by_label = {}
            for i, label in enumerate(labels):
                by_label.setdefault(label, []).append(i)
            groups = list(by_label.values())
        groups = [sorted(group) for group in groups]  # energy order (conf_ids is)
        groups.sort(key=lambda group: group[0])
        clusters = [[conf_ids[i] for i in group] for group in groups]
        return clusters, matrix, conf_ids

    def prune(self, mol: Chem.Mol) -> Chem.Mol:
        clusters, matrix, conf_ids = self._cluster(mol)
        index = {conf_id: k for k, conf_id in enumerate(conf_ids)}
        keep = set()
        for cluster in clusters:
            if self.representative == "centroid" and len(cluster) > 2:
                rows = [index[c] for c in cluster]
                sums = matrix[np.ix_(rows, rows)].sum(axis=1)
                keep.add(cluster[int(np.argmin(sums))])
            else:
                keep.add(cluster[0])
        logger.info(
            "%s clustering at %.2f A: %d conformers in %d clusters",
            self.method,
            self.threshold,
            mol.GetNumConformers(),
            len(clusters),
        )
        for conf in list(mol.GetConformers()):
            if conf.GetId() not in keep:
                mol.RemoveConformer(conf.GetId())
        return mol


def _by_energy(mol: Chem.Mol) -> List[int]:
    """Conformer ids by energy; in conformer order if an energy is missing."""
    conformers = list(mol.GetConformers())
    if all(c.HasProp("energy") for c in conformers):
        conformers.sort(key=lambda c: c.GetDoubleProp("energy"))
    return [c.GetId() for c in conformers]


class FamilySelector:
    """
    Keeps up to n_max conformers spread over structural families: the
    clusters of clusterer (default ClusterPruner(): Butina at 1.5 A), ordered by
    their lowest member, are filled round-robin (the best of each family, then the
    second best, ...). The result is ordered by energy. Every conformer needs an
    energy.
    """

    name = "select_families"

    def __init__(
        self,
        n_max: int,
        clusterer: Optional[ClusterPruner] = None,
        renumber: bool = False,
    ):
        from .stage import PruneCount

        PruneCount(n_max)  # checks n_max
        self.n_max = int(n_max)
        self.clusterer = clusterer if clusterer is not None else ClusterPruner()
        self.renumber = renumber

    def run(self, ctx, ensemble: ConformerEnsemble) -> ConformerEnsemble:
        energies = ensemble.energies()
        if not np.isfinite(energies).all():
            raise ValueError(
                "FamilySelector needs a finite energy for every conformer."
            )
        families = self.clusterer.clusters(ensemble.mol)
        ordered = []
        for rank in range(max((len(f) for f in families), default=0)):
            ordered.extend(f[rank] for f in families if rank < len(f))
        chosen = set(ordered[: self.n_max])
        logger.info(
            "%d families (sizes %s); keeping %d of %d conformers",
            len(families),
            [len(f) for f in families],
            len(chosen),
            len(ensemble),
        )
        by_energy = [i for i in _by_energy(ensemble.mol) if i in chosen]
        return ensemble.filter(by_energy, renumber=self.renumber)
