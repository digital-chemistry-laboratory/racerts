"""MMFF and UFF refinement with the anchor atoms held at the reference positions."""

import copy
import logging
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Optional, Sequence

from rdkit import Chem
from rdkit.Chem.AllChem import (
    MMFFGetMoleculeForceField,  # type: ignore
    MMFFGetMoleculeProperties,  # type: ignore
    UFFGetMoleculeForceField,  # type: ignore
)

from racerts.restraints.model import PositionRestraint

from .base import BaseOptimizer

logger = logging.getLogger(__name__)

ENERGY_TOLERANCE = 1e-4  # kcal/mol


def minimize(ff, max_rounds: int, converge: bool = False) -> int:
    """
    Minimize ff; returns the number of Minimize calls that did not converge.

    Legacy racerts calls Minimize until one call converges. Next to stiff anchor terms,
    a BFGS run can report convergence long before the free atoms reach a minimum; with
    converge, a converged call is restarted (with a fresh Hessian estimate) as long as
    it still lowers the energy by more than ENERGY_TOLERANCE.
    """
    failures = 0
    energy = ff.CalcEnergy()
    for _ in range(max_rounds):
        done = ff.Minimize() == 0
        if not converge:
            if done:
                break
            failures += 1
            continue
        previous, energy = energy, ff.CalcEnergy()
        if done and previous - energy < ENERGY_TOLERANCE:
            break
        failures += not done
    return failures


class ForceFieldOptimizer(BaseOptimizer):
    """
    Minimizes each conformer with an RDKit force field. Each anchor atom is tied to a
    fixed extra point at its reference position by a distance constraint
    (force_constant, kcal/mol/A^2); conformers are aligned on the anchors to the
    reference before and after. Minimize is called up to maxIter times per conformer.

    Args:
        num_threads: Threads over the conformers. They gain little: RDKit's minimizer
            does not release Python's lock.
        num_workers: Worker processes over the conformers (1: none; None: all CPUs of
            the job), each with a share of them; the results are those of one process.
            Starting a worker costs about a second, so they pay off for hundreds of
            conformers of a large molecule.
        converge: Restart minimizations that stop early next to the anchors (see
            minimize); legacy racerts stops at the first converged call.
        anchor_free_energies: Report the energy of the force field without the anchor
            terms; legacy racerts includes them (0.02-0.18 kcal/mol on test systems).
            With restraints or soft atoms the reported energies always leave out
            their terms and those of the anchors.
    """

    def __init__(
        self,
        verbose=False,
        conf_id_ref=-1,
        force_constant=1000000,
        num_threads=1,
        converge: bool = False,
        anchor_free_energies: bool = False,
        num_workers: Optional[int] = 1,
    ):
        if num_workers is not None and num_workers < 1:
            raise ValueError("num_workers must be positive, or None for all CPUs.")
        self.verbose = verbose
        self.conf_id_ref = conf_id_ref
        self.force_constant = force_constant
        self.num_threads = num_threads
        self.converge = converge
        self.anchor_free_energies = anchor_free_energies
        self.num_workers = num_workers
        self.maxIter = 100

    def _setup(self, mol: Chem.Mol):
        """Per-molecule data for _force_field (e.g. MMFF parameters)."""
        return None

    def _force_field(self, mol: Chem.Mol, conf_id: int, setup):
        raise NotImplementedError

    def _add_restraint(self, ff, restraint) -> None:
        """A flat-bottom distance term (RDKit: 1/2 k (d - bound)^2)."""
        raise NotImplementedError

    def _refine(
        self,
        mol: Chem.Mol,
        reference: Optional[Chem.Mol],
        anchors: Sequence[int],
        restraints: Sequence = (),
    ) -> int:
        """
        Returns the number of Minimize calls that did not converge. With restraints,
        the reported energies leave out their terms and those of the anchors.
        Position restraints (soft atoms) hold atoms near points of the reference
        frame: without anchors, the conformers are aligned on their atoms.
        """
        conformer_ids = [c.GetId() for c in mol.GetConformers()]
        workers = getattr(self, "num_workers", 1)  # legacy subclasses set no attribute
        if workers is None:
            workers = len(os.sched_getaffinity(0))
        workers = min(workers, len(conformer_ids))
        if workers > 1:
            return self._refine_in_processes(
                mol, reference, anchors, restraints, workers
            )
        positions = [r for r in restraints if isinstance(r, PositionRestraint)]
        restraints = [r for r in restraints if not isinstance(r, PositionRestraint)]
        anchors = list(anchors)
        align_indices = anchors or [r.atom for r in positions]
        if positions and reference is None:
            raise ValueError("Position restraints need a reference geometry.")
        coordinates_ref = None
        if anchors:
            coordinates_ref = reference.GetConformer(self.conf_id_ref).GetPositions()
        setup = self._setup(mol)

        def _optimize(conf_id: int) -> int:
            """
            Returns the number of failed minimisation attempts for this conformer.
            """
            self.align_mols(
                mol=mol,
                reference=reference,
                align_indices=align_indices,
                conf_id=conf_id,
            )
            ff = self._force_field(mol, conf_id, setup)

            for idx in anchors:
                point = coordinates_ref[idx]
                ep_idx = ff.AddExtraPoint(*point, fixed=True) - 1
                ff.AddDistanceConstraint(ep_idx, idx, 0, 0, self.force_constant)
            for position in positions:
                ep_idx = ff.AddExtraPoint(*position.point, fixed=True) - 1
                ff.AddDistanceConstraint(
                    ep_idx,
                    position.atom,
                    0,
                    position.tolerance,
                    position.force_constant,
                )
            for restraint in restraints:
                self._add_restraint(ff, restraint)

            ff.Initialize()
            local_fail = minimize(ff, self.maxIter, converge=self.converge)

            if restraints or positions or (self.anchor_free_energies and anchors):
                energy = self._force_field(mol, conf_id, setup).CalcEnergy()
            else:
                energy = ff.CalcEnergy()
            mol.GetConformer(conf_id).SetDoubleProp("energy", energy)

            self.align_mols(
                mol=mol,
                reference=reference,
                align_indices=align_indices,
                conf_id=conf_id,
            )

            return local_fail

        if self.num_threads > 1:
            with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                failures = sum(pool.map(_optimize, conformer_ids))
        else:
            failures = sum(_optimize(cid) for cid in conformer_ids)

        logger.info("%s: %d unconverged Minimize calls", type(self).__name__, failures)
        return failures

    def _refine_in_processes(self, mol, reference, anchors, restraints, workers) -> int:
        """_refine with the conformers shared among spawned worker processes."""
        serial = copy.copy(self)
        serial.num_workers, serial.num_threads = 1, 1
        conformer_ids = [c.GetId() for c in mol.GetConformers()]
        shared = (
            None if reference is None else _pack(reference),
            list(anchors),
            list(restraints),
        )
        jobs = [
            (serial, _pack(mol, conformer_ids[k::workers]), *shared)
            for k in range(workers)
        ]
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
            results = list(pool.map(_refine_share, jobs))
        failures = 0
        for share_failures, conformers in results:
            failures += share_failures
            for conf_id, positions, energy in conformers:
                conf = mol.GetConformer(conf_id)
                for atom, position in enumerate(positions):
                    conf.SetAtomPosition(atom, position.tolist())
                if energy is not None:
                    conf.SetDoubleProp("energy", energy)
        return failures


def _pack(mol: Chem.Mol, conf_ids=None):
    """The graph and the positions of the conformers (all, or conf_ids) for a worker.
    The positions travel as arrays: RDKit's pickle keeps them in single precision."""
    graph = Chem.Mol(mol)
    graph.RemoveAllConformers()
    conformers = [
        (conf.GetId(), conf.GetPositions())
        for conf in mol.GetConformers()
        if conf_ids is None or conf.GetId() in conf_ids
    ]
    return graph.ToBinary(Chem.PropertyPickleOptions.AllProps), conformers


def _unpack(packed) -> Chem.Mol:
    graph, conformers = packed
    mol = Chem.Mol(graph)
    for conf_id, positions in conformers:
        conf = Chem.Conformer(mol.GetNumAtoms())
        for atom, position in enumerate(positions):
            conf.SetAtomPosition(atom, position.tolist())
        conf.SetId(conf_id)
        mol.AddConformer(conf, assignId=False)
    return mol


def _refine_share(job):
    """A worker's share: (failures, [(conformer id, positions, energy)])."""
    optimizer, mol, reference, anchors, restraints = job
    mol = _unpack(mol)
    reference = None if reference is None else _unpack(reference)
    failures = optimizer._refine(mol, reference, anchors, restraints)
    return failures, [
        (
            conf.GetId(),
            conf.GetPositions(),
            conf.GetDoubleProp("energy") if conf.HasProp("energy") else None,
        )
        for conf in mol.GetConformers()
    ]


class UFFOptimizer(ForceFieldOptimizer):
    def _force_field(self, mol, conf_id, setup):
        return UFFGetMoleculeForceField(
            mol, confId=conf_id, ignoreInterfragInteractions=False
        )

    def _add_restraint(self, ff, r):
        ff.UFFAddDistanceConstraint(
            r.first, r.second, False, r.lower, r.upper, r.force_constant
        )


DIELECTRIC_MODELS = {"constant": 1, "distance": 2}  # RDKit's numbering


class MMFFOptimizer(ForceFieldOptimizer):
    """
    MMFF94 refinement (see ForceFieldOptimizer).

    Args:
        dielectric_model: "constant" or "distance" (distance-dependent).
        dielectric_constant: The dielectric constant of the electrostatics; e.g.
            "distance" with 4.0 damps the salt bridges of charged systems in vacuum.
    """

    def __init__(
        self,
        verbose=False,
        conf_id_ref=-1,
        force_constant=1000000,
        num_threads=1,
        converge: bool = False,
        anchor_free_energies: bool = False,
        dielectric_model: str = "constant",
        dielectric_constant: float = 1.0,
        num_workers: Optional[int] = 1,
    ):
        super().__init__(
            verbose=verbose,
            conf_id_ref=conf_id_ref,
            force_constant=force_constant,
            num_threads=num_threads,
            converge=converge,
            anchor_free_energies=anchor_free_energies,
            num_workers=num_workers,
        )
        if dielectric_model not in DIELECTRIC_MODELS:
            raise ValueError(
                f"dielectric_model must be one of {sorted(DIELECTRIC_MODELS)}."
            )
        if not dielectric_constant > 0:
            raise ValueError("dielectric_constant must be positive.")
        self.dielectric_model = dielectric_model
        self.dielectric_constant = dielectric_constant

    def _setup(self, mol):
        mmffVerbosity = 2 if self.verbose else 0
        ff_props = MMFFGetMoleculeProperties(mol, mmffVerbosity=mmffVerbosity)
        if ff_props is None:
            raise ValueError("MMFF parameters are not available for this molecule")
        # Only set when not the default, so that the legacy call is unchanged.
        if (self.dielectric_model, self.dielectric_constant) != ("constant", 1.0):
            ff_props.SetMMFFDielectricModel(DIELECTRIC_MODELS[self.dielectric_model])
            ff_props.SetMMFFDielectricConstant(self.dielectric_constant)
        return ff_props

    def _force_field(self, mol, conf_id, setup):
        return MMFFGetMoleculeForceField(
            mol,
            setup,
            confId=conf_id,
            ignoreInterfragInteractions=False,
        )

    def _add_restraint(self, ff, r):
        ff.MMFFAddDistanceConstraint(
            r.first, r.second, False, r.lower, r.upper, r.force_constant
        )
