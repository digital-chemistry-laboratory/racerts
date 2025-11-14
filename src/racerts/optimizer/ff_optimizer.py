from abc import abstractmethod, ABC
from typing import Optional, List

from rdkit import Chem
from rdkit.Chem.AllChem import (
    AlignMol,  # type: ignore
    UFFGetMoleculeForceField,  # type: ignore
    MMFFGetMoleculeProperties,  # type: ignore
    MMFFGetMoleculeForceField,  # type: ignore
)
from concurrent.futures import ThreadPoolExecutor


class BaseOptimizer(ABC):
    @abstractmethod
    def __init__(
        self,
        verbose: bool = False,
        conf_id_ref: int = -1,
        force_constant: float = 1e6,
        num_threads: int = 1,
        **kwargs,
    ):
        raise NotImplementedError

    @abstractmethod
    def tune_ts_conformers(
        self, mol: Chem.Mol, reference: Chem.Mol, align_indices: Optional[List[int]]
    ):
        """
        Embed a transition state
        """


class UFFOptimizer(BaseOptimizer):
    def __init__(
        self, verbose=False, conf_id_ref=-1, force_constant=1000000, num_threads=1
    ):
        self.verbose = verbose
        self.conf_id_ref = conf_id_ref
        self.force_constant = force_constant
        self.num_threads = num_threads
        self.maxIter = 100

    def tune_ts_conformers(
        self,
        mol: Chem.Mol,
        reference: Chem.Mol,
        align_indices: List[int],
    ):
        atom_map = [(i, i) for i in align_indices]

        if self.conf_id_ref == -1:
            self.conf_id_ref = reference.GetConformer().GetId()

        coordinates_ref = reference.GetConformer(self.conf_id_ref).GetPositions()

        def _optimise(conf_id: int) -> int:
            """
            Returns the number of failed minimisation attempts for this conformer.
            """
            AlignMol(
                mol,
                reference,
                atomMap=atom_map,
                prbCid=conf_id,
                refCid=self.conf_id_ref,
            )

            ff = UFFGetMoleculeForceField(
                mol, confId=conf_id, ignoreInterfragInteractions=False
            )

            for i in align_indices:
                point = coordinates_ref[i]
                idx_point = ff.AddExtraPoint(*point, fixed=True) - 1
                ff.AddDistanceConstraint(idx_point, i, 0, 0, self.force_constant)

            ff.Initialize()

            local_fail = 0
            for _ in range(self.maxIter):
                if ff.Minimize() == 0:
                    break
                local_fail += 1

            mol.GetConformer(conf_id).SetDoubleProp("energy", ff.CalcEnergy())

            AlignMol(
                mol,
                reference,
                atomMap=atom_map,
                prbCid=conf_id,
                refCid=self.conf_id_ref,
            )

            return local_fail

        conformer_ids = [c.GetId() for c in mol.GetConformers()]
        if self.num_threads > 1:
            with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                count_ff_failure = sum(pool.map(_optimise, conformer_ids))
        else:
            count_ff_failure = sum(_optimise(cid) for cid in conformer_ids)

        if self.verbose:
            print(f"FF failures: {count_ff_failure}")


class MMFFOptimizer(BaseOptimizer):
    def __init__(
        self,
        verbose=False,
        conf_id_ref=-1,
        force_constant=1000000,
        num_threads: int = 1,
    ):
        self.verbose = verbose
        self.conf_id_ref = conf_id_ref
        self.force_constant = force_constant
        self.num_threads = num_threads
        self.maxIter = 100

    def tune_ts_conformers(
        self,
        mol: Chem.Mol,
        reference: Chem.Mol,
        align_indices,
        num_threads: int = 1,
    ):

        atom_map = [(i, i) for i in align_indices]

        if self.conf_id_ref == -1:
            self.conf_id_ref = reference.GetConformer().GetId()

        mmffVerbosity = 2 if self.verbose else 0
        coordinates_ref = reference.GetConformer(self.conf_id_ref).GetPositions()

        ff_props = MMFFGetMoleculeProperties(mol, mmffVerbosity=mmffVerbosity)

        def _optimise(conf_id: int) -> int:
            """
            Returns the number of failed minimisation attempts for this conformer.
            """
            AlignMol(
                mol,
                reference,
                atomMap=atom_map,
                prbCid=conf_id,
                refCid=self.conf_id_ref,
            )
            ff = MMFFGetMoleculeForceField(
                mol,
                ff_props,
                confId=conf_id,
                ignoreInterfragInteractions=False,
            )

            for idx in align_indices:
                point = coordinates_ref[idx]
                ep_idx = ff.AddExtraPoint(*point, fixed=True) - 1
                ff.AddDistanceConstraint(ep_idx, idx, 0, 0, self.force_constant)

            ff.Initialize()

            local_fail = 0
            for _ in range(self.maxIter):
                if ff.Minimize() == 0:
                    break
                local_fail += 1

            mol.GetConformer(conf_id).SetDoubleProp("energy", ff.CalcEnergy())

            AlignMol(
                mol,
                reference,
                atomMap=atom_map,
                prbCid=conf_id,
                refCid=self.conf_id_ref,
            )

            return local_fail

        conformer_ids = [c.GetId() for c in mol.GetConformers()]

        if self.num_threads > 1:
            with ThreadPoolExecutor(max_workers=self.num_threads) as pool:
                ff_failures = sum(pool.map(_optimise, conformer_ids))
        else:
            ff_failures = sum(_optimise(cid) for cid in conformer_ids)

        if self.verbose:
            print(f"FF failures: {ff_failures}")
