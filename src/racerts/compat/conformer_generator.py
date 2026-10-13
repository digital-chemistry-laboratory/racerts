"""
racerts.conformer_generator of legacy racerts: ConformerGenerator, which runs its steps
as stages of a Pipeline and gives the same results, and the names this module had.
"""

import logging
from dataclasses import dataclass
from typing import Callable, List, Optional

from rdkit import Chem

from racerts.embed import DEFAULT_CONF_FACTOR, conformer_count
from racerts.embed.stage import no_conformers_error
from racerts.io.xyz import write_xyz as _write_xyz
from racerts.pipeline import ConformerEnsemble, Context, Pipeline
from racerts.prune import BasePruner, EnergyPruner, RMSDPruner
from racerts.refine import refine_with_fallback
from racerts.system import (
    BaseMolGetter,
    MolGetterBonds,
    MolGetterConnectivity,
    MolGetterSMILES,
    build_mol,
)
from racerts.task import TransitionState
from racerts.utils.log import verbose_logging

from .embedder import BaseEmbedder, CmapEmbedder
from .optimizer import BaseOptimizer, MMFFOptimizer, UFFOptimizer
from .utils import (
    EV_TO_KCAL_MOL,
    atom_idx_input_validation,
    count_electrons,
    get_frozen_atoms,
    infer_charge_and_multiplicity,
)

logger = logging.getLogger(__name__)

# The legacy value; racerts.utils.units.HARTREE_TO_KCAL_MOL is the CODATA 2022 one.
KCAL_TO_HARTREE = 627.509

__all__ = [
    "EV_TO_KCAL_MOL",
    "KCAL_TO_HARTREE",
    "BaseEmbedder",
    "BaseMolGetter",
    "BaseOptimizer",
    "BasePruner",
    "CmapEmbedder",
    "ConformerGenerator",
    "EnergyPruner",
    "MMFFOptimizer",
    "MolGetterBonds",
    "MolGetterConnectivity",
    "MolGetterSMILES",
    "RMSDPruner",
    "UFFOptimizer",
    "atom_idx_input_validation",
    "count_electrons",
    "get_frozen_atoms",
    "infer_charge_and_multiplicity",
]


class ConformerGenerator(object):
    """
    TS conformer ensembles with exchangeable components (legacy racerts API).

    generate_conformers runs the steps get_mol, embed_TS, optimize and prune as stages
    of a racerts Pipeline, so subclasses that override a step, and components set on
    the generator (mol_getter, embedder, optimizer, energy_pruner, rmsd_pruner), are
    used as before. New code can use racerts.generate_ts instead.
    """

    def __init__(
        self,
        verbose: bool = False,
        randomSeed=12,
        num_threads=1,
        energy_pruner_kwargs={},
        rmsd_pruner_kwargs={},
    ) -> None:
        self.num_threads = num_threads
        self.randomSeed = randomSeed
        self._verbose = verbose
        self.charge = None

        self._mol_getter: BaseMolGetter = MolGetterSMILES()
        self._mol_getter_kwargs = {}
        self._embedder = CmapEmbedder(
            randomSeed=self.randomSeed, num_threads=num_threads
        )
        self._optimizer = MMFFOptimizer(num_threads=num_threads)
        self._energy_pruner_kwargs = energy_pruner_kwargs
        self._rmsd_pruner_kwargs = rmsd_pruner_kwargs

        self._rmsd_pruner: BasePruner = RMSDPruner(**rmsd_pruner_kwargs)
        self._energy_pruner: BasePruner = EnergyPruner(**energy_pruner_kwargs)

    @property
    def energy_pruner(self) -> BasePruner:
        return self._energy_pruner

    @energy_pruner.setter
    def energy_pruner(self, energy_pruner: BasePruner) -> None:
        self._energy_pruner = energy_pruner

    @property
    def rmsd_pruner(self) -> BasePruner:
        return self._rmsd_pruner

    @rmsd_pruner.setter
    def rmsd_pruner(self, rmsd_pruner: BasePruner) -> None:
        self._rmsd_pruner = rmsd_pruner

    @property
    def embedder(self) -> BaseEmbedder:
        return self._embedder

    @embedder.setter
    def embedder(self, embedder: BaseEmbedder) -> None:
        self._embedder = embedder

    @property
    def optimizer(self) -> BaseOptimizer:
        return self._optimizer

    @optimizer.setter
    def optimizer(self, optimizer: BaseOptimizer) -> None:
        self._optimizer = optimizer

    @property
    def mol_getter(self) -> BaseMolGetter:
        return self._mol_getter

    @mol_getter.setter
    def mol_getter(self, mol_getter: BaseMolGetter) -> None:
        self._mol_getter = mol_getter

    @property
    def mol_getter_kwargs(self) -> dict:
        return self._mol_getter_kwargs

    @mol_getter_kwargs.setter
    def mol_getter_kwargs(self, mol_getter_kwargs: dict) -> None:
        self._mol_getter_kwargs = mol_getter_kwargs

    @property
    def energy_pruner_kwargs(self) -> dict:
        return self._energy_pruner_kwargs

    @energy_pruner_kwargs.setter
    def energy_pruner_kwargs(self, energy_pruner_kwargs: dict) -> None:
        self._energy_pruner_kwargs = energy_pruner_kwargs

    @property
    def rmsd_pruner_kwargs(self) -> dict:
        return self._rmsd_pruner_kwargs

    @rmsd_pruner_kwargs.setter
    def rmsd_pruner_kwargs(self, rmsd_pruner_kwargs: dict) -> None:
        self._rmsd_pruner_kwargs = rmsd_pruner_kwargs

    def get_mol(
        self,
        file_name: str,
        charge: int,
        reacting_atoms: List,
        input_smiles=None,
        auto_fallback=True,
    ) -> Chem.Mol:
        """
        Gets the mol object from file_name with the mol_getter (see
        racerts.system.build_mol, which also describes the fallbacks).

        Args:
            file_name (str): The path to the XYZ or SDF/MOL file.
            charge (int): The molecular charge.
            reacting_atoms (list): The atoms that are part of the reaction.
            input_smiles (list[str] or str): The input SMILES of the TS topology.

        Returns:
            mol_ts (Chem.Mol): The molecule object.
        """
        self.charge = charge
        return build_mol(
            file_name,
            charge,
            reacting_atoms,
            input_smiles=input_smiles,
            mol_getter=self._mol_getter,
            auto_fallback=auto_fallback,
        )

    def embed_TS(
        self,
        mol_ts: Chem.Mol,
        new_mol: Chem.Mol,
        reacting_atoms: List[str],
        frozen_atoms: List,
        number_of_conformers: int = -1,
        conf_factor: int = DEFAULT_CONF_FACTOR,
    ):
        self._embedder.embed_TS(
            mol_ts=mol_ts,
            mol=new_mol,
            reacting_atoms=reacting_atoms,
            frozen_atoms=frozen_atoms,
            n=conformer_count(new_mol, number_of_conformers, conf_factor),
            verbose=self._verbose,
        )

        if new_mol.GetNumConformers() == 0:
            return None

        return new_mol

    def optimize(
        self,
        new_mol: Chem.Mol,
        mol_ts: Chem.Mol,
        frozen_atoms: List,
        auto_fallback: bool = True,
    ):
        energy_method = refine_with_fallback(
            self._optimizer,
            lambda optimizer: optimizer.tune_ts_conformers(
                mol=new_mol, reference=mol_ts, align_indices=frozen_atoms
            ),
            fallback=UFFOptimizer if auto_fallback is True else None,
            num_threads=self.num_threads,
        )
        new_mol.SetProp("energy_method", energy_method)
        return new_mol

    def write_xyz(
        self, file_name: str, use_energy=False, comment: Optional[str] = None
    ):
        """
        Write all conformers to a multi-structure xyz file (see
        racerts.io.write_xyz): extended XYZ by default, CREST-style energies (Hartree)
        with use_energy, or a given comment line.
        """
        _write_xyz(self.mol, file_name, use_energy=use_energy, comment=comment)

    def prune(self, mol):

        pruned_mol = Chem.Mol(mol)

        old_num_confs = pruned_mol.GetNumConformers()

        if self._energy_pruner is not None:
            self._energy_pruner.prune(mol=pruned_mol)

        logger.info(
            "Energy pruning reduced conformer number from %d to %d",
            old_num_confs,
            pruned_mol.GetNumConformers(),
        )

        if self._rmsd_pruner is not None:
            self._rmsd_pruner.prune(mol=pruned_mol)

        logger.info(
            "Pruning reduced conformer number from %d to %d",
            old_num_confs,
            pruned_mol.GetNumConformers(),
        )

        return pruned_mol

    def generate_conformers(
        self,
        file_name: str,
        charge: int = 0,
        reacting_atoms: List = [],
        frozen_atoms: List = [],
        input_smiles=None,
        number_of_conformers: int = -1,
        conf_factor: int = DEFAULT_CONF_FACTOR,
        auto_fallback=True,
        multiplicity: Optional[int] = None,
    ) -> Chem.Mol:
        with verbose_logging(self._verbose):
            mol_ts = self.get_mol(
                file_name,
                charge,
                reacting_atoms,
                input_smiles,
                auto_fallback=auto_fallback,
            )
            if mol_ts is None:
                raise ValueError("No valid mol object could be generated.")

            ctx = Context.create(
                mol_ts,
                TransitionState(reacting_atoms, frozen_atoms),
                charge=charge,
                multiplicity=multiplicity,
            )
            frozen = list(ctx.frozen.hard)

            def embed(ctx, ensemble):
                new_mol = self.embed_TS(
                    mol_ts=ctx.mol,
                    new_mol=ctx.graph(),
                    reacting_atoms=reacting_atoms,
                    frozen_atoms=frozen,
                    number_of_conformers=number_of_conformers,
                    conf_factor=conf_factor,
                )
                if new_mol is None:
                    raise no_conformers_error(ctx.frozen)
                return ConformerEnsemble(new_mol)

            def optimize(ctx, ensemble):
                new_mol = self.optimize(
                    new_mol=ensemble.mol,
                    mol_ts=ctx.mol,
                    frozen_atoms=frozen,
                    auto_fallback=auto_fallback,
                )
                return ConformerEnsemble(new_mol)

            def prune(ctx, ensemble):
                return ConformerEnsemble(self.prune(ensemble.mol))

            steps = [("embed", embed), ("optimize", optimize), ("prune", prune)]
            pipeline = Pipeline(_Step(name, run) for name, run in steps)
            self.mol = pipeline.run(ctx).mol

        return self.mol


@dataclass
class _Step:
    """A pipeline stage that calls a step of ConformerGenerator (overridable)."""

    name: str
    run: Callable
