from typing import Type, List
from copy import deepcopy

from rdkit import Chem
from rdkit.Chem import Descriptors

from .mol_getter import (
    BaseMolGetter,
    MolGetterBonds,
    MolGetterConnectivity,
    MolGetterSMILES,
)
from .embedder import CmapEmbedder, BaseEmbedder
from .optimizer import MMFFOptimizer, UFFOptimizer, BaseOptimizer
from .pruner import BasePruner, EnergyPruner, RMSDPruner

from racerts.utils import atom_idx_input_validation, get_frozen_atoms

KCAL_TO_HARTREE = 627.509


class ConformerGenerator(object):

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
    def bounds_generator(self) -> Type[BaseEmbedder]:
        return self._bounds_generator

    @bounds_generator.setter
    def bounds_generator(self, bounds_generator: Type[BaseEmbedder]) -> None:
        self._bounds_generator = bounds_generator

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
        Gets mol object from file_name using the mol_getter object.

        This function takes the mol_getter class (defaults to mol_getter_bonds). I
        f given method throws an error it tries with mol_getter_SMILES or mol_getter_connectivity,
        depending on the input of SMILES:

        Args:
            file_name (str): The path to the XYZ file.
            charge (int): The molecular charge.
            reacting_atoms (list): The atoms that are part of the reaction.
            input_smiles (list[str] or str): The input SMILES of the TS topology.

        Returns:
            mol_ts (Chem.Mol): The molecule object.
        """

        self.charge = charge
        get_mol_kwargs = {}
        get_mol_kwargs["charge"] = charge
        get_mol_kwargs["reacting_atoms"] = reacting_atoms
        if input_smiles is not None and len(input_smiles) > 0:
            get_mol_kwargs["input_smiles"] = input_smiles

        try:
            mol_ts = self._mol_getter.get_mol(file_name=file_name, **get_mol_kwargs)
            return mol_ts
        except Exception as e:
            if auto_fallback is False:
                print(e)
                raise e
            try:
                if not isinstance(self._mol_getter, MolGetterBonds):
                    print("Using mol based on DetermineBonds.")
                    mol_ts = MolGetterBonds().get_mol(
                        file_name=file_name, **get_mol_kwargs
                    )
                    return mol_ts
            except Exception as e:
                print(e)

            print(
                "Using mol based on DetermineConnectivity. No bond information is inferred."
            )
            mol_ts = MolGetterConnectivity().get_mol(
                file_name=file_name, **get_mol_kwargs
            )

        return mol_ts

    def embed_TS(
        self,
        mol_ts: Chem.Mol,
        new_mol: Chem.Mol,
        reacting_atoms: List[str],
        frozen_atoms: List,
        number_of_conformers: int = -1,
        conf_factor: int = 80,
    ):

        if number_of_conformers == -1:
            number_of_conformers = (
                Descriptors.NumRotatableBonds(new_mol)  # type: ignore[attr-defined]
            ) * conf_factor + 30

        cids, error_counts = self._embedder.embed_TS(
            mol_ts=mol_ts,
            mol=new_mol,
            reacting_atoms=reacting_atoms,
            frozen_atoms=frozen_atoms,
            n=number_of_conformers,
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

        try:
            self._optimizer.tune_ts_conformers(
                mol=new_mol, reference=mol_ts, align_indices=frozen_atoms
            )
        except Exception as e:
            print("Optimization failed: ", e)
            if not isinstance(self._optimizer, UFFOptimizer) and auto_fallback is True:
                print("UFF is used for the refinement")
                verbose = getattr(self._optimizer, "verbose", False)
                conf_id_ref = getattr(self._optimizer, "conf_id_ref", -1)
                force_constant = getattr(self._optimizer, "force_constant", 1e6)

                optimizer = UFFOptimizer(
                    verbose=verbose,
                    conf_id_ref=conf_id_ref,
                    force_constant=force_constant,  # type: ignore
                    num_threads=self.num_threads,
                )
                optimizer.tune_ts_conformers(
                    mol=new_mol, reference=mol_ts, align_indices=frozen_atoms
                )

        return new_mol

    def write_xyz(self, file_name: str, use_energy=False, comment="0 1"):
        with open(file_name, "w") as f:
            for conf in self.mol.GetConformers():

                mol_block = Chem.rdmolfiles.MolToXYZBlock(
                    self.mol, confId=conf.GetId()
                ).strip()
                lines = mol_block.split("\n")

                if use_energy:
                    if conf.HasProp("energy"):
                        energy_val = conf.GetDoubleProp("energy") / KCAL_TO_HARTREE
                    else:
                        energy_val = 0.0
                    lines[1] = f"{energy_val:.6f}"
                else:
                    new_comment = comment
                    if hasattr(self, "charge") and self.charge is not None:
                        new_comment = comment.replace("0", str(self.charge))
                    lines[1] = new_comment

                f.write("\n".join(lines) + "\n")

    def prune(self, mol):

        pruned_mol = Chem.Mol(mol)

        old_num_confs = pruned_mol.GetNumConformers()

        if self._energy_pruner is not None:

            self._energy_pruner.prune(mol=pruned_mol)

        print(
            f"Energy pruning reduced conformer number from {old_num_confs} to {pruned_mol.GetNumConformers()}"
        )

        if self._rmsd_pruner is not None:

            self._rmsd_pruner.prune(mol=pruned_mol)

        if self._verbose:
            print(
                f"Pruning reduced conformer number from {old_num_confs} to {pruned_mol.GetNumConformers()}"
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
        conf_factor=30,
        auto_fallback=True,
    ) -> Chem.Mol:

        mol_ts = self.get_mol(
            file_name, charge, reacting_atoms, input_smiles, auto_fallback=auto_fallback
        )
        if mol_ts is None:
            if self._verbose is True:
                print("No valid mol object could be generated.")
            raise ValueError("No valid mol object could be generated.")

        new_mol = deepcopy(mol_ts)
        new_mol.RemoveAllConformers()

        if atom_idx_input_validation(new_mol, reacting_atoms) is False:
            raise ValueError("Invalid reacting atoms provided.")

        frozen_atoms = get_frozen_atoms(mol_ts, reacting_atoms, frozen_atoms)

        new_mol = self.embed_TS(
            mol_ts=mol_ts,
            new_mol=new_mol,
            reacting_atoms=reacting_atoms,
            frozen_atoms=frozen_atoms,
            number_of_conformers=number_of_conformers,
            conf_factor=conf_factor,
        )

        new_mol = self.optimize(
            new_mol=new_mol,
            mol_ts=mol_ts,
            frozen_atoms=frozen_atoms,
            auto_fallback=auto_fallback,
        )

        new_mol = self.prune(new_mol)

        self.mol = new_mol

        return new_mol
