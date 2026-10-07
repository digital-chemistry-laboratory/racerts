import logging
from copy import deepcopy
from typing import List, Optional

from rdkit import Chem
from rdkit.Chem import Descriptors

from racerts.utils import (
    EV_TO_KCAL_MOL,
    atom_idx_input_validation,
    count_electrons,
    get_frozen_atoms,
    infer_charge_and_multiplicity,
)

from .embedder import BaseEmbedder, CmapEmbedder
from .mol_getter import (
    BaseMolGetter,
    MolGetterBonds,
    MolGetterConnectivity,
    MolGetterSMILES,
)
from .optimizer import BaseOptimizer, MMFFOptimizer, UFFOptimizer
from .pruner import BasePruner, EnergyPruner, RMSDPruner

logger = logging.getLogger(__name__)

KCAL_TO_HARTREE = 627.509
DEFAULT_CONF_FACTOR = 80


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
            file_name (str): The path to the XYZ or SDF/MOL file.
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

        mol_ts = None
        if file_name.endswith(".sdf") or file_name.endswith(".mol"):
            mol_ts = Chem.MolFromMolFile(file_name, removeHs=False)
            if mol_ts is not None and charge != sum(
                [
                    mol_ts.GetAtomWithIdx(i).GetFormalCharge()
                    for i in range(mol_ts.GetNumAtoms())
                ]
            ):
                raise ValueError(
                    f"Charge {charge} does not match the formal charges in the file {file_name}."
                )

        elif file_name.endswith(".xyz"):
            try:
                mol_ts = self._mol_getter.get_mol(file_name=file_name, **get_mol_kwargs)
                return mol_ts
            except Exception as e:
                if auto_fallback is False:
                    raise
                # Without SMILES, the default SMILES getter always fails; that is expected.
                if "input_smiles" in get_mol_kwargs or not isinstance(
                    self._mol_getter, MolGetterSMILES
                ):
                    logger.warning("%s failed: %s", type(self._mol_getter).__name__, e)
                if not isinstance(self._mol_getter, MolGetterBonds):
                    try:
                        mol_ts = MolGetterBonds().get_mol(
                            file_name=file_name, **get_mol_kwargs
                        )
                        logger.info("Using the bonds perceived by DetermineBonds.")
                        return mol_ts
                    except Exception as e:
                        logger.warning("MolGetterBonds failed: %s", e)

                logger.warning(
                    "Using the connectivity of the xyz file only (DetermineConnectivity), "
                    "without bond orders or formal charges."
                )
                mol_ts = MolGetterConnectivity().get_mol(
                    file_name=file_name, **get_mol_kwargs
                )
        else:
            raise ValueError(
                "Only file extensions .sdf and .xyz are supported"
            )
        if mol_ts is None:
            raise ValueError(
                f"Failed to create molecule from {file_name}. Check the file format and content."
            )
        return mol_ts

    def embed_TS(
        self,
        mol_ts: Chem.Mol,
        new_mol: Chem.Mol,
        reacting_atoms: List[str],
        frozen_atoms: List,
        number_of_conformers: int = -1,
        conf_factor: int = DEFAULT_CONF_FACTOR,
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

        energy_method = type(self._optimizer).__name__
        try:
            self._optimizer.tune_ts_conformers(
                mol=new_mol, reference=mol_ts, align_indices=frozen_atoms
            )
        except Exception as e:
            # Only MMFF falls back (to UFF); other errors are passed on to the caller.
            if isinstance(self._optimizer, MMFFOptimizer) and auto_fallback is True:
                logger.warning(
                    "%s failed (%s); falling back to UFF. UFF energies are less "
                    "reliable for ranking conformers.",
                    energy_method,
                    e,
                )
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
                energy_method = type(optimizer).__name__
            else:
                raise

        new_mol.SetProp("energy_method", energy_method)
        return new_mol

    def write_xyz(
        self, file_name: str, use_energy=False, comment: Optional[str] = None
    ):
        """
        Write all conformers to a multi-structure xyz file.

        By default, the comment lines are in extended XYZ format (e.g. for ase.io.read)
        with the charge, the spin multiplicity (as "spin" and "multiplicity"), the
        energy in eV as "racerts_energy" and the method that produced it as
        "energy_method". The key is not "energy", which ASE would read as the potential
        energy of the structure, although it is usually a force-field energy.

        Args:
            file_name (str): Output path.
            use_energy (bool): Instead, write only the energy in Hartree, as in CREST
                ensembles; conformers without an energy are left out.
            comment (str): Instead, write this comment line.
        """
        info = infer_charge_and_multiplicity(self.mol)
        extxyz = [
            f"charge={info['charge']}",
            f"spin={info['multiplicity']}",
            f"multiplicity={info['multiplicity']}",
        ]
        if self.mol.HasProp("energy_method"):
            extxyz.append(f"energy_method={self.mol.GetProp('energy_method')}")
        extxyz.append('pbc="F F F"')

        missing_energy = []
        with open(file_name, "w") as f:
            for conf in self.mol.GetConformers():
                energy = (
                    conf.GetDoubleProp("energy") if conf.HasProp("energy") else None
                )
                if energy is None and (use_energy or comment is None):
                    missing_energy.append(conf.GetId())
                    if use_energy:
                        continue
                mol_block = Chem.rdmolfiles.MolToXYZBlock(
                    self.mol, confId=conf.GetId()
                ).strip()
                lines = mol_block.split("\n")

                if use_energy:
                    lines[1] = f"{energy / KCAL_TO_HARTREE:.6f}"
                elif comment is not None:
                    lines[1] = comment
                else:
                    fields = ["Properties=species:S:1:pos:R:3"]
                    if energy is not None:
                        fields.append(f"racerts_energy={energy / EV_TO_KCAL_MOL:.8f}")
                    lines[1] = " ".join(fields + extxyz)

                f.write("\n".join(lines) + "\n")

        if missing_energy:
            logger.warning(
                "Conformers %s have no energy%s.",
                missing_energy,
                " and are left out" if use_energy else "",
            )

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
        conf_factor: int = DEFAULT_CONF_FACTOR,
        auto_fallback=True,
        multiplicity: Optional[int] = None,
    ) -> Chem.Mol:

        mol_ts = self.get_mol(
            file_name, charge, reacting_atoms, input_smiles, auto_fallback=auto_fallback
        )
        if mol_ts is None:
            if self._verbose is True:
                print("No valid mol object could be generated.")
            raise ValueError("No valid mol object could be generated.")

        # Keep charge and multiplicity on the molecule for later steps (ASE, write_xyz),
        # also for graphs without formal charges (MolGetterConnectivity).
        state = infer_charge_and_multiplicity(mol_ts, charge, multiplicity)
        electrons = count_electrons(mol_ts, state["charge"])
        if (electrons + state["multiplicity"]) % 2 == 0:
            logger.warning(
                "Multiplicity %d does not fit the %d electrons of the TS at charge %d.",
                state["multiplicity"],
                electrons,
                state["charge"],
            )
        elif multiplicity is None and state["multiplicity"] == 2:
            logger.warning(
                "The TS has an odd number of electrons (%d at charge %d), so the "
                "multiplicity is 2. Check the charge, or pass multiplicity=2 for a "
                "radical.",
                electrons,
                state["charge"],
            )
        mol_ts.SetIntProp("charge", state["charge"])
        mol_ts.SetIntProp("multiplicity", state["multiplicity"])

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
        if new_mol is None:
            raise RuntimeError(
                "Embedding produced no conformers. Check the reacting atoms and the TS "
                "geometry, or try another embedder."
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
