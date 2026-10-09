"""ConformerEnsemble: a molecule with its conformers, energies and provenance."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

import numpy as np
from rdkit import Chem

from racerts.io.xyz import write_xyz

PROVENANCE = "provenance"


@dataclass(frozen=True)
class ConformerRecord:
    """What is known about one conformer (a read-only view)."""

    conf_id: int
    energy: Optional[float] = None  # kcal/mol
    energy_method: Optional[str] = None
    provenance: dict = field(default_factory=dict)


class ConformerEnsemble:
    """
    A molecule with its conformers.

    Everything known about the conformers stays in RDKit properties: "energy"
    (kcal/mol) and "provenance" (a JSON object, e.g. the embedder and seed) of each
    conformer, and "energy_method" of the molecule. The ensemble only reads and writes
    them, so ensemble.mol is always a complete result for RDKit code and the legacy API.
    """

    def __init__(self, mol: Chem.Mol):
        self.mol = mol

    def copy(self) -> ConformerEnsemble:
        return ConformerEnsemble(Chem.Mol(self.mol))

    # RDKit's default pickling drops all properties (energies, provenance, charge).
    def __getstate__(self):
        return {"mol": self.mol.ToBinary(Chem.PropertyPickleOptions.AllProps)}

    def __setstate__(self, state):
        self.mol = Chem.Mol(state["mol"])

    def __len__(self) -> int:
        return self.mol.GetNumConformers()

    def __repr__(self) -> str:
        return f"ConformerEnsemble({len(self)} conformers)"

    @property
    def conf_ids(self) -> List[int]:
        return [conf.GetId() for conf in self.mol.GetConformers()]

    @property
    def energy_method(self) -> Optional[str]:
        """The method of the energies, e.g. "MMFFOptimizer"."""
        mol = self.mol
        return mol.GetProp("energy_method") if mol.HasProp("energy_method") else None

    def energy(self, conf_id: int) -> Optional[float]:
        return _energy(self.mol.GetConformer(conf_id))

    def energies(self) -> np.ndarray:
        """Energies in kcal/mol in the order of conf_ids; NaN where missing."""
        return np.array([_energy(c) for c in self.mol.GetConformers()], dtype=float)

    def best(self) -> int:
        """The id of the conformer with the lowest energy."""
        energies = self.energies()
        if np.isnan(energies).all():
            raise ValueError("No conformer has an energy.")
        return self.conf_ids[int(np.nanargmin(energies))]

    def provenance(self, conf_id: int) -> dict:
        return _provenance(self.mol.GetConformer(conf_id))

    def add_provenance(self, conf_id: Optional[int] = None, **items) -> None:
        """Add JSON-serializable items to the provenance of a conformer (None: all)."""
        if conf_id is None:
            conformers = self.mol.GetConformers()
        else:
            conformers = [self.mol.GetConformer(conf_id)]
        for conf in conformers:
            conf.SetProp(PROVENANCE, json.dumps({**_provenance(conf), **items}))

    def record(self, conf_id: int) -> ConformerRecord:
        return self._record(self.mol.GetConformer(conf_id))

    @property
    def records(self) -> Dict[int, ConformerRecord]:
        """All records by conformer id (for one conformer, record is cheaper)."""
        return {conf.GetId(): self._record(conf) for conf in self.mol.GetConformers()}

    def _record(self, conf: Chem.Conformer) -> ConformerRecord:
        return ConformerRecord(
            conf_id=conf.GetId(),
            energy=_energy(conf),
            energy_method=self.energy_method,
            provenance=_provenance(conf),
        )

    def filter(self, conf_ids: Iterable[int]) -> ConformerEnsemble:
        """A copy with only the given conformers (ids and data kept)."""
        keep = set(conf_ids)
        unknown = keep - set(self.conf_ids)
        if unknown:
            raise ValueError(f"No conformers with ids {sorted(unknown)}.")
        mol = Chem.Mol(self.mol)
        for conf_id in self.conf_ids:
            if conf_id not in keep:
                mol.RemoveConformer(conf_id)
        return ConformerEnsemble(mol)

    def merge(self, other: ConformerEnsemble) -> ConformerEnsemble:
        """
        A copy with the conformers of both ensembles, which must share the molecular
        graph. The conformers of other get new ids. Energies from different methods
        cannot be ranked together, so ensembles with different energy_method values
        are not merged.
        """
        if not _same_graph(self.mol, other.mol):
            raise ValueError(
                "Only ensembles of the same molecular graph can be merged."
            )
        methods = {e.energy_method for e in (self, other)} - {None}
        if len(methods) > 1:
            raise ValueError(
                f"The energies come from different methods ({', '.join(sorted(methods))})"
                " and cannot be ranked together."
            )
        mol = Chem.Mol(self.mol)
        if methods:
            mol.SetProp("energy_method", methods.pop())
        for conf in other.mol.GetConformers():
            mol.AddConformer(conf, assignId=True)  # AddConformer copies
        return ConformerEnsemble(mol)

    def write_xyz(
        self, file_name: str, use_energy: bool = False, comment: Optional[str] = None
    ) -> None:
        """Write all conformers to a multi-structure xyz file (see io.xyz.write_xyz)."""
        write_xyz(self.mol, file_name, use_energy=use_energy, comment=comment)

    def to_ase(self, conf_id: int):
        """One conformer as ASE Atoms, with charge and multiplicity (needs ASE)."""
        from racerts.io.ase import rdkit_conformer_to_ase_atoms

        return rdkit_conformer_to_ase_atoms(self.mol, conf_id)

    def summary(self) -> str:
        energies = self.energies()
        if np.isnan(energies).all():
            return f"{len(self)} conformers"
        i = int(np.nanargmin(energies))
        method = self.energy_method or "unknown method"
        window = np.nanmax(energies) - energies[i]
        return (
            f"{len(self)} conformers; energies ({method}): lowest {energies[i]:.4f} "
            f"kcal/mol (conformer {self.conf_ids[i]}), window {window:.2f} kcal/mol"
        )


def _energy(conf: Chem.Conformer) -> Optional[float]:
    return conf.GetDoubleProp("energy") if conf.HasProp("energy") else None


def _provenance(conf: Chem.Conformer) -> dict:
    return json.loads(conf.GetProp(PROVENANCE)) if conf.HasProp(PROVENANCE) else {}


def _same_graph(a: Chem.Mol, b: Chem.Mol) -> bool:
    def graph(mol):
        atoms = [atom.GetAtomicNum() for atom in mol.GetAtoms()]
        bonds = sorted(
            (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), str(bond.GetBondType()))
            for bond in mol.GetBonds()
        )
        return atoms, bonds

    return graph(a) == graph(b)
