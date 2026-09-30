"""ConformerEnsemble: a molecule with its conformers, energies and provenance."""

import json
from dataclasses import dataclass, field
from numbers import Integral
from typing import Any, Dict, Iterable, List, Optional, Sequence

import numpy as np
from rdkit import Chem

from racerts.io.xyz import write_xyz
from racerts.utils.units import ENERGY_UNITS

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

    def copy(self) -> "ConformerEnsemble":
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

    def energies(self, unit: str = "kcal/mol") -> np.ndarray:
        """
        Energies in the order of conf_ids; NaN where missing. unit: "kcal/mol" (as
        stored), "kJ/mol", "eV" or "hartree".
        """
        if unit not in ENERGY_UNITS:
            raise ValueError(f"unit must be one of {sorted(ENERGY_UNITS)}.")
        energies = np.array([_energy(c) for c in self.mol.GetConformers()], dtype=float)
        return energies / ENERGY_UNITS[unit]

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

    def filter(
        self, conf_ids: Iterable[int], renumber: bool = False
    ) -> "ConformerEnsemble":
        """
        A copy with only the given conformers, in the given order, with their data.
        They keep their ids, or with renumber get the ids 0, 1, ... in that order.
        """
        conf_ids = list(conf_ids)
        if any(isinstance(i, bool) or not isinstance(i, Integral) for i in conf_ids):
            raise TypeError("Conformer ids must be integers.")
        conf_ids = [int(i) for i in conf_ids]
        if len(set(conf_ids)) != len(conf_ids):
            raise ValueError(f"Repeated conformer ids: {conf_ids}.")
        unknown = set(conf_ids) - set(self.conf_ids)
        if unknown:
            raise ValueError(f"No conformers with ids {sorted(unknown)}.")
        mol = Chem.Mol(self.mol)
        mol.RemoveAllConformers()
        for conf_id in conf_ids:
            conf = Chem.Conformer(self.mol.GetConformer(conf_id))  # with its data
            mol.AddConformer(conf, assignId=renumber)
        return ConformerEnsemble(mol)

    def merge(
        self, other: "ConformerEnsemble", identity: str = "graph"
    ) -> "ConformerEnsemble":
        """
        A copy with the conformers of both ensembles, atom by atom. The conformers of
        other get new ids; self supplies the molecule.

        identity: What must agree at every atom index: "graph" (elements, isotopes,
            formal charges, radical electrons and bonds) or "elements" (elements and
            isotopes only, e.g. for conformers of the same system under another
            graph, as catmlp merges TS guesses).

        Energies from different methods cannot be ranked together, so ensembles with
        different energy_method values are not merged.
        """
        if identity not in ("graph", "elements"):
            raise ValueError("identity must be 'graph' or 'elements'.")
        same = _same_graph if identity == "graph" else _same_elements
        if not same(self.mol, other.mol):
            what = "molecular graph" if identity == "graph" else "elements"
            raise ValueError(
                f"Only ensembles with the same {what} (atom by atom) can be merged."
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

    @classmethod
    def from_frames(
        cls,
        template: Chem.Mol,
        frames: Iterable[Any],
        atom_order: Optional[Sequence[int]] = None,
    ) -> "ConformerEnsemble":
        """
        Conformers of template from external geometries (catmlp import_conformers):
        ASE Atoms or arrays of positions (A, one row per atom, hydrogens included).

        atom_order[i] is the index in the frames of template atom i (default: the
        same order). The elements (for Atoms) and finite, isolated, unconstrained
        frames are checked; no chemistry or stereo is (see racerts.validate).
        Energies and other results of the frames are not taken over; the provenance
        records source "external" and the frame number. The template keeps its graph
        and its charge and multiplicity properties; its conformers and other
        properties are dropped.
        """
        n_atoms = template.GetNumAtoms()
        numbers = np.array([atom.GetAtomicNum() for atom in template.GetAtoms()])
        if not n_atoms or 0 in numbers:
            raise ValueError("The template must contain real atoms, not placeholders.")
        order = list(range(n_atoms)) if atom_order is None else list(atom_order)
        if any(isinstance(i, bool) or not isinstance(i, Integral) for i in order):
            raise TypeError("atom_order must contain integer indices.")
        if sorted(order) != list(range(n_atoms)):
            raise ValueError("atom_order must be a permutation of the atom indices.")

        mol = Chem.Mol(template)
        mol.RemoveAllConformers()
        for key in list(mol.GetPropNames(includePrivate=True, includeComputed=False)):
            if key not in ("charge", "multiplicity"):
                mol.ClearProp(key)
        ensemble = cls(mol)
        for index, frame in enumerate(frames):
            positions = _frame_positions(frame, index, numbers, order)
            conf = Chem.Conformer(n_atoms)
            for atom, position in enumerate(positions):
                conf.SetAtomPosition(atom, position.tolist())
            conf_id = mol.AddConformer(conf, assignId=True)
            ensemble.add_provenance(conf_id, source="external", frame=index)
        if not mol.GetNumConformers():
            raise ValueError("The external ensemble is empty.")
        return ensemble

    def write_xyz(
        self, file_name: str, use_energy: bool = False, comment: Optional[str] = None
    ) -> None:
        """Write all conformers to a multi-structure xyz file (see io.xyz.write_xyz)."""
        write_xyz(self.mol, file_name, use_energy=use_energy, comment=comment)

    def write_sdf(self, file_name: str) -> None:
        """
        Write all conformers to an SDF file: one record per conformer, with the graph,
        and the properties conf_id, energy (kcal/mol), energy_method and provenance
        (JSON). V3000 for molecules with dative bonds.
        """
        mol = self.mol
        writer = Chem.SDWriter(file_name)
        writer.SetKekulize(False)
        if any(b.GetBondType() == Chem.BondType.DATIVE for b in mol.GetBonds()):
            writer.SetForceV3000(True)
        try:
            for conf in mol.GetConformers():
                record = Chem.Mol(mol, confId=conf.GetId())
                record.SetIntProp("conf_id", conf.GetId())
                if conf.HasProp("energy"):
                    record.SetDoubleProp("energy", conf.GetDoubleProp("energy"))
                if self.energy_method:
                    record.SetProp("energy_method", self.energy_method)
                if conf.HasProp(PROVENANCE):
                    record.SetProp(PROVENANCE, conf.GetProp(PROVENANCE))
                writer.write(record)
        finally:
            writer.close()

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


def _same_elements(a: Chem.Mol, b: Chem.Mol) -> bool:
    def elements(mol):
        return [(atom.GetAtomicNum(), atom.GetIsotope()) for atom in mol.GetAtoms()]

    return elements(a) == elements(b)


def _same_graph(a: Chem.Mol, b: Chem.Mol) -> bool:
    def graph(mol):
        atoms = [
            (
                atom.GetAtomicNum(),
                atom.GetIsotope(),
                atom.GetFormalCharge(),
                atom.GetNumRadicalElectrons(),
            )
            for atom in mol.GetAtoms()
        ]
        bonds = sorted(
            (
                *sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())),
                str(bond.GetBondType()),
            )
            for bond in mol.GetBonds()
        )
        return atoms, bonds

    return graph(a) == graph(b)


def _frame_positions(frame, index: int, numbers: np.ndarray, order) -> np.ndarray:
    """The positions of one external frame in template order, checked."""
    if hasattr(frame, "get_positions") and hasattr(frame, "numbers"):
        if len(frame) != len(numbers) or not np.array_equal(
            np.asarray(frame.numbers)[order], numbers
        ):
            raise ValueError(f"Frame {index} has a different atom count or order.")
        if np.any(frame.pbc) or frame.constraints:
            raise ValueError(f"Frame {index} must be isolated and unconstrained.")
        positions = np.asarray(frame.get_positions(), dtype=float)
    elif isinstance(frame, (np.ndarray, list, tuple)):
        positions = np.asarray(frame, dtype=float)
        if positions.shape != (len(numbers), 3):
            raise ValueError(
                f"Frame {index} has positions of shape {positions.shape}, not "
                f"({len(numbers)}, 3)."
            )
    else:
        raise TypeError("Frames must be ASE Atoms or arrays of positions.")
    positions = positions[order]
    if not np.isfinite(positions).all():
        raise ValueError(f"Frame {index} has nonfinite coordinates.")
    return positions
