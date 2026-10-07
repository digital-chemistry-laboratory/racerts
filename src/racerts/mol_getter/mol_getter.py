import itertools
import logging
from abc import abstractmethod
from collections import Counter
from typing import Dict, Iterator, List, Union

import numpy as np
from rdkit import Chem
from rdkit.Chem.AllChem import SanitizeMol  # type: ignore
from rdkit.Chem import rdDetermineBonds
from rdkit.Chem import rdFMCS
from racerts.utils import suppress_std

logger = logging.getLogger(__name__)


class BaseMolGetter:
    @abstractmethod
    def __init__(self, assignBonds: bool, allowChargedFragments: bool, **kwargs):
        raise NotImplementedError

    @abstractmethod
    def get_mol(self, file_name: str, **kwargs) -> Union[Chem.Mol, None]:
        """
        Create a molecular graph from an XYZ file.
        """
        raise NotImplementedError


class MolGetterBonds(BaseMolGetter):
    def __init__(
        self, assignBonds: bool = True, allowChargedFragments: bool = True, **kwargs
    ):
        self.assignBonds = assignBonds
        self.allowChargedFragments = allowChargedFragments

    def get_mol(self, file_name: str, **kwargs) -> Union[Chem.Mol, None]:
        """
        Create a molecular graph from an XYZ file.

        Args:
            file_name (str): The path to the XYZ file.
            charge (int): The molecular charge.
        Returns:
            Chem.Mol: The molecule object.
        """
        charge = kwargs.get("charge", 0)
        if "charge" not in kwargs:
            print("No charge provided, defaulting to 0")

        mol_ts = Chem.MolFromXYZFile(file_name)

        if mol_ts is None:
            return None

        if self.assignBonds:
            with suppress_std():
                rdDetermineBonds.DetermineBonds(
                    mol_ts,
                    charge=charge,
                    allowChargedFragments=self.allowChargedFragments,
                )

        if mol_ts is None:
            raise ValueError(
                f"Failed to create molecule from {file_name}. Check the file format and content."
            )

        return mol_ts


class MolGetterConnectivity(BaseMolGetter):

    def __init__(self, **kwargs):
        pass

    def get_mol(self, file_name: str, **kwargs) -> Union[Chem.Mol, None]:
        """
        Create a molecular graph from an XYZ file.

        Args:
            file_name (str): The path to the XYZ file.
            charge (int): The molecular charge.
        Returns:
            Chem.Mol: The molecule object.
        """

        mol_ts = Chem.MolFromXYZFile(file_name)

        if mol_ts is None:
            return None

        # if self.assignBonds:
        rdDetermineBonds.DetermineConnectivity(mol_ts)
        SanitizeMol(
            mol_ts,
            Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
            | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
            | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
            | Chem.SanitizeFlags.SANITIZE_SYMMRINGS,
        )
        Chem.AssignStereochemistryFrom3D(mol_ts)

        if mol_ts is None:
            raise ValueError(
                f"Failed to create molecule from {file_name}. Check the file format and content."
            )
        return mol_ts


class MolGetterSMILES(BaseMolGetter):

    def __init__(self, **kwargs):
        pass

    def combine_mols(self, smiles_list) -> Chem.Mol:
        if not isinstance(smiles_list, list):
            raise ValueError("Input SMILES must be provided as a list.")
        if not smiles_list:
            raise ValueError("No input SMILES provided.")

        combined_mol = None
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")
            mol = Chem.AddHs(mol)
            combined_mol = (
                mol if combined_mol is None else Chem.CombineMols(combined_mol, mol)
            )

        return combined_mol

    def match_AtomMapNum(self, ref_mol: Chem.Mol, mol: Chem.Mol) -> List[Chem.Mol]:
        """
        Match the atom map numbers for the reference molecule and the molecule of interest.
        The reference atoms are numbered by their index (atom maps given in the SMILES are
        replaced), and the numbers are transferred to the molecule of interest by
        iteratively searching for the MCS. SMILES with atom maps are matched by
        match_by_atom_maps (every heavy atom mapped) or match_by_substructure instead.

        Developer info: MCS works fine until two or more possible MCS
        are possible for the structure (Try GetSubstructureMatches to get this info)...
        this usually happens for reacting atoms, which are anyway fixed later.

        Args:
            ref_mol (Chem.Mol): The molecule of reference
            mol (Chem.Mol): The molecule of interest

        Returns:
            List[Chem.Mol]: The atom-maped molecule objects that are returned.
        """
        # Always renumber: maps from the SMILES do not decide the matching (the MCS does),
        # and hydrogens added by AddHs have none, which led to duplicate numbers.
        for atom in ref_mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(-atom.GetIdx() - 1)

        truncate_ref = Chem.RWMol(ref_mol)
        truncate_mol = Chem.RWMol(mol)
        while truncate_ref.GetNumAtoms() > 0 and truncate_mol.GetNumAtoms() > 0:
            MCresult = rdFMCS.FindMCS(
                [truncate_ref, truncate_mol], bondCompare=rdFMCS.BondCompare.CompareAny
            )
            mc_mol = MCresult.queryMol
            highlight_mcs_r = truncate_ref.GetSubstructMatch(mc_mol, useChirality=False)
            highlight_mcs_p = truncate_mol.GetSubstructMatch(mc_mol, useChirality=False)
            for id in range(len(highlight_mcs_r)):
                id_map_r = truncate_ref.GetAtomWithIdx(
                    highlight_mcs_r[id]
                ).GetAtomMapNum()
                id_map_p = truncate_mol.GetAtomWithIdx(
                    highlight_mcs_p[id]
                ).GetAtomMapNum()
                for atom in mol.GetAtoms():
                    if atom.GetAtomMapNum() == id_map_p:
                        atom.SetAtomMapNum(id_map_r)
            truncate_ref.BeginBatchEdit()
            for id in highlight_mcs_r:
                truncate_ref.RemoveAtom(id)
            truncate_ref.CommitBatchEdit()
            truncate_mol.BeginBatchEdit()
            for id in highlight_mcs_p:
                truncate_mol.RemoveAtom(id)
            truncate_mol.CommitBatchEdit()

        return [ref_mol, mol]

    def match_by_atom_maps(
        self, ref_mol: Chem.Mol, mol: Chem.Mol, reacting_atoms: List[int]
    ) -> List[Chem.Mol]:
        """
        Match the atoms by the atom map numbers of the SMILES: number n is atom n of
        the xyz file (1-based). Hydrogens without a number are matched to the nearest
        free hydrogens of their heavy atom. Afterwards, atoms of both molecules carry
        their 1-based atom number in the xyz file as map number (RDKit index + 1, since
        map number 0 means "unmapped"), as for match_AtomMapNum.

        Raises:
            ValueError: If the numbers do not fit the xyz file: out of range, repeated,
                another element, or a bond of the SMILES that is missing from the
                perceived connectivity of the xyz file (unless between reacting atoms).
        """
        target = self.mapped_atoms(ref_mol, mol)  # template index -> xyz index

        # Unmapped hydrogens: nearest free xyz hydrogens of their heavy atom.
        positions = mol.GetConformer().GetPositions()
        free = {
            a.GetIdx()
            for a in mol.GetAtoms()
            if a.GetAtomicNum() == 1 and a.GetIdx() not in target.values()
        }
        candidates = []
        for atom in ref_mol.GetAtoms():
            if atom.GetIdx() in target:
                continue
            heavy = atom.GetNeighbors()[0].GetIdx() if atom.GetDegree() == 1 else None
            if atom.GetAtomicNum() != 1 or heavy not in target:
                raise ValueError(
                    f"atom {atom.GetIdx()} of the SMILES has no map number"
                )
            anchor = positions[target[heavy]]
            candidates += [
                (float(np.linalg.norm(positions[h] - anchor)), atom.GetIdx(), h)
                for h in free
            ]
        for _, template_h, xyz_h in sorted(candidates):
            if template_h not in target and xyz_h in free:
                target[template_h] = xyz_h
                free.discard(xyz_h)
        if len(target) != ref_mol.GetNumAtoms():
            raise ValueError("the numbers of hydrogens differ")

        for atom in ref_mol.GetAtoms():
            atom.SetAtomMapNum(target[atom.GetIdx()] + 1)
        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx() + 1)
        missing = self.bonds_missing_from_geometry(ref_mol, mol, reacting_atoms)
        if missing:
            i, j = missing[0]
            raise ValueError(
                f"atoms {i + 1} and {j + 1} are bonded in the SMILES but not in the xyz "
                "file"
            )
        return [ref_mol, mol]

    @staticmethod
    def mapped_atoms(ref_mol: Chem.Mol, mol: Chem.Mol) -> Dict[int, int]:
        """
        Template atom index -> xyz atom index for the atoms of the SMILES with a map
        number (number n is atom n of the xyz file, 1-based).

        The element is only a first check: match_by_atom_maps and match_by_substructure
        then check the numbers against the bonds of the xyz file.

        Raises:
            ValueError: If a number is out of range, repeated or of another element.
        """
        target = {}
        for atom in ref_mol.GetAtoms():
            number = atom.GetAtomMapNum()
            if not number:
                continue
            if not 1 <= number <= mol.GetNumAtoms():
                raise ValueError(
                    f"atom map number {number} is not an atom of the xyz file"
                )
            other = mol.GetAtomWithIdx(number - 1)
            if other.GetAtomicNum() != atom.GetAtomicNum():
                raise ValueError(
                    f"atom map number {number} is {atom.GetSymbol()} in the SMILES but "
                    f"{other.GetSymbol()} in the xyz file"
                )
            target[atom.GetIdx()] = number - 1
        if len(set(target.values())) != len(target):
            raise ValueError("atom map numbers are repeated")
        return target

    def match_by_substructure(
        self, ref_mol: Chem.Mol, mol: Chem.Mol, reacting_atoms: List[int]
    ) -> List[Chem.Mol]:
        """
        Match the atoms by a substructure search of the SMILES in the connectivity of
        the xyz file, bond orders and charges aside, with the atoms that have a map
        number held at their xyz atom (see mapped_atoms). Bonds between reacting atoms
        may be missing from the connectivity, so they are added in steps (none, the
        pairs closer than 1.7 x their covalent radii, all pairs) until a match gives a
        valid molecule. The fragments of the SMILES are matched one at a time (see
        _fragments_in_search_order), so identical fragments such as solvent molecules
        are never permuted against each other. Afterwards, atoms of both molecules
        carry their 1-based atom number in the xyz file as map number, as for
        match_AtomMapNum.

        Raises:
            ValueError: If the map numbers do not fit the xyz file or nothing matches.
        """
        pinned = self.mapped_atoms(ref_mol, mol)
        query = _plain_graph(ref_mol)
        for label, index in enumerate(pinned, start=1):
            query.GetAtomWithIdx(index).SetIsotope(label)
        fragments = _fragments_in_search_order(query)
        for target in _connectivity_with_reacting_bonds(mol, reacting_atoms):
            for label, index in enumerate(pinned.values(), start=1):
                target.GetAtomWithIdx(index).SetIsotope(label)
            match = _match_fragments(fragments, target)
            if match is None:
                continue
            template, geometry = Chem.Mol(ref_mol), Chem.Mol(mol)
            for atom in template.GetAtoms():
                atom.SetAtomMapNum(match[atom.GetIdx()] + 1)
            for atom in geometry.GetAtoms():
                atom.SetAtomMapNum(atom.GetIdx() + 1)
            try:  # e.g. an aromatic ring broken at the reaction centre
                self.set_coords(Chem.Mol(template), Chem.Mol(geometry), reacting_atoms)
            except ValueError as error:  # includes RDKit's sanitization errors
                logger.debug("Substructure match rejected: %s", error)
                continue
            return [template, geometry]
        raise ValueError(
            "the SMILES does not match the connectivity of the xyz file with the "
            "mapped atoms in place"
        )

    @staticmethod
    def bonds_missing_from_geometry(
        ref_mol: Chem.Mol, mol: Chem.Mol, reacting_atoms: List[int]
    ) -> List[tuple]:
        """
        Bonds of the matched SMILES template (ref_mol) that the perceived connectivity
        of the xyz file (mol) lacks, as pairs of xyz atom indices. Pairs of reacting
        atoms are skipped, as their bonds may form or break in the TS.
        """
        where = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol.GetAtoms()}
        reacting = set(reacting_atoms)
        missing = []
        for bond in ref_mol.GetBonds():
            i = where.get(bond.GetBeginAtom().GetAtomMapNum())
            j = where.get(bond.GetEndAtom().GetAtomMapNum())
            if i is None or j is None or {i, j} <= reacting:
                continue
            if mol.GetBondBetweenAtoms(i, j) is None:
                missing.append((min(i, j), max(i, j)))
        return missing

    def set_coords(
        self, pmol: Chem.Mol, mol_ts: Chem.Mol, reacting_atoms: List[int]
    ) -> Chem.Mol:
        """
        Assign the topoly (connectivity and bond orders) from the SMILES structure to the xyz file of the transition state.
        Only bonds present between reacting atoms presents in pmol and in mol_ts after determining the connectivity will be copied.

        Args:
            pmol (Chem.Mol): Mol object from which the topology (connectivity and bond orders) is taken
            mol_ts (Chem.Mol): Mol object with 3D coordinates but no topology
            reacting_atoms (List[int]): List of the reacting atoms based on the indexes of mol_ts

        Returns:
            Chem.Mol: The final molecule with coordinates and bond order information
        """
        map_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in mol_ts.GetAtoms()}
        for atom in pmol.GetAtoms():
            idx = map_to_idx.get(atom.GetAtomMapNum())
            if idx is not None:
                mol_ts.GetAtomWithIdx(idx).SetFormalCharge(atom.GetFormalCharge())
        emol = Chem.EditableMol(mol_ts)
        for bond in mol_ts.GetBonds():
            emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
        for bond in pmol.GetBonds():
            m1 = bond.GetBeginAtom().GetAtomMapNum()
            m2 = bond.GetEndAtom().GetAtomMapNum()
            if m1 not in map_to_idx or m2 not in map_to_idx:
                raise ValueError(
                    f"Bond {m1}-{m2} of the SMILES template has no counterpart in the "
                    "TS geometry. Check that the SMILES matches the xyz file."
                )
            id1, id2 = map_to_idx[m1], map_to_idx[m2]
            if id1 not in reacting_atoms or id2 not in reacting_atoms:
                emol.AddBond(id1, id2, order=bond.GetBondType())
            elif mol_ts.GetBondBetweenAtoms(id1, id2) is not None:
                emol.AddBond(id1, id2, order=bond.GetBondType())
        new_mol = emol.GetMol()
        Chem.SanitizeMol(new_mol)
        Chem.SetDoubleBondNeighborDirections(new_mol, new_mol.GetConformer())
        Chem.DetectBondStereochemistry(new_mol)
        for (
            atom
        ) in (
            new_mol.GetAtoms()
        ):  # <- We need to remove the atom map numbers to avoid the artifacts
            atom.SetAtomMapNum(0)
        Chem.AssignStereochemistryFrom3D(
            new_mol
        )  # <- Now we use the 3D conformer to set the stereochemistry
        Chem.AssignCIPLabels(new_mol)
        # Chem.AssignStereochemistry(new_mol) <- This is deprecated in favor of Chem.AssignCIPLabels(new_mol) above
        return new_mol

    def get_mol(self, file_name: str, **kwargs) -> Chem.Mol:

        if "input_smiles" not in kwargs:
            raise ValueError("No input SMILES provided in the SMILES method.")
        if "reacting_atoms" not in kwargs:
            raise ValueError("No reacting atoms provided.")

        input_smiles = kwargs.get("input_smiles", None)
        reacting_atoms = kwargs.get("reacting_atoms", [])

        if type(input_smiles) is str:
            input_smiles = [input_smiles]
        if not isinstance(input_smiles, (list, tuple)) or not isinstance(
            reacting_atoms, (list, tuple)
        ):
            raise ValueError(
                "Input SMILES and reacting atoms must be provided as lists (or tuples)."
            )
        input_smiles, reacting_atoms = list(input_smiles), list(reacting_atoms)
        input_mol = self.combine_mols(input_smiles)
        charge = kwargs.get("charge")
        if charge is not None and charge != Chem.GetFormalCharge(input_mol):
            raise ValueError(
                f"Charge {charge} does not match the formal charges of the SMILES "
                f"({Chem.GetFormalCharge(input_mol)})."
            )

        mol_ts = Chem.MolFromXYZFile(file_name)
        if mol_ts is None:
            raise ValueError(f"Failed to read {file_name}.")
        # Otherwise atoms missing from the SMILES are silently left without bonds.
        in_smiles = Counter(atom.GetSymbol() for atom in input_mol.GetAtoms())
        in_xyz = Counter(atom.GetSymbol() for atom in mol_ts.GetAtoms())
        if in_smiles != in_xyz:
            raise ValueError(
                "The SMILES and the xyz file do not match. Extra atoms in the xyz file: "
                f"{dict(in_xyz - in_smiles)}, extra atoms in the SMILES: "
                f"{dict(in_smiles - in_xyz)}."
            )
        new_mol = self.setup_mol(mol_ts, reacting_atoms, input_mol)
        if new_mol is None:
            raise ValueError(
                f"Failed to create molecule from {file_name}. Check the file format and content."
            )
        return new_mol

    def setup_mol(self, mol_ts, reacting_atoms, input_mol):
        rdDetermineBonds.DetermineConnectivity(mol_ts)
        # Chem.AssignStereochemistryFrom3D(mol_ts) <- This is not needed at this point as chirality is not used in atom mapping, we will do it later.
        heavy_maps = [
            a.GetAtomMapNum() for a in input_mol.GetAtoms() if a.GetAtomicNum() > 1
        ]
        matched = None
        if any(a.GetAtomMapNum() for a in input_mol.GetAtoms()):
            # Complete maps fix every atom; partial ones (e.g. only the reacting atoms)
            # pin a substructure search.
            if heavy_maps and all(heavy_maps):
                match = self.match_by_atom_maps
            else:
                match = self.match_by_substructure
            try:
                matched = match(input_mol, mol_ts, reacting_atoms)
            except ValueError as error:
                logger.warning(
                    "The atom maps of the SMILES do not fit the xyz file (%s); matching "
                    "the atoms by maximum common substructure instead.",
                    error,
                )
        if matched is None:
            matched = self.match_AtomMapNum(input_mol, mol_ts)
            missing = self.bonds_missing_from_geometry(*matched, reacting_atoms)
            if missing:
                logger.warning(
                    "The atoms of the SMILES were matched with bonds that the TS geometry "
                    "does not have (atom pairs %s, 0-based); check the SMILES or give "
                    "every heavy atom an atom map number (its atom in the xyz file).",
                    missing,
                )
        [input_mol, mol_ts] = matched

        new_mol = self.set_coords(input_mol, mol_ts, reacting_atoms)

        return new_mol


def _plain_graph(mol: Chem.Mol) -> Chem.RWMol:
    """Copy with single bonds only and without charges, radicals, isotopes or maps."""
    plain = Chem.RWMol(mol)
    for atom in plain.GetAtoms():
        atom.SetFormalCharge(0)
        atom.SetNumRadicalElectrons(0)
        atom.SetIsotope(0)
        atom.SetAtomMapNum(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
    for bond in plain.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    return plain


def _connectivity_with_reacting_bonds(
    mol: Chem.Mol, reacting_atoms: List[int]
) -> Iterator[Chem.RWMol]:
    """
    The connectivity of mol as a plain graph, first as it is, then with bonds between
    the reacting atoms closer than 1.7 x their covalent radii, then between all of them.
    """
    table = Chem.GetPeriodicTable()
    positions = mol.GetConformer().GetPositions()
    pairs = [
        (i, j)
        for i, j in itertools.combinations(sorted(set(reacting_atoms)), 2)
        if mol.GetBondBetweenAtoms(i, j) is None
    ]
    close = [
        (i, j)
        for i, j in pairs
        if np.linalg.norm(positions[i] - positions[j])
        < 1.7
        * sum(table.GetRcovalent(mol.GetAtomWithIdx(k).GetAtomicNum()) for k in (i, j))
    ]
    steps = [[]]
    for added in (close, pairs):
        if added != steps[-1]:
            steps.append(added)
    for added in steps:
        target = _plain_graph(mol)
        for i, j in added:
            target.AddBond(i, j, Chem.BondType.SINGLE)
        yield target


def _fragments_in_search_order(query: Chem.Mol) -> List[tuple]:
    """
    The connected fragments of the query as (atom indices, fragment): fragments with
    pinned (isotope-labelled) atoms first, then larger before smaller, each with its
    atoms in breadth-first order from the pinned atoms. A mismatch at the reaction
    centre is then found before symmetric parts are tried in all their permutations.
    """
    fragments = []
    for atoms in Chem.GetMolFrags(query):
        pins = [i for i in atoms if query.GetAtomWithIdx(i).GetIsotope()]
        order, seen = [], set(pins or atoms[:1])
        queue = list(pins or atoms[:1])
        while queue:
            index = queue.pop(0)
            order.append(index)
            for neighbor in query.GetAtomWithIdx(index).GetNeighbors():
                if neighbor.GetIdx() not in seen:
                    seen.add(neighbor.GetIdx())
                    queue.append(neighbor.GetIdx())
        rest = [i for i in range(query.GetNumAtoms()) if i not in seen]
        fragment = Chem.RWMol(Chem.RenumberAtoms(query, order + rest))
        for index in range(fragment.GetNumAtoms() - 1, len(order) - 1, -1):
            fragment.RemoveAtom(index)
        fragment.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(fragment)
        fragments.append((not pins, -len(order), order, fragment))
    fragments.sort(key=lambda item: item[:2])
    return [(order, fragment) for _, _, order, fragment in fragments]


def _match_fragments(fragments: List[tuple], target: Chem.RWMol) -> Union[dict, None]:
    """
    Query atom index -> target atom index, matching the fragments one after the other;
    target atoms used by earlier fragments become dummy atoms, which no atom matches.
    None if a fragment has no match.
    """
    assignment, used = {}, set()
    for order, fragment in fragments:
        free = Chem.RWMol(target)
        for index in used:
            free.GetAtomWithIdx(index).SetAtomicNum(0)
        free.UpdatePropertyCache(strict=False)
        Chem.FastFindRings(free)
        match = free.GetSubstructMatch(fragment, useChirality=False)
        if not match:
            return None
        assignment.update(zip(order, match))
        used.update(match)
    return assignment
