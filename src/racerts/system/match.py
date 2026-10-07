"""
Matching the atoms of a SMILES template to a geometry, by atom map numbers or by a
substructure search, and putting the template's bonds and charges on the geometry
(apply_template). MolGetterSMILES uses these.
"""

import itertools
import logging
from typing import Dict, Iterator, List, Union

import numpy as np
from rdkit import Chem

from racerts.errors import MoleculeError

logger = logging.getLogger(__name__)


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
            raise MoleculeError(
                f"atom map number {number} is not an atom of the xyz file"
            )
        other = mol.GetAtomWithIdx(number - 1)
        if other.GetAtomicNum() != atom.GetAtomicNum():
            raise MoleculeError(
                f"atom map number {number} is {atom.GetSymbol()} in the SMILES but "
                f"{other.GetSymbol()} in the xyz file"
            )
        target[atom.GetIdx()] = number - 1
    if len(set(target.values())) != len(target):
        raise MoleculeError("atom map numbers are repeated")
    return target


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


def match_by_atom_maps(
    ref_mol: Chem.Mol, mol: Chem.Mol, reacting_atoms: List[int]
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
    target = mapped_atoms(ref_mol, mol)  # template index -> xyz index

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
            raise MoleculeError(f"atom {atom.GetIdx()} of the SMILES has no map number")
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
        raise MoleculeError("the numbers of hydrogens differ")

    for atom in ref_mol.GetAtoms():
        atom.SetAtomMapNum(target[atom.GetIdx()] + 1)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx() + 1)
    missing = bonds_missing_from_geometry(ref_mol, mol, reacting_atoms)
    if missing:
        i, j = missing[0]
        raise MoleculeError(
            f"atoms {i + 1} and {j + 1} are bonded in the SMILES but not in the xyz "
            "file"
        )
    return [ref_mol, mol]


def match_by_substructure(
    ref_mol: Chem.Mol, mol: Chem.Mol, reacting_atoms: List[int]
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
    pinned = mapped_atoms(ref_mol, mol)
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
            apply_template(Chem.Mol(template), Chem.Mol(geometry), reacting_atoms)
        except ValueError as error:  # includes RDKit's sanitization errors
            logger.debug("Substructure match rejected: %s", error)
            continue
        return [template, geometry]
    raise MoleculeError(
        "the SMILES does not match the connectivity of the xyz file with the "
        "mapped atoms in place"
    )


def apply_template(
    template: Chem.Mol, geometry: Chem.Mol, reacting_atoms: List[int]
) -> Chem.Mol:
    """
    The molecule of the geometry with the bonds, bond orders and formal charges of the
    matched template (atoms paired by their map numbers). Bonds between reacting atoms
    are only kept where the perceived connectivity of the geometry has them.

    Raises:
        ValueError: If a template bond has no counterpart in the geometry, or the result
            cannot be sanitized.
    """
    map_to_idx = {atom.GetAtomMapNum(): atom.GetIdx() for atom in geometry.GetAtoms()}
    for atom in template.GetAtoms():
        idx = map_to_idx.get(atom.GetAtomMapNum())
        if idx is not None:
            geometry.GetAtomWithIdx(idx).SetFormalCharge(atom.GetFormalCharge())
    emol = Chem.EditableMol(geometry)
    for bond in geometry.GetBonds():
        emol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
    for bond in template.GetBonds():
        m1 = bond.GetBeginAtom().GetAtomMapNum()
        m2 = bond.GetEndAtom().GetAtomMapNum()
        if m1 not in map_to_idx or m2 not in map_to_idx:
            raise MoleculeError(
                f"Bond {m1}-{m2} of the SMILES template has no counterpart in the "
                "TS geometry. Check that the SMILES matches the xyz file."
            )
        id1, id2 = map_to_idx[m1], map_to_idx[m2]
        if id1 not in reacting_atoms or id2 not in reacting_atoms:
            emol.AddBond(id1, id2, order=bond.GetBondType())
        elif geometry.GetBondBetweenAtoms(id1, id2) is not None:
            emol.AddBond(id1, id2, order=bond.GetBondType())
    new_mol = emol.GetMol()
    Chem.SanitizeMol(new_mol)
    Chem.SetDoubleBondNeighborDirections(new_mol, new_mol.GetConformer())
    Chem.DetectBondStereochemistry(new_mol)
    for atom in (
        new_mol.GetAtoms()
    ):  # <- We need to remove the atom map numbers to avoid the artifacts
        atom.SetAtomMapNum(0)
    Chem.AssignStereochemistryFrom3D(
        new_mol
    )  # <- Now we use the 3D conformer to set the stereochemistry
    Chem.AssignCIPLabels(new_mol)
    # Chem.AssignStereochemistry(new_mol) <- This is deprecated in favor of Chem.AssignCIPLabels(new_mol) above
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
