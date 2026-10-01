"""Molecular graphs from a geometry and an explicit-hydrogen SMILES."""

from typing import Any, Optional, Sequence

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdDetermineBonds


def mol_from_explicit_h_smiles(smiles: str) -> Chem.Mol:
    """
    Parse a SMILES keeping its explicit hydrogens, with no implicit hydrogens allowed
    on any atom: a missing neighbour then stays a radical.

    Raises:
        ValueError: If the SMILES cannot be parsed.
    """
    params = Chem.SmilesParserParams()
    params.removeHs = False
    mol = Chem.MolFromSmiles(smiles, params)
    if mol is None:
        raise ValueError(f"Invalid SMILES: {smiles!r}")
    for atom in mol.GetAtoms():
        atom.SetNoImplicit(True)
    mol.UpdatePropertyCache(strict=False)
    return mol


def bondless_mol(symbols: Sequence[str], positions: Any) -> Chem.Mol:
    """A molecule of the atoms (elements and one conformer), without bonds."""
    positions = np.asarray(positions, dtype=float)
    if positions.shape != (len(symbols), 3):
        raise ValueError(
            f"{len(symbols)} atoms need positions of shape ({len(symbols)}, 3), not "
            f"{positions.shape}."
        )
    mol = Chem.RWMol()
    for symbol in symbols:
        atom = Chem.Atom(symbol)
        atom.SetNoImplicit(True)
        mol.AddAtom(atom)
    conf = Chem.Conformer(len(symbols))
    for i, position in enumerate(positions):
        conf.SetAtomPosition(i, position.tolist())
    mol.AddConformer(conf, assignId=True)
    mol = mol.GetMol()
    mol.UpdatePropertyCache(strict=False)
    return mol


def mol_from_geometry(
    structure: Any,
    smiles: str,
    charge: Optional[int] = None,
    multiplicity: Optional[int] = None,
) -> Chem.Mol:
    """
    The molecule of an explicit-hydrogen SMILES on a geometry:
    the connectivity is perceived from the distances, and the SMILES' bond orders,
    charges and radicals are assigned onto it. The atoms keep the order of the
    geometry.

    Args:
        structure: ASE Atoms, or a molecule with a conformer (e.g. from an xyz file).
        smiles: The molecule, with explicit hydrogens.
        charge, multiplicity: If given, they must agree with the SMILES (the sum of the
            formal charges; 1 + the radical electrons).

    Raises:
        ValueError: If the geometry does not have the connectivity of the SMILES, or
            charge or multiplicity disagree.
    """
    template = mol_from_explicit_h_smiles(smiles)
    expected_charge = Chem.GetFormalCharge(template)
    expected_multiplicity = 1 + sum(
        a.GetNumRadicalElectrons() for a in template.GetAtoms()
    )
    if charge is not None and charge != expected_charge:
        raise ValueError(
            f"Charge mismatch: {charge} given, the SMILES has {expected_charge}."
        )
    if multiplicity is not None and multiplicity != expected_multiplicity:
        raise ValueError(
            f"Multiplicity mismatch: {multiplicity} given, the SMILES has "
            f"{expected_multiplicity} (1 + radical electrons)."
        )

    if isinstance(structure, Chem.Mol):
        symbols = [atom.GetSymbol() for atom in structure.GetAtoms()]
        positions = structure.GetConformer().GetPositions()
    else:
        symbols = structure.get_chemical_symbols()
        positions = structure.get_positions()
    geometry = bondless_mol(symbols, positions)
    rdDetermineBonds.DetermineConnectivity(geometry, useHueckel=False)
    geometry.UpdatePropertyCache(strict=False)

    # Perceived connectivity leaves a radical atom merely under-coordinated, which
    # AssignBondOrdersFromTemplate cannot match: copy the radicals of the SMILES onto
    # the geometry first, by a match of the plain connectivity.
    query = Chem.RWMol(template)
    for bond in query.GetBonds():
        bond.SetBondType(Chem.BondType.SINGLE)
        bond.SetIsAromatic(False)
    for atom in query.GetAtoms():
        atom.SetNumRadicalElectrons(0)
        atom.SetFormalCharge(0)
        atom.SetIsAromatic(False)
        atom.SetNoImplicit(True)
    query = query.GetMol()
    query.UpdatePropertyCache(strict=False)
    if geometry.GetNumAtoms() != template.GetNumAtoms():
        raise ValueError(
            f"The geometry has {geometry.GetNumAtoms()} atoms, the SMILES "
            f"{template.GetNumAtoms()} (hydrogens must be explicit)."
        )
    match = geometry.GetSubstructMatch(query)
    if not match:
        raise ValueError(f"The geometry does not have the connectivity of {smiles!r}.")
    for template_index, geometry_index in enumerate(match):
        geometry.GetAtomWithIdx(geometry_index).SetNumRadicalElectrons(
            template.GetAtomWithIdx(template_index).GetNumRadicalElectrons()
        )
    geometry.UpdatePropertyCache(strict=False)
    try:
        mol = AllChem.AssignBondOrdersFromTemplate(template, geometry)
    except ValueError as error:
        raise ValueError(
            f"The geometry does not have the connectivity of {smiles!r}: {error}"
        ) from error
    Chem.AssignStereochemistryFrom3D(mol)
    return mol


def radical_multiplicity(mol: Chem.Mol) -> int:
    """The multiplicity of the graph: 1 + the radical electrons of mol."""
    return 1 + sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
