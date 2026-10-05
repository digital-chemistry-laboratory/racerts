"""Sites on hydrogens, and several groups replaced at once."""

from typing import Mapping

import numpy as np
from rdkit import Chem

from .apply import apply_swap
from .model import Swap, SwapError
from .selection import _site


def label_hydrogen(mol: Chem.Mol, anchor: int, label: int) -> Chem.Mol:
    """
    A copy of mol whose only hydrogen on anchor carries the map number label (a site
    for Swap(site=label)). An anchor with several hydrogens raises: prochiral sites are
    chosen by the caller.
    """
    if isinstance(label, bool) or not isinstance(label, int):
        raise TypeError("Site labels are positive integers.")
    if label <= 0:
        raise SwapError("Site labels are positive integers.")
    if isinstance(anchor, bool) or not 0 <= anchor < mol.GetNumAtoms():
        raise IndexError(f"No atom {anchor} (the molecule has {mol.GetNumAtoms()}).")
    if any(atom.GetAtomMapNum() == label for atom in mol.GetAtoms()):
        raise SwapError(f"Map number {label} is already used.")
    hydrogens = [
        a for a in mol.GetAtomWithIdx(anchor).GetNeighbors() if a.GetAtomicNum() == 1
    ]
    if len(hydrogens) != 1:
        raise SwapError(
            f"Atom {anchor} has {len(hydrogens)} explicit hydrogens, not exactly one."
        )
    if hydrogens[0].GetAtomMapNum():
        raise SwapError(f"The hydrogen of atom {anchor} already has an atom map.")
    result = Chem.Mol(mol)
    result.GetAtomWithIdx(hydrogens[0].GetIdx()).SetAtomMapNum(label)
    return result


def substitute_groups(
    mol: Chem.Mol, substitutions: Mapping[int, str], random_seed: int = 0xF00D
) -> Chem.Mol:
    """
    Each labelled terminal H or dummy (map number -> "[*]R") becomes the group R,
    grafted rigidly on every conformer; "[H]" caps a site with a hydrogen at the sum
    of the covalent radii along its bond. Kept atoms keep their indices and
    coordinates; the group's first atom takes the label. The properties of the
    molecule and its conformers are cleared (results do not carry over), except its
    charge and multiplicity as the swaps settle them; without substitutions, an
    unchanged copy.
    """
    result = Chem.Mol(mol)
    if not substitutions:
        return result
    for label, smiles in substitutions.items():
        if isinstance(label, bool) or not isinstance(label, int) or label <= 0:
            raise SwapError("Site labels are positive integers.")
        if smiles == "[H]":
            result = _cap(result, label)
            continue
        result = apply_swap(result, Swap(smiles, site=label), seed=random_seed).mol
    state = {
        key: result.GetIntProp(key)
        for key in ("charge", "multiplicity")
        if result.HasProp(key)
    }
    for carrier in [result, *result.GetConformers()]:
        for key in list(
            carrier.GetPropNames(includePrivate=True, includeComputed=False)
        ):
            carrier.ClearProp(key)
    for key, value in state.items():
        result.SetIntProp(key, value)
    return result


def _cap(mol: Chem.Mol, label: int) -> Chem.Mol:
    """The site with map number label as a hydrogen, at the covalent distance."""
    site = _site(mol, label)
    anchor = mol.GetAtomWithIdx(site).GetNeighbors()[0].GetIdx()
    capped = Chem.RWMol(mol)
    hydrogen = Chem.Atom(1)
    hydrogen.SetAtomMapNum(label)
    capped.ReplaceAtom(site, hydrogen)
    table = Chem.GetPeriodicTable()
    length = table.GetRcovalent(1) + table.GetRcovalent(
        mol.GetAtomWithIdx(anchor).GetAtomicNum()
    )
    for conf in capped.GetConformers():
        positions = conf.GetPositions()
        direction = positions[site] - positions[anchor]
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm < 1e-8:
            raise SwapError("The attachment has coincident or invalid coordinates.")
        conf.SetAtomPosition(
            site, (positions[anchor] + length * direction / norm).tolist()
        )
    result = capped.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception as error:
        raise SwapError(f"The capped molecule is invalid: {error}") from None
    return result
