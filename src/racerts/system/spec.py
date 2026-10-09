"""Charge and spin multiplicity of the system, stored as properties of the Mol."""

import logging
from typing import Dict, Optional

from rdkit import Chem

logger = logging.getLogger(__name__)


def infer_charge_and_multiplicity(
    mol: Chem.Mol, charge: Optional[int] = None, multiplicity: Optional[int] = None
) -> Dict[str, int]:
    """
    Charge and spin multiplicity to use for mol.

    Given values come first, then the "charge" and "multiplicity" properties of mol
    (set by generate_conformers). Otherwise, the charge is the sum of the formal
    charges and the multiplicity the lowest one for the number of electrons (1 or 2).
    Radical electrons of the graph are not used: in a TS graph, they mostly stand for
    bonds that are forming or breaking.
    """
    if charge is None:
        charge = (
            mol.GetIntProp("charge")
            if mol.HasProp("charge")
            else Chem.GetFormalCharge(mol)
        )
    if multiplicity is None:
        multiplicity = (
            mol.GetIntProp("multiplicity")
            if mol.HasProp("multiplicity")
            else 1 + count_electrons(mol, charge) % 2
        )
    return {"charge": int(charge), "multiplicity": int(multiplicity)}


def count_electrons(mol: Chem.Mol, charge: int) -> int:
    """Number of electrons of mol (all atoms explicit) at the given charge."""
    return sum(atom.GetAtomicNum() for atom in mol.GetAtoms()) - charge


def set_charge_and_multiplicity(
    mol: Chem.Mol, charge: Optional[int] = None, multiplicity: Optional[int] = None
) -> None:
    """
    Settle charge and multiplicity of mol (see infer_charge_and_multiplicity) and store
    them as its properties "charge" and "multiplicity", for later steps (ASE,
    write_xyz), also for graphs without formal charges (connectivity only).

    A multiplicity that does not fit the number of electrons, and an odd number of
    electrons without a given multiplicity (often a forgotten charge), are logged as
    warnings.
    """
    state = infer_charge_and_multiplicity(mol, charge, multiplicity)
    electrons = count_electrons(mol, state["charge"])
    if (electrons + state["multiplicity"]) % 2 == 0:
        logger.warning(
            "Multiplicity %d does not fit the %d electrons of the system at charge %d.",
            state["multiplicity"],
            electrons,
            state["charge"],
        )
    elif multiplicity is None and state["multiplicity"] == 2:
        logger.warning(
            "The system has an odd number of electrons (%d at charge %d), so the "
            "multiplicity is 2. Check the charge, or pass multiplicity=2 for a "
            "radical.",
            electrons,
            state["charge"],
        )
    mol.SetIntProp("charge", state["charge"])
    mol.SetIntProp("multiplicity", state["multiplicity"])
