"""The system: charge and spin multiplicity (stored as properties of the Mol), fragments."""

import logging
from typing import Dict, List, Optional, Sequence, Tuple

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
    warnings; a multiplicity below 1 raises a ValueError.
    """
    if multiplicity is not None and multiplicity < 1:
        raise ValueError(
            f"The multiplicity must be at least 1 (2S + 1), not {multiplicity}."
        )
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


def split_fragments(
    mol: Chem.Mol, core_atoms: Sequence[int]
) -> Tuple[List[int], List[List[int]]]:
    """
    Split mol into the core and the fragments that move relative to it.

    The core holds the atoms of all fragments with a core atom (e.g. the reacting
    atoms of a TS); every other fragment (e.g. a solvent molecule or counterion) is
    free. Without core atoms, the largest fragment is the core.

    Returns:
        The core atom indices and the free fragments (lists of atom indices).
    """
    core_set = set(core_atoms)
    fragments = [list(fragment) for fragment in Chem.GetMolFrags(mol)]
    free = [f for f in fragments if not core_set.intersection(f)]
    core = [i for f in fragments if core_set.intersection(f) for i in f]
    if not core and free:
        core = max(free, key=len)
        free.remove(core)
    return core, free


def rigid_body_dof(mol: Chem.Mol, core_atoms: Sequence[int], fragments=None) -> int:
    """
    Rigid-body degrees of freedom of the fragments that move relative to the core (see
    split_fragments): 3 translations for a single atom, 6 for a molecule. With
    fragments (racerts.system.roles.Fragment), by their roles: an anchored fragment
    only turns (3, a single atom 0); reactive ones do not count.
    """
    if fragments is not None:
        dof = 0
        for fragment in fragments:
            single = len(fragment.atoms) == 1
            if fragment.role == "anchored":
                dof += 0 if single else 3
            elif fragment.role in ("contained", "free"):
                dof += 3 if single else 6
        return dof
    _, free = split_fragments(mol, core_atoms)
    return sum(3 if len(fragment) == 1 else 6 for fragment in free)
