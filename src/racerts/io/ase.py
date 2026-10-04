"""Conversion between RDKit conformers and ASE Atoms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

from rdkit import Chem
from rdkit.Geometry import Point3D

from racerts.system.spec import infer_charge_and_multiplicity
from racerts.utils.optional import require

if TYPE_CHECKING:
    from ase import Atoms


def rdkit_conformer_to_ase_atoms(
    mol: Chem.Mol,
    conf_id: int,
    multiplicity: Optional[int] = None,
    charge: Optional[int] = None,
) -> Atoms:
    """
    Convert one conformer of an RDKit Mol to an ASE Atoms object.

    Charge and multiplicity (from infer_charge_and_multiplicity when None) are stored
    in atoms.info as charge, spin (the multiplicity, as read by UMA models) and
    multiplicity, and as initial charges and magnetic moments (read e.g. by tblite).
    """
    ase = require("ase", "ase")
    if multiplicity is None or charge is None:
        state = infer_charge_and_multiplicity(mol, charge, multiplicity)
        charge, multiplicity = state["charge"], state["multiplicity"]

    conf = mol.GetConformer(conf_id)
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    # Codes such as tblite only read the totals; put what the formal charges do not
    # explain, and all unpaired electrons, on the first atom.
    charges = [float(atom.GetFormalCharge()) for atom in mol.GetAtoms()]
    charges[0] += charge - sum(charges)
    magmoms = [0.0] * mol.GetNumAtoms()
    magmoms[0] = float(multiplicity - 1)
    atoms = ase.Atoms(
        symbols=symbols,
        positions=conf.GetPositions(),
        charges=charges,
        magmoms=magmoms,
    )
    atoms.info.update(
        {
            "charge": int(charge),
            "spin": int(multiplicity),  # for uma models
            "multiplicity": int(multiplicity),
        }
    )
    return atoms


def is_calculator_factory(calculator) -> bool:
    """Whether calculator is a class or a callable that returns an ASE calculator,
    not a calculator itself."""
    return isinstance(calculator, type) or (
        callable(calculator) and not hasattr(calculator, "get_property")
    )


def check_calculator(calculator) -> bool:
    """
    Whether calculator is a factory (see is_calculator_factory); raises ValueError if
    it is neither a factory nor an ASE calculator.
    """
    factory = is_calculator_factory(calculator)
    if not factory and not hasattr(calculator, "get_property"):
        raise ValueError(
            "`calculator` must be an ASE calculator instance or a callable "
            "returning one."
        )
    return factory


def set_positions(conf: Chem.Conformer, positions) -> None:
    """Write positions (one row of x, y, z per atom, in Angstrom) into a conformer."""
    for idx, xyz in enumerate(positions):
        conf.SetAtomPosition(idx, Point3D(float(xyz[0]), float(xyz[1]), float(xyz[2])))


def write_ase_positions_to_rdkit(atoms: Atoms, mol: Chem.Mol, conf_id: int) -> None:
    set_positions(mol.GetConformer(conf_id), atoms.get_positions())
