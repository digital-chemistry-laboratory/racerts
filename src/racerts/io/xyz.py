"""Multi-structure xyz files."""

import logging
from typing import Optional

from rdkit import Chem

from racerts.system.spec import infer_charge_and_multiplicity
from racerts.utils.units import EV_TO_KCAL_MOL, HARTREE_TO_KCAL_MOL

logger = logging.getLogger(__name__)


def write_xyz(
    mol: Chem.Mol, file_name: str, use_energy=False, comment: Optional[str] = None
) -> None:
    """
    Write all conformers of mol to a multi-structure xyz file.

    By default, the comment lines are in extended XYZ format (e.g. for ase.io.read)
    with the charge, the spin multiplicity (as "spin" and "multiplicity"), the
    energy in eV as "racerts_energy" and the method that produced it as
    "energy_method". The key is not "energy", which ASE would read as the potential
    energy of the structure, although it is usually a force-field energy.

    Args:
        mol (Chem.Mol): The molecule with its conformers (energies in kcal/mol as the
            conformer property "energy").
        file_name (str): Output path.
        use_energy (bool): Instead, write only the energy in Hartree, as in CREST
            ensembles; conformers without an energy are left out.
        comment (str): Instead, write this comment line (use_energy comes first).

    Conformers without an energy are reported if others have one (not for an
    ensemble without any energies, e.g. embedded only).
    """
    info = infer_charge_and_multiplicity(mol)
    extxyz = [
        f"charge={info['charge']}",
        f"spin={info['multiplicity']}",
        f"multiplicity={info['multiplicity']}",
    ]
    if mol.HasProp("energy_method"):
        method = mol.GetProp("energy_method")
        if any(c.isspace() for c in method):  # extended XYZ: quoted
            method = '"' + method.replace('"', "'") + '"'
        extxyz.append(f"energy_method={method}")
    extxyz.append('pbc="F F F"')

    missing_energy = []
    any_energy = any(conf.HasProp("energy") for conf in mol.GetConformers())
    with open(file_name, "w") as f:
        for conf in mol.GetConformers():
            energy = conf.GetDoubleProp("energy") if conf.HasProp("energy") else None
            if energy is None and (use_energy or comment is None):
                missing_energy.append(conf.GetId())
                if use_energy:
                    continue
            mol_block = Chem.rdmolfiles.MolToXYZBlock(mol, confId=conf.GetId()).strip()
            lines = mol_block.split("\n")

            if use_energy:
                lines[1] = f"{energy / HARTREE_TO_KCAL_MOL:.6f}"
            elif comment is not None:
                lines[1] = comment
            else:
                fields = ["Properties=species:S:1:pos:R:3"]
                if energy is not None:
                    fields.append(f"racerts_energy={energy / EV_TO_KCAL_MOL:.8f}")
                lines[1] = " ".join(fields + extxyz)

            f.write("\n".join(lines) + "\n")

    if missing_energy and (any_energy or use_energy):
        logger.warning(
            "Conformers %s have no energy%s.",
            missing_energy,
            " and are left out" if use_energy else "",
        )
