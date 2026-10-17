"""CREST's conformer search (iMTD-GC with xtb levels) as an ensemble of racerts."""

from pathlib import Path
from typing import List, Optional, Tuple

from attrs import define, field
from attrs.validators import gt, in_, instance_of, optional
from rdkit import Chem

from racerts.pipeline.ensemble import ConformerEnsemble
from racerts.system.spec import infer_charge_and_multiplicity

from .external import (
    ensemble_from_xyz,
    run_program,
    search_folder,
    task_constraints,
    write_start,
)

METHODS = ("gfn2", "gfn1", "gfn0", "gfnff", "gfn2//gfnff")
MODES = ("quick", "squick", "mquick")
ENSEMBLE = "crest_conformers.xyz"  # the conformers, each with its energy (hartree)
CONSTRAINTS = "constraints.inp"


def read_crest(path, mol: Chem.Mol, method: str = "CREST gfn2") -> ConformerEnsemble:
    """
    The conformers of a CREST run as conformers of mol (atoms in the same order):
    path is its crest_conformers.xyz or the folder that holds it; the energies are
    CREST's (hartree, stored in kcal/mol) under the name method.
    """
    path = Path(path)
    return ensemble_from_xyz(
        mol, path / ENSEMBLE if path.is_dir() else path, method, "crest"
    )


@define(frozen=True)
class Crest:
    """
    A CREST run (3.x) from one conformer of a molecule; run returns its conformers.

    Attributes:
        executable: The CREST binary (a name on PATH or a path); CREST 3. Give the
            path where an older CREST comes first on PATH.
        method: "gfn2", "gfn1", "gfn0", "gfnff" or "gfn2//gfnff".
        solvent: An ALPB solvent; None in the gas phase.
        threads: For CREST's -T.
        energy_window: CREST's -ewin (kcal/mol); None for its default.
        mode: None for the default search; "quick", "squick" or "mquick".
        extra: Further arguments, passed as they are.

    The charge and the multiplicity come from the molecule (its "charge" and
    "multiplicity" properties, else its formal charges). What a task holds comes in
    by --cinp: racerts.restraints.export's "crest" input (frozen atoms constrained to
    the reference geometry, windows as harmonic distances, $metadyn on the rest).
    That is xtb's documented syntax, which CREST 3.0.1 (release build) reads; the
    3.0.2 module build of Euler (2026-10) crashes on its force constant lines, atom
    ranges and $metadyn block, so runs with a task need another build there.
    """

    executable: str = "crest"
    method: str = field(default="gfn2", validator=in_(METHODS))
    solvent: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    threads: int = field(default=1, validator=[instance_of(int), gt(0)])
    energy_window: Optional[float] = field(default=None, validator=optional(gt(0)))
    mode: Optional[str] = field(default=None, validator=optional(in_(MODES)))
    extra: Tuple[str, ...] = field(default=(), converter=tuple)

    @property
    def label(self) -> str:
        """The name of the energies: "CREST gfn2", "CREST gfn2/ALPB(water)"."""
        return f"CREST {self.method}" + (
            f"/ALPB({self.solvent})" if self.solvent else ""
        )

    def arguments(
        self,
        start: str,
        charge: int,
        multiplicity: int,
        constraints: Optional[str] = None,
    ) -> List[str]:
        """The command line after the executable."""
        args = [
            start,
            f"--{self.method}",
            "--chrg",
            str(charge),
            "--uhf",
            str(multiplicity - 1),
        ]
        if self.solvent:
            args += ["--alpb", self.solvent]
        args += ["-T", str(self.threads)]
        if self.energy_window is not None:
            args += ["-ewin", str(self.energy_window)]
        if self.mode:
            args.append(f"--{self.mode}")
        if constraints:
            args += ["--cinp", constraints]
        return args + list(self.extra)

    def run(
        self, mol: Chem.Mol, task=None, conf_id: int = -1, workdir=None, env=None
    ) -> ConformerEnsemble:
        """
        CREST from conformer conf_id of mol, with what task holds held; workdir: the
        folder of the run, new (default: a temporary one, removed afterwards); env:
        further environment variables.
        """
        spin = infer_charge_and_multiplicity(mol)
        with search_folder(workdir, "crest") as folder:
            start = write_start(mol, folder, conf_id)
            block = task_constraints(mol, task, "crest", folder)
            if block:
                (folder / CONSTRAINTS).write_text(block)
            args = self.arguments(
                start,
                spin["charge"],
                spin["multiplicity"],
                CONSTRAINTS if block else None,
            )
            run_program([self.executable, *args], folder, env)
            return read_crest(folder / ENSEMBLE, mol, self.label)
