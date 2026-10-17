"""ORCA's GOAT conformer search as an ensemble of racerts."""

from pathlib import Path
from typing import Mapping, Optional

from attrs import define, field
from attrs.validators import gt, instance_of, optional
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

BASENAME = "goat"  # of the input: the ensemble is goat.finalensemble.xyz
ENSEMBLE = f"{BASENAME}.finalensemble.xyz"


def read_goat(path, mol: Chem.Mol, method: str = "GOAT XTB") -> ConformerEnsemble:
    """
    The final ensemble of a GOAT run as conformers of mol (atoms in the same order):
    path is its <basename>.finalensemble.xyz or the folder of a run of Goat; the
    energies are ORCA's (hartree, stored in kcal/mol) under the name method.
    """
    path = Path(path)
    return ensemble_from_xyz(
        mol, path / ENSEMBLE if path.is_dir() else path, method, "goat"
    )


@define(frozen=True)
class Goat:
    """
    A GOAT run of ORCA (6.x) from one conformer of a molecule; run returns its final
    ensemble.

    Attributes:
        executable: ORCA, by its full path where it runs in parallel (ORCA's rule).
        method: The method keywords of the search, e.g. "XTB" (GFN2-xTB through
            ORCA's xtb) or "XTB1".
        variant: "GOAT" or one of ORCA's variants ("GOAT-ENTROPY", "GOAT-EXPLORE",
            ...).
        solvent: An ALPB solvent; None in the gas phase.
        nprocs, maxcore: ORCA's %pal nprocs and %maxcore (MB).
        workers: GOAT's NWORKERS (a multiple of 4); None for ORCA's choice.
        min_global_iterations, max_global_iterations, max_iterations: GOAT's
            MINGLOBALITER, MAXGLOBALITER and MAXITER (its cost); None for ORCA's.
        options: Further lines of the %goat block, e.g. {"MAXEN": "6.0"}.

    The charge and the multiplicity come from the molecule. What a task holds comes
    in as racerts.restraints.export's "orca" block (frozen atoms as Cartesian
    constraints, windows as distance constraints at their centres). With nprocs > 1
    under Slurm, ORCA's MPI may need OMPI_MCA_rmaps_base_oversubscribe=1 in env.
    """

    executable: str = "orca"
    method: str = field(default="XTB", validator=instance_of(str))
    variant: str = field(default="GOAT", validator=instance_of(str))
    solvent: Optional[str] = field(default=None, validator=optional(instance_of(str)))
    nprocs: int = field(default=1, validator=[instance_of(int), gt(0)])
    maxcore: int = field(default=500, validator=[instance_of(int), gt(0)])
    workers: Optional[int] = field(default=None, validator=optional(gt(0)))
    min_global_iterations: Optional[int] = field(
        default=None, validator=optional(gt(0))
    )
    max_global_iterations: Optional[int] = field(
        default=None, validator=optional(gt(0))
    )
    max_iterations: Optional[int] = field(default=None, validator=optional(gt(0)))
    options: Mapping[str, str] = field(factory=dict, converter=dict)

    @property
    def label(self) -> str:
        """The name of the energies: "GOAT XTB", "GOAT XTB/ALPB(water)"."""
        return f"GOAT {self.method}" + (
            f"/ALPB({self.solvent})" if self.solvent else ""
        )

    def input_text(
        self,
        start: str,
        charge: int,
        multiplicity: int,
        constraints: Optional[str] = None,
    ) -> str:
        """ORCA's input for the run; constraints: a %geom block."""
        keywords = f"! {self.variant} {self.method}"
        if self.solvent:
            keywords += f" ALPB({self.solvent})"
        lines = [keywords, f"%pal nprocs {self.nprocs} end", f"%maxcore {self.maxcore}"]
        goat = dict(self.options)
        for key, value in (
            ("NWORKERS", self.workers),
            ("MINGLOBALITER", self.min_global_iterations),
            ("MAXGLOBALITER", self.max_global_iterations),
            ("MAXITER", self.max_iterations),
        ):
            if value is not None:
                goat[key] = str(value)
        if goat:
            lines += ["%goat", *(f"  {k} {v}" for k, v in goat.items()), "end"]
        if constraints:
            lines += constraints.rstrip("\n").splitlines()
        lines.append(f"* xyzfile {charge} {multiplicity} {start}")
        return "\n".join(lines) + "\n"

    def run(
        self, mol: Chem.Mol, task=None, conf_id: int = -1, workdir=None, env=None
    ) -> ConformerEnsemble:
        """
        GOAT from conformer conf_id of mol, with what task holds held; workdir: the
        folder of the run, new (default: a temporary one, removed afterwards); env:
        further environment variables.
        """
        spin = infer_charge_and_multiplicity(mol)
        with search_folder(workdir, "goat") as folder:
            start = write_start(mol, folder, conf_id)
            constraints = task_constraints(mol, task, "orca", folder)
            text = self.input_text(
                start, spin["charge"], spin["multiplicity"], constraints
            )
            (folder / f"{BASENAME}.inp").write_text(text)
            run_program([self.executable, f"{BASENAME}.inp"], folder, env)
            return read_goat(folder / ENSEMBLE, mol, self.label)
