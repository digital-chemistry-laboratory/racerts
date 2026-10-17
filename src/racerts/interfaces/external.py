"""What the interfaces to external conformer searches share: their xyz ensembles read
back as conformers, the atoms a task holds, and a run of the program in its own folder."""

import logging
import os
import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem

from racerts.errors import ExternalSearchError
from racerts.pipeline.ensemble import ConformerEnsemble
from racerts.utils.units import HARTREE_TO_KCAL_MOL

logger = logging.getLogger(__name__)

START = "start.xyz"  # the structure a search starts from, in its folder
LOG = "program.log"  # what the program writes to stdout and stderr
REFERENCE = "reference.xyz"  # the geometry that CREST's constraints refer to


def read_frames(path) -> List[Tuple[List[str], np.ndarray, str]]:
    """The structures of a multi-frame xyz file: (symbols, positions (A), comment)."""
    lines = Path(path).read_text().splitlines()
    frames, i = [], 0
    while i < len(lines):
        if not lines[i].strip():
            i += 1
            continue
        n = int(lines[i].split()[0])
        rows = [line.split() for line in lines[i + 2 : i + 2 + n]]
        if len(rows) != n or any(len(row) < 4 for row in rows):
            raise ValueError(f"{path}: the structure at line {i + 1} is cut short.")
        symbols = [row[0].capitalize() for row in rows]
        positions = np.array([[float(x) for x in row[1:4]] for row in rows])
        frames.append((symbols, positions, lines[i + 1] if i + 1 < len(lines) else ""))
        i += 2 + n
    return frames


def ensemble_from_xyz(
    mol: Chem.Mol, path, method: str, source: str
) -> ConformerEnsemble:
    """
    The structures of an ensemble file as conformers of mol (the same atoms in the
    same order): each with the energy of its comment line (its first number, in
    hartree) in kcal/mol, the energies named method, and the provenance source and
    frame. Coordinates only are checked (as ConformerEnsemble.from_frames); whether a
    structure kept the graph is for racerts.validate.
    """
    path = Path(path)
    if not path.exists():
        log = path.parent / LOG
        tail = log.read_text().splitlines()[-15:] if log.exists() else []
        raise ExternalSearchError(f"{path} was not written.\n" + "\n".join(tail))
    frames = read_frames(path)
    if not frames:
        raise ExternalSearchError(f"{path} holds no structures.")
    symbols = [atom.GetSymbol() for atom in mol.GetAtoms()]
    for index, (found, _, _) in enumerate(frames):
        if found != symbols:
            raise ValueError(
                f"Structure {index} of {path} has other atoms or another atom order "
                "than the molecule."
            )
    ensemble = ConformerEnsemble.from_frames(
        mol, [positions for _, positions, _ in frames]
    )
    for index, (conf_id, (_, _, comment)) in enumerate(zip(ensemble.conf_ids, frames)):
        try:
            energy = float(comment.split()[0]) * HARTREE_TO_KCAL_MOL
        except (IndexError, ValueError):
            raise ValueError(
                f"Structure {index} of {path} has no energy in its comment line "
                f"({comment!r})."
            ) from None
        ensemble.mol.GetConformer(conf_id).SetDoubleProp("energy", energy)
    ensemble.mol.SetProp("energy_method", method)
    ensemble.add_provenance(source=source)
    return ensemble


def task_constraints(mol: Chem.Mol, task, fmt: str, folder: Path) -> Optional[str]:
    """
    What task holds, as an input block for the program (racerts.restraints.export:
    the frozen atoms and the windows of the refinement, fmt "crest" or "orca"); None
    if task is None or holds nothing. For CREST the reference geometry is written to
    folder (REFERENCE).
    """
    if task is None:
        return None
    from racerts.pipeline.context import Context
    from racerts.restraints.export import _windows, export_restraints

    ctx = Context.create(mol, task)
    frozen = ctx.frozen
    if not (frozen.hard or frozen.core or frozen.soft or _windows(ctx)):
        return None
    reference = None
    if fmt == "crest" and ctx.reference is not None:
        Chem.MolToXYZFile(ctx.reference, str(folder / REFERENCE))
        reference = REFERENCE
    return export_restraints(ctx, fmt, reference=reference)


@contextmanager
def search_folder(workdir, program: str) -> Iterator[Path]:
    """
    The folder of one run: workdir, which must not exist yet (CREST leaves files
    there that a later run reads), or a temporary folder removed afterwards.
    """
    if workdir is None:
        folder = Path(tempfile.mkdtemp(prefix=f"racerts-{program}-"))
        try:
            yield folder
        finally:
            shutil.rmtree(folder, ignore_errors=True)
        return
    folder = Path(workdir)
    if folder.exists():
        raise FileExistsError(
            f"{folder} exists: every {program} run needs a new folder."
        )
    folder.mkdir(parents=True)
    yield folder


def write_start(mol: Chem.Mol, folder: Path, conf_id: int = -1) -> str:
    """The conformer conf_id of mol as the start structure (START) in folder."""
    Chem.MolToXYZFile(mol, str(folder / START), confId=conf_id)
    return START


def run_program(
    arguments: Sequence[str], folder: Path, env: Optional[Mapping[str, str]] = None
) -> None:
    """Run a program in folder (stdout and stderr to LOG there); its failure raises
    ExternalSearchError with the end of the log."""
    logger.info("Running %s in %s", " ".join(arguments), folder)
    with open(folder / LOG, "w") as log:
        try:
            done = subprocess.run(
                list(arguments),
                cwd=folder,
                env={**os.environ, **dict(env or {})},
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        except OSError as error:
            raise ExternalSearchError(
                f"{arguments[0]} could not start: {error}"
            ) from error
    if done.returncode != 0:
        tail = (folder / LOG).read_text().splitlines()[-15:]
        raise ExternalSearchError(
            f"{arguments[0]} stopped with exit code {done.returncode}:\n"
            + "\n".join(tail)
        )
