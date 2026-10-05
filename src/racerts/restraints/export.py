"""
Restraints for other programs: xtb, CREST, ORCA and JSON. The exit of the "constraints
only" use: build the restraints of a system, then sample with another program.
"""

import json
import logging
from typing import List, Optional

from rdkit import Chem

from racerts.pipeline import ConformerEnsemble
from racerts.restraints.model import DEFAULT_FORCE_CONSTANT
from racerts.utils.units import HARTREE_TO_KCAL_MOL

logger = logging.getLogger(__name__)

FORMATS = ("xtb", "crest", "orca", "json")
BOHR = 0.529177210903  # A
XTB_DEFAULT_FORCE_CONSTANT = 0.5  # Eh/bohr^2, xtb's default for $constrain
DEFAULT_PATHS = {
    "xtb": "restraints.xcontrol",
    "crest": "restraints.xcontrol",
    "orca": "restraints.inp",
    "json": "restraints.json",
}


def xtb_force_constant(k: float) -> float:
    """
    racerts' force constant k (kcal/(mol A^2), E = 1/2 k (d - bound)^2) in xtb's units:
    xtb's harmonic constraint is E = fc (d - d0)^2 with fc in Eh/bohr^2 (checked with
    xtb 6.6.1).
    """
    return k / 2 / HARTREE_TO_KCAL_MOL * BOHR**2


def _ranges(atoms, base: int = 1) -> str:
    """Atom indices as ranges, e.g. "1-4,7" (1-based by default)."""
    atoms = sorted(int(i) + base for i in atoms)
    parts, start = [], None
    for k, i in enumerate(atoms):
        if start is None:
            start = i
        if k + 1 == len(atoms) or atoms[k + 1] != i + 1:
            parts.append(f"{start}" if start == i else f"{start}-{i}")
            start = None
    return ",".join(parts)


def _windows(ctx) -> List:
    """The distance windows of the refinement that other programs can take."""
    windows = []
    for r in ctx.restraints.for_stage("refine"):
        if not hasattr(r, "pair"):
            continue
        if r.lower <= 0:
            logger.warning(
                "Restraint %s has no lower bound (e.g. a containment): other programs "
                "take no one-sided windows, so it is left out.",
                r.label,
            )
            continue
        windows.append(r)
    return windows


def _held(ctx) -> List[int]:
    if ctx.frozen.soft:
        logger.warning(
            "The soft atoms %s are exported as frozen atoms: other programs take no "
            "position restraints with a tolerance.",
            list(ctx.frozen.soft),
        )
    return sorted({*ctx.frozen.hard, *ctx.frozen.soft})


def _force_constant(windows) -> float:
    """The one force constant (kcal/(mol A^2)) of the windows: xtb takes one."""
    constants = sorted({r.force_constant for r in windows})
    if len(constants) > 1:
        logger.warning(
            "The restraints have several force constants %s, but xtb and CREST take one "
            "force constant for all: the largest is used.",
            constants,
        )
    return constants[-1] if constants else DEFAULT_FORCE_CONSTANT


def export_restraints(ctx, fmt: str, reference: Optional[str] = None) -> str:
    """
    The frozen atoms and the restraints of a context for another program:

    - "xtb": a detailed input ($fix for the frozen atoms, $constrain for the windows),
      for xtb --input;
    - "crest": the same for crest --cinp, with the frozen atoms under $constrain (CREST
      advises against $fix), reference= the reference geometry (a file written by the
      caller) and $metadyn on the other atoms;
    - "orca": a %geom block with the frozen atoms and the windows as constraints;
    - "json": the frozen atoms and the restraints (RestraintSet.to_json).

    None of these programs has flat-bottom distances: every window becomes a harmonic
    constraint at its centre (ORCA: an exact one), with the restraint's force constant
    (xtb takes one for all). Held values (the frozen atoms) export without loss.
    Windows without a lower bound (containment) are left out. Atoms are 1-based for
    xtb and CREST, 0-based for ORCA and JSON.
    """
    if fmt not in FORMATS:
        raise ValueError(f"Unknown export format {fmt!r}; use one of {FORMATS}.")
    if fmt == "json":
        return json.dumps(
            {
                "frozen": {
                    "hard": sorted(ctx.frozen.hard),
                    "core": sorted(ctx.frozen.core),
                    "soft": sorted(ctx.frozen.soft),
                },
                "restraints": json.loads(ctx.restraints.to_json()),
                "units": {"distance": "A", "force_constant": "kcal/(mol A^2)"},
                "atoms": "0-based",
            },
            indent=1,
        )
    held, windows = _held(ctx), _windows(ctx)
    if fmt == "orca":
        lines = ["%geom", "  Constraints"]
        lines += [f"    {{C {i} C}}" for i in held]
        lines += [
            f"    {{B {r.first} {r.second} {(r.lower + r.upper) / 2:.3f} C}}"
            for r in windows
        ]
        return "\n".join(lines + ["  end", "end"]) + "\n"
    distances = [
        f"   distance: {r.first + 1}, {r.second + 1}, {(r.lower + r.upper) / 2:.3f}"
        for r in windows
    ]
    fc = xtb_force_constant(_force_constant(windows))
    if fmt == "xtb":
        lines = []
        if held:
            lines += ["$fix", f"   atoms: {_ranges(held)}"]
        if distances:
            lines += ["$constrain", f"   force constant={fc:.6f}", *distances]
        return "\n".join(lines + ["$end"]) + "\n"
    if held and fc < XTB_DEFAULT_FORCE_CONSTANT:
        logger.warning(
            "CREST takes one force constant for the frozen atoms and the windows: "
            "%g Eh/bohr^2 (xtb's default) holds the frozen atoms, and the windows as "
            "stiffly.",
            XTB_DEFAULT_FORCE_CONSTANT,
        )
        fc = XTB_DEFAULT_FORCE_CONSTANT
    lines = ["$constrain", f"   force constant={fc:.6f}"]
    if held:
        lines.append(f"   atoms: {_ranges(held)}")
        if reference:
            lines.append(f"   reference={reference}")
    lines += distances
    rest = sorted(set(range(ctx.mol.GetNumAtoms())) - set(held))
    if held and rest:
        lines += ["$metadyn", f"   atoms: {_ranges(rest)}"]
    return "\n".join(lines + ["$end"]) + "\n"


class ExportRestraints:
    """
    A stage that writes the restraints of the context for another program (see
    export_restraints) and returns an empty ensemble: a pipeline of this stage alone
    stops before any embedding. For CREST it writes the reference geometry next to
    path (path + ".ref.xyz").
    """

    name = "export_restraints"

    def __init__(self, fmt: str, path: str):
        if fmt not in FORMATS:
            raise ValueError(f"Unknown export format {fmt!r}; use one of {FORMATS}.")
        self.fmt = fmt
        self.path = path

    def run(self, ctx, ensemble=None) -> ConformerEnsemble:
        reference = None
        if self.fmt == "crest" and ctx.reference is not None:
            reference = self.path + ".ref.xyz"
            Chem.MolToXYZFile(ctx.reference, reference)
        with open(self.path, "w") as handle:
            handle.write(export_restraints(ctx, self.fmt, reference=reference))
        logger.info("Restraints for %s written to %s", self.fmt, self.path)
        empty = Chem.Mol(ctx.mol)
        empty.RemoveAllConformers()
        return ConformerEnsemble(empty)
