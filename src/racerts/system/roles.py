"""
Fragment roles: how the molecules of a multi-fragment system (solvent, counterions, a
separate substrate) may move as a whole, and containment restraints.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem

from racerts.restraints.model import DistanceRestraint, RestraintSet

logger = logging.getLogger(__name__)

REACTIVE = "reactive"  # holds core atoms (or, without any, is the largest fragment)
ANCHORED = "anchored"  # a restraint of the refinement ties it to the core side
CONTAINED = "contained"  # only a containment restraint holds it near the core
FREE = "free"
ROLE_OVERRIDES = (ANCHORED, CONTAINED, FREE)  # reactive fragments follow from the task
CONTAIN = "contain"  # the source of containment restraints


@dataclass(frozen=True)
class Fragment:
    """
    A fragment (molecule) of the system and its role. anchors: its atoms in the
    restraints that anchor it (for ANCHORED), or its central atom (for CONTAINED).
    """

    atoms: Tuple[int, ...]
    role: str
    anchors: Tuple[int, ...] = ()


def _frozen_atoms(frozen) -> set:
    """The frozen atoms of a FrozenSet, or a collection of atoms as it is."""
    if frozen is None:
        return set()
    if hasattr(frozen, "hard"):
        return {*frozen.hard, *frozen.core, *frozen.soft}
    return set(frozen)


def _core_fragments(fragments, frozen) -> List[int]:
    """The fragments with frozen atoms, or else the largest one."""
    held = _frozen_atoms(frozen)
    core = [k for k, atoms in enumerate(fragments) if held & set(atoms)]
    if not core and fragments:
        core = [max(range(len(fragments)), key=lambda k: len(fragments[k]))]
    return core


def central_atom(mol: Chem.Mol, atoms: Sequence[int]) -> int:
    """The heavy atom (any atom, if there is none) of atoms that lies closest to their
    centroid in the geometry of mol, or that is the topological center without one."""
    atoms = list(atoms)
    heavy = [i for i in atoms if mol.GetAtomWithIdx(i).GetAtomicNum() > 1] or atoms
    if mol.GetNumConformers():
        positions = mol.GetConformer().GetPositions()
        center = positions[atoms].mean(axis=0)
        return min(heavy, key=lambda i: np.linalg.norm(positions[i] - center))
    distances = Chem.GetDistanceMatrix(mol)
    return min(heavy, key=lambda i: max(distances[i][j] for j in atoms))


def fragment_roles(
    mol: Chem.Mol,
    frozen=None,
    restraints: Iterable = (),
    override: Optional[Mapping[int, str]] = None,
) -> List[Fragment]:
    """
    The role of every fragment of mol, in the order of Chem.GetMolFrags:
    - REACTIVE: it holds frozen atoms of the task (for a TS the reacting atoms); if no
      fragment does, the largest fragment is the reactive one (the frame of reference);
    - ANCHORED: a restraint of the refinement (not the embedding-only hints, not a
      containment restraint) ties it to a reactive fragment or to an anchored one; its
      anchors are its atoms in those restraints;
    - CONTAINED: only a containment restraint (source "contain") holds it; its anchor
      is its central atom;
    - FREE: nothing holds it.

    override: {atom: role} for the fragment of each atom (not for reactive ones).
    """
    fragments = [tuple(f) for f in Chem.GetMolFrags(mol)]
    of_atom = {i: k for k, atoms in enumerate(fragments) for i in atoms}
    pairs, contained = [], set()
    for r in restraints:
        if not hasattr(r, "pair") or r.stage == "embed":
            continue
        if r.source == CONTAIN:
            contained.update(of_atom[i] for i in r.pair)
        else:
            pairs.append(r.pair)
    core = _core_fragments(fragments, frozen)
    roles: Dict[int, Fragment] = {k: Fragment(fragments[k], REACTIVE) for k in core}
    front = set(core)
    while front:  # anchor fragment by fragment, outwards from the core
        found: Dict[int, set] = {}
        for a, b in pairs:
            for inside, outside in ((a, b), (b, a)):
                k = of_atom[outside]
                if of_atom[inside] in front and k not in roles:
                    found.setdefault(k, set()).add(outside)
        for k, anchors in found.items():
            roles[k] = Fragment(fragments[k], ANCHORED, tuple(sorted(anchors)))
        front = set(found)
    for k, atoms in enumerate(fragments):
        if k in roles:
            continue
        if k in contained:
            roles[k] = Fragment(atoms, CONTAINED, (central_atom(mol, atoms),))
        else:
            roles[k] = Fragment(atoms, FREE)
    for atom, role in (override or {}).items():
        if role not in ROLE_OVERRIDES:
            raise ValueError(
                f"Unknown fragment role {role!r} for atom {atom}: use one of "
                f"{ROLE_OVERRIDES}."
            )
        if atom not in of_atom:
            raise ValueError(
                f"Atom {atom} of the role override is not in the molecule."
            )
        k = of_atom[atom]
        if roles[k].role == REACTIVE:
            raise ValueError(
                f"The fragment of atom {atom} is reactive (it holds core atoms); its "
                "role cannot change."
            )
        anchors = roles[k].anchors if role == roles[k].role else ()
        if role == CONTAINED and not anchors:
            anchors = (central_atom(mol, fragments[k]),)
        roles[k] = Fragment(fragments[k], role, anchors)
    return [roles[k] for k in range(len(fragments))]


def describe(fragments: Sequence[Fragment]) -> str:
    """The roles for the log, e.g. "atoms 6-8: anchored at 7"."""
    parts = []
    for f in fragments:
        span = f"{f.atoms[0]}-{f.atoms[-1]}" if len(f.atoms) > 1 else f"{f.atoms[0]}"
        where = f" at {', '.join(map(str, f.anchors))}" if f.anchors else ""
        parts.append(f"atoms {span}: {f.role}{where}")
    return "; ".join(parts)


def containment(
    mol: Chem.Mol,
    frozen,
    radius: float,
    restraints: Iterable = (),
    force_constant: float = 20.0,
) -> RestraintSet:
    """
    Containment restraints: for every fragment that is neither reactive nor anchored by
    restraints, a window [0, radius] (A) between its central atom and the central atom
    of the core (the core atoms of the task, else the reactive fragments), source
    "contain", in embedding and refinement.
    """
    if not radius > 0:
        raise ValueError("The containment radius must be positive.")
    fragments = fragment_roles(mol, frozen, restraints)
    core_atoms = list(getattr(frozen, "core", None) or [])
    if not core_atoms:
        core_atoms = [i for f in fragments if f.role == REACTIVE for i in f.atoms]
    center = central_atom(mol, core_atoms)
    return RestraintSet(
        DistanceRestraint(
            central_atom(mol, f.atoms),
            center,
            0.0,
            radius,
            force_constant=force_constant,
            source=CONTAIN,
        )
        for f in fragments
        if f.role == FREE
    )
