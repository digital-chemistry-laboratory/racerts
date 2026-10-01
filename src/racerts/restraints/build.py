"""build_restraints: the restraints of a task from the user and the seed geometry."""

import logging
from typing import Iterable, Optional, Sequence

from rdkit import Chem

from racerts.task import FrozenSet

from . import sources
from .model import (
    DEFAULT_FORCE_CONSTANT,
    DEFAULT_HALF_WIDTH,
    DistanceRestraint,
    RestraintSet,
)

logger = logging.getLogger(__name__)


def build_restraints(
    mol: Chem.Mol,
    frozen: FrozenSet = FrozenSet(),
    user: Iterable[Sequence] = (),
    half_width: float = DEFAULT_HALF_WIDTH,
    force_constant: float = DEFAULT_FORCE_CONSTANT,
    hbonds: bool = False,
    contacts: Iterable[Sequence[int]] = (),
    keep_fragments: bool = False,
    fragment_links: Optional[Iterable[Sequence[int]]] = None,
    link_fragments: bool = False,
    seed: int = 0xF00D,
    hints: bool = False,
    max_hints: int = 8,
) -> RestraintSet:
    """
    The distance restraints for mol, which carries the reference geometry if a source
    needs one.

    Args:
        frozen: The frozen atoms of the task: restraints between two hard atoms are
            left out (their distance is fixed), with a warning for user restraints.
        user: (atom, atom, target) triplets: windows target +/- half_width.
        half_width, force_constant: Of the user and seed windows (A, kcal/(mol A^2)).
        hbonds: Keep the hydrogen bonds of the reference geometry (H...A and D...A).
        contacts: (atom, atom) pairs of non-covalent contacts to keep as in the
            reference geometry (with the neighbours, for the orientation).
        keep_fragments: Keep every fragment without core atoms (e.g. solvent) at the
            core by its closest contact in the reference geometry.
        fragment_links, link_fragments: For molecules of several fragments without a
            reference (catmlp's reactant complexes): windows [1.0, 1.3] x the vdW sum
            for the given links (fragment_links), and with link_fragments for links
            chosen to join every fragment (user pairs between fragments first, then
            charged pairs, then the least buried atoms). If the windows cannot be
            smoothed, the lower factor 0.8 is tried once; user windows never widen.
        hints: Candidate hydrogen bonds from the graph (sources.graph_hints, at most
            max_hints) as embedding-only windows (source "hint", stage "embed"): Embed
            uses them in some batches only (see Embed hint_share).

    User restraints win over generated ones for the same pair; generated ones for the
    same pair must agree.
    """
    user_set = RestraintSet(
        DistanceRestraint.around(
            i, j, target, half_width, force_constant=force_constant, source="user"
        )
        for i, j, target in sources.check_triplets(mol, user)
    )
    seed_triplets = []
    if hbonds:
        seed_triplets += [(*t, "hbond") for t in sources.hydrogen_bonds(mol)]
    pairs = [sources.check_pair(mol, pair, "Contact") for pair in contacts]
    seed_triplets += [(*t, "contact") for t in sources.contacts(mol, pairs)]
    if keep_fragments:
        links = sources.fragment_contacts(mol, frozen.core)
        seed_triplets += [(*t, "fragment") for t in sources.contacts(mol, links)]
    generated = RestraintSet()
    for i, j, target, source in seed_triplets:
        restraint = DistanceRestraint.around(
            i, j, target, half_width, force_constant=force_constant, source=source
        )
        generated.add(restraint, override=True)  # the same pair from two sources

    hard = set(frozen.hard)
    generated = _consistent(mol, frozen, user_set, generated)
    frozen_user = [r.pair for r in user_set if r.first in hard and r.second in hard]
    if frozen_user:
        logger.warning(
            "Restraints %s are ignored: both atoms are frozen at the reference.",
            frozen_user,
        )
    restraints = generated.merge(user_set).without_pairs_within(frozen.hard)

    if fragment_links is not None or link_fragments:
        restraints = _with_fragment_links(
            mol, restraints, user_set, fragment_links, link_fragments, frozen, seed,
            force_constant,
        )  # fmt: skip
    if hints:
        taken = {r.pair for r in restraints}
        candidates = RestraintSet(
            DistanceRestraint(h, a, lower, upper, stage="embed", source="hint")
            for h, a, lower, upper in sources.graph_hints(mol, max_hints=max_hints)
            if (min(h, a), max(h, a)) not in taken and not (h in hard and a in hard)
        )
        for hint in _consistent(mol, frozen, restraints, candidates, each=True):
            restraints.add(hint)
    if restraints:
        logger.info(
            "Restraints (atoms: window in A): %s",
            ", ".join(
                f"{r.first}-{r.second}: {r.lower:.2f}-{r.upper:.2f}" for r in restraints
            ),
        )
    return restraints


def _with_fragment_links(
    mol, restraints, user_set, fragment_links, link_fragments, frozen, seed, k
):
    from racerts.embed.bounds import bounds_matrix

    explicit = [
        sources.check_pair(mol, pair, "Fragment link") for pair in fragment_links or ()
    ]
    links = list(explicit)
    if link_fragments:
        fragment_of = {
            i: n for n, fragment in enumerate(Chem.GetMolFrags(mol)) for i in fragment
        }
        between = [
            r.pair for r in user_set if fragment_of[r.first] != fragment_of[r.second]
        ]
        candidates = explicit + between + sources.fallback_links(mol, frozen.core, seed)
        links += [
            pair for pair in sources.carrier_links(mol, candidates) if pair not in links
        ]
    for lower_factor in sources.LINK_LOWER_FACTORS:
        windows = RestraintSet(
            DistanceRestraint(
                *pair,
                *sources.link_window(mol, pair, lower_factor),
                force_constant=k,
                source="link",
            )  # fmt: skip
            for pair in {tuple(sorted(p)) for p in links}
        )
        candidate = windows.merge(restraints)  # user and seed windows win
        try:
            bounds_matrix(mol, windows=candidate)
        except ValueError:
            continue
        return candidate
    raise ValueError(
        "The fragment links cannot be embedded with the other restraints, even with "
        "the widened lower bounds (0.8 x the vdW sum)."
    )


def _consistent(mol, frozen, kept, generated, each=False) -> RestraintSet:
    """
    The generated restraints that the bounds take with kept (e.g. the user's): one by
    one, each with the ones taken before (each=True: each with kept alone, e.g. hints,
    which are embedded one at a time). The others are left out with a warning; a
    window that smoothing could only fit by moving other bounds (e.g. a short-range
    contact against the covalent geometry) would otherwise make embedding raise.
    """
    from racerts.embed.bounds import bounds_matrix

    if not generated:
        return generated
    reference = mol if mol.GetNumConformers() else None
    hard = list(frozen.hard) if reference is not None else []
    pairs = [(a, b) for k, a in enumerate(hard) for b in hard[k + 1 :]]

    def fits(windows) -> bool:
        try:
            bounds_matrix(mol, reference, pairs=pairs, windows=windows)
        except ValueError:
            return False
        return True

    taken, left_out = list(kept), []
    result = RestraintSet()
    for restraint in generated:
        if fits((list(kept) if each else taken) + [restraint]):
            result.add(restraint)
            taken.append(restraint)
        else:
            left_out.append(restraint.label)
    if left_out:
        logger.warning(
            "Restraints %s are left out: the bounds cannot take them together with "
            "the others and the frozen atoms.",
            left_out,
        )
    return result
