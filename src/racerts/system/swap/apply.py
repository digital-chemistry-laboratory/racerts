"""apply_swap: the steps of a swap in their order."""

import logging

from rdkit import Chem

from ..stereo import TETRAHEDRAL
from .configuration import (
    _given_stereo,
    _perceive_stereo,
    _settle_stereo,
    _stereo_candidates,
)
from .coordinates import _coordinates
from .graph import _build, _carry_restraints, _charge_and_multiplicity, _layout
from .model import MIN_KEPT_SHARE, Swap, SwapResult
from .selection import _fragment, _pair, _removed_atoms

logger = logging.getLogger(__name__)


def apply_swap(
    mol: Chem.Mol, swap: Swap, seed: int = 0xF00D, restraints=None
) -> SwapResult:
    """
    The molecule with the swap applied (mol is not changed). seed: of the fragment
    geometry that is grafted. restraints (a RestraintSet of the reference, e.g. its
    hydrogen bonds): those between kept atoms carry over (SwapResult.restraints), the
    others are reported (lost_contacts); a contact to an atom that a fragment atom
    replaces does not carry over, its distance would not fit.

    Raises:
        SwapError: For a selector that does not match exactly once, a fragment that
            is not one connected piece with its dummies, radicals or atom maps in the
            fragment, or valences that do not work out.
    """
    fragment, dummies = _fragment(swap)
    removed, cuts, anchors = _removed_atoms(mol, swap)
    attachments = _pair(mol, swap, fragment, dummies, removed, cuts, anchors)
    new_order, frag_to_new, ref_to_new = _layout(
        mol, swap, fragment, removed, attachments
    )
    result, carried_atoms, carried_bonds, dropped = _build(
        mol, swap, fragment, attachments, new_order, frag_to_new, ref_to_new
    )

    placed = False
    anchored = set()
    if mol.GetNumConformers():
        placed = len(attachments) == 1 and attachments[0].partner is not None
        _coordinates(
            mol, result, fragment, attachments, frag_to_new, ref_to_new, placed, seed
        )
        # atoms whose coordinates mean something: the kept ones and the atoms that
        # replace one (not the rest of a graft, whose rotation is arbitrary)
        anchored = set(ref_to_new.values()) | {
            frag_to_new[a.root] for a in attachments if a.partner is not None
        }
    settled = _settle_stereo(result, anchored, carried_atoms, carried_bonds)
    given = _given_stereo(swap, result, ref_to_new, attachments, frag_to_new)
    _perceive_stereo(result, carried_bonds | settled | given)
    for i in sorted(dropped):  # a tag that was neither carried nor settled
        if result.GetAtomWithIdx(i).GetChiralTag() not in TETRAHEDRAL and (
            i in _stereo_candidates(result)[0]
        ):
            logger.warning(
                "Atom %d changes its neighbours in the swap and the reference does "
                "not define its configuration; its stereo is left unspecified.",
                i,
            )
    _charge_and_multiplicity(mol, result, fragment, frag_to_new, removed)

    conserved = [ref_to_new[i] for i in sorted(ref_to_new)]
    new_atoms = sorted(frag_to_new.values())
    kept_ends = sorted({ref_to_new[a.kept] for a in attachments})
    junction = set(kept_ends)
    for i in kept_ends:
        junction.update(
            n.GetIdx()
            for n in result.GetAtomWithIdx(i).GetNeighbors()
            if n.GetIdx() in ref_to_new.values()
        )
    warnings = []
    share = len(conserved) / result.GetNumAtoms()
    if share < MIN_KEPT_SHARE:
        warnings.append(
            f"The swap keeps {len(conserved)} of {result.GetNumAtoms()} atoms "
            f"({share:.0%}): close to a new embedding."
        )
    for warning in warnings:
        logger.warning(warning)
    return SwapResult(
        mol=result,
        conserved=conserved,
        new_atoms=new_atoms,
        junction=sorted(junction),
        ref_to_new=ref_to_new,
        attachments=[(ref_to_new[a.kept], frag_to_new[a.root]) for a in attachments],
        placed=placed,
        warnings=warnings,
        positioned=new_atoms
        if placed
        else sorted(
            {
                frag_to_new[a.root]
                for a in attachments
                if a.partner is not None and mol.GetNumConformers()
            }
        ),
        fragment=fragment,
        fragment_map=frag_to_new,
        replaced={
            a.partner: frag_to_new[a.root] for a in attachments if a.partner is not None
        },
        **_carry_restraints(restraints, ref_to_new),
    )
