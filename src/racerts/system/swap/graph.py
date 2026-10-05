"""The graph of the result, its charge, multiplicity and restraints, and its checks."""

import logging

from rdkit import Chem

from ..spec import infer_charge_and_multiplicity
from ..stereo import TETRAHEDRAL, UNSPECIFIED_BOND
from .configuration import FLIPPED_STEREO, GEOMETRIC, _ends, _odd
from .model import SwapError

logger = logging.getLogger(__name__)


def _carry_restraints(restraints, ref_to_new) -> dict:
    if restraints is None:
        return {}
    from racerts.restraints import RestraintSet

    restraints = RestraintSet(restraints)
    kept = restraints.remap(ref_to_new)
    lost = [
        r.label
        for r in restraints
        if not (r.first in ref_to_new and r.second in ref_to_new)
    ]
    if lost:
        logger.warning("Restraints %s lose an atom in the swap and are left out.", lost)
    return {"restraints": kept, "lost_contacts": lost}


def _layout(mol, swap, fragment, removed, attachments):
    """
    The atoms of the new molecule in order, as ("ref" | "frag", index), and the index
    maps of the fragment and the kept atoms.
    """
    dummies = {a.dummy for a in attachments}
    roots = list(dict.fromkeys(a.root for a in attachments))
    new = roots + [
        a.GetIdx()
        for a in fragment.GetAtoms()
        if a.GetIdx() not in dummies and a.GetIdx() not in roots
    ]
    if swap.mode == "renumber":
        slots = [("ref", i) for i in range(mol.GetNumAtoms()) if i not in removed]
        slots += [("frag", j) for j in new]
    else:
        # A root takes the slot of the atom it replaces, the other new atoms the
        # remaining slots of removed atoms, then the end; unused slots close up.
        fill = {}
        for a in attachments:
            if (
                a.partner is not None
                and a.partner not in fill
                and a.root not in fill.values()
            ):
                fill[a.partner] = a.root
        rest = [j for j in new if j not in fill.values()]
        free = [i for i in removed if i not in fill]
        fill.update(zip(free, rest))
        slots = []
        for i in range(mol.GetNumAtoms()):
            if i not in removed:
                slots.append(("ref", i))
            elif i in fill:
                slots.append(("frag", fill[i]))
        slots += [("frag", j) for j in rest[len(free) :]]
    frag_to_new = {j: k for k, (kind, j) in enumerate(slots) if kind == "frag"}
    ref_to_new = {i: k for k, (kind, i) in enumerate(slots) if kind == "ref"}
    return slots, frag_to_new, ref_to_new


def _build(mol, swap, fragment, attachments, slots, frag_to_new, ref_to_new):
    """
    The new graph, and the atoms and double bonds (atom pairs) whose stereo was
    carried over from the reference or the fragment.
    """
    result = Chem.RWMol()
    site_label = swap.site
    roots = {a.root for a in attachments}
    for kind, i in slots:
        source = mol if kind == "ref" else fragment
        atom = Chem.Atom(source.GetAtomWithIdx(i))
        if kind == "frag":
            atom.SetAtomMapNum(site_label if site_label and i in roots else 0)
        result.AddAtom(atom)
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        if i in ref_to_new and j in ref_to_new:
            _copy_bond(result, bond, ref_to_new[i], ref_to_new[j])
    dummy_to_kept = {a.dummy: ref_to_new[a.kept] for a in attachments}
    for bond in fragment.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        ends = [frag_to_new.get(k, dummy_to_kept.get(k)) for k in (i, j)]
        if i in dummy_to_kept or j in dummy_to_kept:
            result.AddBond(ends[0], ends[1], bond.GetBondType())
        else:
            _copy_bond(result, bond, *ends)
    # Stereo refers to neighbour atoms: map them. A removed atom is replaced, at each
    # kept atom it was bound to, by the fragment atom attached there (one atom between
    # two kept atoms can become a chain: each end has its own new neighbour).
    took_over = {
        (a.kept, a.partner): frag_to_new[a.root]
        for a in attachments
        if a.partner is not None
    }
    frag_map = {**frag_to_new, **dummy_to_kept}

    def ref_image(atom, seen_from):
        if atom in ref_to_new:
            return ref_to_new[atom]
        return took_over.get((seen_from, atom))

    def frag_image(atom, seen_from):
        return frag_map.get(atom)

    carried_bonds, carried_atoms, dropped = set(), set(), set()
    for source, image in ((mol, ref_image), (fragment, frag_image)):
        for bond in source.GetBonds():
            stereo_atoms = list(bond.GetStereoAtoms())
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            new_i, new_j = image(i, j), image(j, i)
            if not stereo_atoms or new_i is None or new_j is None:
                continue
            target = result.GetBondBetweenAtoms(new_i, new_j)
            if target is None or bond.GetStereo() in UNSPECIFIED_BOND:
                continue
            # as cis/trans of the stereo atoms, which the swap keeps (E/Z may not)
            stereo = GEOMETRIC.get(bond.GetStereo(), bond.GetStereo())
            ends = []
            for end, new_end, other, atom in (
                (i, new_i, j, stereo_atoms[0]),
                (j, new_j, i, stereo_atoms[1]),
            ):
                found = image(atom, end)
                if found is None:  # it leaves: the other neighbour of its end
                    others = [
                        image(n.GetIdx(), end)
                        for n in source.GetAtomWithIdx(end).GetNeighbors()
                        if n.GetIdx() not in (other, atom)
                    ]
                    others = [k for k in others if k is not None]
                    if not others:
                        break
                    found = others[0]
                    stereo = FLIPPED_STEREO[stereo]
                if result.GetBondBetweenAtoms(new_end, found) is None:
                    break  # the end itself was replaced by a chain: no such neighbour
                ends.append(found)
            if len(ends) < 2:
                continue
            if target.GetBeginAtomIdx() != new_i:  # RDKit: the begin atom's side first
                ends.reverse()
            target.SetStereoAtoms(*ends)
            target.SetStereo(stereo)
            carried_bonds.add(frozenset(_ends(target)))
    # Chiral tags refer to the order of the neighbours: carry them over by parity.
    for source, image, kind in (
        (mol, ref_image, "ref"),
        (fragment, frag_image, "frag"),
    ):
        own = ref_to_new if kind == "ref" else frag_to_new
        for old, new_index in own.items():
            atom = source.GetAtomWithIdx(old)
            if atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED:
                continue
            expected = [image(n.GetIdx(), old) for n in atom.GetNeighbors()]
            target = result.GetAtomWithIdx(new_index)
            actual = [n.GetIdx() for n in target.GetNeighbors()]
            if None in expected or sorted(expected) != sorted(actual):
                target.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                if atom.GetChiralTag() in TETRAHEDRAL:
                    dropped.add(new_index)  # settled from the geometry, or warned
                else:
                    logger.warning(
                        "Atom %d changes its neighbours in the swap; its "
                        "non-tetrahedral stereo is left unspecified.",
                        new_index,
                    )
                continue
            if atom.GetChiralTag() not in TETRAHEDRAL:
                # e.g. square planar: only the parity of tetrahedral tags is remapped
                if expected != actual:
                    logger.warning(
                        "Atom %d gets its neighbours in another order; its "
                        "non-tetrahedral stereo is left unspecified.",
                        new_index,
                    )
                    target.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                continue
            if _odd([expected.index(k) for k in actual]):
                target.InvertChirality()
            carried_atoms.add(new_index)
    flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS
    trial = Chem.RWMol(result)  # a failed sanitization can leave bonds changed
    try:  # radicals as given (kept atoms) or none (fragment), not guessed
        Chem.SanitizeMol(trial, flags)
        result = trial
    except Exception as error:
        _accept_problems_of_the_reference(mol, result, ref_to_new, error)
        Chem.SanitizeMol(result, flags ^ Chem.SanitizeFlags.SANITIZE_PROPERTIES)
        result.UpdatePropertyCache(strict=False)
    _check_valences(mol, result, ref_to_new)
    result = result.GetMol()
    return result, carried_atoms, carried_bonds, dropped


def _charge_and_multiplicity(mol, result, fragment, frag_to_new, removed) -> None:
    """
    The charge of the reference (property or formal charges) changed by the formal
    charges of the fragment and of the removed atoms; the multiplicity of the
    reference, if it has one and the swap keeps the parity of the electrons.
    """
    reference_charge = infer_charge_and_multiplicity(mol)["charge"]
    charge = reference_charge
    charge += sum(fragment.GetAtomWithIdx(j).GetFormalCharge() for j in frag_to_new)
    charge -= sum(mol.GetAtomWithIdx(i).GetFormalCharge() for i in removed)
    result.SetIntProp("charge", int(charge))
    if mol.HasProp("multiplicity"):
        multiplicity = mol.GetIntProp("multiplicity")
        change = _electrons(result) - charge - (_electrons(mol) - reference_charge)
        if change % 2 == 0:  # the parity of the electrons is the same
            result.SetIntProp("multiplicity", multiplicity)
        else:
            logger.warning(
                "The multiplicity %d of the reference does not fit the swap (%+d "
                "electrons); it is not carried over.",
                multiplicity,
                change,
            )


def _electrons(mol: Chem.Mol) -> int:
    """Protons of the atoms and their implicit hydrogens, a dummy counting as one."""
    total = 0
    for atom in mol.GetAtoms():
        total += atom.GetAtomicNum() or 1
        try:
            total += atom.GetTotalNumHs()
        except RuntimeError:  # valences not computed: explicit counts only
            total += atom.GetNumExplicitHs()
    return total


def _accept_problems_of_the_reference(mol, result, ref_to_new, error) -> None:
    """
    A swapped molecule that does not sanitize is accepted only where the reference has
    the same problem (e.g. a TS connectivity graph with a hypervalent atom): valence
    problems of kept atoms that the reference has too. Anything else raises.
    """
    try:
        own = Chem.DetectChemistryProblems(mol)
    except RuntimeError:
        own = []
    accepted = {
        ref_to_new[p.GetAtomIdx()]
        for p in own
        if p.GetType() == "AtomValenceException" and p.GetAtomIdx() in ref_to_new
    }
    problems = Chem.DetectChemistryProblems(result)
    for problem in problems:
        if problem.GetType() != "AtomValenceException" or (
            problem.GetAtomIdx() not in accepted
        ):
            raise SwapError(f"The swapped molecule is invalid: {problem.Message()}")
    if not problems:
        raise SwapError(f"The swapped molecule is invalid: {error}")
    logger.info("The reference has the valence problems of the swap: %s", error)


def _check_valences(mol, result, ref_to_new) -> None:
    """
    A kept atom may not lose bond order where that leaves it short: a hydrogen
    without coordinates (a reference with a geometry and explicit hydrogens), or an
    open valence on an atom without implicit hydrogens (not guessed as a radical).
    Metals (no default valence) are not checked; nor, for the first, graphs without
    a geometry or without explicit hydrogens, where implicit hydrogens are normal.
    """
    table = Chem.GetPeriodicTable()
    geometry = mol.GetNumConformers() > 0 and any(
        a.GetAtomicNum() == 1 for a in mol.GetAtoms()
    )
    for i, k in ref_to_new.items():
        old, new = mol.GetAtomWithIdx(i), result.GetAtomWithIdx(k)
        if table.GetDefaultValence(new.GetAtomicNum()) < 0:
            continue
        try:
            lost = old.GetTotalValence() - new.GetTotalValence()
            gained_h = new.GetNumImplicitHs() - old.GetNumImplicitHs()
        except RuntimeError:  # a reference without computed valences
            continue
        if (geometry and gained_h > 0) or (lost > 0 and new.GetNoImplicit()):
            raise SwapError(
                f"Atom {i} would lose bond order in the swap (it would need "
                f"{max(lost, gained_h)} more hydrogen(s) or unpaired electrons): the "
                "new bond is of lower order than the removed one; give bond_types, "
                "or attach where bonds are cut."
            )


def _copy_bond(result, bond, i, j):
    """Add the bond i-j (i the begin atom of bond) with its type and direction."""
    result.AddBond(i, j, bond.GetBondType())
    added = result.GetBondBetweenAtoms(i, j)
    added.SetIsAromatic(bond.GetIsAromatic())
    added.SetBondDir(bond.GetBondDir())
