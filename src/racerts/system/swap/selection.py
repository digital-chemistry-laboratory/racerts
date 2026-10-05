"""The fragment that comes, the atoms that leave, and how the two are paired."""

import logging
from typing import Dict, List, Tuple

from rdkit import Chem

from racerts.utils.checks import is_integer

from .configuration import FLIPPED_DIRECTIONS, _odd
from .model import BOND_TYPES, Swap, SwapError, _Attachment

logger = logging.getLogger(__name__)


def _fragment(swap: Swap) -> Tuple[Chem.Mol, Dict[int, int]]:
    """The fragment with explicit hydrogens, and its dummies (number -> index)."""
    params = Chem.SmilesParserParams()
    params.removeHs = False
    params.sanitize = False
    fragment = Chem.MolFromSmiles(swap.new_fragment, params)
    if fragment is None:
        raise SwapError(f"Invalid fragment SMILES {swap.new_fragment!r}.")
    fragment = Chem.RWMol(fragment)
    dummy_atoms = [a for a in fragment.GetAtoms() if a.GetAtomicNum() == 0]
    if not dummy_atoms:
        raise SwapError(f"The fragment {swap.new_fragment!r} has no dummy atom [*].")
    if len(Chem.GetMolFrags(fragment)) != 1:
        raise SwapError(f"The fragment {swap.new_fragment!r} is not one piece.")
    numbers = [a.GetAtomMapNum() for a in dummy_atoms]
    if len(dummy_atoms) == 1 and numbers == [0]:
        numbers = [1]
    elif 0 in numbers or len(set(numbers)) != len(numbers):
        raise SwapError("Several dummies need distinct numbers: [*:1], [*:2], ...")
    dummies = dict(zip(numbers, (a.GetIdx() for a in dummy_atoms)))
    for number, index in dummies.items():
        dummy = fragment.GetAtomWithIdx(index)
        if dummy.GetDegree() != 1:
            raise SwapError(f"Dummy {number} must have exactly one bond.")
        if any(n.GetAtomicNum() == 0 for n in dummy.GetNeighbors()):
            raise SwapError("Dummies cannot bind to each other.")
        kind = (swap.bond_types or {}).get(number)
        if kind is not None:
            _set_bond_type(fragment, index, kind)
    unknown = set(swap.bond_types or {}) - set(dummies)
    if unknown:
        raise SwapError(f"bond_types for dummies the fragment lacks: {sorted(unknown)}")
    for atom in fragment.GetAtoms():
        if atom.GetAtomicNum() and atom.GetAtomMapNum():
            raise SwapError("Only the dummies of the fragment may carry map numbers.")
    try:
        Chem.SanitizeMol(fragment)
    except Exception as error:
        raise SwapError(f"Invalid fragment {swap.new_fragment!r}: {error}") from None
    if any(a.GetNumRadicalElectrons() for a in fragment.GetAtoms()):
        raise SwapError(f"The fragment {swap.new_fragment!r} has radical electrons.")
    with_h = Chem.AddHs(fragment)  # hydrogens are appended: indices stay
    Chem.AssignStereochemistry(with_h, cleanIt=True, force=True)
    return with_h, dummies


def _set_bond_type(fragment: Chem.RWMol, dummy: int, kind: str) -> None:
    """
    The dummy's bond as kind (dative: from the fragment atom). RDKit cannot turn a bond
    around, so it is removed and added again; the root's chiral tag and the bond's
    direction (E/Z of a double bond next to it), which refer to the neighbour order
    and the bond's begin atom, are carried over.
    """
    bond = fragment.GetAtomWithIdx(dummy).GetBonds()[0]
    root = bond.GetOtherAtomIdx(dummy)
    before = [n.GetIdx() for n in fragment.GetAtomWithIdx(root).GetNeighbors()]
    direction, begin = bond.GetBondDir(), bond.GetBeginAtomIdx()
    fragment.RemoveBond(dummy, root)
    first, second = (root, dummy) if kind == "dative" else (dummy, root)
    fragment.AddBond(first, second, BOND_TYPES[kind])
    added = fragment.GetBondBetweenAtoms(first, second)
    if first != begin:
        direction = FLIPPED_DIRECTIONS.get(direction, direction)
    added.SetBondDir(direction)
    after = [n.GetIdx() for n in fragment.GetAtomWithIdx(root).GetNeighbors()]
    if _odd([before.index(k) for k in after]):
        fragment.GetAtomWithIdx(root).InvertChirality()


def _removed_atoms(mol: Chem.Mol, swap: Swap):
    """
    The atoms that leave, the cut bonds as (kept atom, removed atom), and for
    old_fragment the kept atom of each map number (else None).
    """
    n = mol.GetNumAtoms()
    anchors = None
    if swap.site is not None:
        removed = {_site(mol, swap.site)}
    elif swap.remove_atoms is not None:
        atoms = list(swap.remove_atoms)
        invalid = [i for i in atoms if not _is_atom(mol, i)]
        if invalid:
            raise SwapError(f"Invalid atoms to remove: {invalid}.")
        removed = set(int(i) for i in atoms)
        for i in list(removed):  # hydrogens leave with their atom
            removed.update(
                h.GetIdx()
                for h in mol.GetAtomWithIdx(i).GetNeighbors()
                if h.GetAtomicNum() == 1 and h.GetDegree() == 1
            )
    elif swap.center is not None:
        groups = _groups(mol, swap.center)
        if not is_integer(swap.substructure):
            raise SwapError(
                f"substructure must be an integer (the number of a group of atom "
                f"{swap.center}), not {swap.substructure!r}."
            )
        if not 0 <= swap.substructure < len(groups):
            raise SwapError(
                f"Atom {swap.center} has {len(groups)} groups (substructure 0 to "
                f"{len(groups) - 1}), not {swap.substructure}."
            )
        removed = set(groups[swap.substructure])
    else:
        removed, ways = _match_old_fragment(mol, swap.old_fragment)
        anchors = ways[0]
        if len(ways) > 1 and swap.attach_map is None:
            logger.warning(
                "old_fragment %r matches the group in %d ways (dummy number: kept "
                "atom): %s. The first is used; give attach_map to choose.",
                swap.old_fragment,
                len(ways),
                ways,
            )
    if len(removed) == n:
        raise SwapError("The swap would remove every atom.")
    cuts = sorted(
        (b.GetOtherAtomIdx(i), i)
        for i in removed
        for b in mol.GetAtomWithIdx(i).GetBonds()
        if b.GetOtherAtomIdx(i) not in removed
    )
    return sorted(removed), cuts, anchors


def _is_atom(mol: Chem.Mol, index) -> bool:
    """Whether index is an integer and an atom of mol."""
    return is_integer(index) and 0 <= index < mol.GetNumAtoms()


def _site(mol: Chem.Mol, label: int) -> int:
    """The terminal hydrogen or dummy atom with the map number label."""
    label = int(label)
    sites = [a for a in mol.GetAtoms() if a.GetAtomMapNum() == label]
    if len(sites) != 1:
        raise SwapError(
            f"Expected exactly one atom with map number {label}, found {len(sites)}."
        )
    if sites[0].GetAtomicNum() not in (0, 1) or sites[0].GetDegree() != 1:
        raise SwapError("A site must be a terminal hydrogen or dummy atom.")
    return sites[0].GetIdx()


def _groups(mol: Chem.Mol, center: int) -> List[List[int]]:
    """The groups bound to center: connected pieces without it, by lowest index."""
    if not _is_atom(mol, center):
        raise SwapError(f"Invalid center atom {center!r}.")
    center = int(center)  # a NumPy integer is no index for RDKit
    seen = {center}
    groups = []
    for start in sorted(n.GetIdx() for n in mol.GetAtomWithIdx(center).GetNeighbors()):
        if start in seen:
            continue
        group, stack = [], [start]
        seen.add(start)
        while stack:
            i = stack.pop()
            group.append(i)
            for n in mol.GetAtomWithIdx(i).GetNeighbors():
                if n.GetIdx() not in seen:
                    seen.add(n.GetIdx())
                    stack.append(n.GetIdx())
        groups.append(sorted(group))
    return sorted(groups)


def _match_old_fragment(mol: Chem.Mol, smarts: str):
    """
    The atoms that leave, and the ways the SMARTS matches them: for each, the kept
    atom of every map number.
    """
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise SwapError(f"Invalid SMARTS {smarts!r}.")
    mapped = {
        a.GetIdx(): a.GetAtomMapNum() for a in query.GetAtoms() if a.GetAtomMapNum()
    }
    if not mapped:
        raise SwapError("old_fragment needs mapped atoms ([c:1]) that stay.")
    repeated = sorted(
        {n for n in mapped.values() if list(mapped.values()).count(n) > 1}
    )
    if repeated:
        raise SwapError(
            f"old_fragment {smarts!r} uses the map number {repeated[0]} more than "
            "once: every atom that stays needs a number of its own."
        )
    candidates = {}
    for match in mol.GetSubstructMatches(query, uniquify=False, maxMatches=10000):
        kept = {match[i] for i in mapped}
        start = [match[i] for i in range(len(match)) if i not in mapped]
        removed, stack = set(start), list(start)
        while stack:  # everything bound to the leaving atoms beyond the kept ones
            i = stack.pop()
            for n in mol.GetAtomWithIdx(i).GetNeighbors():
                j = n.GetIdx()
                if j not in removed and j not in kept:
                    removed.add(j)
                    stack.append(j)
        anchors = {number: match[i] for i, number in mapped.items()}
        ways = candidates.setdefault(frozenset(removed), [])
        if anchors not in ways:
            ways.append(anchors)
    if not candidates:
        raise SwapError(f"old_fragment {smarts!r} does not match the molecule.")
    if len(candidates) > 1:
        raise SwapError(
            f"old_fragment {smarts!r} matches {len(candidates)} different groups "
            f"{sorted(sorted(c) for c in candidates)}; use remove_atoms."
        )
    removed, ways = next(iter(candidates.items()))
    return set(removed), ways


def _pair(mol, swap, fragment, dummies, removed, cuts, anchors) -> List[_Attachment]:
    """Which kept atom each dummy binds to, and the removed atom it replaces."""
    numbers = sorted(dummies)
    if swap.attach_map is not None:
        attach_map = dict(swap.attach_map)
    elif anchors is not None:
        attach_map = anchors
    else:
        if len(cuts) != len(numbers):
            raise SwapError(
                f"The swap cuts {len(cuts)} bonds but the fragment has {len(numbers)} "
                "dummies; give attach_map."
            )
        attach_map = {number: cut[0] for number, cut in zip(numbers, cuts)}
    if set(attach_map) != set(numbers):
        raise SwapError(
            f"The dummies {numbers} of the fragment need a kept atom each; given for "
            f"{sorted(attach_map)}."
        )
    open_cuts = list(cuts)
    attachments = []
    for number in numbers:
        kept = attach_map[number]
        if not _is_atom(mol, kept):
            raise SwapError(
                f"Dummy {number} would bind to atom {kept!r}, which the molecule does "
                "not have."
            )
        if kept in removed:
            raise SwapError(f"Dummy {number} would bind to atom {kept}, which leaves.")
        cut = next((c for c in open_cuts if c[0] == kept), None)
        partner = None
        if cut is not None:
            open_cuts.remove(cut)
            partner = cut[1]
        dummy = dummies[number]
        root = fragment.GetAtomWithIdx(dummy).GetNeighbors()[0].GetIdx()
        attachments.append(_Attachment(number, dummy, root, int(kept), partner))
    for kept, leaving in open_cuts:  # e.g. a ring atom that leaves, one side closed
        logger.warning(
            "The swap cuts the bond %d-%d without binding anything to atom %d: it is "
            "left with one bond less (a hydrogen more where the graph has implicit "
            "ones).",
            kept,
            leaving,
            kept,
        )
    bonds = [(a.kept, a.root) for a in attachments]
    if len(set(bonds)) < len(bonds):
        raise SwapError(
            "Two dummies on one fragment atom would form the same bond to a kept atom."
        )
    return attachments
