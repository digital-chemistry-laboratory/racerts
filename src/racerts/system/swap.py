"""
Swaps: replace a group of a molecule by a new fragment (graph surgery), keeping the
geometry of the other atoms (as catmlp's substitutions do).

apply_swap returns the new graph with every conformer of the reference: the kept atoms
at their coordinates and, for a single attachment, the fragment grafted rigidly along
the removed bond (as catmlp). racerts.swap then samples the new atoms (api.py).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem

logger = logging.getLogger(__name__)

BOND_TYPES = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
    "dative": Chem.BondType.DATIVE,
}
MODES = ("append", "renumber")
TETRAHEDRAL = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
MIN_KEPT_SHARE = 0.3  # below: warn, the swap is close to a new embedding


class SwapError(ValueError):
    """A swap that cannot be done as given (selector, fragment, valences, settings)."""


@dataclass(frozen=True)
class Swap:
    """
    What to replace, and by what. One selector:

    - site: the map number of a terminal H or dummy atom that leaves (catmlp's
      form); the fragment binds to its neighbour.
    - remove_atoms: the atoms that leave (their hydrogens leave with them); the
      fragment binds where bonds were cut. [] with attach_map adds a fragment.
    - center, substructure: the substructure-th group bound to center (groups are
      numbered by their lowest atom index), which leaves.
    - old_fragment: a SMARTS matched once in the molecule; its mapped atoms ([c:1])
      stay and bind to the dummy of the same number, the unmapped ones leave with
      everything bound to them beyond the mapped atoms.

    Args:
        new_fragment: SMILES with one dummy per attachment ([*] or [*:1], [*:2], ...).
            Hydrogens are added. A dummy bond "[*:1]<-P" is dative from P.
        attach_map: Dummy number -> the atom (reference index) it binds to. Default:
            the cut bonds, by kept atom index, pair with the dummies in number order.
        bond_types: Dummy number -> "single", "double", "triple" or "dative" (from the
            fragment atom), overriding the bond of the SMILES.
        mode: "append": new atoms take the slots of removed ones and the rest are
            appended (catmlp's layout), so the kept atoms keep their indices when the
            fragment has at least as many atoms as leave; otherwise the unused slots
            close up and later atoms move down (see SwapResult.ref_to_new).
            "renumber" puts the kept atoms first, in their order.
    """

    new_fragment: str
    site: Optional[int] = None
    remove_atoms: Optional[Sequence[int]] = None
    center: Optional[int] = None
    substructure: Optional[int] = None
    old_fragment: Optional[str] = None
    attach_map: Optional[Mapping[int, int]] = None
    bond_types: Optional[Mapping[int, str]] = None
    mode: str = "append"

    def __post_init__(self):
        selectors = [
            self.site is not None,
            self.remove_atoms is not None,
            self.center is not None or self.substructure is not None,
            self.old_fragment is not None,
        ]
        if sum(selectors) != 1:
            raise SwapError(
                "A Swap needs exactly one selector: site, remove_atoms, center and "
                "substructure, or old_fragment."
            )
        if (self.center is None) != (self.substructure is None):
            raise SwapError("center and substructure go together.")
        if self.mode not in MODES:
            raise SwapError(f"mode must be one of {MODES}, not {self.mode!r}.")
        for number, kind in (self.bond_types or {}).items():
            if kind not in BOND_TYPES:
                raise SwapError(
                    f"Unknown bond type {kind!r} for dummy {number}; use one of "
                    f"{sorted(BOND_TYPES)}."
                )


@dataclass
class SwapResult:
    """
    Attributes:
        mol: The new molecule with every conformer of the reference (same IDs): kept
            atoms at their coordinates, new atoms grafted (placed) or at the origin.
        conserved: New indices of the kept atoms, in reference order.
        new_atoms: New indices of the fragment atoms.
        junction: Kept atoms at an attachment and their kept neighbours.
        ref_to_new: Reference index -> new index of the kept atoms.
        attachments: (kept atom, fragment atom) of each attachment bond, new indices.
        placed: Whether the new atoms have coordinates (a single attachment that
            replaces a bond, grafted rigidly).
        positioned: The new atoms with coordinates: all if placed, else none.
        warnings: Diagnostics, also logged.
        fragment: The fragment with its hydrogens and dummies (for further poses,
            see racerts.embed.rigid_attach).
        fragment_map: Fragment index -> new index of its atoms (not the dummies).
        replaced: Removed atom (reference index) -> the new atom that took its bond
            to a kept atom; it takes the removed atom's role in a task (see
            index_map).
    """

    mol: Chem.Mol
    conserved: List[int]
    new_atoms: List[int]
    junction: List[int]
    ref_to_new: Dict[int, int]
    attachments: List[Tuple[int, int]]
    placed: bool
    warnings: List[str] = field(default_factory=list)
    positioned: List[int] = field(default_factory=list)
    fragment: Optional[Chem.Mol] = None
    fragment_map: Dict[int, int] = field(default_factory=dict)
    replaced: Dict[int, int] = field(default_factory=dict)

    @property
    def index_map(self) -> Dict[int, int]:
        """Reference index -> new index of the kept and the replaced atoms, e.g. to
        remap a task (a leaving group Cl replaced by Br stays a reacting atom)."""
        return {**self.ref_to_new, **self.replaced}


@dataclass
class _Attachment:
    number: int  # dummy number
    dummy: int  # fragment index of the dummy
    root: int  # fragment index of the atom bound to the dummy
    kept: int  # reference index of the kept atom
    partner: Optional[int]  # reference index of the removed atom bound to kept


def apply_swap(mol: Chem.Mol, swap: Swap, seed: int = 0xF00D) -> SwapResult:
    """
    The molecule with the swap applied (mol is not changed). seed: of the fragment
    geometry that is grafted.

    Raises:
        ValueError: For a selector that does not match exactly once, a fragment that
            is not one connected piece with its dummies, radicals or atom maps in the
            fragment, or valences that do not work out.
    """
    fragment, dummies = _fragment(swap)
    removed, cuts, anchors = _removed_atoms(mol, swap)
    attachments = _pair(mol, swap, fragment, dummies, removed, cuts, anchors)
    new_order, frag_to_new, ref_to_new = _layout(
        mol, swap, fragment, removed, attachments
    )
    result, carried_atoms, carried_bonds = _build(
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
    _settle_new_stereo(mol, result, ref_to_new, anchored, carried_atoms, carried_bonds)

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
        positioned=new_atoms if placed else [],
        fragment=fragment,
        fragment_map=frag_to_new,
        replaced={
            a.partner: frag_to_new[a.root] for a in attachments if a.partner is not None
        },
    )


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


FLIPPED_DIRECTIONS = {
    Chem.BondDir.ENDUPRIGHT: Chem.BondDir.ENDDOWNRIGHT,
    Chem.BondDir.ENDDOWNRIGHT: Chem.BondDir.ENDUPRIGHT,
}


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
        sites = [a for a in mol.GetAtoms() if a.GetAtomMapNum() == swap.site]
        if len(sites) != 1:
            raise SwapError(
                f"Expected exactly one atom with map number {swap.site}, found "
                f"{len(sites)}."
            )
        site = sites[0]
        if site.GetAtomicNum() not in (0, 1) or site.GetDegree() != 1:
            raise SwapError("A site must be a terminal hydrogen or dummy atom.")
        removed = {site.GetIdx()}
    elif swap.remove_atoms is not None:
        atoms = list(swap.remove_atoms)
        invalid = [
            i for i in atoms if not (isinstance(i, (int, np.integer)) and 0 <= i < n)
        ]
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
        if not 0 <= swap.substructure < len(groups):
            raise SwapError(
                f"Atom {swap.center} has {len(groups)} groups (substructure 0 to "
                f"{len(groups) - 1}), not {swap.substructure}."
            )
        removed = set(groups[swap.substructure])
    else:
        removed, anchors = _match_old_fragment(mol, swap.old_fragment)
    if len(removed) == n:
        raise SwapError("The swap would remove every atom.")
    cuts = sorted(
        (b.GetOtherAtomIdx(i), i)
        for i in removed
        for b in mol.GetAtomWithIdx(i).GetBonds()
        if b.GetOtherAtomIdx(i) not in removed
    )
    return sorted(removed), cuts, anchors


def _groups(mol: Chem.Mol, center: int) -> List[List[int]]:
    """The groups bound to center: connected pieces without it, by lowest index."""
    if not 0 <= center < mol.GetNumAtoms():
        raise SwapError(f"Invalid center atom {center}.")
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
    """The atoms that leave, and the kept atom of each map number of the SMARTS."""
    query = Chem.MolFromSmarts(smarts)
    if query is None:
        raise SwapError(f"Invalid SMARTS {smarts!r}.")
    mapped = {
        a.GetIdx(): a.GetAtomMapNum() for a in query.GetAtoms() if a.GetAtomMapNum()
    }
    if not mapped:
        raise SwapError("old_fragment needs mapped atoms ([c:1]) that stay.")
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
        if len(removed) + len(kept) >= mol.GetNumAtoms():
            continue  # not a substituent at the mapped atoms (e.g. a ring)
        anchors = {number: match[i] for i, number in mapped.items()}
        candidates.setdefault(frozenset(removed), anchors)
    if not candidates:
        raise SwapError(f"old_fragment {smarts!r} matches no substituent.")
    if len(candidates) > 1:
        raise SwapError(
            f"old_fragment {smarts!r} matches {len(candidates)} different groups "
            f"{sorted(sorted(c) for c in candidates)}; use remove_atoms."
        )
    removed, anchors = next(iter(candidates.items()))
    return set(removed), anchors


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
        if not 0 <= kept < mol.GetNumAtoms() or kept in removed:
            raise SwapError(f"Dummy {number} would bind to atom {kept}, which leaves.")
        cut = next((c for c in open_cuts if c[0] == kept), None)
        partner = None
        if cut is not None:
            open_cuts.remove(cut)
            partner = cut[1]
        dummy = dummies[number]
        root = fragment.GetAtomWithIdx(dummy).GetNeighbors()[0].GetIdx()
        attachments.append(_Attachment(number, dummy, root, kept, partner))
    return attachments


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
    # Double-bond stereo refers to neighbour atoms: map them.
    partner_to_root = {
        a.partner: frag_to_new[a.root] for a in attachments if a.partner is not None
    }
    ref_map = {**{i: ref_to_new[i] for i in ref_to_new}, **partner_to_root}
    frag_map = {**frag_to_new, **dummy_to_kept}
    carried_bonds, carried_atoms = set(), set()
    for source, index_map in ((mol, ref_map), (fragment, frag_map)):
        for bond in source.GetBonds():
            stereo_atoms = list(bond.GetStereoAtoms())
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            if not stereo_atoms or i not in index_map or j not in index_map:
                continue
            target = result.GetBondBetweenAtoms(index_map[i], index_map[j])
            if target is None or bond.GetStereo() in UNSPECIFIED_BOND:
                continue
            # as cis/trans of the stereo atoms, which the swap keeps (E/Z may not)
            stereo = GEOMETRIC.get(bond.GetStereo(), bond.GetStereo())
            ends = []
            for end, other, atom in ((i, j, stereo_atoms[0]), (j, i, stereo_atoms[1])):
                if atom not in index_map:  # it leaves: the other neighbour of its end
                    others = [
                        n.GetIdx()
                        for n in source.GetAtomWithIdx(end).GetNeighbors()
                        if n.GetIdx() not in (other, atom) and n.GetIdx() in index_map
                    ]
                    if not others:
                        break
                    atom = others[0]
                    stereo = FLIPPED_STEREO[stereo]
                ends.append(index_map[atom])
            if len(ends) < 2:
                continue
            target.SetStereoAtoms(*ends)
            target.SetStereo(stereo)
            carried_bonds.add(frozenset(target_pair(target)))
    # Chiral tags refer to the order of the neighbours: carry them over by parity.
    for source, index_map, kind in (
        (mol, ref_map, "ref"),
        (fragment, frag_map, "frag"),
    ):
        own = ref_to_new if kind == "ref" else frag_to_new
        for old, new_index in own.items():
            atom = source.GetAtomWithIdx(old)
            if atom.GetChiralTag() not in TETRAHEDRAL:
                continue
            expected = [index_map.get(n.GetIdx()) for n in atom.GetNeighbors()]
            target = result.GetAtomWithIdx(new_index)
            actual = [n.GetIdx() for n in target.GetNeighbors()]
            if None in expected or sorted(expected) != sorted(actual):
                logger.warning(
                    "Atom %d changes its neighbours in the swap; its stereo is left "
                    "unspecified.",
                    new_index,
                )
                target.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
                continue
            if _odd([expected.index(k) for k in actual]):
                target.InvertChirality()
            carried_atoms.add(new_index)
    flags = Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_FINDRADICALS
    try:  # radicals as given (kept atoms) or none (fragment), not guessed
        Chem.SanitizeMol(result, flags)
    except Exception as error:
        raise SwapError(f"The swapped molecule is invalid: {error}") from None
    result = result.GetMol()
    _perceive_stereo(result, carried_bonds)
    return result, carried_atoms, carried_bonds


UNSPECIFIED_BOND = (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
FLIPPED_STEREO = {
    Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS,
}
# E/Z of stereo atoms that are the CIP-highest neighbours, as geometry: E = trans
GEOMETRIC = {
    Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOCIS,
}


def target_pair(bond) -> Tuple[int, int]:
    return bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()


def _perceive_stereo(mol: Chem.Mol, keep_bonds) -> None:
    """
    Legacy stereo perception from the bond stereo that is set: bond directions are
    rebuilt from it alone; a double bond whose directions come only from its
    neighbours (not in keep_bonds) is left unspecified.
    """
    for bond in mol.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)
    Chem.SetDoubleBondNeighborDirections(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for bond in mol.GetBonds():
        if (
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetStereo() not in UNSPECIFIED_BOND
            and frozenset(target_pair(bond)) not in keep_bonds
        ):
            bond.SetStereo(Chem.BondStereo.STEREONONE)


def _stereo_candidates(mol: Chem.Mol):
    """
    The atoms and double bonds (atom pairs) that may carry stereo in the graph,
    including those whose stereogenicity depends on other stereo (flagPossible).
    """
    atoms, bonds = set(), set()
    try:
        infos = Chem.FindPotentialStereo(Chem.Mol(mol), cleanIt=True, flagPossible=True)
    except RuntimeError:  # e.g. an unsanitized molecule
        return atoms, bonds
    for info in infos:
        if info.type == Chem.StereoType.Atom_Tetrahedral:
            atoms.add(info.centeredOn)
        elif info.type == Chem.StereoType.Bond_Double:
            bonds.add(frozenset(target_pair(mol.GetBondWithIdx(info.centeredOn))))
    return atoms, bonds


def _stereo_3d(mol: Chem.Mol, conf_id: int, candidates):
    """
    The stereo of conformer conf_id: chiral tags of the stereocentres, and for the
    double bonds (atom pairs) stereo atoms and cis/trans. Candidates come from the
    graph (see _stereo_candidates) and from the cleaned 3D perception (RDKit 2025.03
    misses ring cis/trans in FindPotentialStereo); the configurations come from the
    geometry itself, so they depend neither on other unspecified stereo nor on the
    stereo perception in use.
    """
    one = Chem.Mol(mol, False, conf_id)
    probe = Chem.Mol(one)
    Chem.AssignStereochemistryFrom3D(probe)
    Chem.AssignStereochemistry(probe, cleanIt=True, force=True)
    atoms_found = {
        a.GetIdx() for a in probe.GetAtoms() if a.GetChiralTag() in TETRAHEDRAL
    }
    bonds_found = {
        frozenset(target_pair(b))
        for b in probe.GetBonds()
        if b.GetBondType() == Chem.BondType.DOUBLE
        and b.GetStereo() not in UNSPECIFIED_BOND
    }
    raw = Chem.Mol(one)
    Chem.AssignAtomChiralTagsFromStructure(raw, replaceExistingTags=True)
    atoms = {}
    for i in atoms_found | candidates[0]:
        tag = raw.GetAtomWithIdx(i).GetChiralTag()
        if tag in TETRAHEDRAL:
            atoms[i] = tag
    positions = one.GetConformer().GetPositions()
    bonds = {}
    for pair in bonds_found | candidates[1]:
        a, b = sorted(pair)
        ends = []
        for end, other in ((a, b), (b, a)):
            neighbours = [
                n.GetIdx()
                for n in mol.GetAtomWithIdx(end).GetNeighbors()
                if n.GetIdx() != other
            ]
            if not neighbours:
                break
            ends.append(min(neighbours))
        if len(ends) < 2:
            continue
        dihedral = abs(_dihedral(positions, ends[0], a, b, ends[1]))
        if 80 < dihedral < 100:  # twisted: neither cis nor trans
            continue
        cis = dihedral < 90
        stereo = Chem.BondStereo.STEREOCIS if cis else Chem.BondStereo.STEREOTRANS
        bonds[pair] = (tuple(ends), stereo)
    return atoms, bonds


def _dihedral(positions, i, j, k, m) -> float:
    b0, b1, b2 = (
        positions[i] - positions[j],
        positions[k] - positions[j],
        positions[m] - positions[k],
    )
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


def _settle_new_stereo(mol, result, ref_to_new, anchored, carried_atoms, carried_bonds):
    """
    Stereo elements that the swap creates (e.g. CH2 -> CH(R), a ring that gets
    cis/trans, a double bond that gets E/Z) take the configuration of the reference
    geometry where it defines them: the element and all its neighbours are anchored
    (kept atoms or atoms that replace one), and every reference conformer agrees on
    it (else a warning). Which hydrogen is replaced thus chooses the configuration
    (catmlp's prochiral sites). Other new elements stay unspecified. New elements are
    those of the result that the reference does not have.
    """
    if not anchored:
        return
    first = mol.GetConformers()[0].GetId()
    old_atoms, old_bonds = _stereo_3d(mol, first, _stereo_candidates(mol))
    old_atoms = {ref_to_new[i] for i in old_atoms if i in ref_to_new}
    old_bonds = {
        frozenset(ref_to_new[i] for i in pair)
        for pair in old_bonds
        if all(i in ref_to_new for i in pair)
    }

    def neighbourhood(atoms):
        return set(atoms) | {
            n.GetIdx() for i in atoms for n in result.GetAtomWithIdx(i).GetNeighbors()
        }

    candidates = _stereo_candidates(result)
    atom_values, bond_values = {}, {}
    for conf in result.GetConformers():
        atoms, bonds = _stereo_3d(result, conf.GetId(), candidates)
        for i in candidates[0] | set(atoms):
            atom_values.setdefault(i, []).append(atoms.get(i))
        for pair in candidates[1] | set(bonds):
            bond_values.setdefault(pair, []).append(bonds.get(pair))
    n = result.GetNumConformers()
    new_atoms = {
        i: values
        for i, values in atom_values.items()
        if i not in old_atoms | carried_atoms and neighbourhood([i]) <= anchored
    }
    new_bonds = {
        pair: values
        for pair, values in bond_values.items()
        if pair not in old_bonds | carried_bonds and neighbourhood(pair) <= anchored
    }
    # the same configuration in every reference conformer (stereo atoms may differ
    # only if the configuration is the same: compare after aligning to the first)
    agreed_atoms = {
        i: values[0]
        for i, values in new_atoms.items()
        if len(values) == n and values[0] is not None and len(set(values)) == 1
    }
    agreed_bonds = {
        pair: values[0]
        for pair, values in new_bonds.items()
        if len(values) == n and values[0] is not None and len(set(values)) == 1
    }
    disputed = sorted(
        {i for i, v in new_atoms.items() if i not in agreed_atoms and any(v)}
        | {
            i
            for p, v in new_bonds.items()
            if p not in agreed_bonds and any(v)
            for i in p
        }
    )
    if disputed:
        logger.warning(
            "The reference conformers disagree on the configuration of new stereo at "
            "atoms %s; it stays unspecified.",
            disputed,
        )
    if not (agreed_atoms or agreed_bonds):
        return
    for i, tag in agreed_atoms.items():
        result.GetAtomWithIdx(i).SetChiralTag(tag)
    for pair, (stereo_atoms, stereo) in agreed_bonds.items():
        bond = result.GetBondBetweenAtoms(*pair)
        bond.SetStereoAtoms(*stereo_atoms)
        bond.SetStereo(stereo)
    for bond in result.GetBonds():  # re-perceive, E/Z by CIP of the new graph
        if bond.GetStereo() in GEOMETRIC:
            bond.SetStereo(GEOMETRIC[bond.GetStereo()])
    _perceive_stereo(result, carried_bonds | set(agreed_bonds))


def _copy_bond(result, bond, i, j):
    """Add the bond i-j (i the begin atom of bond) with its type and direction."""
    result.AddBond(i, j, bond.GetBondType())
    added = result.GetBondBetweenAtoms(i, j)
    added.SetIsAromatic(bond.GetIsAromatic())
    added.SetBondDir(bond.GetBondDir())


def _odd(order: Sequence[int]) -> bool:
    inversions = sum(
        order[a] > order[b] for a in range(len(order)) for b in range(a + 1, len(order))
    )
    return bool(inversions % 2)


def _coordinates(
    mol, result, fragment, attachments, frag_to_new, ref_to_new, placed, seed
):
    """Every reference conformer, with the fragment grafted if placed."""
    xyz = None
    if placed:
        embedded = Chem.Mol(fragment)
        with rdBase.BlockLogs():  # UFF typer messages for the dummies
            if AllChem.EmbedMolecule(embedded, randomSeed=seed) < 0:
                raise SwapError("Could not embed the fragment.")
        xyz = embedded.GetConformer().GetPositions()
        a = attachments[0]
    for conf in mol.GetConformers():
        positions = conf.GetPositions()
        new_positions = np.zeros((result.GetNumAtoms(), 3))
        for i, k in ref_to_new.items():
            new_positions[k] = positions[i]
        if placed:
            anchor = positions[a.kept]
            direction = positions[a.partner] - anchor
            norm = np.linalg.norm(direction)
            if not np.isfinite(norm) or norm < 1e-8:
                raise SwapError("The attachment has coincident or invalid coordinates.")
            direction /= norm
            length = _bond_length(mol, fragment, a, norm)
            rotation = rotation_between(xyz[a.root] - xyz[a.dummy], direction)
            grafted = (xyz - xyz[a.root]) @ rotation.T + anchor + length * direction
            for j, k in frag_to_new.items():
                new_positions[k] = grafted[j]
        new_conf = Chem.Conformer(result.GetNumAtoms())
        for k, p in enumerate(new_positions):
            new_conf.SetAtomPosition(k, p.tolist())
        new_conf.SetId(conf.GetId())
        new_conf.Set3D(conf.Is3D())
        result.AddConformer(new_conf, assignId=False)


def _bond_length(mol, fragment, attachment, removed_length: float) -> float:
    """
    The length of the new bond: the removed one scaled by covalent radii, so that a
    bond stretched in the reference (e.g. a leaving group of a TS) stays stretched;
    for a dummy that leaves (a template site), the sum of the covalent radii.
    """
    table = Chem.GetPeriodicTable()
    kept = table.GetRcovalent(mol.GetAtomWithIdx(attachment.kept).GetAtomicNum())
    root = table.GetRcovalent(fragment.GetAtomWithIdx(attachment.root).GetAtomicNum())
    partner = mol.GetAtomWithIdx(attachment.partner).GetAtomicNum()
    if partner == 0:
        return kept + root
    return removed_length * (kept + root) / (kept + table.GetRcovalent(partner))


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The proper rotation taking the direction source onto target (catmlp)."""
    source = source / np.linalg.norm(source)
    target = target / np.linalg.norm(target)
    cosine = float(np.clip(source @ target, -1, 1))
    if cosine < -1 + 1e-12:  # antiparallel: a half turn about a perpendicular axis
        axis = np.cross(source, np.eye(3)[np.argmin(np.abs(source))])
        axis /= np.linalg.norm(axis)
        return 2 * np.outer(axis, axis) - np.eye(3)
    cross = np.cross(source, target)
    skew = np.array(
        [[0, -cross[2], cross[1]], [cross[2], 0, -cross[0]], [-cross[1], cross[0], 0]]
    )
    return np.eye(3) + skew + skew @ skew / (1 + cosine)
