"""
Swaps: replace a group of a molecule by a new fragment (graph surgery), keeping the
geometry of the other atoms.

apply_swap returns the new graph with every conformer of the reference: the kept atoms
at their coordinates and, for a single attachment, the fragment grafted rigidly along
the removed bond. racerts.swap then samples the new atoms (api.py).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
from rdkit import Chem, rdBase
from rdkit.Chem import AllChem, rdCIPLabeler

from racerts.utils import seeds
from racerts.utils.checks import is_integer

from .spec import infer_charge_and_multiplicity
from .stereo import TETRAHEDRAL, UNSPECIFIED_BOND, geometry_tags

logger = logging.getLogger(__name__)

BOND_TYPES = {
    "single": Chem.BondType.SINGLE,
    "double": Chem.BondType.DOUBLE,
    "triple": Chem.BondType.TRIPLE,
    "dative": Chem.BondType.DATIVE,
}
MODES = ("append", "renumber")
BOND_STEREO = ("cis", "trans", "E", "Z")
ATOM_STEREO = ("R", "S")
MIN_KEPT_SHARE = 0.3  # below: warn, the swap is close to a new embedding
FLIPPED_DIRECTIONS = {
    Chem.BondDir.ENDUPRIGHT: Chem.BondDir.ENDDOWNRIGHT,
    Chem.BondDir.ENDDOWNRIGHT: Chem.BondDir.ENDUPRIGHT,
}
FLIPPED_STEREO = {
    Chem.BondStereo.STEREOCIS: Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOTRANS: Chem.BondStereo.STEREOCIS,
}
# E/Z of stereo atoms that are the CIP-highest neighbours, as geometry: E = trans
GEOMETRIC = {
    Chem.BondStereo.STEREOE: Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOZ: Chem.BondStereo.STEREOCIS,
}


class SwapError(ValueError):
    """A swap that cannot be done as given (selector, fragment, valences, settings)."""


@dataclass(frozen=True)
class Swap:
    """
    What to replace, and by what. One selector:

    - site: the map number of a terminal H or dummy atom that leaves (see
      label_hydrogen); the fragment binds to its neighbour.
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
        stereo: The configuration of stereo that the swap creates at kept atoms,
            instead of the one that the reference geometry gives it (which hydrogen
            leaves): {atom: "R" or "S"} for a centre and {(atom, atom): "E" or "Z"}
            for a double bond, by the CIP rules of the result; {(atom, atom): "cis" or
            "trans"} for the two atoms that the fragment binds with at the ends of a
            double bond (a ring closed on it). Atoms are reference indices of kept
            atoms. Kept neighbours on the other side are sampled to fit; every
            conformer is checked.
        mode: "append": new atoms take the slots of removed ones and the rest are
            appended, so the kept atoms keep their indices when the
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
    stereo: Optional[Mapping] = None

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
        for key, word in (self.stereo or {}).items():
            pair = isinstance(key, (tuple, list)) and len(key) == 2
            if not (is_integer(key) or (pair and all(is_integer(i) for i in key))):
                raise SwapError(
                    f"stereo is given for an atom or a pair of atoms, not {key!r}."
                )
            if pair and word not in BOND_STEREO:
                raise SwapError(
                    f"stereo of the double bond {tuple(key)} is cis, trans, E or Z, "
                    f"not {word!r}."
                )
            if not pair and word not in ATOM_STEREO:
                raise SwapError(f"stereo of atom {key} is R or S, not {word!r}.")
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
        positioned: The new atoms with coordinates: all if placed; else the fragment
            atoms that replace a removed atom, at its position if of the same element
            (a donor atom of a ligand) or along its bond (the others are at the origin
            until sampled).
        warnings: Diagnostics, also logged.
        fragment: The fragment with its hydrogens and dummies (for further poses,
            see racerts.embed.rigid_attach).
        fragment_map: Fragment index -> new index of its atoms (not the dummies).
        replaced: Removed atom (reference index) -> the new atom that took its bond
            to a kept atom; it takes the removed atom's role in a task (see
            index_map).
        restraints, lost_contacts: The restraints given to apply_swap between kept
            atoms, in new indices, and the labels of the others.
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
    restraints: Optional[object] = None
    lost_contacts: List[str] = field(default_factory=list)

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


def _ends(bond) -> Tuple[int, int]:
    return bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()


def _perceive_stereo(mol: Chem.Mol, keep_bonds) -> None:
    """
    Legacy stereo perception from the bond stereo that is set: bond directions are
    rebuilt from it alone; a double bond whose directions come only from its
    neighbours (not in keep_bonds) is left unspecified.
    """
    given = {}
    for bond in mol.GetBonds():
        bond.SetBondDir(Chem.BondDir.NONE)
        if frozenset(_ends(bond)) in keep_bonds and bond.GetStereo() in FLIPPED_STEREO:
            given[bond.GetIdx()] = (list(bond.GetStereoAtoms()), bond.GetStereo())
    Chem.SetDoubleBondNeighborDirections(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    for bond in mol.GetBonds():
        if (
            bond.GetBondType() == Chem.BondType.DOUBLE
            and bond.GetStereo() not in UNSPECIFIED_BOND
            and frozenset(_ends(bond)) not in keep_bonds
        ):
            bond.SetStereo(Chem.BondStereo.STEREONONE)
        elif bond.GetIdx() in given and bond.GetStereo() in UNSPECIFIED_BOND:
            # RDKit perceives no cis/trans in a ring of fewer than eight atoms: the
            # bond keeps what was set, with its stereo atoms
            atoms, stereo = given[bond.GetIdx()]
            if len(atoms) == 2:
                bond.SetStereoAtoms(*atoms)
                bond.SetStereo(stereo)


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
            bonds.add(frozenset(_ends(mol.GetBondWithIdx(info.centeredOn))))
    # RDKit leaves out the double bonds in rings of fewer than eight atoms.
    ranks = list(Chem.CanonicalRankAtoms(mol, breakTies=False))
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE or not bond.IsInRing():
            continue
        ends = _ends(bond)
        sides = [
            [
                ranks[n.GetIdx()]
                for n in mol.GetAtomWithIdx(end).GetNeighbors()
                if n.GetIdx() != other
            ]
            for end, other in (ends, ends[::-1])
        ]
        if all(side and len(set(side)) == len(side) for side in sides):
            bonds.add(frozenset(ends))
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
        frozenset(_ends(b))
        for b in probe.GetBonds()
        if b.GetBondType() == Chem.BondType.DOUBLE
        and b.GetStereo() not in UNSPECIFIED_BOND
    }
    found = geometry_tags(one, one.GetConformer().GetId(), atoms_found | candidates[0])
    atoms = {i: tag for i, tag in found.items() if tag in TETRAHEDRAL}
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
        if np.linalg.norm(positions[a] - positions[b]) < 1e-8:
            continue  # atoms without coordinates yet (at the origin)
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


def _settle_stereo(result, anchored, carried_atoms, carried_bonds) -> set:
    """
    Every stereo element of the result that the graphs leave unspecified (not carried
    over from the reference or the fragment) takes the configuration of the reference
    geometry where it defines it: the element is anchored (kept atoms or atoms that
    replace one) with all its neighbours, or, for a centre, with all but one and at
    least three (the missing one then lies opposite the three: a phosphine made a
    phosphine oxide), and every reference conformer agrees (else a warning). Which
    hydrogen is replaced thus chooses the configuration of a prochiral site. Returns
    the double bonds (atom pairs) that were set.
    """
    if not anchored:
        return set()
    candidates = _stereo_candidates(result)

    def neighbours(i):
        return [n.GetIdx() for n in result.GetAtomWithIdx(i).GetNeighbors()]

    atom_values, bond_values, virtual = {}, {}, {}
    for i in candidates[0] - carried_atoms:
        loose = [k for k in neighbours(i) if k not in anchored]
        if i in anchored and len(loose) == 1 and len(neighbours(i)) - 1 >= 3:
            virtual[i] = loose[0]
    for conf in result.GetConformers():
        probe = Chem.Mol(result, False, conf.GetId())
        positions = probe.GetConformer().GetPositions()
        for i in set(range(result.GetNumAtoms())) - anchored:
            # New atoms without coordinates lie on top of each other at the origin,
            # which RDKit cannot read stereo around. Apart from each other they can be
            # read past: no element next to them is taken from the geometry.
            spot = (50.0 + 1.7 * i, 0.9 * (i % 5), 1.3 * (i % 7))
            probe.GetConformer().SetAtomPosition(i, spot)
        for i, k in virtual.items():  # the missing neighbour opposite the others
            arms = [positions[j] - positions[i] for j in neighbours(i) if j != k]
            arms = [arm / np.linalg.norm(arm) for arm in arms]
            away = -np.sum(arms, axis=0)
            if np.linalg.norm(away) > 0.3:
                spot = positions[i] + 1.5 * away / np.linalg.norm(away)
                probe.GetConformer().SetAtomPosition(k, spot.tolist())
        atoms, bonds = _stereo_3d(probe, probe.GetConformer().GetId(), candidates)
        for i in candidates[0] | set(atoms):
            atom_values.setdefault(i, []).append(atoms.get(i))
        for pair in candidates[1] | set(bonds):
            bond_values.setdefault(pair, []).append(bonds.get(pair))
    n = result.GetNumConformers()

    def defined(atoms):
        return all(
            i in anchored
            and all(k in anchored or virtual.get(i) == k for k in neighbours(i))
            for i in atoms
        )

    open_atoms = {
        i: values
        for i, values in atom_values.items()
        if i not in carried_atoms and defined([i])
    }
    open_bonds = {
        pair: values
        for pair, values in bond_values.items()
        if pair not in carried_bonds
        and set(pair) <= anchored
        and all(k in anchored for i in pair for k in neighbours(i))
    }
    agreed_atoms = {
        i: values[0]
        for i, values in open_atoms.items()
        if len(values) == n and values[0] is not None and len(set(values)) == 1
    }
    agreed_bonds = {
        pair: values[0]
        for pair, values in open_bonds.items()
        if len(values) == n and values[0] is not None and len(set(values)) == 1
    }
    disputed = sorted(
        {i for i, v in open_atoms.items() if i not in agreed_atoms and any(v)}
        | {
            i
            for p, v in open_bonds.items()
            if p not in agreed_bonds and any(v)
            for i in p
        }
    )
    if disputed:
        logger.warning(
            "The reference conformers disagree on the configuration of the "
            "unspecified stereo at atoms %s; it stays unspecified.",
            disputed,
        )
    for i, tag in agreed_atoms.items():
        result.GetAtomWithIdx(i).SetChiralTag(tag)
    for pair, (stereo_atoms, stereo) in agreed_bonds.items():
        bond = result.GetBondBetweenAtoms(*pair)
        if bond.GetBeginAtomIdx() != min(pair):  # RDKit: the begin atom's side first
            stereo_atoms = stereo_atoms[::-1]
        bond.SetStereoAtoms(*stereo_atoms)
        bond.SetStereo(stereo)
    return set(agreed_bonds)


def _given_stereo(swap, result, ref_to_new, attachments, frag_to_new) -> set:
    """
    Sets the configurations of swap.stereo in the result, over what the geometry gave,
    and returns the double bonds (atom pairs) among them.
    """
    bound = {}  # kept atom -> the fragment atoms bound to it
    for a in attachments:
        bound.setdefault(ref_to_new[a.kept], []).append(frag_to_new[a.root])

    def kept(i):
        if int(i) not in ref_to_new:
            raise SwapError(f"stereo: atom {i} leaves in the swap or is no atom.")
        return ref_to_new[int(i)]

    bonds = set()
    for key, word in (swap.stereo or {}).items():
        if is_integer(key):
            _set_centre(result, kept(key), key, word)
            continue
        a, b = (kept(i) for i in key)
        bond = result.GetBondBetweenAtoms(a, b)
        if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
            raise SwapError(
                f"stereo: atoms {tuple(key)} are not joined by a double bond in the "
                "result."
            )
        if word in ("cis", "trans"):
            if len(bound.get(a, [])) != 1 or len(bound.get(b, [])) != 1:
                raise SwapError(
                    f"stereo: cis and trans of {tuple(key)} are those of the two atoms "
                    "that the fragment binds with at both ends of the double bond; "
                    "here give E or Z."
                )
            ends, cis = [bound[a][0], bound[b][0]], word == "cis"
        else:
            ends, cis = _highest_neighbours(result, a, b, key), word == "Z"
        if bond.GetBeginAtomIdx() != a:  # RDKit: the begin atom's side first
            ends.reverse()
        bond.SetStereoAtoms(*ends)
        bond.SetStereo(
            Chem.BondStereo.STEREOCIS if cis else Chem.BondStereo.STEREOTRANS
        )
        bonds.add(frozenset((a, b)))
    return bonds


def _highest_neighbours(result, a, b, key) -> list:
    """The neighbour of highest CIP rank at each end of the double bond a=b: E and Z
    are trans and cis of these two."""
    probe = Chem.Mol(result)
    Chem.AssignStereochemistry(
        probe, cleanIt=False, force=True, flagPossibleStereoCenters=True
    )
    ends = []
    for end, other in ((a, b), (b, a)):
        ranked = sorted(
            (int(n.GetProp("_CIPRank")), n.GetIdx())
            for n in probe.GetAtomWithIdx(end).GetNeighbors()
            if n.GetIdx() != other
        )
        if not ranked or (len(ranked) == 2 and ranked[0][0] == ranked[1][0]):
            raise SwapError(
                f"stereo: the double bond {tuple(key)} has no E or Z: an end has no "
                "substituent, or two equal substituents."
            )
        ends.append(ranked[-1][1])
    return ends


def _set_centre(result, atom: int, key, word: str) -> None:
    """The chiral tag of atom for which its CIP label in the result is word."""
    target = result.GetAtomWithIdx(atom)
    for tag in TETRAHEDRAL:
        target.SetChiralTag(tag)
        probe = Chem.Mol(result)
        rdCIPLabeler.AssignCIPLabels(probe)
        labelled = probe.GetAtomWithIdx(atom)
        if labelled.HasProp("_CIPCode") and labelled.GetProp("_CIPCode") == word:
            return
    target.SetChiralTag(Chem.ChiralType.CHI_UNSPECIFIED)
    raise SwapError(f"stereo: atom {key} is no stereocentre in the result (no R or S).")


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
            start = seeds.derive(seed, "swap fragment")
            if AllChem.EmbedMolecule(embedded, randomSeed=start) < 0:
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
        else:  # roots that replace an atom: at its position (same element) or along
            for b in attachments:  # its bond at the sum of the covalent radii
                if b.partner is None:
                    continue
                new_positions[frag_to_new[b.root]] = _root_position(
                    mol, fragment, b, positions
                )
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


def _root_position(mol, fragment, attachment, positions) -> np.ndarray:
    anchor = positions[attachment.kept]
    direction = positions[attachment.partner] - anchor
    norm = np.linalg.norm(direction)
    if not np.isfinite(norm) or norm < 1e-8:
        raise SwapError("The attachment has coincident or invalid coordinates.")
    return anchor + direction / norm * _bond_length(mol, fragment, attachment, norm)


def rotation_between(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The proper rotation taking the direction source onto target."""
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


def label_hydrogen(mol: Chem.Mol, anchor: int, label: int) -> Chem.Mol:
    """
    A copy of mol whose only hydrogen on anchor carries the map number label (a site
    for Swap(site=label)). An anchor with several hydrogens raises: prochiral sites are
    chosen by the caller.
    """
    if isinstance(label, bool) or not isinstance(label, int):
        raise TypeError("Site labels are positive integers.")
    if label <= 0:
        raise SwapError("Site labels are positive integers.")
    if isinstance(anchor, bool) or not 0 <= anchor < mol.GetNumAtoms():
        raise IndexError(f"No atom {anchor} (the molecule has {mol.GetNumAtoms()}).")
    if any(atom.GetAtomMapNum() == label for atom in mol.GetAtoms()):
        raise SwapError(f"Map number {label} is already used.")
    hydrogens = [
        a for a in mol.GetAtomWithIdx(anchor).GetNeighbors() if a.GetAtomicNum() == 1
    ]
    if len(hydrogens) != 1:
        raise SwapError(
            f"Atom {anchor} has {len(hydrogens)} explicit hydrogens, not exactly one."
        )
    if hydrogens[0].GetAtomMapNum():
        raise SwapError(f"The hydrogen of atom {anchor} already has an atom map.")
    result = Chem.Mol(mol)
    result.GetAtomWithIdx(hydrogens[0].GetIdx()).SetAtomMapNum(label)
    return result


def substitute_groups(
    mol: Chem.Mol, substitutions: Mapping[int, str], random_seed: int = 0xF00D
) -> Chem.Mol:
    """
    Each labelled terminal H or dummy (map number -> "[*]R") becomes the group R,
    grafted rigidly on every conformer; "[H]" caps a site with a hydrogen at the sum
    of the covalent radii along its bond. Kept atoms keep their indices and
    coordinates; the group's first atom takes the label. The properties of the
    molecule and its conformers are cleared (results do not carry over), except its
    charge and multiplicity as the swaps settle them; without substitutions, an
    unchanged copy.
    """
    result = Chem.Mol(mol)
    if not substitutions:
        return result
    for label, smiles in substitutions.items():
        if isinstance(label, bool) or not isinstance(label, int) or label <= 0:
            raise SwapError("Site labels are positive integers.")
        if smiles == "[H]":
            result = _cap(result, label)
            continue
        result = apply_swap(result, Swap(smiles, site=label), seed=random_seed).mol
    state = {
        key: result.GetIntProp(key)
        for key in ("charge", "multiplicity")
        if result.HasProp(key)
    }
    for carrier in [result, *result.GetConformers()]:
        for key in list(
            carrier.GetPropNames(includePrivate=True, includeComputed=False)
        ):
            carrier.ClearProp(key)
    for key, value in state.items():
        result.SetIntProp(key, value)
    return result


def _cap(mol: Chem.Mol, label: int) -> Chem.Mol:
    """The site with map number label as a hydrogen, at the covalent distance."""
    site = _site(mol, label)
    anchor = mol.GetAtomWithIdx(site).GetNeighbors()[0].GetIdx()
    capped = Chem.RWMol(mol)
    hydrogen = Chem.Atom(1)
    hydrogen.SetAtomMapNum(label)
    capped.ReplaceAtom(site, hydrogen)
    table = Chem.GetPeriodicTable()
    length = table.GetRcovalent(1) + table.GetRcovalent(
        mol.GetAtomWithIdx(anchor).GetAtomicNum()
    )
    for conf in capped.GetConformers():
        positions = conf.GetPositions()
        direction = positions[site] - positions[anchor]
        norm = np.linalg.norm(direction)
        if not np.isfinite(norm) or norm < 1e-8:
            raise SwapError("The attachment has coincident or invalid coordinates.")
        conf.SetAtomPosition(
            site, (positions[anchor] + length * direction / norm).tolist()
        )
    result = capped.GetMol()
    try:
        Chem.SanitizeMol(result)
    except Exception as error:
        raise SwapError(f"The capped molecule is invalid: {error}") from None
    return result
