"""What a swap is and what it returns."""

from dataclasses import dataclass, field
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from rdkit import Chem

from racerts.utils.checks import is_integer

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
