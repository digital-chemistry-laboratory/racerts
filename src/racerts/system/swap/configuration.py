"""
The stereo of the new graph: read from the geometry of the reference, carried over
from the two graphs, or given by the swap.
"""

import logging
from typing import Sequence, Tuple

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdCIPLabeler

from racerts.utils.checks import is_integer

from ..stereo import TETRAHEDRAL, UNSPECIFIED_BOND, geometry_tags
from .model import SwapError

logger = logging.getLogger(__name__)

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


def _odd(order: Sequence[int]) -> bool:
    inversions = sum(
        order[a] > order[b] for a in range(len(order)) for b in range(a + 1, len(order))
    )
    return bool(inversions % 2)
