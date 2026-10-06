"""Stereo of conformers compared with the stereo their graph specifies."""

from typing import Collection, Dict, List, Optional

import numpy as np
from rdkit import Chem

UNSPECIFIED_BOND = (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
TETRAHEDRAL = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)
# Below this volume of the three unit vectors to its neighbours a centre is flat.
FLAT_VOLUME = 0.05


def geometry_tags(
    mol: Chem.Mol, conf_id: int, atoms: Collection[int]
) -> Dict[int, Chem.ChiralType]:
    """
    The chiral tags that conformer conf_id gives the atoms (unspecified where it gives
    none). RDKit reads them from the structure (AssignAtomChiralTagsFromStructure),
    except for a stereocentre with three neighbours and a lone pair (P, As, a ring N),
    which it leaves without a tag. There the tag follows from the volume of the three
    neighbours in the order of the atom's bonds: counter-clockwise if it is positive,
    as RDKit has it for a centre with an implicit hydrogen.
    """
    probe = Chem.Mol(mol)
    Chem.AssignAtomChiralTagsFromStructure(
        probe, confId=conf_id, replaceExistingTags=True
    )
    positions = mol.GetConformer(conf_id).GetPositions()
    tags = {}
    for i in atoms:
        atom = probe.GetAtomWithIdx(i)
        tags[i] = atom.GetChiralTag()
        if tags[i] in TETRAHEDRAL or atom.GetDegree() != 3:
            continue
        arms = [positions[b.GetOtherAtomIdx(i)] - positions[i] for b in atom.GetBonds()]
        arms = [arm / np.linalg.norm(arm) for arm in arms]
        volume = float(np.dot(arms[0], np.cross(arms[1], arms[2])))
        if abs(volume) > FLAT_VOLUME:
            tags[i] = TETRAHEDRAL[1] if volume > 0 else TETRAHEDRAL[0]
    return tags


class StereoCheck:
    """
    Compares conformers with the stereo that graph specifies (unspecified stereo is a
    wildcard), except that of the exempt atoms.

    - Stereocentres: the chiral tag from the geometry (geometry_tags) must equal the
      graph's. Tags refer to the order of the neighbours, which a conformer of the same
      graph shares, so this also covers stereo without CIP labels, such as cis/trans on
      rings, and centres with a lone pair.
    - Double bonds: the CIP label (E/Z) from the geometry must equal the graph's, where
      the graph has one.
    """

    def __init__(self, graph: Chem.Mol, exempt: Collection[int] = ()):
        exempt = set(exempt)
        self.graph = Chem.Mol(graph, True)  # without conformers and properties
        self.atoms: Dict[int, Chem.ChiralType] = {
            a.GetIdx(): a.GetChiralTag()
            for a in self.graph.GetAtoms()
            if a.GetChiralTag() in TETRAHEDRAL and a.GetIdx() not in exempt
        }
        specified = [
            b.GetIdx()
            for b in self.graph.GetBonds()
            if b.GetStereo() not in UNSPECIFIED_BOND
            and b.GetBeginAtomIdx() not in exempt
            and b.GetEndAtomIdx() not in exempt
        ]
        self.bonds = _bond_configurations(self.graph, specified) if specified else {}

    def __bool__(self) -> bool:
        return bool(self.atoms or self.bonds)

    def mismatch(self, conf: Chem.Conformer) -> Optional[str]:
        """What conf has inverted ("stereo of atom 1 inverted"), or None."""
        if not self:
            return None
        geometry = Chem.Mol(self.graph)
        conf_id = geometry.AddConformer(Chem.Conformer(conf), assignId=True)
        wrong = []
        if self.atoms:
            found = geometry_tags(geometry, conf_id, self.atoms)
            wrong += [f"atom {i}" for i, tag in self.atoms.items() if found[i] != tag]
        if self.bonds:
            positions = conf.GetPositions()
            for i, (quad, cis) in self.bonds.items():
                if (abs(dihedral(positions, *quad)) < 90.0) != cis:
                    wrong.append(f"bond {i}")
        return f"stereo of {', '.join(wrong)} inverted" if wrong else None


def reference_tags(mol: Chem.Mol, reference: Chem.Mol, atoms: Collection[int]) -> dict:
    """The chiral tags that the reference geometry gives the atoms of mol."""
    probe = Chem.Mol(mol, True)
    conf_id = probe.AddConformer(
        Chem.Conformer(reference.GetConformer()), assignId=True
    )
    return geometry_tags(probe, conf_id, atoms)


def reference_fixed(mol: Chem.Mol, frozen) -> List[int]:
    """
    The frozen (hard or soft) atoms whose configuration the reference fixes: those
    with at most one neighbour that is not frozen. With two or more free neighbours
    the configuration is open, and the chiral tag has to set it.
    """
    held = set(frozen.hard) | set(frozen.soft)
    return [
        i
        for i in (*frozen.hard, *frozen.soft)
        if sum(n.GetIdx() not in held for n in mol.GetAtomWithIdx(i).GetNeighbors())
        <= 1
    ]


def stereo_anchors(mol: Chem.Mol, frozen) -> List[int]:
    """
    Free atoms that alone set the configuration of a frozen stereocentre: the only
    neighbour of a tagged hard (or soft) atom that is not frozen. With the centre and
    its other neighbours fixed, such a substituent on the wrong side is the other
    stereoisomer; in "frozen_first" mode the coordinate-map embedder places it at the
    reference too, and Refine(stereo_anchors=True) holds it there.
    """
    held = set(frozen.hard) | set(frozen.soft)
    anchors = set()
    for i in held:
        atom = mol.GetAtomWithIdx(i)
        if atom.GetChiralTag() not in TETRAHEDRAL:
            continue
        free = [n.GetIdx() for n in atom.GetNeighbors() if n.GetIdx() not in held]
        if len(free) == 1:
            anchors.add(free[0])
    return sorted(anchors)


CIS = (Chem.BondStereo.STEREOCIS, Chem.BondStereo.STEREOZ)
TRANS = (Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOE)


def _bond_configurations(graph: Chem.Mol, bonds) -> dict:
    """
    Bond index -> ((x, a, b, y), cis): the stereo atoms x and y of the double bond a=b
    and whether they are cis. E/Z as RDKit perceives it (its stereo atoms are then the
    neighbours of highest CIP rank); a bond whose cis/trans is set with its stereo atoms
    but that RDKit does not perceive (a ring of fewer than eight atoms) as it is set.
    """
    cleaned = Chem.Mol(graph)
    Chem.AssignStereochemistry(cleaned, cleanIt=True, force=True)
    found = {}
    for i in bonds:
        for source in (cleaned, graph):
            bond = source.GetBondWithIdx(i)
            atoms = list(bond.GetStereoAtoms())
            if bond.GetStereo() in UNSPECIFIED_BOND or len(atoms) != 2:
                continue
            quad = (atoms[0], bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), atoms[1])
            found[i] = (quad, bond.GetStereo() in CIS)
            break
    return found


def dihedral(positions, i, j, k, m) -> float:
    """The dihedral angle i-j-k-m in degrees, between -180 and 180."""
    b0, b1, b2 = (
        positions[i] - positions[j],
        positions[k] - positions[j],
        positions[m] - positions[k],
    )
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return float(np.degrees(np.arctan2(np.dot(np.cross(b1, v), w), np.dot(v, w))))


SMALL_RING = 8  # RDKit has cis and trans for the double bonds of rings from this size


def trans_in_small_rings(mol: Chem.Mol) -> List[tuple]:
    """
    The double bonds that are set trans in a ring of fewer than SMALL_RING atoms, as
    (begin atom, end atom, ring size): configurations that RDKit's stereo perception
    does not have, that its embedding follows only without torsion preferences, and
    that a force field may not hold (MMFF holds a ring of seven, not of six).
    """
    found = []
    rings = mol.GetRingInfo().AtomRings()
    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        atoms = list(bond.GetStereoAtoms())
        if stereo not in (*CIS, *TRANS) or len(atoms) != 2:
            continue
        a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        for ring in rings:
            if a in ring and b in ring and len(ring) < SMALL_RING:
                outside = sum(atom not in ring for atom in atoms)
                if (stereo in TRANS) != (outside % 2 == 1):
                    found.append((a, b, len(ring)))
                    break
    return found
