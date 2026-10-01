"""Stereo of conformers compared with the stereo their graph specifies."""

from typing import Collection, Dict, Optional

from rdkit import Chem

UNSPECIFIED_BOND = (Chem.BondStereo.STEREONONE, Chem.BondStereo.STEREOANY)
TETRAHEDRAL = (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW)


class StereoCheck:
    """
    Compares conformers with the stereo that graph specifies (unspecified stereo is a
    wildcard), except that of the exempt atoms.

    - Stereocentres: the chiral tag from the geometry
      (AssignAtomChiralTagsFromStructure) must equal the graph's. Tags refer to the
      order of the neighbours, which a conformer of the same graph shares, so this also
      covers stereo without CIP labels, such as cis/trans on rings.
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
        labels = _bond_labels(self.graph) if specified else {}
        self.bonds = {i: labels[i] for i in specified if i in labels}

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
            probe = Chem.Mol(geometry)
            Chem.AssignAtomChiralTagsFromStructure(
                probe, confId=conf_id, replaceExistingTags=True
            )
            wrong += [
                f"atom {i}"
                for i, tag in self.atoms.items()
                if probe.GetAtomWithIdx(i).GetChiralTag() != tag
            ]
        if self.bonds:
            probe = Chem.Mol(geometry)
            for bond in probe.GetBonds():
                bond.SetStereo(Chem.BondStereo.STEREONONE)
            Chem.AssignStereochemistryFrom3D(probe, confId=conf_id)
            found = _bond_labels(probe)
            wrong += [
                f"bond {i}" for i, label in self.bonds.items() if found.get(i) != label
            ]
        return f"stereo of {', '.join(wrong)} inverted" if wrong else None


def stereo_mismatch(
    graph: Chem.Mol, conf: Chem.Conformer, exempt: Collection[int] = ()
) -> Optional[str]:
    """
    The specified stereo of graph (stereocentres and double bonds, except those of
    the exempt atoms) that conformer conf does not have, as a reason ("stereo of
    atom 1 inverted"), or None; see StereoCheck, which is faster for many conformers.
    """
    return StereoCheck(graph, exempt).mismatch(conf)


def _bond_labels(mol: Chem.Mol) -> Dict[int, str]:
    """The E/Z labels of the double bonds with stereo, as RDKit assigns them."""
    mol = Chem.Mol(mol)
    Chem.AssignStereochemistry(mol, cleanIt=True, force=True)
    return {
        b.GetIdx(): str(b.GetStereo())
        for b in mol.GetBonds()
        if b.GetStereo() not in UNSPECIFIED_BOND
    }
