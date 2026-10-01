"""Swaps: graph surgery (apply_swap) and catmlp's substitutions."""

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from racerts.system.stereo import StereoCheck
from racerts.system.swap import Swap, apply_swap


def embedded(smiles, n=1, seed=0):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ids = AllChem.EmbedMultipleConfs(mol, n, randomSeed=seed)
    assert len(ids) == n
    AllChem.MMFFOptimizeMoleculeConfs(mol)
    return mol


def identity(mol):
    """Canonical SMILES without hydrogens and map numbers."""
    copy = Chem.Mol(mol)
    for atom in copy.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(Chem.RemoveHs(copy))


def canonical(smiles):
    return Chem.MolToSmiles(Chem.MolFromSmiles(smiles))


@pytest.fixture(scope="module")
def methylbiphenyl():
    """2-methylbiphenyl (C13H12, 25 atoms), methyl C = atom 0, ipso C = atom 1."""
    return embedded("Cc1ccccc1-c1ccccc1")


BUTYL = canonical("CCCCc1ccccc1-c1ccccc1")


def test_renumber_puts_the_kept_atoms_first(methylbiphenyl):
    result = apply_swap(
        methylbiphenyl, Swap("[*:1]CCCC", remove_atoms=[0], mode="renumber")
    )
    assert identity(result.mol) == BUTYL
    assert result.conserved == list(range(21))
    assert result.new_atoms == list(range(21, 34))
    assert result.ref_to_new[1] == 0 and result.attachments == [(0, 21)]


def test_reverse_swap_restores_the_graph(methylbiphenyl):
    forward = apply_swap(methylbiphenyl, Swap("[*:1]CCCC", old_fragment="[CH3][c:1]"))
    back = apply_swap(forward.mol, Swap("[*:1]C", old_fragment="[CH2;!R]([CH2])[c:1]"))
    assert identity(back.mol) == identity(methylbiphenyl)
    # 13 atoms leave, 4 come: the other 9 slots close up; kept atoms keep their order.
    assert back.mol.GetNumAtoms() == methylbiphenyl.GetNumAtoms()
    kept = sorted(back.ref_to_new)
    assert [back.ref_to_new[i] for i in kept] == sorted(back.ref_to_new.values())


def test_ambiguous_and_invalid_selectors(methylbiphenyl):
    mol = methylbiphenyl
    with pytest.raises(ValueError, match="different groups"):
        apply_swap(mol, Swap("[*:1]C", old_fragment="[H][c:1]"))
    with pytest.raises(ValueError, match="no substituent"):
        apply_swap(mol, Swap("[*:1]C", old_fragment="[N][c:1]"))
    with pytest.raises(ValueError, match="exactly one selector"):
        Swap("[*]C")
    with pytest.raises(ValueError, match="exactly one selector"):
        Swap("[*]C", site=1, remove_atoms=[0])
    with pytest.raises(ValueError, match="mode"):
        Swap("[*]C", site=1, mode="keep")
    with pytest.raises(ValueError, match="Invalid atoms"):
        apply_swap(mol, Swap("[*]C", remove_atoms=[99]))
    with pytest.raises(ValueError, match="groups"):
        apply_swap(mol, Swap("[*]C", center=1, substructure=5))
    with pytest.raises(ValueError, match="attach_map"):
        apply_swap(mol, Swap("[*:1]C", remove_atoms=[1]))  # three cut bonds


def _stereo_agrees(mol):
    check = StereoCheck(mol)
    return all(check.mismatch(conf) is None for conf in mol.GetConformers())


# Metal complexes: dative bonds (the graphs).

PD_DMPE = "CP(C)(CCP(C)(C)->[Pd]1(Cl)Cl)->1"


def test_bidentate_to_bidentate_with_dative_bonds():
    mol = Chem.AddHs(Chem.MolFromSmiles(PD_DMPE))
    pd = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Pd")
    dmpe = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in "CP"]
    dppe = "[*:1]P(c1ccccc1)(c1ccccc1)CCP([*:2])(c1ccccc1)c1ccccc1"
    result = apply_swap(
        mol,
        Swap(
            dppe,
            remove_atoms=dmpe,
            attach_map={1: pd, 2: pd},
            bond_types={1: "dative", 2: "dative"},
        ),
    )
    datives = [
        b for b in result.mol.GetBonds() if b.GetBondType() == Chem.BondType.DATIVE
    ]
    assert len(datives) == 2
    assert {b.GetEndAtom().GetSymbol() for b in datives} == {"Pd"}
    assert {b.GetBeginAtom().GetSymbol() for b in datives} == {"P"}
    assert Chem.MolToSmiles(Chem.RemoveHs(result.mol)) == Chem.MolToSmiles(
        Chem.MolFromSmiles(
            "c1ccc(cc1)P1(c2ccccc2)CCP(c2ccccc2)(c2ccccc2)->[Pd]<-1(Cl)Cl"
        )
    )
    assert not result.placed  # two attachments: the new atoms are sampled later


def test_addition_without_removal():
    mol = Chem.AddHs(Chem.MolFromSmiles("Cl[Pd]Cl"))
    result = apply_swap(
        mol,
        Swap("[*:1]<-P(C)(C)C", remove_atoms=[], attach_map={1: 1}),
    )
    assert identity(result.mol) == canonical("CP(C)(C)->[Pd](Cl)Cl")
    assert result.conserved == [0, 1, 2]


def test_single_bond_to_a_phosphine_needs_the_bond_type():
    # A plain single bond makes a P(V) with an extra H: the dative bond is explicit.
    mol = Chem.AddHs(Chem.MolFromSmiles("Cl[Pd]Cl"))
    plain = apply_swap(mol, Swap("[*:1]P(C)(C)C", remove_atoms=[], attach_map={1: 1}))
    assert "[PH]" in Chem.MolToSmiles(Chem.RemoveHs(plain.mol))
    dative = apply_swap(
        mol,
        Swap(
            "[*:1]P(C)(C)C",
            remove_atoms=[],
            attach_map={1: 1},
            bond_types={1: "dative"},
        ),
    )
    assert "[PH]" not in Chem.MolToSmiles(Chem.RemoveHs(dative.mol))


def test_small_kept_share_warns(caplog):
    mol = embedded("CO")
    result = apply_swap(mol, Swap("[*]CCCCCCCC", remove_atoms=[0]))
    assert result.warnings and "close to a new embedding" in result.warnings[0]


# Regression tests.


@pytest.mark.parametrize(
    "fragment", ["[*:1][C@](F)(Cl)Br", "F[C@H]([*:1])Cl", "[*:1]/C=C/F"]
)
def test_bond_types_keep_the_fragment_stereo(fragment):
    mol = embedded("CCl")
    plain = apply_swap(mol, Swap(fragment, old_fragment="[C:1]Cl"))
    typed = apply_swap(
        mol, Swap(fragment, old_fragment="[C:1]Cl", bond_types={1: "single"})
    )
    expected = canonical(fragment.replace("[*:1]", "C"))
    assert identity(plain.mol) == identity(typed.mol) == expected
    assert _stereo_agrees(typed.mol)


def test_a_dative_bond_type_keeps_the_donor_stereo():
    pdcl2 = Chem.AddHs(Chem.MolFromSmiles("Cl[Pd]Cl"))
    written = apply_swap(
        pdcl2, Swap("[*:1]<-[P@](C)(CC)c1ccccc1", remove_atoms=[], attach_map={1: 1})
    )
    typed = apply_swap(
        pdcl2,
        Swap(
            "[*:1][P@](C)(CC)c1ccccc1",
            remove_atoms=[],
            attach_map={1: 1},
            bond_types={1: "dative"},
        ),
    )
    assert identity(written.mol) == identity(typed.mol)


def test_e_z_survives_when_a_stereo_atom_leaves():
    # Cl (a stereo atom of the double bond) leaves without replacement: the other
    # neighbour of that end carries the E/Z.
    reference = Chem.MolFromSmiles("F/C(Cl)=C/CC")
    result = apply_swap(reference, Swap("[*:1]Br", remove_atoms=[2], attach_map={1: 5}))
    assert identity(result.mol) == canonical("F/C=C/CCBr")
