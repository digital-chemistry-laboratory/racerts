"""Swaps: graph surgery (apply_swap) and catmlp's substitutions."""

import numpy as np
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


def test_methyl_to_butyl(methylbiphenyl):
    mol = methylbiphenyl
    result = apply_swap(mol, Swap("[*:1]CCCC", old_fragment="[CH3][c:1]"))
    assert identity(result.mol) == BUTYL
    # 25 - CH3 (4) = 21 kept atoms; C4H9 = 13 new atoms; C16H18.
    assert len(result.conserved) == 21 and len(result.new_atoms) == 13
    assert result.mol.GetNumAtoms() == 34
    assert result.attachments == [(1, 0)]  # the butyl C1 takes the slot of the methyl C
    # append: the kept atoms keep their indices, and their coordinates.
    assert all(i == k for i, k in result.ref_to_new.items())
    kept = sorted(result.ref_to_new)
    np.testing.assert_array_equal(
        result.mol.GetConformer().GetPositions()[kept],
        mol.GetConformer().GetPositions()[kept],
    )
    # junction: the ipso C and its kept neighbours.
    assert result.junction == sorted(
        [
            1,
            *[
                n.GetIdx()
                for n in mol.GetAtomWithIdx(1).GetNeighbors()
                if n.GetIdx() != 0
            ],
        ]
    )
    assert result.placed
    # C for C: the bond length of the reference (scaled by covalent radii otherwise).
    positions = mol.GetConformer().GetPositions()
    length = np.linalg.norm(result.mol.GetConformer().GetPositions()[0] - positions[1])
    assert length == pytest.approx(np.linalg.norm(positions[0] - positions[1]))
    assert not result.warnings


def test_selectors_are_equivalent(methylbiphenyl):
    mol = methylbiphenyl
    methyl_h = [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(0).GetNeighbors()
        if n.GetAtomicNum() == 1
    ]
    groups_of_ipso = sorted(
        (n.GetIdx() for n in mol.GetAtomWithIdx(1).GetNeighbors()),
    )
    swaps = [
        Swap("[*:1]CCCC", old_fragment="[CH3][c:1]"),
        Swap("[*:1]CCCC", remove_atoms=[0]),  # its hydrogens leave with it
        Swap("[*:1]CCCC", remove_atoms=[0, *methyl_h]),
        Swap("[*:1]CCCC", center=1, substructure=0),  # the group of atom 0
        Swap("[*]CCCC", remove_atoms=[0], attach_map={1: 1}),
    ]
    assert groups_of_ipso[0] == 0
    results = [apply_swap(mol, swap) for swap in swaps]
    for result in results:
        assert identity(result.mol) == BUTYL
        assert result.ref_to_new == results[0].ref_to_new
        np.testing.assert_allclose(
            result.mol.GetConformer().GetPositions(),
            results[0].mol.GetConformer().GetPositions(),
        )


def test_renumber_puts_the_kept_atoms_first(methylbiphenyl):
    result = apply_swap(
        methylbiphenyl, Swap("[*:1]CCCC", remove_atoms=[0], mode="renumber")
    )
    assert identity(result.mol) == BUTYL
    assert result.conserved == list(range(21))
    assert result.new_atoms == list(range(21, 34))
    assert result.ref_to_new[1] == 0 and result.attachments == [(0, 21)]


def test_every_conformer_is_transferred_with_its_id():
    mol = embedded("CCO", n=3)
    for conf, new_id in zip(list(mol.GetConformers()), (4, 9, 2)):
        conf.SetId(new_id)
        conf.SetDoubleProp("energy", -1.0)
    oh = next(
        a.GetIdx()
        for a in mol.GetAtoms()
        if a.GetAtomicNum() == 1 and a.GetNeighbors()[0].GetAtomicNum() == 8
    )
    result = apply_swap(mol, Swap("[*]C", remove_atoms=[oh]))
    assert identity(result.mol) == "CCOC"
    assert [c.GetId() for c in result.mol.GetConformers()] == [4, 9, 2]
    assert not result.mol.GetConformer(4).HasProp("energy")
    for conf_id in (4, 9, 2):
        before = mol.GetConformer(conf_id).GetPositions()
        after = result.mol.GetConformer(conf_id).GetPositions()
        kept = sorted(result.ref_to_new)
        np.testing.assert_array_equal(after[kept], before[kept])


def test_reverse_swap_restores_the_graph(methylbiphenyl):
    forward = apply_swap(methylbiphenyl, Swap("[*:1]CCCC", old_fragment="[CH3][c:1]"))
    back = apply_swap(forward.mol, Swap("[*:1]C", old_fragment="[CH2;!R]([CH2])[c:1]"))
    assert identity(back.mol) == identity(methylbiphenyl)
    # 13 atoms leave, 4 come: the other 9 slots close up; kept atoms keep their order.
    assert back.mol.GetNumAtoms() == methylbiphenyl.GetNumAtoms()
    kept = sorted(back.ref_to_new)
    assert [back.ref_to_new[i] for i in kept] == sorted(back.ref_to_new.values())


def test_identity_swap_keeps_the_geometry(methylbiphenyl):
    result = apply_swap(methylbiphenyl, Swap("[*:1]C", old_fragment="[CH3][c:1]"))
    assert identity(result.mol) == identity(methylbiphenyl)
    before = methylbiphenyl.GetConformer().GetPositions()
    after = result.mol.GetConformer().GetPositions()
    kept = sorted(result.ref_to_new)
    assert np.abs(after[kept] - before[kept]).max() < 1e-3
    assert np.linalg.norm(after[0] - before[0]) < 0.05  # the methyl C (1.52 vs 1.51 A)


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


# Tiers: soft atoms (coordinate map in embedding, position restraints in refinement).

import racerts  # noqa: E402
from racerts.restraints import PositionRestraint  # noqa: E402
from racerts.task import Constrained, FrozenSet  # noqa: E402


def test_frozen_set_and_constrained_with_soft_atoms():
    assert FrozenSet(soft=(2,)) and not FrozenSet()
    with pytest.raises(ValueError, match="both hard and soft"):
        FrozenSet(hard=(1, 2), soft=(2,))
    with pytest.raises(ValueError, match="both hard and soft"):
        Constrained(hard=[1], soft=[1])
    with pytest.raises(ValueError, match="at least one"):
        Constrained()
    task = Constrained(hard=[0], soft=[1, 2], core=[0])
    frozen = task.frozen_atoms(None)
    assert (frozen.hard, frozen.soft, frozen.core) == ((0,), (1, 2), (0,))
    moved = task.remap({0: 5, 1: 6, 2: 7})
    assert (moved.hard, moved.soft, moved.core) == ((5,), (6, 7), (5,))
    with pytest.raises(ValueError, match=r"soft atoms \[2\]"):
        task.remap({0: 5, 1: 6})


def _eclipsed_octane():
    from rdkit.Chem import rdMolTransforms

    mol = embedded("CCCCCCCC", seed=1)
    for k in range(5):  # every C-C-C-C torsion anticlinal: far from a minimum
        rdMolTransforms.SetDihedralDeg(
            mol.GetConformer(), k, k + 1, k + 2, k + 3, 120.0
        )
    return mol


def test_soft_atoms_start_at_the_reference_and_are_restrained(monkeypatch):
    from racerts.refine import MMFFOptimizer

    mol = _eclipsed_octane()
    carbons = list(range(8))
    reference = mol.GetConformer().GetPositions()
    ctx = racerts.Context.create(mol, Constrained(soft=carbons))
    embedded_ = racerts.Embed(n_conformers=3).run(ctx)
    for conf_id in embedded_.conf_ids:  # the coordinate map places them exactly
        positions = embedded_.mol.GetConformer(conf_id).GetPositions()
        assert np.abs(positions[carbons] - reference[carbons]).max() < 1e-6

    passed = []
    original = MMFFOptimizer._refine

    def spy(self, mol, reference, anchors, restraints=()):
        passed.append((list(anchors), list(restraints)))
        return original(self, mol, reference, anchors, restraints)

    monkeypatch.setattr(MMFFOptimizer, "_refine", spy)
    refined = racerts.Refine().run(ctx, embedded_.copy())
    [(anchors, restraints)] = passed
    assert anchors == []
    assert [r.atom for r in restraints] == carbons
    for r in restraints:
        assert r.point == pytest.approx(tuple(reference[r.atom]))
        assert (r.tolerance, r.force_constant) == (0.3, 5.0)
    for conf_id in refined.conf_ids:
        # The reported energy leaves out the restraint terms.
        props = AllChem.MMFFGetMoleculeProperties(refined.mol)
        plain = AllChem.MMFFGetMoleculeForceField(refined.mol, props, confId=conf_id)
        assert refined.energy(conf_id) == pytest.approx(plain.CalcEnergy(), abs=1e-6)


def test_position_restraints_hold_with_their_force_constant():
    from racerts.refine import MMFFOptimizer

    mol = _eclipsed_octane()
    reference = mol.GetConformer().GetPositions()
    deviations = []
    for k in (5.0, 1000.0):
        probe = Chem.Mol(mol)
        restraints = [
            PositionRestraint(i, tuple(reference[i]), 0.3, k) for i in range(8)
        ]
        MMFFOptimizer(converge=True).refine(probe, mol, (), restraints)
        positions = probe.GetConformer().GetPositions()
        deviations.append(np.linalg.norm(positions[:8] - reference[:8], axis=1).max())
    assert deviations[1] < 0.45 < deviations[0]  # beyond 0.3 A only a little at k 1000
    with pytest.raises(ValueError, match="reference"):
        MMFFOptimizer().refine(Chem.Mol(mol), None, (), restraints)
    with pytest.raises(ValueError):
        PositionRestraint(0, (0.0, 0.0, float("nan")))


def test_soft_atoms_need_the_coordinate_map_embedder():
    mol = _eclipsed_octane()
    ctx = racerts.Context.create(mol, Constrained(soft=[0, 1]))
    embedder = racerts.embed.BoundsMatrixEmbedder()
    with pytest.raises(ValueError, match="coordinate map"):
        racerts.Embed(embedder, n_conformers=2).run(ctx)


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


def test_the_graft_keeps_a_stretched_bond(sn2_ts):
    # Cl -> Br as leaving group: the TS bond stays stretched (2.15 A for C-Cl in the
    # seed), scaled by the covalent radii, not the covalent C-Br length.
    from racerts.system import build_mol

    ts = build_mol(sn2_ts, -1, [0, 1, 2], input_smiles=["CCl", "[Cl-]"])
    result = apply_swap(ts, Swap("[*]Br", remove_atoms=[1]))
    table = Chem.GetPeriodicTable()
    r = {z: table.GetRcovalent(z) for z in (6, 17, 35)}
    positions = result.mol.GetConformer().GetPositions()
    assert np.linalg.norm(positions[1] - positions[0]) == pytest.approx(
        2.15 * (r[6] + r[35]) / (r[6] + r[17])
    )


def test_custom_tasks_may_return_lists():
    class ListTask:
        needs_reference = True

        def frozen_atoms(self, mol):
            return FrozenSet(hard=[0, 1, 2])

    ctx = racerts.Context.create(embedded("CCCC"), ListTask())
    assert ctx.frozen.hard == (0, 1, 2) and ctx.frozen.core == (0, 1, 2)


def test_soft_atoms_dropped_by_an_optimizer_are_reported(caplog):
    from racerts.refine.base import BaseOptimizer

    class Plain(BaseOptimizer):  # takes no restraints, like the ASE optimizer
        def _refine(self, mol, reference, anchors):
            return 0

    mol = _eclipsed_octane()
    with caplog.at_level("WARNING"):
        Plain().refine(Chem.Mol(mol), mol, (), [PositionRestraint(0, (0.0, 0.0, 0.0))])
    assert "soft atoms" in caplog.text


def test_e_z_survives_when_a_stereo_atom_leaves():
    # Cl (a stereo atom of the double bond) leaves without replacement: the other
    # neighbour of that end carries the E/Z.
    reference = Chem.MolFromSmiles("F/C(Cl)=C/CC")
    result = apply_swap(reference, Swap("[*:1]Br", remove_atoms=[2], attach_map={1: 5}))
    assert identity(result.mol) == canonical("F/C=C/CCBr")
