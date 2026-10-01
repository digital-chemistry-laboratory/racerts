"""Swaps: graph surgery (apply_swap) and catmlp's substitutions."""

import json

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

from racerts.system.stereo import StereoCheck
from racerts.system.swap import Swap, apply_swap, label_hydrogen, substitute_groups


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


# Ported from catmlp (tests/test_templates.py, draft d9381f1): substitute_groups and
# label_hydrogen; the graph-level tests (substitute_graph) stay in catmlp.


def quinoline():
    # The mapped c is quinoline C6.
    mol = embedded("[cH:6]1ccc2ncccc2c1", seed=42)
    anchor = next(a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() == 6)
    mol.GetAtomWithIdx(anchor).SetAtomMapNum(0)
    return label_hydrogen(mol, anchor, 100), anchor


def test_quinoline_to_methyl_preserves_core_and_clears_results():
    mol, anchor = quinoline()
    source = mol.GetConformer().GetPositions().copy()
    mol.GetConformer().SetDoubleProp("energy", -100)
    mol.SetProp("acceptance", "passed")
    grafted = substitute_groups(mol, {100: "[*]C"})
    assert identity(grafted) == identity(Chem.MolFromSmiles("c1(C)ccc2ncccc2c1"))
    core = [a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() != 100]
    np.testing.assert_array_equal(
        grafted.GetConformer().GetPositions()[core], source[core]
    )
    assert grafted.GetNumAtoms() == mol.GetNumAtoms() + 3
    assert not grafted.GetConformer().HasProp("energy")
    assert not grafted.HasProp("acceptance")
    assert mol.GetConformer().GetDoubleProp("energy") == -100
    assert mol.GetAtomWithIdx(anchor).GetAtomicNum() == 6


@pytest.mark.parametrize("fragment", ["[*][C@H](F)Cl", "F[C@@H](Cl)[*]", "[*]/C=C/Cl"])
def test_substituent_stereochemistry_is_preserved(fragment):
    parent = Chem.MolFromSmiles("Br[*:100]")
    actual = substitute_groups(parent, {100: fragment})
    expected = Chem.MolFromSmiles(fragment.replace("[*]", "Br"))
    assert identity(actual) == identity(expected)


def test_existing_stereocenter_is_not_inverted():
    parent = embedded("N[C@@H](C)C(=O)O", seed=42)
    labeled = label_hydrogen(parent, 1, 100)
    grafted = substitute_groups(labeled, {100: "[*]F"})
    assert identity(grafted) == identity(Chem.MolFromSmiles("N[C@@](C)(C(=O)O)F"))


@pytest.mark.parametrize("fragment", ["[*][C@H](F)Cl", "F[C@@H](Cl)[*]", "[*]/C=C/Cl"])
def test_grafted_coordinates_agree_with_substituent_stereo(fragment):
    parent = label_hydrogen(embedded("BrC(F)Cl", seed=42), 1, 100)
    grafted = substitute_groups(parent, {100: fragment})
    # The stereo the graph specifies is that of the geometry (the anchor, a new
    # stereocentre, is unspecified).
    check = StereoCheck(grafted)
    assert check and check.mismatch(grafted.GetConformer()) is None
    if "@" in fragment:  # the mirror image fails
        mirror = Chem.Conformer(grafted.GetConformer())
        for i, p in enumerate(mirror.GetPositions()):
            mirror.SetAtomPosition(i, (-p[0], p[1], p[2]))
        assert "inverted" in check.mismatch(mirror)


@pytest.mark.parametrize(
    "fragment", ["C", "[*]C.[Cl-]", "[*:1]C[*:2]", "[*]=C", "[*][CH2]"]
)
def test_bad_fragments_fail_without_mutating_parent(fragment):
    mol, _ = quinoline()
    before = mol.ToBinary()
    with pytest.raises(ValueError):
        substitute_groups(mol, {100: fragment})
    assert mol.ToBinary() == before


def _stereo_agrees(mol):
    check = StereoCheck(mol)
    return all(check.mismatch(conf) is None for conf in mol.GetConformers())


@pytest.mark.parametrize("extra_site", [False, True])
def test_quinoline_methyl_transfer_preserves_core_and_invalidates_results(extra_site):
    mol = label_hydrogen(embedded("c1ccc2ncccc2c1", seed=42), 0, 100)
    substitutions = {100: "[*]C"}
    if extra_site:
        mol = label_hydrogen(mol, 1, 101)
        substitutions[101] = "[*]F"
    source = mol.GetConformer().GetPositions()
    mol.GetConformer().SetDoubleProp("energy", -100)
    mol.GetConformer().SetProp("hessian", "stale")
    second = Chem.Conformer(mol.GetConformer())
    second.SetId(7)
    for index, position in enumerate(source + 2):
        second.SetAtomPosition(index, position)
    mol.AddConformer(second, assignId=False)
    mol.SetProp("acceptance", "passed")
    before = mol.ToBinary(Chem.PropertyPickleOptions.AllProps)
    grafted = substitute_groups(mol, substitutions)
    expected = "c1(C)c(F)cc2ncccc2c1" if extra_site else "c1(C)ccc2ncccc2c1"
    assert identity(grafted) == canonical(expected)
    core = [
        a.GetIdx() for a in mol.GetAtoms() if a.GetAtomMapNum() not in substitutions
    ]
    assert [c.GetId() for c in grafted.GetConformers()] == [0, 7]
    for conformer, reference in zip(grafted.GetConformers(), (source, source + 2)):
        np.testing.assert_array_equal(conformer.GetPositions()[core], reference[core])
        assert not conformer.GetPropsAsDict(includePrivate=True)
    assert grafted.GetNumAtoms() == mol.GetNumAtoms() + 3
    assert not grafted.HasProp("acceptance")
    assert mol.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_dummy_cap_then_enlarge_and_empty_copy():
    template = Chem.MolFromSmiles("c1ccccc1[*:100]")
    parent = substitute_groups(template, {100: "[H]"})
    variant = substitute_groups(parent, {100: "[*]CC"})
    assert identity(variant) == "CCc1ccccc1"
    assert variant.GetNumConformers() == 0
    parent.SetProp("keep", "unchanged")
    copied = substitute_groups(parent, {})
    assert copied is not parent and copied.GetProp("keep") == "unchanged"


@pytest.mark.parametrize(
    "fragment", ["[*][C@H](F)Cl", "F[C@@H](Cl)[*]", "[*]/C=C/Cl", "[*]/C=C\\Cl"]
)
def test_graft_stereo_is_verified_from_coordinates(fragment):
    parent = label_hydrogen(embedded("Br", seed=42), 0, 100)
    grafted = substitute_groups(parent, {100: fragment})
    assert identity(grafted) == canonical(fragment.replace("[*]", "Br"))
    assert StereoCheck(grafted) and _stereo_agrees(grafted)


def test_existing_stereocenter_and_attachment_distance_are_preserved():
    parent = label_hydrogen(embedded("N[C@@H](C)C(=O)O", seed=42), 1, 100)
    grafted = substitute_groups(parent, {100: "[*]F"})
    assert identity(grafted) == canonical("N[C@@](C)(C(=O)O)F")
    assert _stereo_agrees(grafted)
    capped = substitute_groups(parent, {100: "[H]"})
    site = next(a.GetIdx() for a in capped.GetAtoms() if a.GetAtomMapNum() == 100)
    xyz = capped.GetConformer().GetPositions()
    radii = Chem.GetPeriodicTable()
    assert np.linalg.norm(xyz[site] - xyz[1]) == pytest.approx(
        radii.GetRcovalent(6) + radii.GetRcovalent(1)
    )


@pytest.mark.parametrize(
    "fragment", ["C", "[*]C.[Cl-]", "[*]C[*]", "[*]=C", "[*][CH2]"]
)
def test_invalid_fragments_leave_parent_unchanged(fragment):
    parent = label_hydrogen(embedded("Br", seed=42), 0, 100)
    before = parent.ToBinary(Chem.PropertyPickleOptions.AllProps)
    with pytest.raises(ValueError):
        substitute_groups(parent, {100: fragment})
    assert parent.ToBinary(Chem.PropertyPickleOptions.AllProps) == before


def test_ambiguous_sites_and_invalid_indices_are_not_guessed():
    parent = embedded("C", seed=42)
    with pytest.raises(ValueError, match="exactly one"):
        label_hydrogen(parent, 0, 100)
    with pytest.raises(IndexError):
        label_hydrogen(parent, -1, 100)
    with pytest.raises(TypeError):
        label_hydrogen(parent, 0, True)
    with pytest.raises(ValueError, match="exactly one atom with map number 100"):
        substitute_groups(parent, {100: "[*]C"})
    marked = label_hydrogen(embedded("Br", seed=42), 0, 100)
    with pytest.raises(ValueError, match="already has an atom map"):
        label_hydrogen(marked, 0, 101)
    coincident = Chem.MolFromSmiles("F[*:100]")
    coincident.AddConformer(Chem.Conformer(2))
    with pytest.raises(ValueError, match="coincident"):
        substitute_groups(coincident, {100: "[H]"})


def test_charged_graft():
    # catmlp: a charged group changes the charge of what follows (here the context).
    mol, _ = quinoline()
    grafted = substitute_groups(mol, {100: "[*][N+](C)(C)C"})
    assert Chem.GetFormalCharge(grafted) == 1
    ctx = racerts.Context.create(grafted, racerts.GroundState())
    assert ctx.mol.GetIntProp("charge") == 1


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
from racerts.task import Constrained, FrozenSet, TransitionState  # noqa: E402


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


def test_transition_state_remap():
    task = TransitionState([0, 1, 2], active_bonds=[(0, 2)], active_window=0.3)
    task.bond_changes = [(0, 2)]
    moved = task.remap({0: 3, 1: 4, 2: 1})
    assert moved.reacting_atoms == [3, 4, 1]
    assert moved.active_bonds == [(1, 3)] and moved.bond_changes == [(1, 3)]
    assert moved.active_window == 0.3
    with pytest.raises(ValueError, match=r"reacting atoms \[2\]"):
        task.remap({0: 3, 1: 4})
    assert racerts.GroundState().remap({}).frozen_atoms(None) == FrozenSet()


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


# racerts.swap: sampling after the swap.

from racerts.embed.rigid_attach import rigid_attach  # noqa: E402
from racerts.prune import aligned_rmsd  # noqa: E402

BUTYL_SWAP = Swap("[*:1]CCCC", old_fragment="[CH3][c:1]")


def _in_frame(positions, target, atoms):
    """positions superposed on target by the atoms (Kabsch)."""
    a, b = positions[atoms], target[atoms]
    ca, cb = a.mean(0), b.mean(0)
    u, _, vt = np.linalg.svd((a - ca).T @ (b - cb))
    rotation = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    return (positions - ca) @ rotation + cb


def _ring_torsion(mol, conf_id=-1):
    from rdkit.Chem import rdMolTransforms

    bond = next(
        b for b in mol.GetBonds()
        if b.GetBeginAtom().GetIsAromatic() and b.GetEndAtom().GetIsAromatic()
        and not b.IsInRing()
    )  # fmt: skip
    a, b = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
    na = min(
        n.GetIdx() for n in mol.GetAtomWithIdx(a).GetNeighbors() if n.GetIdx() != b
    )
    nb = min(
        n.GetIdx() for n in mol.GetAtomWithIdx(b).GetNeighbors() if n.GetIdx() != a
    )
    return rdMolTransforms.GetDihedralDeg(mol.GetConformer(conf_id), na, a, b, nb)


def test_swap_soft_samples_the_chain_around_the_kept_skeleton(methylbiphenyl):
    mol = methylbiphenyl
    grafted = apply_swap(mol, BUTYL_SWAP).mol.GetConformer().GetPositions()
    skeleton = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    assert {
        racerts.swap(mol, BUTYL_SWAP, n_conformers=4).provenance(c)["route"]
        for c in racerts.swap(mol, BUTYL_SWAP, n_conformers=4).conf_ids
    } == {"dg"}  # the default route
    ensemble = racerts.swap(mol, BUTYL_SWAP, n_conformers=30, routes=["dg", "rigid"])
    assert identity(ensemble.mol) == BUTYL
    routes = {ensemble.provenance(c)["route"] for c in ensemble.conf_ids}
    assert routes == {"dg", "rigid"}
    reference_torsion = _ring_torsion(mol)
    butyl = [
        a.GetIdx()
        for a in ensemble.mol.GetAtoms()
        if a.GetAtomicNum() == 6 and not a.GetIsAromatic()
    ]
    poses = []
    for conf_id in ensemble.conf_ids:
        positions = ensemble.mol.GetConformer(conf_id).GetPositions()
        # The skeleton stays in the basin of the reference; the 0.3 A flat bottom of
        # the position restraints lets the ring torsion adapt (15 deg measured).
        assert aligned_rmsd(positions[skeleton], grafted[skeleton]) < 0.2
        turn = (
            _ring_torsion(ensemble.mol, conf_id) - reference_torsion + 180
        ) % 360 - 180
        assert abs(turn) < 25
        pose = _in_frame(positions, grafted, skeleton)[butyl]
        if all(np.sqrt(((pose - other) ** 2).sum(1).mean()) > 0.3 for other in poses):
            poses.append(pose)
    assert len(poses) >= 4  # distinct chain conformers


def test_swap_free_and_hard(methylbiphenyl):
    mol = methylbiphenyl
    skeleton = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    result = apply_swap(mol, BUTYL_SWAP, seed=racerts.PipelineConfig().seed)
    grafted = result.mol.GetConformer().GetPositions()
    free = racerts.swap(
        mol, BUTYL_SWAP, conserve="free", n_conformers=30, routes=["dg"]
    )
    # Resampled as a whole: the ring torsion takes other values too.
    turns = {
        round(abs((_ring_torsion(free.mol, c) - _ring_torsion(mol) + 180) % 360 - 180))
        for c in free.conf_ids
    }
    assert max(turns) > 90
    hard = racerts.swap(mol, BUTYL_SWAP, conserve="hard")
    assert len(hard) == 1 and hard.provenance(0) == {"route": "graft", "reference": 0}
    np.testing.assert_array_equal(hard.mol.GetConformer(0).GetPositions(), grafted)
    assert (
        np.abs(
            hard.mol.GetConformer(0).GetPositions()[skeleton]
            - mol.GetConformer().GetPositions()[skeleton]
        ).max()
        == 0
    )


def test_swap_in_a_transition_state(sn2_ts, caplog):
    # As in catmlp: SN2 TS, H -> 4-hydroxybutyl on the reacting carbon. The first chain
    # atom is a neighbour of a reacting atom: it is held where the graft puts it.
    from racerts.system import build_mol

    mol = build_mol(sn2_ts, -1, [0, 1, 2], input_smiles=["CCl", "[Cl-]"])
    task = TransitionState([0, 1, 2])
    change = Swap("[*]CCCCO", remove_atoms=[3])
    result = apply_swap(mol, change)
    held = task.remap(result.ref_to_new).frozen_atoms(result.mol).hard
    assert sorted(held) == [0, 1, 2, 3, 4, 5]  # atom 3 is now the chain's C1
    with caplog.at_level("WARNING"):
        ensemble = racerts.swap(mol, change, task=task, n_conformers=20)
    assert "attaches at the frozen atoms" in caplog.text
    assert "clash" not in caplog.text  # the forming C...Cl pair is not a clash
    assert identity(ensemble.mol) == identity(Chem.MolFromSmiles("OCCCCCCl.[Cl-]"))
    grafted = result.mol.GetConformer().GetPositions()
    assert len(ensemble) >= 5
    for conf_id in ensemble.conf_ids:
        positions = ensemble.mol.GetConformer(conf_id).GetPositions()
        assert np.abs(positions[list(held)] - grafted[list(held)]).max() < 1e-3


def test_swap_every_reference_conformer():
    mol = embedded("CCCO", n=2, seed=5)
    change = Swap("[*]CC", remove_atoms=[0])
    ensemble = racerts.swap(mol, change, n_conformers=4, routes=["dg"])
    references = {ensemble.provenance(c).get("reference") for c in ensemble.conf_ids}
    assert references == {0, 1}


def test_swap_errors(methylbiphenyl, sn2_ts):
    from racerts.system import build_mol

    mol = methylbiphenyl
    with pytest.raises(ValueError, match="conserve"):
        racerts.swap(mol, BUTYL_SWAP, conserve="all")
    with pytest.raises(ValueError, match="routes"):
        racerts.swap(mol, BUTYL_SWAP, routes=["etkdg"])
    with pytest.raises(ValueError, match="conformers"):
        racerts.swap(Chem.MolFromSmiles("CC"), BUTYL_SWAP)
    methyl_h = next(
        n.GetIdx()
        for n in mol.GetAtomWithIdx(0).GetNeighbors()
        if n.GetAtomicNum() == 1
    )
    with pytest.raises(ValueError, match="hard atoms"):
        racerts.swap(mol, BUTYL_SWAP, hard=[methyl_h])  # leaves, nothing replaces it
    ts = build_mol(sn2_ts, -1, [0, 1, 2], input_smiles=["CCl", "[Cl-]"])
    with pytest.raises(ValueError, match="windows"):
        racerts.swap(
            ts,
            Swap("[*]C", remove_atoms=[3]),
            task=TransitionState([0, 1, 2], active_window=0.3),
        )
    with pytest.raises(ValueError, match="reacting atoms"):  # the nucleophile leaves
        racerts.swap(
            ts,
            Swap("[*]C", remove_atoms=[2, 3], attach_map={1: 0}),
            task=TransitionState([0, 1, 2]),
        )
    pd = Chem.AddHs(Chem.MolFromSmiles("Cl[Pd]Cl"))
    AllChem.Compute2DCoords(pd)
    with pytest.raises(ValueError, match="single attachment"):
        racerts.swap(
            pd,
            Swap("[*:1]<-P(C)(C)C", remove_atoms=[], attach_map={1: 1}),
            conserve="hard",
        )


def test_rigid_attach_poses(methylbiphenyl):
    result = apply_swap(methylbiphenyl, BUTYL_SWAP)
    poses = rigid_attach(result, n_fragment_conformers=2, n_rotations=6)
    assert 0 < len(poses) <= 12
    kept = result.conserved
    reference = result.mol.GetConformer().GetPositions()
    anchor, root = result.attachments[0]
    seen = set()
    for conf_id in poses.conf_ids:
        positions = poses.mol.GetConformer(conf_id).GetPositions()
        np.testing.assert_allclose(positions[kept], reference[kept])
        assert np.linalg.norm(positions[root] - positions[anchor]) == pytest.approx(
            np.linalg.norm(reference[root] - reference[anchor])
        )
        provenance = poses.provenance(conf_id)
        assert provenance["route"] == "rigid" and provenance["reference"] == 0
        seen.add(tuple(provenance["pose"]))
    assert len(seen) == len(poses)
    # A strict clash filter leaves fewer poses.
    assert len(rigid_attach(result, 2, 6, clash_factor=1.2)) < len(poses)


def test_a_replaced_atom_passes_its_role_on(sn2_ts):
    # The leaving group Cl -> Br: Br takes the slot and the role of Cl.
    from racerts.system import build_mol

    ts = build_mol(sn2_ts, -1, [0, 1, 2], input_smiles=["CCl", "[Cl-]"])
    result = apply_swap(ts, Swap("[*]Br", remove_atoms=[1]))
    assert result.replaced == {1: 1} and result.index_map[1] == 1
    assert result.mol.GetAtomWithIdx(1).GetSymbol() == "Br"
    task = TransitionState([0, 1, 2]).remap(result.index_map)
    assert task.reacting_atoms == [0, 1, 2]
    ensemble = racerts.swap(
        ts,
        Swap("[*]Br", remove_atoms=[1]),
        task=TransitionState([0, 1, 2]),
        conserve="hard",
    )
    assert len(ensemble) == 1


def _square_planar_pd_dmpe():
    """cis-[PdCl2(dmpe)] with Pd, Cl and P held square planar."""
    from rdkit.Geometry import Point3D

    mol = Chem.AddHs(Chem.MolFromSmiles(PD_DMPE))
    pd = next(a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Pd")
    cl = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "Cl"]
    p = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() == "P"]
    square = {pd: (0, 0, 0), cl[0]: (2.35, 0, 0), cl[1]: (0, 2.35, 0)}
    square.update({p[0]: (-2.25, 0, 0), p[1]: (0, -2.25, 0)})
    coord_map = {i: Point3D(*x) for i, x in square.items()}
    assert (
        AllChem.EmbedMolecule(
            mol, coordMap=coord_map, randomSeed=1, useRandomCoords=True
        )
        == 0
    )
    ff = AllChem.UFFGetMoleculeForceField(mol)
    for i in square:
        ff.UFFAddPositionConstraint(i, 0.0, 1e4)
    ff.Minimize(maxIts=2000)
    return mol, pd, cl, p


# Regression tests.

TETRAHEDRAL_TAGS = (
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
)


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


def test_new_stereo_at_the_attachment_comes_from_the_reference():
    # Which hydrogen is replaced chooses the configuration (catmlp): the graph takes it
    # from the reference, so embedding cannot mix the two.
    mol = embedded("OCc1ccccc1")
    hydrogens = [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(1).GetNeighbors()
        if n.GetAtomicNum() == 1
    ]
    graphs = set()
    for h in hydrogens:
        result = apply_swap(mol, Swap("[*:1]C", remove_atoms=[h]))
        assert result.mol.GetAtomWithIdx(1).GetChiralTag() in TETRAHEDRAL_TAGS
        assert _stereo_agrees(result.mol)
        graphs.add(identity(result.mol))
    assert len(graphs) == 2  # the enantiomers
    ensemble = racerts.swap(
        mol, Swap("[*:1]C", remove_atoms=[hydrogens[0]]), routes=["dg"], n_conformers=10
    )
    assert ensemble.mol.GetAtomWithIdx(1).GetChiralTag() in TETRAHEDRAL_TAGS
    assert _stereo_agrees(ensemble.mol)
    # The same for a double bond: one H of CH2= of styrene gives E, the other Z.
    styrene = embedded("C=Cc1ccccc1")
    hydrogens = [
        n.GetIdx()
        for n in styrene.GetAtomWithIdx(0).GetNeighbors()
        if n.GetAtomicNum() == 1
    ]
    graphs = {
        identity(apply_swap(styrene, Swap("[*]C", remove_atoms=[h])).mol)
        for h in hydrogens
    }
    assert graphs == {canonical("C/C=C/c1ccccc1"), canonical("C/C=C\\c1ccccc1")}


def test_charge_and_multiplicity_carry_over(sn2_ts):
    from racerts.system import GRAPH_METHODS, build_mol

    # A connectivity graph: the charge is only a property, no formal charges.
    ts = build_mol(sn2_ts, -1, [0, 1, 2], mol_getter=GRAPH_METHODS["connect"]())
    result = apply_swap(ts, Swap("[*]C", remove_atoms=[3]))
    assert result.mol.GetIntProp("charge") == -1
    assert not any(a.GetNumRadicalElectrons() for a in result.mol.GetAtoms())
    ensemble = racerts.swap(
        ts,
        Swap("[*]C", remove_atoms=[3]),
        task=TransitionState([0, 1, 2]),
        n_conformers=3,
    )
    assert (
        ensemble.mol.GetIntProp("charge"),
        ensemble.mol.GetIntProp("multiplicity"),
    ) == (-1, 1)
    triplet = embedded("CCO")
    triplet.SetIntProp("multiplicity", 3)
    for conserve in ("soft", "hard"):
        swapped = racerts.swap(
            triplet,
            Swap("[*]CC", remove_atoms=[0]),
            conserve=conserve,
            **({} if conserve == "hard" else {"n_conformers": 3}),
        )
        assert swapped.mol.GetIntProp("multiplicity") == 3
    charged = apply_swap(embedded("CO"), Swap("[*][N+](C)(C)C", remove_atoms=[0]))
    assert charged.mol.GetIntProp("charge") == 1


def test_a_lower_bond_order_raises():
    mol = embedded("CC(=O)C")  # atom 2 is O
    with pytest.raises(ValueError, match="bond order"):
        apply_swap(mol, Swap("[*]F", remove_atoms=[2]))  # C=O -> C-F: C would gain an H
    assert identity(apply_swap(mol, Swap("[*]=C", remove_atoms=[2])).mol) == "C=C(C)C"


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


def test_swap_rejects_settings_it_does_not_use(methylbiphenyl):
    with pytest.raises(ValueError, match="restraints"):
        racerts.swap(
            methylbiphenyl,
            BUTYL_SWAP,
            config=racerts.PipelineConfig.from_dict({"restraints": {"hbonds": True}}),
        )
    with pytest.raises(ValueError, match="cmap"):
        racerts.swap(
            methylbiphenyl,
            BUTYL_SWAP,
            config=racerts.PipelineConfig.from_dict({"embed": {"mode": "bounds"}}),
        )
    with pytest.raises(ValueError, match="hard"):
        racerts.swap(methylbiphenyl, BUTYL_SWAP, conserve="hard", routes=["dg"])
    with pytest.raises(ValueError, match="Invalid"):
        racerts.swap(methylbiphenyl, BUTYL_SWAP, hard=[999])


def test_hard_grafts_are_checked_for_clashes(methylbiphenyl, caplog):
    # An ortho-tolyl for the methyl: the rigid graft runs into the other ring.
    with caplog.at_level("WARNING"):
        racerts.swap(
            methylbiphenyl,
            Swap("[*:1]c1ccccc1C", old_fragment="[CH3][c:1]"),
            conserve="hard",
        )
    assert "clash" in caplog.text


def test_soft_atoms_dropped_by_an_optimizer_are_reported(caplog):
    from racerts.refine.base import BaseOptimizer

    class Plain(BaseOptimizer):  # takes no restraints, like the ASE optimizer
        def _refine(self, mol, reference, anchors):
            return 0

    mol = _eclipsed_octane()
    with caplog.at_level("WARNING"):
        Plain().refine(Chem.Mol(mol), mol, (), [PositionRestraint(0, (0.0, 0.0, 0.0))])
    assert "soft atoms" in caplog.text


def _labels(mol):
    """Canonical SMILES of the graph and of the geometry of each conformer."""
    graph = identity(mol)
    geometry = []
    for conf in mol.GetConformers():
        probe = Chem.Mol(mol)
        Chem.AssignStereochemistryFrom3D(probe, confId=conf.GetId())
        geometry.append(identity(probe))
    return graph, geometry


def test_no_e_z_from_the_rotation_of_a_graft():
    # C=O -> C=CHF: the E/Z of the new double bond would come from where the graft
    # happens to put F; it stays unspecified whatever the seed.
    mol = embedded("CCC=O")
    graphs = {
        identity(apply_swap(mol, Swap("[*]=CF", remove_atoms=[3]), seed=s).mol)
        for s in range(6)
    }
    assert graphs == {canonical("CCC=CF")}


def test_new_ring_stereo_comes_from_the_reference():
    # Methylcyclohexane C4-H -> CH3: C1 and C4 get cis/trans, both from the reference.
    mol = embedded("CC1CCCCC1")
    c4 = 4
    hydrogens = [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(c4).GetNeighbors()
        if n.GetAtomicNum() == 1
    ]
    graphs = set()
    for h in hydrogens:
        result = apply_swap(mol, Swap("[*]C", remove_atoms=[h]))
        graph, geometry = _labels(result.mol)
        assert geometry == [graph]
        graphs.add(graph)
    assert graphs == {
        canonical("C[C@H]1CC[C@H](C)CC1"),
        canonical("C[C@H]1CC[C@@H](C)CC1"),
    }
    ensemble = racerts.swap(
        mol, Swap("[*]C", remove_atoms=[hydrogens[0]]), conserve="free", n_conformers=8
    )
    assert _stereo_agrees(ensemble.mol)  # one isomer only


def test_no_spurious_e_z_between_stereo_double_bonds():
    # (E)-penta-1,3-diene, a terminal H -> /C=C/F: the new E/Z at C1=C2 is that of the
    # reference, not read from bond directions set for the neighbouring double bonds.
    mol = embedded("C=C/C=C/C")
    for h in (5, 6):
        graph, geometry = _labels(
            apply_swap(mol, Swap("[*]/C=C/F", remove_atoms=[h])).mol
        )
        assert geometry == [graph]


def test_a_dummy_site_gets_the_covalent_bond_length():
    template = Chem.AddHs(Chem.MolFromSmiles("[*:1]c1ccccc1"))
    AllChem.EmbedMolecule(template, randomSeed=1)
    grafted = substitute_groups(template, {1: "[*]C"})
    root = next(a.GetIdx() for a in grafted.GetAtoms() if a.GetAtomMapNum() == 1)
    positions = grafted.GetConformer().GetPositions()
    assert np.linalg.norm(positions[root] - positions[1]) == pytest.approx(
        2 * Chem.GetPeriodicTable().GetRcovalent(6)
    )


def test_valence_check_spares_metals_and_heavy_atom_templates():
    mol, pd, cl, p = _square_planar_pd_dmpe()
    dmpe = [a.GetIdx() for a in mol.GetAtoms() if a.GetSymbol() in ("C", "P")]
    one = apply_swap(
        mol,
        Swap(
            "[*:1]P(C)(C)C",
            remove_atoms=dmpe,
            attach_map={1: pd},
            bond_types={1: "dative"},
        ),
    )
    assert identity(one.mol) == canonical("CP(C)(C)->[Pd](Cl)Cl")
    template = Chem.MolFromSmiles("CC=O")  # no explicit hydrogens
    AllChem.EmbedMolecule(template, randomSeed=1)
    assert identity(apply_swap(template, Swap("[*]C", remove_atoms=[2])).mol) == "CCC"


def test_hard_mode_takes_charge_and_multiplicity(methylbiphenyl):
    hard = racerts.swap(
        methylbiphenyl, BUTYL_SWAP, conserve="hard", charge=1, multiplicity=2
    )
    assert (hard.mol.GetIntProp("charge"), hard.mol.GetIntProp("multiplicity")) == (
        1,
        2,
    )
    index_map = json.loads(hard.mol.GetProp("swap_index_map"))
    assert index_map["1"] == 1


def test_a_multiplicity_that_no_longer_fits_is_dropped(caplog):
    radical = Chem.AddHs(Chem.MolFromSmiles("[CH2]C"))
    AllChem.EmbedMolecule(radical, randomSeed=1)
    radical.SetIntProp("multiplicity", 2)
    h = next(
        n.GetIdx()
        for n in radical.GetAtomWithIdx(0).GetNeighbors()
        if n.GetAtomicNum() == 1
    )
    with caplog.at_level("WARNING"):
        # the radical centre gets a methyl in place of an H: still a radical, fits
        result = apply_swap(radical, Swap("[*]C", remove_atoms=[h]))
    assert result.mol.GetIntProp("multiplicity") == 2
    # The radical CH2 group itself leaves for a methyl: closed shell, dropped.
    with caplog.at_level("WARNING"):
        result = apply_swap(radical, Swap("[*]C", remove_atoms=[0]))
    assert not result.mol.HasProp("multiplicity") and "does not fit" in caplog.text


def test_swap_rejects_rigid_settings_without_the_route(methylbiphenyl):
    with pytest.raises(ValueError, match="rigid route"):
        racerts.swap(methylbiphenyl, BUTYL_SWAP, n_rotations=6, n_conformers=2)
    with pytest.raises(ValueError, match="Invalid hard"):
        racerts.swap(methylbiphenyl, BUTYL_SWAP, hard=[1.5], n_conformers=2)


def test_a_reference_that_does_not_sanitize_is_swapped_like_it():
    # Connectivity graphs of TSs can have hypervalent atoms (benchmark propargylation:
    # Si with six bonds); the swap takes what the reference takes.
    mol = Chem.MolFromSmiles("C[Si](F)(F)(F)(F)F", sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    mol = Chem.AddHs(mol)
    h = next(
        n.GetIdx()
        for n in mol.GetAtomWithIdx(0).GetNeighbors()
        if n.GetAtomicNum() == 1
    )
    result = apply_swap(mol, Swap("[*]C", remove_atoms=[h]))
    assert result.mol.GetAtomWithIdx(1).GetDegree() == 6
    with pytest.raises(ValueError, match="invalid"):  # a new valence error still raises
        apply_swap(embedded("CCO"), Swap("[*]=C", remove_atoms=[2]))


def test_contacts_of_the_reference_are_no_clashes(caplog):
    # Benchmark Pd_carbofluorination: an O...Pd contact of 2.08 A that the graph
    # lacks (two fragments) is part of the reference, not a clash of the swap.
    from rdkit.Geometry import Point3D

    mol = Chem.RWMol(Chem.AddHs(Chem.MolFromSmiles("CCCCO.O")))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    conf = mol.GetConformer()
    water = 5  # the O of the water, placed 1.5 A from C0
    target = conf.GetAtomPosition(0)
    shift = target - conf.GetAtomPosition(water) + Point3D(1.5, 0, 0)
    for i in [water, *(n.GetIdx() for n in mol.GetAtomWithIdx(water).GetNeighbors())]:
        conf.SetAtomPosition(i, conf.GetAtomPosition(i) + shift)
    h = next(
        n.GetIdx()
        for n in mol.GetAtomWithIdx(4).GetNeighbors()
        if n.GetAtomicNum() == 1
    )
    with caplog.at_level("WARNING"):
        racerts.swap(mol.GetMol(), Swap("[*]C", remove_atoms=[h]), conserve="hard")
    assert "clash" not in caplog.text


def _hypervalent_si():
    mol = Chem.MolFromSmiles("C[Si](F)(F)(F)(F)F", sanitize=False)
    mol.UpdatePropertyCache(strict=False)
    return Chem.AddHs(mol)


def test_the_sanitization_fallback_hides_no_new_errors():
    mol = _hypervalent_si()
    h = next(
        n.GetIdx()
        for n in mol.GetAtomWithIdx(0).GetNeighbors()
        if n.GetAtomicNum() == 1
    )
    for fragment in ("[*]=C", "[*]#N"):  # a five- or six-bonded carbon
        with pytest.raises(ValueError, match="invalid"):
            apply_swap(mol, Swap(fragment, remove_atoms=[h]))
    # A reference that does not kekulize: raises, instead of saturating the ring.
    ring = Chem.MolFromSmiles("Cc1cccc1", sanitize=False)
    ring.UpdatePropertyCache(strict=False)
    with pytest.raises(ValueError, match="invalid"):
        apply_swap(ring, Swap("[*]F", remove_atoms=[0]))


@pytest.mark.parametrize(
    "reference, fragment",
    [
        ("C[C@@H](O)CC", "[*]C(O)C"),
        ("C[C@@H](F)CC", "[*]C(F)C"),
        ("C/C=C\\CC", "[*]C=CC"),
    ],
)
def test_new_stereo_does_not_depend_on_the_seed(reference, fragment):
    # C3 becomes a stereocentre whose stereogenicity depends on the unspecified stereo
    # of the graft; its configuration comes from the reference whatever the seed.
    mol = embedded(reference)
    hydrogens = [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(3).GetNeighbors()
        if n.GetAtomicNum() == 1
    ]
    results = [
        apply_swap(mol, Swap(fragment, remove_atoms=[hydrogens[0]]), seed=s)
        for s in range(8)
    ]
    assert len({identity(r.mol) for r in results}) == 1
    assert results[0].mol.GetAtomWithIdx(3).GetChiralTag() in TETRAHEDRAL_TAGS


@pytest.mark.parametrize("legacy", [True, False])
def test_new_stereo_with_either_stereo_perception(legacy):
    Chem.SetUseLegacyStereoPerception(legacy)
    try:
        mol = embedded("C/C=C/C(C)C")  # a methyl H of the isopropyl -> CH3
        h = next(
            n.GetIdx()
            for n in mol.GetAtomWithIdx(4).GetNeighbors()
            if n.GetAtomicNum() == 1
        )
        result = apply_swap(mol, Swap("[*]C", remove_atoms=[h]))
        assert result.mol.GetAtomWithIdx(3).GetChiralTag() in TETRAHEDRAL_TAGS
    finally:
        Chem.SetUseLegacyStereoPerception(True)


def test_the_multiplicity_parity_counts_hydrogens_and_dummies(caplog):
    template = Chem.AddHs(Chem.MolFromSmiles("[*:1]c1ccc([*:2])cc1"))
    AllChem.EmbedMolecule(template, randomSeed=1)
    template.SetIntProp("multiplicity", 3)
    once = apply_swap(template, Swap("[*]C", site=1))
    assert once.mol.GetIntProp("multiplicity") == 3
    heavy = Chem.MolFromSmiles("CCO")  # implicit hydrogens
    AllChem.EmbedMolecule(heavy, randomSeed=1)
    heavy.SetIntProp("multiplicity", 1)
    with caplog.at_level("WARNING"):
        chloro = apply_swap(heavy, Swap("[*]Cl", remove_atoms=[2]))
    assert (
        chloro.mol.GetIntProp("multiplicity") == 1 and "does not fit" not in caplog.text
    )


def test_clashes_are_judged_against_each_reference(caplog):
    # A water 1.5 A beyond C0 in reference conformer 1 only (away from the swap at the
    # other end): not a clash of conformer 1, which has it in its own reference.
    mol = Chem.RWMol(Chem.AddHs(Chem.MolFromSmiles("CCCCO.O")))
    AllChem.EmbedMultipleConfs(mol, 2, randomSeed=3)
    water = 5
    for conf_id, offset in ((0, 8.0), (1, 1.5)):
        conf = mol.GetConformer(conf_id)
        x = conf.GetPositions()
        away = x[0] - x[:5].mean(axis=0)
        shift = x[0] + offset * away / np.linalg.norm(away) - x[water]
        for i in [
            water,
            *(n.GetIdx() for n in mol.GetAtomWithIdx(water).GetNeighbors()),
        ]:
            conf.SetAtomPosition(i, (x[i] + shift).tolist())
    h = next(
        n.GetIdx()
        for n in mol.GetAtomWithIdx(4).GetNeighbors()
        if n.GetAtomicNum() == 1
    )
    with caplog.at_level("WARNING"):
        racerts.swap(mol.GetMol(), Swap("[*]C", remove_atoms=[h]), conserve="hard")
    assert "clash" not in caplog.text


def test_e_z_survives_when_a_stereo_atom_leaves():
    # Cl (a stereo atom of the double bond) leaves without replacement: the other
    # neighbour of that end carries the E/Z.
    reference = Chem.MolFromSmiles("F/C(Cl)=C/CC")
    result = apply_swap(reference, Swap("[*:1]Br", remove_atoms=[2], attach_map={1: 5}))
    assert identity(result.mol) == canonical("F/C=C/CCBr")
