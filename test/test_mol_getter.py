import logging
import os
import time

import pytest
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds

import racerts.mol_getter.mol_getter as mol_getter_module
from racerts import ConformerGenerator

DATA = os.path.join(os.path.dirname(__file__), "data")


def test_smiles_that_does_not_match_the_xyz_file_raises():
    # ex.xyz is hept-1-ene; hex-1-ene lacks a CH2 group of the geometry.
    with pytest.raises(ValueError, match="do not match"):
        ConformerGenerator().get_mol(
            os.path.join(DATA, "ex.xyz"),
            0,
            [3, 4, 5],
            input_smiles=["CCCCC=C"],
            auto_fallback=False,
        )


def test_rejected_smiles_is_reported_before_falling_back(caplog):
    with caplog.at_level(logging.WARNING):
        mol = ConformerGenerator().get_mol(
            os.path.join(DATA, "ex.xyz"), 0, [3, 4, 5], input_smiles=["CCCCC=C"]
        )

    assert mol.GetNumAtoms() == 21  # graph from the fallback, not from the SMILES
    assert "do not match" in caplog.text


def test_missing_smiles_falls_back_without_warning(caplog):
    with caplog.at_level(logging.WARNING):
        ConformerGenerator().get_mol(os.path.join(DATA, "ex.xyz"), 0, [3, 4, 5])

    assert not caplog.records


def _bonds(path, charge, reacting, smiles):
    mol = ConformerGenerator().get_mol(
        path, charge, reacting, input_smiles=smiles, auto_fallback=False
    )
    return sorted(
        (b.GetBeginAtomIdx(), b.GetEndAtomIdx(), str(b.GetBondType()))
        for b in mol.GetBonds()
    ), [a.GetFormalCharge() for a in mol.GetAtoms()]


def test_complete_atom_maps_replace_the_mcs_search(sn2_ts, monkeypatch):
    # Map number n is atom n of the xyz file; unmapped hydrogens go to their carbon.
    expected = _bonds(sn2_ts, -1, [0, 1, 2], ["CCl", "[Cl-]"])

    def no_mcs(*args, **kwargs):
        raise AssertionError("MCS search despite complete atom maps")

    monkeypatch.setattr(mol_getter_module.rdFMCS, "FindMCS", no_mcs)
    assert _bonds(sn2_ts, -1, [0, 1, 2], ["[CH3:1][Cl:2]", "[Cl-:3]"]) == expected


def test_partial_atom_maps_replace_the_mcs_search(monkeypatch):
    # Typical use: only the reacting atoms (3, 4, 5; 0-based) have map numbers.
    path = os.path.join(DATA, "ex.xyz")
    expected = _bonds(path, 0, [3, 4, 5], ["CCCCCC=C"])

    def no_mcs(*args, **kwargs):
        raise AssertionError("MCS search despite atom maps")

    monkeypatch.setattr(mol_getter_module.rdFMCS, "FindMCS", no_mcs)
    assert _bonds(path, 0, [3, 4, 5], ["CCC[CH2:4][CH2:5][CH:6]=C"]) == expected


@pytest.mark.parametrize("chloride", [2, 3])
def test_partial_atom_maps_decide_ambiguous_matches(sn2_ts_symmetric, chloride):
    # Both chlorines are 2.32 A from carbon; the map number says which is the chloride.
    _, charges = _bonds(sn2_ts_symmetric, -1, [0, 1, 2], ["CCl", f"[Cl-:{chloride}]"])

    assert charges[chloride - 1] == -1


def _in_water(xyz_path, n_water):
    """The structure plus n waters on a grid 4 A apart, 6 A above and below it."""
    lines = open(xyz_path).read().strip().splitlines()[2:]
    grid = [
        (4.0 * i, 4.0 * j, z)
        for i in range(-3, 4)
        for j in range(-3, 4)
        for z in (-6, 6)
    ]
    for x, y, z in grid[:n_water]:
        lines += [
            f"O {x} {y} {z}",
            f"H {x + 0.96} {y} {z}",
            f"H {x - 0.24} {y + 0.93} {z}",
        ]
    return f"{len(lines)}\n\n" + "\n".join(lines) + "\n"


def test_partial_atom_maps_with_many_identical_fragments(
    sn2_ts_symmetric, tmp_path, monkeypatch
):
    # 60 atoms, waters listed first, and the first connectivity step cannot match
    # (no C-Cl bond in the symmetric TS). One search over the whole SMILES tries the
    # waters in all orders (seconds for 8 waters); fragment by fragment it is fast.
    path = tmp_path / "sn2_in_18_waters.xyz"
    path.write_text(_in_water(sn2_ts_symmetric, 18))

    def no_mcs(*args, **kwargs):
        raise AssertionError("MCS search despite atom maps")

    monkeypatch.setattr(mol_getter_module.rdFMCS, "FindMCS", no_mcs)
    start = time.time()
    _, charges = _bonds(str(path), -1, [0, 1, 2], ["O"] * 18 + ["CCl", "[Cl-:3]"])

    assert charges[2] == -1 and sum(charges) == -1
    assert time.time() - start < 10


@pytest.mark.parametrize(
    "smiles",
    [
        "[CH3:1][CH2:2][CH2:3][CH2:4][CH2:5][CH:6]=[CH2:30]",  # no atom 30
        "[CH3:1][CH2:3][CH2:2][CH2:4][CH2:5][CH:6]=[CH2:7]",  # atoms 1 and 3 not bonded
        "[CH3:30]CCCCC=C",  # partial maps: no atom 30
        "[CH3:2]CCCCC=C",  # partial maps: atom 2 is inside the chain
    ],
)
def test_atom_maps_that_do_not_fit_fall_back_to_the_mcs(smiles, caplog):
    path = os.path.join(DATA, "ex.xyz")
    with caplog.at_level(logging.WARNING):
        mapped = _bonds(path, 0, [3, 4, 5], [smiles])

    assert mapped == _bonds(path, 0, [3, 4, 5], ["CCCCCC=C"])
    assert "maximum common substructure" in caplog.text


def test_mcs_matches_with_bonds_missing_from_the_geometry_are_flagged(sn2_ts, caplog):
    getter = mol_getter_module.MolGetterSMILES()
    template = getter.combine_mols(["CCl", "[Cl-]"])
    xyz = Chem.MolFromXYZFile(sn2_ts)
    rdDetermineBonds.DetermineConnectivity(xyz)
    template, xyz = getter.match_AtomMapNum(template, xyz)
    assert getter.bonds_missing_from_geometry(template, xyz, [0, 1, 2]) == []

    # Swap the numbers of a methyl hydrogen and the nucleophile (2.6 A from carbon).
    h, cl = xyz.GetAtomWithIdx(3), xyz.GetAtomWithIdx(2)
    h_number, cl_number = h.GetAtomMapNum(), cl.GetAtomMapNum()
    h.SetAtomMapNum(cl_number)
    cl.SetAtomMapNum(h_number)
    assert (0, 2) in getter.bonds_missing_from_geometry(template, xyz, [0, 1])


def test_invalid_smiles_raises_clearly(sn2_ts):
    with pytest.raises(ValueError, match="Invalid SMILES"):
        ConformerGenerator().get_mol(
            sn2_ts, -1, [0, 1, 2], input_smiles=["C1CCl", "[Cl-]"], auto_fallback=False
        )


def test_smiles_that_does_not_match_the_charge_raises(sn2_ts):
    # A forgotten charge=-1: the SMILES says chloride, the charge says neutral.
    with pytest.raises(ValueError, match="formal charges of the SMILES"):
        ConformerGenerator().get_mol(
            sn2_ts, 0, [0, 1, 2], input_smiles=["CCl", "[Cl-]"], auto_fallback=False
        )


def test_mol_file_keeps_hydrogens():
    mol = ConformerGenerator().get_mol(os.path.join(DATA, "ex.mol"), 0, [3, 4, 5])

    assert mol.GetNumAtoms() == 21


@pytest.mark.parametrize(
    "smiles, reacting_atoms",
    [(["CCCCCC=C"], 3), ("CCCCCC=C", None), (7, [3, 4, 5])],
    ids=["one atom", "no atoms", "no smiles"],
)
def test_smiles_and_reacting_atoms_must_be_lists(smiles, reacting_atoms):
    # Either argument of the wrong type raises, not only both.
    with pytest.raises(ValueError, match="must be provided as lists"):
        mol_getter_module.MolGetterSMILES().get_mol(
            os.path.join(DATA, "ex.xyz"),
            input_smiles=smiles,
            reacting_atoms=reacting_atoms,
            charge=0,
        )


def test_tuples_work_as_lists():
    getter = mol_getter_module.MolGetterSMILES()
    path = os.path.join(DATA, "ex.xyz")
    kwargs = dict(charge=0)
    from_tuples = getter.get_mol(
        path, input_smiles=("CCCCCC=C",), reacting_atoms=(3, 4, 5), **kwargs
    )
    from_lists = getter.get_mol(
        path, input_smiles=["CCCCCC=C"], reacting_atoms=[3, 4, 5], **kwargs
    )
    assert Chem.MolToSmiles(from_tuples) == Chem.MolToSmiles(from_lists)
