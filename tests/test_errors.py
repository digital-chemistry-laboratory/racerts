"""The errors of racerts that a caller can tell apart from a wrong call."""

import os

import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import racerts
import racerts.embed.rigid_attach as rigid
from racerts import (
    InconsistentRestraints,
    MoleculeError,
    NoConformersError,
    RacerTSError,
    SwapError,
    TransitionState,
)
from racerts.restraints import build_restraints
from racerts.system import build_mol, mol_from_explicit_h_smiles, mol_from_geometry
from racerts.system.match import mapped_atoms
from racerts.system.swap import Swap, apply_swap

from .conftest import DATA, EX

ALDOL = os.path.join(DATA, "aldol_ts.xyz")
ALDOL_SMILES = ["OC(=O)[C@@H]1CCCN1C(C)=C", "O=Cc1ccccc1"]
REACTING = [0, 10, 11, 12, 19]
ETHANOL = "[H]OC([H])([H])C([H])([H])[H]"


def _ethanol():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    return mol


def test_one_base_and_the_builtin_types_as_before():
    builtin = {
        NoConformersError: RuntimeError,
        InconsistentRestraints: ValueError,
        SwapError: ValueError,
        MoleculeError: ValueError,
    }
    for error, base in builtin.items():
        assert issubclass(error, RacerTSError) and issubclass(error, base)
    assert not issubclass(RacerTSError, (ValueError, RuntimeError))
    # The names they had are the same classes.
    assert racerts.embed.InconsistentRestraints is InconsistentRestraints
    assert racerts.embed.bounds.InconsistentRestraints is InconsistentRestraints
    assert racerts.system.swap.SwapError is SwapError
    assert racerts.errors.NoConformersError is NoConformersError
    assert "inconsistent" in str(InconsistentRestraints())  # its own message


def test_a_molecule_that_cannot_be_used_as_given(tmp_path):
    # -- a SMILES that is not one
    with pytest.raises(MoleculeError, match="Invalid SMILES"):
        racerts.generate_gs("C1CC")
    with pytest.raises(MoleculeError, match="Invalid SMILES"):
        mol_from_explicit_h_smiles("C(")

    # -- a geometry that is not the molecule of the SMILES, or not its state
    geometry = _ethanol()
    for smiles, state, message in [
        ("[H]C([H])([H])OC([H])([H])[H]", {}, "connectivity"),  # dimethyl ether
        ("[H]OC([H])([H])[H]", {}, "atoms"),
        (ETHANOL, {"charge": -1}, "Charge mismatch"),
        (ETHANOL, {"multiplicity": 3}, "Multiplicity mismatch"),
    ]:
        with pytest.raises(MoleculeError, match=message):
            mol_from_geometry(geometry, smiles, **state)

    # -- a SMILES of a transition state that is not the molecule of its file
    with pytest.raises(MoleculeError, match="do not match"):  # hex-1-ene
        build_mol(EX, 0, [3, 4, 5], input_smiles=["CCCCC=C"], auto_fallback=False)
    with pytest.raises(MoleculeError, match="does not match the formal charges"):
        build_mol(EX, 1, [3, 4, 5], input_smiles=["CCCCCC=C"], auto_fallback=False)
    with pytest.raises(MoleculeError, match="atom map number 4 is not an atom"):
        mapped_atoms(Chem.MolFromSmiles("[CH3:1]C[OH:4]"), Chem.MolFromSmiles("CCO"))

    # -- a file that cannot be read
    path = tmp_path / "broken.xyz"
    path.write_text("3\n\nC 0 0 0\n")
    with pytest.raises(MoleculeError, match="No valid mol object"):
        racerts.generate_ts(str(path), [0], mol_getter=racerts.compat.MolGetterBonds())
    with pytest.raises(MoleculeError, match="file extensions"):
        build_mol(str(tmp_path / "ts.pdb"), 0, [0], input_smiles=["C"])

    # -- two endpoints that are not one reaction
    ethene = Chem.MolFromSmiles("C=C")
    with pytest.raises(MoleculeError, match="no bond forms or breaks"):
        TransitionState.from_endpoints(ethene, Chem.Mol(ethene))
    with pytest.raises(MoleculeError, match="same atoms"):
        TransitionState.from_endpoints(ethene, Chem.MolFromSmiles("CC=C"))


def test_a_wrong_call_is_not_an_error_of_racerts():
    # What is wrong whatever the molecule stays a ValueError or TypeError alone.
    for wrong in (
        lambda: racerts.Embed(n_conformers=0),
        lambda: racerts.prune.RMSDPruner(hydrogens="some"),
        lambda: racerts.prune.RMSDPruner(filter_energies=True),
        lambda: TransitionState([3, 4, 5], active_window=-0.1),
    ):
        with pytest.raises((ValueError, TypeError)) as caught:
            wrong()
        assert not isinstance(caught.value, RacerTSError)


@pytest.mark.parametrize("stratify", [0, 5])
def test_restraints_that_cannot_be_met_together(stratify):
    # -- a window that the frozen atoms cannot take
    # They let the forming C-C bond of the aldol TS reach about 3.9 A. Found by the
    # bounds (with targets) or by the embedded lengths (without): one error.
    aldol = build_mol(ALDOL, 0, REACTING, input_smiles=ALDOL_SMILES)
    far = TransitionState(
        REACTING, active_bonds=[(10, 12)], active_window=(4.5, 5.5), stratify=stratify
    )
    with pytest.raises(InconsistentRestraints, match="narrower active_window"):
        racerts.Embed(n_conformers=6).run(racerts.Context.create(aldol, far))

    # -- a fragment link that the other restraints leave no room for
    # H...Cl at 1.2 A puts O...Cl below the widest contact window.
    mol = Chem.AddHs(Chem.MolFromSmiles("O.[Cl-]"))
    with pytest.raises(InconsistentRestraints, match="fragment links cannot"):
        build_restraints(mol, user=[(2, 1, 1.2)], fragment_links=[(0, 1)])


def test_a_swap_whose_fragment_cannot_be_embedded(monkeypatch):
    mol = Chem.AddHs(Chem.MolFromSmiles("Cc1ccccc1-c1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=0)
    result = apply_swap(mol, Swap("[*:1]CCCC", old_fragment="[CH3][c:1]"))
    monkeypatch.setattr(rigid.AllChem, "EmbedMultipleConfs", lambda *a, **k: [])
    with pytest.raises(SwapError, match="Could not embed the fragment"):
        rigid.rigid_attach(result, 2, 3)
