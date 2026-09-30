"""Tasks decide which atoms stay at the reference geometry."""

import pytest
from rdkit import Chem

from racerts import Constrained, Context, FrozenSet, GroundState, Task, TransitionState
from racerts.embed.bounds import fixed_distance_pairs
from racerts.task.transition_state import get_frozen_atoms


def test_transition_state_freezes_the_reacting_atoms_and_neighbours(hept_1_ene_ts):
    frozen = TransitionState([3, 4, 5]).frozen_atoms(hept_1_ene_ts)

    # Order as in legacy racerts (neighbours of each reacting atom, then the atom): it enters
    # the force-field sums and the alignment, so sorting would change the results.
    assert frozen.hard == (2, 4, 14, 15, 3, 5, 16, 17, 6, 18)
    assert list(frozen.hard) == get_frozen_atoms(hept_1_ene_ts, [3, 4, 5])
    assert frozen.core == (3, 4, 5)


def test_given_frozen_atoms_replace_the_neighbours(hept_1_ene_ts):
    frozen = TransitionState([3, 4, 5], frozen_atoms=[3, 4, 5, 6]).frozen_atoms(
        hept_1_ene_ts
    )

    assert frozen.hard == (3, 4, 5, 6) and frozen.core == (3, 4, 5)


def test_invalid_reacting_atoms_raise(hept_1_ene_ts):
    with pytest.raises(ValueError, match="Invalid reacting atoms"):
        TransitionState([3, 100]).frozen_atoms(hept_1_ene_ts)


def test_ground_state_freezes_nothing(hept_1_ene_ts):
    frozen = GroundState().frozen_atoms(hept_1_ene_ts)

    assert frozen == FrozenSet() and not frozen
    assert GroundState.needs_reference is False


def test_constrained_fixes_all_distances_between_the_hard_atoms(hept_1_ene_ts):
    frozen = Constrained(hard=[3, 4, 5]).frozen_atoms(hept_1_ene_ts)

    assert frozen.hard == (3, 4, 5) and frozen.core == (3, 4, 5)  # core: default hard
    assert sorted(fixed_distance_pairs(frozen)) == [
        (3, 4),
        (3, 5),
        (4, 3),
        (4, 5),
        (5, 3),
        (5, 4),
    ]


@pytest.mark.parametrize("hard, message", [([], "at least one"), ([3, 3], "Repeated")])
def test_constrained_checks_its_atoms(hard, message):
    with pytest.raises(ValueError, match=message):
        Constrained(hard=hard)


@pytest.mark.parametrize(
    "task",
    [Constrained(hard=[3, 99]), TransitionState([3, 4, 5], frozen_atoms=[3, 4, 99])],
)
def test_frozen_atoms_of_every_task_are_checked(hept_1_ene_ts, task):
    with pytest.raises(ValueError, match=r"Invalid frozen atoms: \[99\]"):
        Context.create(hept_1_ene_ts, task)


@pytest.mark.parametrize(
    "task", [TransitionState([0]), GroundState(), Constrained([0])]
)
def test_tasks_follow_the_protocol(task):
    assert isinstance(task, Task)


def _sn2_endpoints():
    """CH3Cl + Cl- and Cl- + CH3Cl, atom-aligned (C 0, Cl 1, Cl 2, H 3-5)."""
    reactant = Chem.AddHs(Chem.MolFromSmiles("CCl.[Cl-]"))
    product = Chem.RWMol(reactant)
    product.RemoveBond(0, 1)
    product.AddBond(0, 2, Chem.BondType.SINGLE)
    product.GetAtomWithIdx(1).SetFormalCharge(-1)
    product.GetAtomWithIdx(2).SetFormalCharge(0)
    return reactant, product.GetMol()


def test_transition_state_from_endpoints():
    reactant, product = _sn2_endpoints()
    task = TransitionState.from_endpoints(reactant, product)
    assert task.reacting_atoms == [0, 1, 2]
    assert task.bond_changes == [(0, 1), (0, 2)]
    assert TransitionState([0]).bond_changes is None


def test_from_endpoints_needs_aligned_endpoints_with_bond_changes():
    ethene = Chem.MolFromSmiles("C=C")
    ethane_graph = Chem.RWMol(ethene)
    ethane_graph.GetBondWithIdx(0).SetBondType(Chem.BondType.SINGLE)
    with pytest.raises(ValueError, match="no bond forms or breaks"):
        TransitionState.from_endpoints(ethene, ethane_graph.GetMol())
    with pytest.raises(ValueError, match="same atoms"):
        TransitionState.from_endpoints(
            Chem.MolFromSmiles("CO"), Chem.MolFromSmiles("OC")
        )
