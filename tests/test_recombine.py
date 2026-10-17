"""Recombine: the torsions of the low conformers in new combinations."""

import logging

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

import racerts
from racerts import Recombine
from racerts.exploit import torsion_moves
from racerts.recombine import set_torsions, torsion_angles, torsions
from racerts.refine import MMFFOptimizer

ANTI, GAUCHE = 180.0, 65.0


def _hydrogens(smiles):
    return Chem.AddHs(Chem.MolFromSmiles(smiles))


def _heptane(*states):
    """Heptane with its C-C-C-C torsions (0-1-2-3, 1-2-3-4, 2-3-4-5, 3-4-5-6) at the
    states given (degrees), one conformer for each tuple of four."""
    mol = _hydrogens("CCCCCCC")
    AllChem.EmbedMolecule(mol, randomSeed=7)
    template = Chem.Conformer(mol.GetConformer())
    mol.RemoveAllConformers()
    for values in states:
        conf = Chem.Conformer(template)
        for first, value in enumerate(values):
            rdMolTransforms.SetDihedralDeg(
                conf, first, first + 1, first + 2, first + 3, value
            )
        mol.AddConformer(conf, assignId=True)
    return mol


def _states(ensemble, conf_id):
    """ "a" or "g" for each of the four torsions of a conformer of heptane."""
    conf = ensemble.mol.GetConformer(conf_id)
    angles = [
        rdMolTransforms.GetDihedralDeg(conf, k, k + 1, k + 2, k + 3) for k in range(4)
    ]
    return "".join("a" if abs(angle) > 120 else "g" for angle in angles)


def _start(mol, optimizer=None, task=None, seed=12):
    """A context and the conformers of mol, refined by optimizer (MMFF94)."""
    ctx = racerts.Context.create(mol, task or racerts.GroundState(), seed=seed)
    ensemble = racerts.ConformerEnsemble(Chem.Mol(mol))
    return ctx, racerts.Refine(optimizer or MMFFOptimizer()).run(ctx, ensemble)


def _apart(angles, others):
    """How far angles are from others (degrees), whichever way round."""
    return np.abs((np.asarray(angles) - np.asarray(others) + 180.0) % 360.0 - 180.0)


def _internal_coordinates(mol, positions, without=()):
    """Every bond length, angle and dihedral of the graph at positions, but for the
    dihedrals about the bonds in without."""
    conf = Chem.Conformer(mol.GetNumAtoms())
    for atom, position in enumerate(positions):
        conf.SetAtomPosition(atom, position.tolist())
    values = []
    for bond in mol.GetBonds():
        j, k = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        values.append(rdMolTransforms.GetBondLength(conf, j, k))
        left = [
            n.GetIdx() for n in bond.GetBeginAtom().GetNeighbors() if n.GetIdx() != k
        ]
        right = [
            n.GetIdx() for n in bond.GetEndAtom().GetNeighbors() if n.GetIdx() != j
        ]
        values += [rdMolTransforms.GetAngleDeg(conf, i, j, k) for i in left]
        values += [rdMolTransforms.GetAngleDeg(conf, j, k, m) for m in right]
        if frozenset((j, k)) not in without:
            values += [
                np.cos(np.radians(rdMolTransforms.GetDihedralDeg(conf, i, j, k, m)))
                for i in left
                for m in right
            ]
    return np.array(values)


def test_the_torsions_and_a_child():
    # -- the torsions: single bonds outside rings whose two heavy atoms each carry
    # another heavy atom
    heptane = _heptane((ANTI, ANTI, ANTI, ANTI))
    found = torsions(heptane)
    assert [t.atoms[1:3] for t in found] == [(2, 1), (3, 2), (3, 4), (4, 5)]
    for smiles, count in (
        ("CCO", 0),  # neither a methyl group nor a hydroxyl group
        ("C1CCCCC1", 0),  # no bond of a ring
        ("c1ccccc1-c1ccccc1", 1),
        ("CC(C)(C)CC", 1),  # a symmetric rotor too
        ("CC#CC", 0),  # nothing turns about a linear atom
    ):
        assert len(torsions(_hydrogens(smiles))) == count
    # The planar ones too, which Exploit does not turn: a child gets the value of a
    # parent, cis or trans, and no value in between.
    amide = _hydrogens("CC(=O)NC")
    assert [t.atoms[1:3] for t in torsions(amide)] == [(1, 3)]
    assert torsion_moves(amide, set()) == []

    # -- the angles of several conformers at once
    parents = _heptane((GAUCHE, ANTI, ANTI, ANTI), (ANTI, ANTI, ANTI, -GAUCHE))
    first, second = (conf.GetPositions() for conf in parents.GetConformers())
    angles = torsion_angles(np.array([first, second]), found)
    assert angles.shape == (2, 4)
    for row, conf in zip(angles, parents.GetConformers()):
        rdkit = [rdMolTransforms.GetDihedralDeg(conf, *t.atoms) for t in found]
        assert _apart(row, rdkit).max() < 1e-9

    # -- a child: one torsion state of each parent, the rest of the first
    # The first parent is gauche about 1-2, the second about 4-5; the child takes the
    # torsion 3-4-5-6 of the second, and is gauche about both.
    child = set_torsions(first, found, {3: angles[1, 3]})
    turned = torsion_angles(child, found)
    assert _apart(turned, [*angles[0, :3], angles[1, 3]]).max() < 1e-9
    assert _apart(np.abs(turned), [GAUCHE, ANTI, ANTI, GAUCHE]).max() < 1e-6
    np.testing.assert_allclose(
        _internal_coordinates(heptane, child, without=[frozenset((4, 5))]),
        _internal_coordinates(heptane, first, without=[frozenset((4, 5))]),
        rtol=1e-9,
        atol=1e-9,
    )
    assert np.abs(child - first).max() > 1.0  # the end of the chain did move
    # Several at once, in any order: each to its value.
    both = set_torsions(first, found, {3: angles[1, 3], 0: angles[1, 0], 1: 100.0})
    values = [angles[1, 0], 100.0, angles[0, 2], angles[1, 3]]
    assert _apart(torsion_angles(both, found), values).max() < 1e-9

    # -- the smaller side turns, or the side without atoms that the task holds
    assert [len(t.moving) for t in found] == [6, 9, 9, 6]  # with the hydrogens
    assert found[3].moving == (6, 18, 19, 20, 21, 22)
    held = {5, 6}
    kept = torsions(heptane, held)
    assert [t.atoms[1:3] for t in kept] == [(2, 1), (3, 2), (4, 3), (5, 4)]
    assert not any(held & set(t.moving) for t in kept)
    assert {0, 1, 2, 3}.issubset(kept[3].moving)  # the larger side, about 4-5
    child = set_torsions(first, kept, {0: -60.0, 2: 80.0, 3: 70.0})
    np.testing.assert_array_equal(child[sorted(held)], first[sorted(held)])
    values = [-60.0, torsion_angles(first, kept)[1], 80.0, 70.0]
    assert _apart(torsion_angles(child, kept), values).max() < 1e-9
    # The atoms of the bond stay on the axis, so they may be held; a bond whose turn
    # would move held atoms on either side is no torsion (3-4 with 2 and 5 held).
    assert [t.atoms[1:3] for t in torsions(heptane, {2, 5})] == [(2, 1), (3, 2), (4, 5)]
    assert torsions(heptane, {0, 6}) == []


def test_the_combination_that_no_parent_has():
    # Two conformers of heptane, each with one gauche torsion: the lowest conformer,
    # all anti, takes the anti state of the one from each.
    mol = _heptane((GAUCHE, ANTI, ANTI, ANTI), (ANTI, ANTI, GAUCHE, ANTI))
    ctx, ensemble = _start(mol)
    assert [_states(ensemble, i) for i in ensemble.conf_ids] == ["gaaa", "aaga"]
    stage = Recombine(MMFFOptimizer())
    out = stage.run(ctx, ensemble)

    # -- the children: both combinations, and the parents again
    assert out.energy_method == "MMFFOptimizer"
    assert sorted(_states(out, i) for i in out.conf_ids) == [
        "aaaa",
        "aaga",
        "gaaa",
        "gaga",
    ]
    best = out.best()
    assert _states(out, best) == "aaaa"
    assert out.energy(best) < min(out.energy(0), out.energy(1)) - 0.5

    # -- the provenance and the statistics
    provenance = out.provenance(best)
    assert provenance["route"] == "cross" and sorted(provenance["parents"]) == [0, 1]
    assert provenance["cross_torsions"] == 1 and provenance["cross_round"] == 0
    assert all("route" not in out.provenance(i) for i in (0, 1))
    stats = stage.stats
    # Two parents that differ in two torsions: three children of each as the first.
    assert (stats["children"], stats["refined"], stats["torsions"]) == (6, 6, 4)
    assert (stats["accepted"], stats["refound"], stats["rejected"]) == (2, 4, 0)
    assert (stats["rounds"], stats["stop"]) == (1, "rounds")  # it found a lower one
    hits = [out.provenance(i)["cross_hits"] for i in out.conf_ids]
    assert hits == [1, 1, 2, 2]  # each parent once, each new one twice

    # -- children with overlapping atoms are not refined
    class Counting(MMFFOptimizer):
        calls = 0

        def _refine(self, mol, reference, anchors, restraints=()):
            Counting.calls += mol.GetNumConformers()
            return super()._refine(mol, reference, anchors, restraints)

    ctx, ensemble = _start(mol, Counting())
    Counting.calls = 0
    stage = Recombine(Counting(), overlap_factor=3.0)  # three times the vdW radii
    out = stage.run(ctx, ensemble)
    assert out.conf_ids == [0, 1] and Counting.calls == 0
    assert stage.stats["overlapping"] == 6 and stage.stats["stop"] == "no children"

    # -- with a force field as with any optimizer; the conformers must come from it
    ctx, ensemble = _start(mol)
    assert len(Recombine(racerts.Refine()).run(ctx, ensemble)) == 4  # MMFF, or UFF
    ctx, ensemble = _start(mol)
    with pytest.raises(ValueError, match="energy_method"):
        Recombine(racerts.refine.UFFOptimizer()).run(ctx, ensemble)


def test_rounds_of_recombination():
    # Three conformers, each anti about one of three torsions and gauche about the
    # other two. No child of two of them is anti about all three: the first round
    # makes those with two anti torsions, the second the lowest conformer.
    parents = [(ANTI, GAUCHE, GAUCHE, ANTI), (GAUCHE, ANTI, GAUCHE, ANTI)]
    mol = _heptane(*parents, (GAUCHE, GAUCHE, ANTI, ANTI))

    def search(rounds, **settings):
        ctx, ensemble = _start(mol)
        stage = Recombine(MMFFOptimizer(), rounds=rounds, **settings)
        out = stage.run(ctx, ensemble)
        return out, stage.stats, _states(out, out.best())

    # -- one round finds what two parents have between them
    out, stats, best = search(1)
    assert best.count("g") == 1 and (stats["rounds"], stats["stop"]) == (1, "rounds")

    # -- another round starts from what the round before found
    out, stats, best = search(2)
    assert best == "aaaa" and (stats["rounds"], stats["stop"]) == (2, "rounds")
    provenance = out.provenance(out.best())
    assert provenance["cross_round"] == 1
    first_round = [
        out.provenance(i).get("cross_round") == 0 for i in provenance["parents"]
    ]
    assert any(first_round)  # a parent that the first round made

    # -- and the rounds end when one finds nothing lower
    out, stats, best = search(5)
    assert best == "aaaa" and (stats["rounds"], stats["stop"]) == (3, "nothing lower")
    rounds = {out.provenance(i).get("cross_round") for i in out.conf_ids}
    assert rounds >= {None, 0, 1}

    # -- parents: within the window of the lowest, and no more than asked for
    out, stats, best = search(1, window=0.5)  # the three, 0.2 kcal/mol apart
    assert stats["parents"] == 3
    energies = sorted(out.energies())
    out, stats, _ = search(1, parents=2)
    assert stats["parents"] == 2 and len(out) < len(energies)
    mol = _heptane((ANTI, ANTI, ANTI, ANTI), (GAUCHE, GAUCHE, GAUCHE, ANTI))
    ctx, ensemble = _start(mol)
    stage = Recombine(MMFFOptimizer(), window=1.0)  # the second lies 2.0 above
    assert stage.run(ctx, ensemble).conf_ids == [0, 1]
    assert stage.stats["stop"] == "no parents" and stage.stats["rounds"] == 0


class Recording(MMFFOptimizer):
    """MMFF94 that keeps the structures it is given."""

    def __init__(self):
        super().__init__()
        self.given = []

    def _refine(self, mol, reference, anchors, restraints=()):
        self.given += [conf.GetPositions() for conf in mol.GetConformers()]
        return super()._refine(mol, reference, anchors, restraints)


def test_what_the_task_holds_is_not_turned(hept_1_ene_ts, caplog):
    # -- a constrained end: its atoms are where they were before the refinement
    mol = _heptane(
        (GAUCHE, ANTI, ANTI, ANTI), (ANTI, ANTI, ANTI, GAUCHE)
    )  # 0, 1, 2 the same
    held = [0, 1, 2]
    optimizer = Recording()
    ctx, ensemble = _start(mol, optimizer, racerts.Constrained(hard=held))
    reference = ctx.mol.GetConformer().GetPositions()
    del optimizer.given[:]
    stage = Recombine(optimizer)
    out = stage.run(ctx, ensemble)
    # Both torsions that differ turn: about 1-2 the atoms 3 to 6 (1 and 2 stay on the
    # axis), about 4-5 those of 5 and 6; each parent with one or both of the other's.
    assert stage.stats["torsions"] == 4 and stage.stats["children"] == 6
    assert len(optimizer.given) == 6
    starts = [ensemble.mol.GetConformer(i).GetPositions()[held] for i in (0, 1)]
    for child in optimizer.given:
        assert any(np.array_equal(child[held], start) for start in starts)
    assert racerts.validate.FrozenCore(0.1).validate(ctx, out) == {}
    assert (
        np.abs(out.mol.GetConformer(0).GetPositions()[held] - reference[held]).max()
        < 0.1
    )

    # -- a transition state: the reacting atoms and their neighbours
    task = racerts.TransitionState([3, 4, 5])
    ctx = racerts.Context.create(hept_1_ene_ts, task)
    frozen = sorted(ctx.frozen.hard)
    optimizer = Recording()
    pipeline = racerts.Pipeline(
        [racerts.Embed(n_conformers=20), racerts.Refine(optimizer)]
    )
    ensemble = pipeline.run(ctx)
    before = {i: ensemble.mol.GetConformer(i).GetPositions() for i in ensemble.conf_ids}
    del optimizer.given[:]
    stage = Recombine(optimizer, window=50.0)
    out = stage.run(ctx, ensemble)
    # 0-1-2-3 and 1-2-3-4, which turn 0 and 1 about frozen 2 and 3; the others would
    # move frozen atoms on either side
    assert stage.stats["torsions"] == 2
    assert stage.stats["children"] == len(optimizer.given) > 0
    for child in optimizer.given:
        assert any(np.array_equal(child[frozen], x[frozen]) for x in before.values())
    assert racerts.validate.FrozenCore(0.1).validate(ctx, out) == {}

    # -- nothing to turn: the ensemble passes
    mol = _heptane((GAUCHE, ANTI, ANTI, ANTI), (ANTI, ANTI, ANTI, GAUCHE))
    ctx, ensemble = _start(mol, None, racerts.Constrained(hard=[0, 6]))
    stage = Recombine(MMFFOptimizer())
    with caplog.at_level(logging.WARNING, logger="racerts"):
        assert stage.run(ctx, ensemble).conf_ids == [0, 1]
    assert "nothing to recombine" in caplog.text
    assert stage.stats["stop"] == "no torsions"


def test_seeds_and_settings():
    # Eight conformers with one or two gauche torsions: many children to choose from.
    states = [(GAUCHE, ANTI, ANTI, ANTI), (ANTI, GAUCHE, ANTI, ANTI)]
    states += [(ANTI, ANTI, GAUCHE, ANTI), (ANTI, ANTI, ANTI, GAUCHE)]
    states += [(-GAUCHE, ANTI, ANTI, -GAUCHE), (GAUCHE, GAUCHE, ANTI, ANTI)]
    states += [(ANTI, -GAUCHE, -GAUCHE, ANTI), (ANTI, ANTI, GAUCHE, GAUCHE)]
    mol = _heptane(*states)

    def search(seed=12, **settings):
        # With this threshold nearly every child is a conformer of the result, with
        # its provenance: (the two parents, the number of torsions turned).
        settings = {"children": 6, "rmsd_threshold": 1e-9, **settings}
        ctx, ensemble = _start(mol, seed=seed)
        stage = Recombine(MMFFOptimizer(), **settings)
        out = stage.run(ctx, ensemble)
        made = [out.provenance(i) for i in out.conf_ids[8:]]
        return out, stage.stats, [(p["parents"], p["cross_torsions"]) for p in made]

    # -- the same seed makes the same children, another seed others
    first, again, other = search(12), search(12), search(13)
    assert len(first[2]) > 2 and first[2] == again[2] != other[2]
    np.testing.assert_array_equal(
        *(
            [c.GetPositions() for c in out.mol.GetConformers()]
            for out in (first[0], again[0])
        )
    )

    # -- children: as many as asked for, by default 50 for every torsion
    assert first[1]["children"] == 6 and search(children=7)[1]["children"] == 7
    out, stats, made = search(children=None)
    assert stats["children"] == 200 and len(made) > 150
    assert all(a != b for (a, b), _ in made)

    # -- share: the part of the differing torsions that a child takes, one at least
    def turned(share):
        return np.mean([count for _, count in search(children=40, share=share)[2]])

    assert turned(1.0) > turned(0.5) > turned(0.01) == 1.0

    # -- temperature: the first parent is a low one, the colder the more so
    energies = first[0].energies()[:8]
    low = set(np.nonzero(energies < energies.min() + 0.2)[0].tolist())
    assert 0 < len(low) < 8

    def from_the_low(temperature):
        made = search(children=40, temperature=temperature, window=20.0)[2]
        return np.mean([parents[0] in low for parents, _ in made])

    assert from_the_low(0.02) == 1.0 > from_the_low(50.0)

    # -- different: torsions closer than this are the same
    out, stats, made = search(different=175.0)
    assert stats["stop"] == "no children" and made == []

    # -- what the settings must be
    for wrong in (
        dict(window=0),
        dict(parents=1),
        dict(parents=2.5),
        dict(temperature=0),
        dict(different=0),
        dict(different=180),
        dict(share=0),
        dict(share=1.5),
        dict(children=0),
        dict(children=2.5),
        dict(rounds=0),
        dict(rounds=True),
        dict(rmsd_threshold=0),
        dict(overlap_factor=0),
    ):
        with pytest.raises(ValueError):
            Recombine(MMFFOptimizer(), **wrong)
    assert "Recombine(" in repr(Recombine(MMFFOptimizer()))
