"""Polish: the lowest families completed one change at a time."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import racerts
from racerts import Polish
from racerts.exploit import ring_moves
from racerts.refine import MMFFOptimizer
from racerts.system.rings import closest_form, mean_plane, pucker_forms

from .test_recombine import ANTI, GAUCHE, Recording, _heptane, _start, _states


def test_polish_completes_the_lowest_families():
    # Heptane one torsion away from its lowest conformer (all anti): the first pass
    # turns that torsion, the second finds nothing lower.
    mol = _heptane((GAUCHE, ANTI, ANTI, ANTI))
    ctx, ensemble = _start(mol)
    stage = Polish(MMFFOptimizer())
    out = stage.run(ctx, ensemble)

    # -- the lowest conformer, in two passes of every single change
    best = out.best()
    assert _states(out, best) == "aaaa" and out.energy(best) < out.energy(0) - 0.5
    stats = stage.stats
    # Four torsions (not the methyl rotors) with two other states each: 8 children of
    # the start, 8 of the lowest.
    assert (stats["families"], stats["passes"], stats["stop"]) == (1, 2, "converged")
    assert stats["children"] == stats["optimizations"] == 16
    assert stats["accepted"] + stats["refound"] + stats["rejected"] == 16
    provenance = out.provenance(best)
    assert provenance["route"] == "polish" and provenance["parents"] == [0]
    assert provenance["polish_change"].startswith("torsion 1-2 ")
    assert provenance["polish_pass"] == 0
    assert provenance["polish_family"] == 0 and provenance["polish_hits"] >= 1
    # No random numbers: the same result again, whatever the seed.
    ctx, ensemble = _start(mol, seed=99)
    again = Polish(MMFFOptimizer()).run(ctx, ensemble)
    np.testing.assert_allclose(again.energies(), out.energies())

    # -- what the task holds does not turn: with the gauche torsion's atoms held, the
    # lowest conformer is out of reach
    optimizer = Recording()
    ctx, ensemble = _start(mol, optimizer, racerts.Constrained(hard=[0, 1, 2, 3]))
    del optimizer.given[:]
    out = Polish(optimizer).run(ctx, ensemble)
    assert "aaaa" not in {_states(out, i) for i in out.conf_ids}
    start = ensemble.mol.GetConformer(0).GetPositions()[:4]
    assert all(np.abs(child[:4] - start).max() < 1e-3 for child in optimizer.given)

    # -- families: the lowest torsion states, each completed in turn; a budget stops it
    mol = _heptane(
        (GAUCHE, GAUCHE, ANTI, ANTI), (ANTI, -GAUCHE, -GAUCHE, ANTI), (GAUCHE,) * 4
    )
    ctx, ensemble = _start(mol)
    stage = Polish(MMFFOptimizer(), families=2)
    out = stage.run(ctx, ensemble)
    assert stage.stats["families"] == 2 and _states(out, out.best()) == "aaaa"
    ctx, ensemble = _start(mol)
    stage = Polish(MMFFOptimizer(), families=3, max_optimizations=10)
    stage.run(ctx, ensemble)
    assert stage.stats["optimizations"] == 10 and stage.stats["stop"] == "budget"
    # mode "lowest": the lowest families again after every round, so the lower ones a
    # pass finds get passes of their own
    ctx, ensemble = _start(mol)
    stage = Polish(MMFFOptimizer(), families=2, mode="lowest")
    out = stage.run(ctx, ensemble)
    assert _states(out, out.best()) == "aaaa" and stage.stats["stop"] == "converged"
    assert (
        stage.stats["families"] > 2 and stage.stats["passes"] >= stage.stats["families"]
    )


def _form(positions, ring):
    """The name of the canonical form closest to the ring's pucker."""
    center, normal = mean_plane(positions[list(ring)])
    forms = pucker_forms(len(ring))
    return forms[closest_form((positions[list(ring)] - center) @ normal, forms)][0]


def test_polish_takes_a_ring_to_its_lowest_form():
    # Cyclohexanol refined in a twist-boat: the chair is one ring change away.
    mol = Chem.AddHs(Chem.MolFromSmiles("OC1CCCCC1"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    (move,) = ring_moves(mol, set())
    positions = mol.GetConformer().GetPositions()
    move.apply(positions, None, form="twist-boat 30")
    for atom, position in enumerate(positions):
        mol.GetConformer().SetAtomPosition(atom, position.tolist())
    ctx, ensemble = _start(mol)
    start = ensemble.mol.GetConformer(0).GetPositions()
    assert _form(start, move.ring).startswith("twist-boat")

    stage = Polish(MMFFOptimizer(), kinds=("pucker",))
    out = stage.run(ctx, ensemble)
    best = out.best()
    assert "chair" in _form(out.mol.GetConformer(best).GetPositions(), move.ring)
    assert out.energy(best) < out.energy(0) - 3.0
    assert out.provenance(best)["polish_change"].startswith("ring ")
    # Seven other forms of the one ring in each pass, no torsion turned.
    assert stage.stats["children"] == 7 * stage.stats["passes"]

    # -- what the settings must be
    for wrong in (
        dict(kinds=()),
        dict(kinds=("rigid",)),
        dict(families=0),
        dict(tolerance=-0.1),
        dict(mode="greedy"),
        dict(max_optimizations=0),
    ):
        with pytest.raises(ValueError):
            Polish(MMFFOptimizer(), **wrong)
    assert "Polish(" in repr(Polish(MMFFOptimizer()))


def test_duplicates_by_the_polar_hydrogens():
    # Butan-1-ol with its O-H rotor turned, or its methyl group: by the heavy atoms
    # both are the same conformer; by the polar hydrogens (include_hs="polar", as the
    # default RMSDPruner counts them) the first is another one; by all, both are.
    from rdkit.Chem import rdMolTransforms

    from racerts.exploit.population import Population

    mol = Chem.AddHs(Chem.MolFromSmiles("OCCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=7)

    def hydrogen(atom):
        return next(
            n.GetIdx()
            for n in mol.GetAtomWithIdx(atom).GetNeighbors()
            if n.GetAtomicNum() == 1
        )

    conf = mol.GetConformer()
    rdMolTransforms.SetDihedralDeg(conf, 2, 1, 0, hydrogen(0), 180.0)
    turned, methyl = Chem.Conformer(conf), Chem.Conformer(conf)
    rdMolTransforms.SetDihedralDeg(turned, 2, 1, 0, hydrogen(0), 60.0)
    now = rdMolTransforms.GetDihedralDeg(conf, 2, 3, 4, hydrogen(4))
    rdMolTransforms.SetDihedralDeg(methyl, 2, 3, 4, hydrogen(4), now + 60.0)
    conf.SetDoubleProp("energy", 0.0)
    ensemble = racerts.ConformerEnsemble(mol)
    ctx = racerts.Context.create(mol, racerts.GroundState())
    for include_hs, twins in (
        (False, (0, 0)),
        ("polar", (None, 0)),
        (True, (None, None)),
    ):
        population = Population(ctx, ensemble, include_hs)
        key = population.bins[0]
        found = [
            population.duplicate(c.GetPositions(), 0.1, key, 0.25)
            for c in (turned, methyl)
        ]
        assert tuple(found) == twins, include_hs
    for stage in (Polish, racerts.Recombine, racerts.Resample):
        assert stage(MMFFOptimizer(), include_hs="polar").include_hs == "polar"
        with pytest.raises(ValueError, match="include_hs"):
            stage(MMFFOptimizer(), include_hs="some")


def test_a_lower_refind_takes_the_place_of_its_twins():
    # A refined candidate that duplicates members but lies lower than all of them takes
    # their place: the lowest of a set of duplicates stays, as RMSDPruner keeps it.
    # Butan-1-ol with its O-H turned: duplicates by the heavy atoms.
    from rdkit.Chem import rdMolTransforms

    from racerts.exploit.population import Population, add_refined

    mol = Chem.AddHs(Chem.MolFromSmiles("OCCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    hydrogen = next(
        n.GetIdx()
        for n in mol.GetAtomWithIdx(0).GetNeighbors()
        if n.GetAtomicNum() == 1
    )

    def turned(angle, energy=None):
        conf = Chem.Conformer(mol.GetConformer())
        rdMolTransforms.SetDihedralDeg(conf, 2, 1, 0, hydrogen, angle)
        if energy is not None:
            conf.SetDoubleProp("energy", energy)
        return conf

    def population_of(*members):
        start = Chem.Mol(mol)
        start.RemoveAllConformers()
        for angle, energy in members:
            start.AddConformer(turned(angle, energy), assignId=True)
        ensemble = racerts.ConformerEnsemble(start)
        ctx = racerts.Context.create(start, racerts.GroundState())
        return ctx, ensemble, Population(ctx, ensemble, False)

    def candidate(ctx, population, angle, energy):
        refined = Chem.Mol(mol)
        refined.RemoveAllConformers()
        refined.AddConformer(turned(angle, energy), assignId=True)
        counts = dict(rejected=0, refound=0, accepted=0, observations=0)
        counts.update(lowered=0, merged=0)
        positions = turned(angle).GetPositions()
        add_refined(
            ctx,
            population,
            racerts.ConformerEnsemble(refined),
            [(0, positions, {"route": "test"})],
            [],
            0.25,
            counts,
        )
        return counts

    # -- higher: a re-find, the member stays as it is
    ctx, ensemble, population = population_of((60.0, 0.5))
    before = ensemble.mol.GetConformer(0).GetPositions()
    counts = candidate(ctx, population, 180.0, 0.8)
    assert (counts["refound"], counts["accepted"], counts["lowered"]) == (1, 0, 0)
    assert ensemble.energy(0) == 0.5
    np.testing.assert_array_equal(ensemble.mol.GetConformer(0).GetPositions(), before)

    # -- lower: still a re-find, but the member takes its structure and energy
    counts = candidate(ctx, population, 180.0, 0.0)
    assert (counts["refound"], counts["lowered"], counts["merged"]) == (1, 1, 0)
    assert ensemble.energy(0) == population.energies[0] == 0.0
    np.testing.assert_allclose(
        ensemble.mol.GetConformer(0).GetPositions(), turned(180.0).GetPositions()
    )
    provenance = ensemble.provenance(0)
    assert provenance["route"] == "test" and provenance["lowered_from"] == 0.5
    assert population.hits[0] == 2
    # within REFIND_LOWER of the member: the same minimum refined again, no change
    counts = candidate(ctx, population, 60.0, -0.005)
    assert (counts["refound"], counts["lowered"]) == (1, 0) and ensemble.energy(
        0
    ) == 0.0

    # -- lower than two members that duplicate it: the lower of them takes its place,
    # the other leaves; lower than one only: a re-find of that one
    ctx, ensemble, population = population_of((60.0, 0.5), (300.0, 0.3))
    counts = candidate(ctx, population, 180.0, 0.45)  # between them: a re-find
    assert (counts["refound"], counts["lowered"], counts["merged"]) == (1, 0, 0)
    assert ensemble.conf_ids == [0, 1] and population.hits == {0: 1, 1: 0}
    counts = candidate(ctx, population, 180.0, 0.0)
    assert (counts["refound"], counts["lowered"], counts["merged"]) == (1, 1, 1)
    assert population.ids == [1] and ensemble.energy(1) == 0.0
    assert population.hits == {1: 2}  # this re-find, and the hit of member 0
    # Member 0 leaves the search at once, its conformer the ensemble only at the end of
    # the stage: a candidate of the same batch may still have it as its parent.
    from racerts.exploit.population import candidates_ensemble

    assert ensemble.conf_ids == [0, 1] and population.survivor(0) == 1
    assert population.survivor(1) == 1
    made = candidates_ensemble(ensemble, [(0, turned(180.0).GetPositions(), {})])
    assert len(made) == 1
    population.drop_merged()
    assert ensemble.conf_ids == [1] and population.survivor(0) == 1
