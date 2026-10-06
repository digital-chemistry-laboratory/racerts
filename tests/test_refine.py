"""Refinement details: the optimizer interface, the UFF fallback, reused optimizers
and the ASE atoms builder."""

import os

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import racerts
from racerts import EmbedConfig, PipelineConfig, TransitionState
from racerts.refine import BaseOptimizer, MMFFOptimizer

from .conftest import DATA

BORONIC_ACID = os.path.join(DATA, "baseline", "boronic_acid.xyz")


def test_force_field_refinement(hept_1_ene_ts):
    # -- an optimizer implements refine
    class Empty(BaseOptimizer):
        pass

    class Zero(BaseOptimizer):
        def _refine(self, mol, reference, anchors):
            for conf in mol.GetConformers():
                conf.SetDoubleProp("energy", 0.0)

    with pytest.raises(TypeError):
        Empty()
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    with pytest.raises(ValueError, match="reference"):
        Zero().refine(mol, anchors=[0])

    # -- refine falls back to uff
    # MMFF has no boron parameters.
    def run(fallback):
        pipeline = racerts.Pipeline(
            [racerts.Embed(n_conformers=3), racerts.Refine(fallback=fallback)]
        )
        return racerts.generate_ts(BORONIC_ACID, [0, 1, 2], pipeline=pipeline)

    assert run(True).energy_method == "UFFOptimizer"
    with pytest.raises(ValueError, match="MMFF"):
        run(False)

    # -- a pipeline can be reused for another reference
    mol = hept_1_ene_ts
    task = TransitionState([3, 4, 5])
    pipeline = PipelineConfig(embed=EmbedConfig(n_conformers=5)).build(task)
    first = racerts.generate(mol, task, pipeline=pipeline)

    # The same TS, but its reference conformer has id 7 (e.g. from an ensemble).
    other = Chem.Mol(mol)
    other.GetConformer().SetId(7)
    second = racerts.generate(other, task, pipeline=pipeline)

    assert second.energies() == pytest.approx(first.energies())

    # -- converged refinement reaches the minimum
    # Legacy racerts stops the minimization next to the stiff anchors early: a second
    # pass still lowers the energies by kcal/mol.
    legacy, ctx = _refined(MMFFOptimizer(energies_without_anchors=True), hept_1_ene_ts)
    assert np.max(_gain_of_a_second_pass(legacy, ctx)) > 1.0

    converged, ctx = _refined(
        MMFFOptimizer(converge=True, energies_without_anchors=True), hept_1_ene_ts
    )
    assert np.max(np.abs(_gain_of_a_second_pass(converged, ctx))) < 1e-2
    # The frozen atoms stay at the reference.
    frozen = list(ctx.frozen.hard)
    reference = ctx.reference.GetConformer().GetPositions()[frozen]
    for conf in converged.mol.GetConformers():
        assert np.abs(conf.GetPositions()[frozen] - reference).max() < 1e-3

    # -- energies without anchors leave out the anchor terms
    def excess(optimizer):
        ensemble, _ = _refined(optimizer, hept_1_ene_ts)
        return [
            conf.GetDoubleProp("energy") - _mmff_energy(ensemble.mol, conf.GetId())
            for conf in ensemble.mol.GetConformers()
        ]

    assert max(excess(MMFFOptimizer())) > 1e-3  # legacy: anchor terms included
    assert max(np.abs(excess(MMFFOptimizer(energies_without_anchors=True)))) < 1e-8

    # -- mmff dielectric settings
    # A zwitterion: a distance-dependent dielectric of 4 weakens the salt bridge, so
    # the energy differs from the default (constant, 1).
    mol = Chem.AddHs(Chem.MolFromSmiles("[NH3+]CCCCC(=O)[O-]"))
    AllChem.EmbedMolecule(mol, randomSeed=3)

    def energy(**settings):
        work = Chem.Mol(mol)
        MMFFOptimizer(**settings).refine(work)
        return work.GetConformer().GetDoubleProp("energy")

    assert energy() == pytest.approx(energy(dielectric_constant=1.0))
    assert energy(dielectric_model="distance", dielectric_constant=4.0) > energy() + 10
    with pytest.raises(ValueError, match="dielectric_model"):
        MMFFOptimizer(dielectric_model="water")
    with pytest.raises(ValueError, match="dielectric_constant"):
        MMFFOptimizer(dielectric_constant=0)

    # -- the uff fallback keeps the new settings
    pipeline = racerts.Pipeline(
        [
            racerts.Embed(n_conformers=3),
            racerts.Refine(MMFFOptimizer(converge=True, energies_without_anchors=True)),
        ]
    )
    ensemble = racerts.generate_ts(BORONIC_ACID, [0, 1, 2], pipeline=pipeline)
    assert ensemble.energy_method == "UFFOptimizer"

    for conf in ensemble.mol.GetConformers():
        uff = AllChem.UFFGetMoleculeForceField(
            ensemble.mol, confId=conf.GetId(), ignoreInterfragInteractions=False
        )
        assert conf.GetDoubleProp("energy") == pytest.approx(uff.CalcEnergy(), abs=1e-8)

    # -- refine without anchors moves the frozen atoms
    ensemble, ctx = _embedded(hept_1_ene_ts, n=3)
    racerts.Refine(anchors=False).run(ctx, ensemble)
    frozen = list(ctx.frozen.hard)
    reference = ctx.reference.GetConformer().GetPositions()[frozen]
    moved = [
        np.abs(conf.GetPositions()[frozen] - reference).max()
        for conf in ensemble.mol.GetConformers()
    ]
    assert min(moved) > 0.05


@pytest.mark.ase
def test_the_ase_atoms_builder_can_be_patched_in_racerts_io(hept_1_ene_ts, monkeypatch):
    pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones

    import racerts.io.ase
    from racerts.refine import ASEOptimizer

    calls = []
    builder = racerts.io.ase.rdkit_conformer_to_ase_atoms

    def counting_builder(*args, **kwargs):
        calls.append(1)
        return builder(*args, **kwargs)

    monkeypatch.setattr(
        racerts.io.ase, "rdkit_conformer_to_ase_atoms", counting_builder
    )
    mol = hept_1_ene_ts
    ensemble = racerts.Pipeline(
        [
            racerts.Embed(n_conformers=2),
            racerts.Refine(ASEOptimizer(calculator=LennardJones(), max_steps=2)),
        ]
    ).run(racerts.Context.create(mol, TransitionState([3, 4, 5])))

    assert len(calls) == 2 and not np.isnan(ensemble.energies()).any()


def _embedded(mol, n=12):
    """n conformers of the TS of ex.xyz, embedded (not refined)."""
    task = TransitionState([3, 4, 5])
    ctx = racerts.Context.create(mol, task)
    return racerts.Embed(n_conformers=n).run(ctx), ctx


def _refined(optimizer, mol):
    ensemble, ctx = _embedded(mol)
    optimizer.refine(ensemble.mol, ctx.reference, ctx.frozen.hard)
    return ensemble, ctx


def _gain_of_a_second_pass(ensemble, ctx):
    """How much a second legacy MMFF pass lowers each energy (kcal/mol)."""
    again = ensemble.copy()
    MMFFOptimizer(energies_without_anchors=True).refine(
        again.mol, ctx.reference, ctx.frozen.hard
    )
    return ensemble.energies() - again.energies()


def _mmff_energy(mol, conf_id):
    props = AllChem.MMFFGetMoleculeProperties(mol)
    return AllChem.MMFFGetMoleculeForceField(
        mol, props, confId=conf_id, ignoreInterfragInteractions=False
    ).CalcEnergy()


def test_worker_processes_of_the_force_field(hept_1_ene_ts):
    # -- worker processes give the results of one process
    # RDKit's minimizer holds Python's lock, so threads gain nothing; processes do.
    task = TransitionState([3, 4, 5])
    ctx = racerts.Context.create(hept_1_ene_ts, task, seed=4)
    embedded = racerts.Embed(n_conformers=7).run(ctx)
    serial = racerts.Refine(MMFFOptimizer()).run(ctx, embedded.copy())
    pooled = racerts.Refine(MMFFOptimizer(num_workers=3)).run(ctx, embedded.copy())

    assert pooled.conf_ids == serial.conf_ids
    assert pooled.energies().tolist() == serial.energies().tolist()
    for conf_id in serial.conf_ids:
        np.testing.assert_array_equal(
            pooled.mol.GetConformer(conf_id).GetPositions(),
            serial.mol.GetConformer(conf_id).GetPositions(),
        )
    assert pooled.energy_method == "MMFFOptimizer"
    # The provenance of the embedding is still there.
    assert pooled.provenance(pooled.conf_ids[0]) == serial.provenance(
        serial.conf_ids[0]
    )

    # -- worker processes with restraints and the uff fallback
    # Restraints travel to the workers.
    task = TransitionState([3, 4, 5])
    ctx = racerts.Context.create(hept_1_ene_ts, task, seed=4)
    restraint = racerts.restraints.DistanceRestraint.around(0, 6, 4.5, 0.1)
    ctx = ctx.with_restraints([restraint])
    embedded = racerts.Embed(n_conformers=4).run(ctx)
    serial = racerts.Refine(MMFFOptimizer()).run(ctx, embedded.copy())
    pooled = racerts.Refine(MMFFOptimizer(num_workers=2)).run(ctx, embedded.copy())
    assert pooled.energies().tolist() == serial.energies().tolist()
    free = racerts.Refine(MMFFOptimizer(num_workers=2)).run(
        ctx.with_restraints([]), embedded.copy()
    )
    assert free.energies().tolist() != pooled.energies().tolist()  # they acted

    # An MMFF failure in the workers falls back to UFF, with the same workers.
    pipeline = racerts.Pipeline(
        [racerts.Embed(n_conformers=4), racerts.Refine(MMFFOptimizer(num_workers=2))]
    )
    ensemble = racerts.generate_ts(BORONIC_ACID, [0, 1, 2], pipeline=pipeline)
    assert ensemble.energy_method == "UFFOptimizer" and len(ensemble) == 4

    with pytest.raises(ValueError, match="num_workers"):
        MMFFOptimizer(num_workers=0)
