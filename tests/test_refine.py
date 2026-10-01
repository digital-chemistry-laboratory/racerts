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


def test_an_optimizer_implements_refine():
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


def test_refine_falls_back_to_uff():
    # MMFF has no boron parameters.
    def run(fallback):
        pipeline = racerts.Pipeline(
            [racerts.Embed(n_conformers=3), racerts.Refine(fallback=fallback)]
        )
        return racerts.generate_ts(BORONIC_ACID, [0, 1, 2], pipeline=pipeline)

    assert run(True).energy_method == "UFFOptimizer"
    with pytest.raises(ValueError, match="MMFF"):
        run(False)


def test_a_pipeline_can_be_reused_for_another_reference(hept_1_ene_ts):
    mol = hept_1_ene_ts
    task = TransitionState([3, 4, 5])
    pipeline = PipelineConfig(embed=EmbedConfig(n_conformers=5)).build(task)
    first = racerts.generate(mol, task, pipeline=pipeline)

    # The same TS, but its reference conformer has id 7 (e.g. from an ensemble).
    other = Chem.Mol(mol)
    other.GetConformer().SetId(7)
    second = racerts.generate(other, task, pipeline=pipeline)

    assert second.energies() == pytest.approx(first.energies())


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


def _mmff_energy(mol, conf_id):
    props = AllChem.MMFFGetMoleculeProperties(mol)
    return AllChem.MMFFGetMoleculeForceField(
        mol, props, confId=conf_id, ignoreInterfragInteractions=False
    ).CalcEnergy()


def test_anchor_free_energies_leave_out_the_anchor_terms(hept_1_ene_ts):
    def excess(optimizer):
        ensemble, _ = _refined(optimizer, hept_1_ene_ts)
        return [
            conf.GetDoubleProp("energy") - _mmff_energy(ensemble.mol, conf.GetId())
            for conf in ensemble.mol.GetConformers()
        ]

    assert max(excess(MMFFOptimizer())) > 1e-3  # legacy: anchor terms included
    assert max(np.abs(excess(MMFFOptimizer(anchor_free_energies=True)))) < 1e-8


def test_refine_without_anchors_moves_the_frozen_atoms(hept_1_ene_ts):
    ensemble, ctx = _embedded(hept_1_ene_ts, n=3)
    racerts.Refine(anchors=False).run(ctx, ensemble)
    frozen = list(ctx.frozen.hard)
    reference = ctx.reference.GetConformer().GetPositions()[frozen]
    moved = [
        np.abs(conf.GetPositions()[frozen] - reference).max()
        for conf in ensemble.mol.GetConformers()
    ]
    assert min(moved) > 0.05
