"""Pipelines run stages on a Context; custom stages slot in."""

import logging

import numpy as np
import pytest
from rdkit import Chem

import racerts
from racerts import Context, GroundState, Pipeline, TransitionState


class KeepLowest:
    """A custom stage: keep the n conformers lowest in energy."""

    name = "keep_lowest"

    def __init__(self, n):
        self.n = n

    def run(self, ctx, ensemble):
        order = sorted(ensemble.conf_ids, key=ensemble.energy)
        return ensemble.filter(order[: self.n])


def test_custom_stages_slot_in(hept_1_ene_ts, caplog):
    ctx = Context.create(hept_1_ene_ts, TransitionState([3, 4, 5]))
    pipeline = Pipeline(
        [racerts.Embed(n_conformers=10), racerts.Refine(), KeepLowest(3)]
    )

    with caplog.at_level(logging.INFO, logger="racerts"):
        ensemble = pipeline.run(ctx)

    assert isinstance(KeepLowest(1), racerts.Stage)
    assert len(ensemble) == 3
    assert [
        r.getMessage().split(":")[0]
        for r in caplog.records
        if r.name == "racerts.pipeline.runner"
    ] == [
        "embed",
        "refine",
        "keep_lowest",
    ]


def test_a_pipeline_does_not_change_the_ensemble_it_gets(hept_1_ene_ts):
    ctx = Context.create(hept_1_ene_ts, TransitionState([3, 4, 5]))
    embedded = Pipeline([racerts.Embed(n_conformers=4)]).run(ctx)
    before = [conf.GetPositions() for conf in embedded.mol.GetConformers()]

    refined = Pipeline([racerts.Refine(), racerts.PruneRMSD()]).run(ctx, embedded)

    assert len(embedded) == 4 and np.isnan(embedded.energies()).all()
    assert all(
        np.array_equal(conf.GetPositions(), positions)
        for conf, positions in zip(embedded.mol.GetConformers(), before)
    )
    assert refined is not embedded and not np.isnan(refined.energies()).any()


def test_embed_starts_an_ensemble(hept_1_ene_ts):
    # A second Embed would draw the same random numbers again (same seed): exact
    # copies of the first conformers. Ensembles of two runs are merged explicitly.
    ctx = Context.create(hept_1_ene_ts, TransitionState([3, 4, 5]))
    embedded = Pipeline([racerts.Embed(n_conformers=2)]).run(ctx)

    with pytest.raises(ValueError, match="Embed starts an ensemble"):
        Pipeline([racerts.Embed(n_conformers=2)]).run(ctx, embedded)
    with pytest.raises(ValueError, match="no stages"):
        Pipeline([]).run(ctx)


def test_context_works_on_a_copy(hept_1_ene_ts):
    ctx = Context.create(hept_1_ene_ts, TransitionState([3, 4, 5]), seed=5, charge=0)

    assert ctx.mol is not hept_1_ene_ts and not hept_1_ene_ts.HasProp("multiplicity")
    assert ctx.mol.GetIntProp("charge") == 0 and ctx.mol.GetIntProp("multiplicity") == 1
    assert ctx.reference is ctx.mol and ctx.seed == 5
    graph = ctx.graph()
    assert graph.GetNumConformers() == 0 and graph.GetIntProp("multiplicity") == 1


def test_tasks_with_a_reference_need_a_geometry():
    graph = Chem.AddHs(Chem.MolFromSmiles("CCCCCC=C"))

    with pytest.raises(ValueError, match="needs a reference geometry"):
        Context.create(graph, TransitionState([3, 4, 5]))
    assert Context.create(graph, GroundState()).reference is None
