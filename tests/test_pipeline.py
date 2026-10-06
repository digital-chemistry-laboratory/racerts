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


def test_stages_ensembles_and_the_context(hept_1_ene_ts, caplog):
    # -- custom stages slot in
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

    # -- a stage must return the ensemble
    class Forgetful:
        name = "forgetful"

        def run(self, ctx, ensemble):
            return None

    ctx = Context.create(hept_1_ene_ts, TransitionState([3, 4, 5]))
    with pytest.raises(TypeError, match="Stage 'forgetful' returned NoneType"):
        Pipeline([racerts.Embed(n_conformers=2), Forgetful()]).run(ctx)

    # -- a pipeline does not change the ensemble it gets
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

    # -- embed starts an ensemble
    # A second Embed would draw the same random numbers again (same seed): exact
    # copies of the first conformers. Ensembles of two runs are merged explicitly.
    ctx = Context.create(hept_1_ene_ts, TransitionState([3, 4, 5]))
    embedded = Pipeline([racerts.Embed(n_conformers=2)]).run(ctx)

    with pytest.raises(ValueError, match="Embed starts an ensemble"):
        Pipeline([racerts.Embed(n_conformers=2)]).run(ctx, embedded)
    with pytest.raises(ValueError, match="no stages"):
        Pipeline([]).run(ctx)

    # -- context works on a copy
    ctx = Context.create(hept_1_ene_ts, TransitionState([3, 4, 5]), seed=5, charge=0)

    assert ctx.mol is not hept_1_ene_ts and not hept_1_ene_ts.HasProp("multiplicity")
    assert ctx.mol.GetIntProp("charge") == 0 and ctx.mol.GetIntProp("multiplicity") == 1
    assert ctx.reference is ctx.mol and ctx.seed == 5
    graph = ctx.graph()
    assert graph.GetNumConformers() == 0 and graph.GetIntProp("multiplicity") == 1

    # -- tasks with a reference need a geometry
    graph = Chem.AddHs(Chem.MolFromSmiles("CCCCCC=C"))

    with pytest.raises(ValueError, match="needs a reference geometry"):
        Context.create(graph, TransitionState([3, 4, 5]))
    assert Context.create(graph, GroundState()).reference is None


def _two_references(mol):
    """mol with a second reference geometry: the first one rotated and shifted."""
    import numpy as np

    positions = mol.GetConformer().GetPositions()
    c, s = np.cos(1.0), np.sin(1.0)
    rotated = positions @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]).T + [5, 0, 0]
    two = Chem.Mol(mol)
    conf = Chem.Conformer(mol.GetNumAtoms())
    for i, p in enumerate(rotated):
        conf.SetAtomPosition(i, p.tolist())
    conf.SetId(7)
    two.AddConformer(conf)
    return two


def test_several_references(hept_1_ene_ts):
    import numpy as np

    from racerts.validate import FrozenCore

    mol = _two_references(hept_1_ene_ts)
    task = racerts.TransitionState([3, 4, 5])
    pipeline = racerts.Pipeline(
        [racerts.Embed(n_conformers=3, references="all"), racerts.Refine()]
    )
    ctx = racerts.Context.create(mol, task)
    ensemble = pipeline.run(ctx)

    references = [ensemble.provenance(i)["reference"] for i in ensemble.conf_ids]
    assert references == [0, 0, 0, 7, 7, 7]
    # Each conformer is refined onto its own reference: the frozen atoms are there.
    frozen = list(ctx.frozen.hard)
    for conf_id, reference in zip(ensemble.conf_ids, references):
        target = mol.GetConformer(reference).GetPositions()[frozen]
        found = ensemble.mol.GetConformer(conf_id).GetPositions()[frozen]
        assert np.abs(found - target).max() < 1e-3
    assert FrozenCore().validate(ctx, ensemble) == {}

    # Each group has its own reference: the optimizer cannot name one conformer.
    from racerts.refine import MMFFOptimizer

    with pytest.raises(ValueError, match="conf_id_ref"):
        racerts.Refine(MMFFOptimizer(conf_id_ref=0)).run(ctx, ensemble.copy())

    one = racerts.Embed(n_conformers=3, references=[7]).run(ctx)
    assert len(one) == 3 and {one.provenance(i)["reference"] for i in one.conf_ids} == {
        7
    }
    with pytest.raises(ValueError, match="No reference conformers with ids"):
        racerts.Embed(n_conformers=3, references=[3]).run(ctx)
    with pytest.raises(ValueError, match="'all'"):
        racerts.Embed(references="first")
