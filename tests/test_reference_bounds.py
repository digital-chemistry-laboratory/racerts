"""Frozen cores that RDKit's distance bounds contradict: the bounds widened to the
reference geometry (reference_bounds of the embedders)."""

import logging

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import racerts
from racerts import TransitionState
from racerts.embed import BoundsMatrixEmbedder, CmapEmbedder
from racerts.embed.bounds import bounds_matrix, fixed_distance_pairs


def _tilted_toluene(tilt):
    """
    Toluene with its methyl group turned by tilt degrees out of the ring plane about the
    ipso carbon, like a metal over the ipso-ortho bond of an oxidative-addition TS.
    RDKit's bounds keep the methyl carbon in the ring plane: at 70 degrees, the meta
    carbons cannot be as far from it as they require.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("Cc1ccccc1"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    AllChem.MMFFOptimizeMolecule(mol)
    conf = mol.GetConformer()
    p = conf.GetPositions()
    ring = p[1:7]
    normal = np.linalg.svd(ring - ring.mean(axis=0))[2][2]
    axis = np.cross(p[0] - p[1], normal)
    axis /= np.linalg.norm(axis)
    t = np.radians(tilt)
    k = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    turn = np.eye(3) + np.sin(t) * k + (1 - np.cos(t)) * k @ k  # Rodrigues
    for i in (0, 7, 8, 9):  # the methyl group
        conf.SetAtomPosition(i, (p[1] + turn @ (p[i] - p[1])).tolist())
    return mol


def _ctx(tilt):
    # Frozen: C0 and C1 and their neighbours (the methyl hydrogens, the ortho carbons).
    return racerts.Context.create(_tilted_toluene(tilt), TransitionState([0, 1]))


def _positions(ensemble):
    return [ensemble.mol.GetConformer(i).GetPositions() for i in ensemble.conf_ids]


def test_bounds_are_widened_to_the_reference(caplog, tmp_path):
    from racerts.cli import main

    # -- rdkit alone cannot embed the tilted core
    with pytest.raises(racerts.NoConformersError, match="no conformers"):
        racerts.Embed(CmapEmbedder(reference_bounds="never"), n_conformers=5).run(
            _ctx(70)
        )

    # -- the bounds are widened to the reference when rdkit cannot
    caplog.clear()
    ctx = _ctx(70)
    embedder = CmapEmbedder()  # reference_bounds="fallback"
    with caplog.at_level(logging.WARNING, logger="racerts"):
        ensemble = racerts.Embed(embedder, n_conformers=10).run(ctx)
    assert len(ensemble) == 10
    assert "bounds widened to the reference" in caplog.text
    # e.g. the methyl carbon and the meta and para carbons (in the ring plane for RDKit)
    assert {(0, 3), (0, 4), (0, 5)} <= {(a, b) for a, b, *_ in embedder.widened}
    reference = ctx.reference.GetConformer().GetPositions()
    held = list(ctx.frozen.hard)
    for conf_id, p in zip(ensemble.conf_ids, _positions(ensemble)):
        assert np.abs(p[held] - reference[held]).max() < 1e-6
        ring = [
            np.linalg.norm(p[a] - p[b]) for a, b in [(2, 3), (3, 4), (4, 5), (5, 6)]
        ]
        assert 1.3 < min(ring) and max(ring) < 1.5
        assert ensemble.provenance(conf_id)["widened_bounds"] > 0

    # -- cores that rdkit embeds are embedded as before
    ctx = _ctx(50)
    before = racerts.Embed(CmapEmbedder(reference_bounds="never"), n_conformers=5)
    now = racerts.Embed(n_conformers=5).run(ctx)
    for a, b in zip(_positions(before.run(ctx)), _positions(now)):
        assert np.array_equal(a, b)
    assert "widened_bounds" not in now.provenance(now.conf_ids[0])

    # -- always widens wherever the reference breaks a bound
    # RDKit embeds the core at 50 degrees, but some of its bounds exclude the reference.
    embedder = CmapEmbedder(reference_bounds="always")
    ensemble = racerts.Embed(embedder, n_conformers=5).run(_ctx(50))
    assert ensemble.provenance(ensemble.conf_ids[0])["widened_bounds"] > 0

    # -- bounds matrices are widened beyond their tolerance
    # The bounds mode and distance windows smooth with a growing tolerance (up to 0.4;
    # the core at 70 degrees needs 0.08), and are widened only beyond it.
    ctx = _ctx(70)
    graph, reference = ctx.graph(), ctx.reference
    pairs = fixed_distance_pairs(ctx.frozen)
    with pytest.raises(Exception, match="Triangle smoothing error"):
        bounds_matrix(graph, reference, pairs, max_tolerance=0.05)
    bounds = bounds_matrix(
        graph, reference, pairs, max_tolerance=0.05, reference_bounds="fallback"
    )
    p = reference.GetConformer().GetPositions()
    d = np.linalg.norm(p[:, None] - p[None], axis=-1)
    upper = np.triu_indices(len(d), 1)
    assert (bounds.T[upper] <= d[upper] + 1e-6).all()  # lower bounds
    assert (d[upper] <= bounds[upper] + 1e-6).all()  # upper bounds
    assert np.allclose(bounds[tuple(np.array(pairs).T)], d[tuple(np.array(pairs).T)])
    assert len(racerts.Embed(BoundsMatrixEmbedder(), n_conformers=3).run(ctx)) == 3

    # -- the setting
    config = racerts.PipelineConfig.from_dict({"embed": {"reference_bounds": "never"}})
    embed = config.build(TransitionState([0, 1])).stages[0]
    assert embed.embedder.reference_bounds == "never"
    assert CmapEmbedder().reference_bounds == "fallback"
    with pytest.raises(ValueError, match="reference_bounds"):
        racerts.PipelineConfig.from_dict({"embed": {"reference_bounds": "sometimes"}})
    with pytest.raises(ValueError, match="reference_bounds"):
        CmapEmbedder(reference_bounds="sometimes")

    # -- the command line
    xyz = str(tmp_path / "tilted.xyz")
    Chem.MolToXYZFile(_tilted_toluene(70), xyz)
    out = str(tmp_path / "out.xyz")
    arguments = ["ts", xyz, "-r", "0", "1", "--smiles", "Cc1ccccc1", "-n", "3"]
    main(arguments + ["-o", out])
    assert open(out).read().startswith("15\n")  # the ring is rigid: one conformer
    with pytest.raises(RuntimeError, match="no conformers"):
        main(arguments + ["--reference-bounds", "never", "-o", out])
