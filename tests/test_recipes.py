"""The staged workflow recipe: its stages, and end-to-end runs with GFN2-xTB."""

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import racerts
from racerts import TransitionState
from racerts.recipes import staged
from racerts.refine import MMFFOptimizer, Refine, Rescore
from racerts.validate import gate

pytestmark = pytest.mark.ase
pytest.importorskip("ase")
from ase.calculators.lj import LennardJones  # noqa: E402

from racerts.refine import ASEOptimizer  # noqa: E402

LJ = ASEOptimizer(LennardJones(), method="lj")
STEPS = [
    "embed",
    "validate",  # clashes
    "refine",  # cheap
    "validate",  # the gate
    "attack_face",  # windowed TSs only
    "prune_energy",
    "prune_rmsd",
    "refine",  # expensive
    "validate",  # the gate
    "prune_energy",
    "prune_rmsd",
    "exploit",
    "prune_rmsd",
    "prune_energy",
]


def _thresholds(pipeline):
    return [s.pruner.threshold for s in pipeline.stages if s.name == "prune_energy"]


def test_the_stages():
    pipeline = staged(LJ)
    assert [s.name for s in pipeline.stages] == STEPS
    assert _thresholds(pipeline) == [25.0, 8.0, 6.0]
    cheap, expensive = [s for s in pipeline.stages if s.name == "refine"]
    # The cheap level is the refinement of the default settings: MMFF, not converged.
    assert isinstance(cheap.optimizer, MMFFOptimizer) and not cheap.optimizer.converge
    assert cheap.optimizer.anchor_free_energies and cheap.stereo_anchors
    assert expensive.optimizer is LJ
    exploit = pipeline.stages[STEPS.index("exploit")]
    assert exploit.refine is expensive  # the same refinement as step 4


def test_embedding_and_cheap_refinement_follow_the_config(hept_1_ene_ts):
    ctx = racerts.Context.create(hept_1_ene_ts, TransitionState([3, 4, 5]), seed=5)

    def embedded(config):
        settings = {"embed": {"n_conformers": 6}}
        pipeline = staged(LJ, config=config(**settings))
        ensemble = pipeline.stages[0].run(ctx, None)
        positions = [
            ensemble.mol.GetConformer(i).GetPositions() for i in ensemble.conf_ids
        ]
        copies = sum(
            np.allclose(a, b) for k, a in enumerate(positions) for b in positions[:k]
        )
        return pipeline, ensemble, copies

    # The defaults: a seed per conformer, from the seed of the context.
    pipeline, ensemble, copies = embedded(racerts.PipelineConfig)
    assert len(ensemble) == 6 and copies == 0
    assert ensemble.provenance(ensemble.conf_ids[0])["seed"] != 12
    # Legacy settings: the first three conformers are embedded twice.
    pipeline, ensemble, copies = embedded(racerts.PipelineConfig.legacy)
    assert len(ensemble) == 6 and copies == 3
    cheap = pipeline.stages[2]
    assert not cheap.optimizer.anchor_free_energies and not cheap.stereo_anchors
    with pytest.raises(ValueError, match="config"):  # nothing left for it to set
        staged(
            LJ,
            embed=racerts.Embed(),
            cheap=MMFFOptimizer(),
            config=racerts.PipelineConfig(),
        )


def test_options():
    rescore = Rescore(LennardJones(), method="lj-sp")
    pipeline = staged(
        Refine(LJ, anchors=False),
        cheap=MMFFOptimizer(),
        windows=(30, 10, 5),
        exploit=None,
        rescore=rescore,
        clash_filter=None,
    )
    names = [s.name for s in pipeline.stages]
    assert "exploit" not in names and names.count("validate") == 2
    assert names[-1] == "rescore" and pipeline.stages[-1] is rescore
    assert _thresholds(pipeline) == [30, 10, 5]
    assert pipeline.stages[names.index("refine", 2)].anchors is False
    exploit = staged(LJ, exploit={"batch": 3, "max_optimizations": 9}).stages[11]
    assert exploit.batch == 3 and exploit.max_optimizations == 9


def test_the_pool_for_the_expensive_level():
    # Without a pool every conformer that passes the cheap level is refined at the
    # expensive one. With one, the conformers are chosen by structural family (the
    # best of each cluster first): the cheap energies often misrank.
    names = [s.name for s in staged(LJ, pool=30).stages]
    cheap_duplicates = names.index("prune_rmsd")
    assert names[cheap_duplicates + 1 : cheap_duplicates + 3] == [
        "select_families", "refine",
    ]  # fmt: skip
    selector = staged(LJ, pool=30).stages[cheap_duplicates + 1]
    assert selector.n_max == 30
    by_energy = staged(LJ, pool=30, pool_by="energy").stages[cheap_duplicates + 1]
    assert by_energy.name == "prune_count" and by_energy.n_max == 30
    assert "select_families" not in [s.name for s in staged(LJ).stages]
    with pytest.raises(ValueError, match="pool_by"):
        staged(LJ, pool=30, pool_by="random")
    with pytest.raises((TypeError, ValueError), match="n_max"):
        staged(LJ, pool=0)


def test_the_pool_limits_the_expensive_refinements():
    class Counting(ASEOptimizer):
        def __init__(self):
            super().__init__(LennardJones(), method="counting")
            self.calls = 0

        def _refine(self, mol, reference, anchors, restraints=()):
            for k, conf in enumerate(mol.GetConformers()):
                self.calls += 1
                conf.SetDoubleProp("energy", float(k))
            return 0

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCCO"))
    counting = Counting()
    config = racerts.PipelineConfig.from_dict({"embed": {"n_conformers": 30}})
    pipeline = staged(counting, pool=4, exploit=None, config=config)
    ensemble = racerts.generate(mol, racerts.GroundState(), pipeline=pipeline)
    assert counting.calls == 4 and 1 <= len(ensemble) <= 4


def test_a_middle_level_with_its_own_search():
    # e.g. GFN-FF or GFN2-xTB between the force field and the MLIP: it removes what the
    # force field misranks before the expensive level, and Exploit there is cheap
    # (ring flips, torsions), so the pool for the expensive level draws on more.
    middle = ASEOptimizer(LennardJones(), method="middle")
    pipeline = staged(
        LJ, middle=middle, middle_window=12.0, middle_exploit={"batch": 4}, pool=20
    )
    names = [s.name for s in pipeline.stages]
    first = names.index("prune_rmsd")  # the end of the cheap level
    assert names[first + 1 : first + 9] == [
        "refine", "validate", "prune_energy", "prune_rmsd",  # the middle level
        "exploit", "prune_rmsd",  # its search
        "select_families",  # the pool
        "refine",  # the expensive level
    ]  # fmt: skip
    middle_stage = pipeline.stages[first + 1]
    assert middle_stage.optimizer is middle
    assert _thresholds(pipeline) == [25.0, 12.0, 8.0, 6.0]
    search = pipeline.stages[first + 5]
    assert search.refine is middle_stage and search.batch == 4
    assert names.count("exploit") == 2  # and the one at the expensive level

    # Without a search at the middle level; a search needs the level.
    plain = [s.name for s in staged(LJ, middle=middle).stages]
    assert plain.count("exploit") == 1 and plain.count("refine") == 3
    with pytest.raises(ValueError, match="middle"):
        staged(LJ, middle_exploit={})
    with pytest.raises(ValueError, match="middle_window"):
        staged(LJ, middle=middle, middle_window=0)


def test_a_ranking_energy_follows_every_expensive_refinement():
    rank = Rescore(LennardJones(), method="lj-rank")
    pipeline = staged(LJ, rank=rank)
    names = [s.name for s in pipeline.stages]
    expensive = names.index("refine", 3)  # after the cheap one
    # refine, gate, the ranking energy, then the window and the duplicates
    assert names[expensive : expensive + 5] == [
        "refine", "validate", "rescore", "prune_energy", "prune_rmsd",
    ]  # fmt: skip
    assert pipeline.stages[expensive + 2] is rank
    exploit = pipeline.stages[names.index("exploit")]
    assert exploit.rank is rank
    # A ready Exploit stage keeps its own setting.
    own = racerts.Exploit(LJ)
    assert staged(LJ, rank=rank, exploit=own).stages[names.index("exploit")] is own


def test_the_saddle_recipe():
    from racerts.recipes import saddles
    from racerts.validate import Connectivity, Converged, ReactionCore, ReactionMode

    pipeline = saddles(LJ)
    assert [s.name for s in pipeline.stages] == ["refine", "validate", "prune_rmsd"]
    search, checks, _ = pipeline.stages
    assert search.optimizer is LJ and search.anchors is False and not search.fallback
    kinds = [type(v) for v in checks.validators]
    assert kinds == [Converged, ReactionMode, ReactionCore, Connectivity]
    assert checks.validators[1].calculator is LJ.calculator  # of the Hessian

    other = LennardJones()
    pipeline = saddles(Refine(LJ, anchors=False), calculator=other, pool=5)
    assert [s.name for s in pipeline.stages][0] == "select_families"
    assert pipeline.stages[2].validators[1].calculator is other
    assert saddles(LJ, pool=5, pool_by="energy").stages[0].name == "prune_count"
    with pytest.raises(ValueError, match="anchors=False"):
        saddles(Refine(LJ))  # a search with the frozen atoms held is no free search


@pytest.mark.parametrize("windows", [(25, 8), (25, 8, -1)])
def test_windows_are_checked(windows):
    with pytest.raises(ValueError, match="windows"):
        staged(LJ, windows=windows)


def _diol(factors):
    """
    Propane-1,3-diol conformers whose O0...O4 (four bonds apart, the only pair Clash
    checks) is factor times their vdW sum.
    """
    mol = Chem.AddHs(Chem.MolFromSmiles("OCCCO"))
    AllChem.EmbedMultipleConfs(mol, len(factors), randomSeed=7)
    vdw = 2 * Chem.GetPeriodicTable().GetRvdw(8)
    moved = [4] + [
        n.GetIdx()
        for n in mol.GetAtomWithIdx(4).GetNeighbors()
        if n.GetAtomicNum() == 1
    ]
    for conf, factor in zip(mol.GetConformers(), factors):
        p = conf.GetPositions()
        direction = (p[4] - p[0]) / np.linalg.norm(p[4] - p[0])
        shift = p[0] + factor * vdw * direction - p[4]
        for i in moved:
            conf.SetAtomPosition(i, (p[i] + shift).tolist())
    graph = Chem.Mol(mol)
    graph.RemoveAllConformers()  # no reference geometry, whose contacts Clash allows
    return racerts.Context.create(graph, racerts.GroundState()), mol


def test_the_clash_filter_drops_overlaps_not_contacts_within_rdkits_bounds():
    # Distance geometry may place atoms four bonds apart down to about 0.55 times their
    # vdW sum (RDKit's lower bounds), and refinement relaxes such contacts: only
    # overlaps are dropped.
    ctx, mol = _diol([0.65, 0.45, 1.0])

    def kept(pipeline):
        return pipeline.stages[1].run(ctx, racerts.ConformerEnsemble(Chem.Mol(mol)))

    assert kept(staged(LJ)).conf_ids == [0, 2]
    assert kept(staged(LJ, clash_filter=0.7)).conf_ids == [2]
    assert "validate" not in [s.name for s in staged(LJ, clash_filter=None).stages[:2]]
    for bad, error in [(True, TypeError), (0.0, ValueError), (1.5, ValueError)]:
        with pytest.raises(error, match="clash_filter"):
            staged(LJ, clash_filter=bad)


def _passes_the_gate(ensemble, task):
    ctx = racerts.Context.create(ensemble.mol, task)
    return all(not check.validate(ctx, ensemble) for check in gate().validators)


@pytest.mark.xtb
def test_a_ground_state_end_to_end():
    TBLite = pytest.importorskip("tblite.ase").TBLite
    gfn2 = ASEOptimizer(TBLite(method="GFN2-xTB", verbosity=0), method="GFN2")
    pipeline = staged(
        gfn2,
        embed=racerts.Embed(n_conformers=6),
        exploit={"max_optimizations": 8, "batch": 4},
    )
    ensemble = racerts.generate_gs("CCCCO", pipeline=pipeline)
    assert len(ensemble) >= 1 and ensemble.energy_method == "GFN2"
    for conf_id in ensemble.conf_ids:
        provenance = ensemble.provenance(conf_id)
        assert "validation" in provenance or provenance.get("route") == "mc"
        assert "mc_usage" in provenance
    assert _passes_the_gate(ensemble, racerts.GroundState())


@pytest.mark.xtb
def test_a_ts_with_waters_end_to_end(sn2_ts_two_waters):
    TBLite = pytest.importorskip("tblite.ase").TBLite
    gfn2 = ASEOptimizer(TBLite(method="GFN2-xTB", verbosity=0), method="GFN2")
    pipeline = staged(
        gfn2,
        embed=racerts.Embed(n_conformers=6),
        exploit={"max_optimizations": 8, "batch": 4},
    )
    ensemble = racerts.generate_ts(
        sn2_ts_two_waters, [0, 1, 2], charge=-1, smiles="CCl.[Cl-].O.O",
        pipeline=pipeline,
    )  # fmt: skip
    assert len(ensemble) >= 1 and ensemble.energy_method == "GFN2"
    assert _passes_the_gate(ensemble, TransitionState([0, 1, 2]))
