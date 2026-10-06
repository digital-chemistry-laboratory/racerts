"""Independent runs of a search: merged, and compared (do they agree?)."""

import math
from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms

import racerts
from racerts import GroundState, PipelineConfig
from racerts.pipeline import ConformerEnsemble
from racerts.pipeline.runs import compare_runs, merge_runs

RT = 0.0019872041 * 298.15


def _pentane(torsions, energies, shift=0.0):
    """Pentane conformers with these C-C-C-C torsions (degrees) and energies; shift
    moves one atom (A), as a second optimization of the same minimum would."""
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCC"))
    AllChem.EmbedMolecule(mol, randomSeed=3)
    template = Chem.Conformer(mol.GetConformer())
    mol.RemoveAllConformers()
    for (first, second), energy in zip(torsions, energies):
        conf = Chem.Conformer(template)
        rdMolTransforms.SetDihedralDeg(conf, 0, 1, 2, 3, first)
        rdMolTransforms.SetDihedralDeg(conf, 1, 2, 3, 4, second)
        if shift:
            conf.SetAtomPosition(0, list(np.array(conf.GetAtomPosition(0)) + shift))
        conf.SetDoubleProp("energy", energy)
        mol.AddConformer(conf, assignId=True)
    mol.SetProp("energy_method", "test")
    return ConformerEnsemble(mol)


ANTI, GAUCHE, TWIST = (180.0, 180.0), (180.0, 60.0), (60.0, 60.0)


@pytest.fixture
def runs():
    """Three runs: all find the lowest conformer; the third finds everything."""
    return [
        _pentane([ANTI, GAUCHE], [0.0, 1.0]),
        _pentane([ANTI, TWIST], [0.1, 0.5], shift=0.02),
        _pentane([ANTI, GAUCHE, TWIST], [0.05, 1.1, 0.45], shift=-0.02),
    ]


def _free_energy(energies):
    energies = np.asarray(energies)
    return -RT * math.log(np.exp(-energies / RT).sum())


def test_merging_and_comparing_runs(runs):
    # -- merged runs keep the lowest copy and who found it
    merged = merge_runs(runs, labels=[11, 12, 13])
    assert merged.energies().tolist() == [0.0, 0.45, 1.0]  # by energy
    provenance = [merged.provenance(i) for i in merged.conf_ids]
    assert [p["run"] for p in provenance] == [11, 13, 11]
    assert [p["found_by"] for p in provenance] == [[11, 12, 13], [12, 13], [11, 13]]
    assert merged.energy_method == "test"
    assert all(len(run) in (2, 3) for run in runs)  # the runs are not changed

    # Further apart in energy than the tolerance: two conformers, whatever the RMSD.
    assert len(merge_runs(runs, energy_tolerance=0.02)) == 7
    with pytest.raises(ValueError, match="labels"):
        merge_runs(runs, labels=[1, 2])

    # -- compare runs
    report = compare_runs(runs, labels=[11, 12, 13])
    union = _free_energy([0.0, 0.45, 1.0])
    assert report.labels == [11, 12, 13] and report.conformers == [2, 2, 3]
    assert report.lowest == pytest.approx([0.0, 0.1, 0.05])
    expected = [
        _free_energy([0.0, 1.0]) - union,
        _free_energy([0.1, 0.5]) - union,
        _free_energy([0.05, 1.1, 0.45]) - union,
    ]
    assert report.free_energy == pytest.approx(expected)
    assert report.spread == pytest.approx(max(expected) - min(expected))
    # Found again: the population of a run (within the window of its minimum) that the
    # other run has too. Run 11 has the anti and a gauche conformer; 12 has no gauche.
    anti, gauche, twist = 1.0, math.exp(-1.0 / RT), math.exp(-0.4 / RT)
    assert report.found_again[(11, 12)] == pytest.approx(anti / (anti + gauche))
    assert report.found_again[(12, 11)] == pytest.approx(anti / (anti + twist))
    assert report.found_again[(11, 13)] == pytest.approx(1.0)
    assert report.found_again[(12, 13)] == pytest.approx(1.0)
    # Merged: the free energy of k runs together, above that of all (mean, worst).
    assert report.merged[3] == pytest.approx((0.0, 0.0))
    assert report.merged[1] == pytest.approx((np.mean(expected), max(expected)))
    pairs = [
        _free_energy([0.0, 0.5, 1.0]) - union,  # 11 and 12
        0.0,  # 11 and 13
        _free_energy([0.05, 0.45, 1.1]) - union,  # 12 and 13
    ]
    assert report.merged[2] == pytest.approx((np.mean(pairs), max(pairs)))
    # Leaving out one run: the most that a run still changes the result.
    assert report.leave_one_out == pytest.approx(max(pairs))
    assert report.converged(tolerance=0.3) and not report.converged(tolerance=0.001)

    text = str(report)
    assert "3 runs" in text and "found again" in text and "11 -> 12" in text
    assert report.to_dict()["labels"] == [11, 12, 13]

    # -- compare runs checks its input
    with pytest.raises(ValueError, match="at least two runs"):
        compare_runs(runs[:1])
    no_energy = runs[1].copy()
    no_energy.mol.GetConformer(no_energy.conf_ids[0]).ClearProp("energy")
    with pytest.raises(ValueError, match="energy"):
        compare_runs([runs[0], no_energy])
    other_method = runs[1].copy()
    other_method.mol.SetProp("energy_method", "other")
    with pytest.raises(ValueError, match="different methods"):
        compare_runs([runs[0], other_method])
    held = runs[1].copy()
    held.add_provenance(held.conf_ids[0], active_bond_targets={"1-2": 2.0})
    with pytest.raises(ValueError, match="held at targets"):
        compare_runs([runs[0], held])


def test_generate_runs_and_what_it_refuses(sn2_ts):
    from racerts.embed import default_embedder
    from racerts.system import build_mol

    # -- generate runs
    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCCO"))
    config = PipelineConfig.from_dict({"embed": {"n_conformers": 20}})
    result = racerts.generate_runs(mol, GroundState(), seeds=[1, 2, 3], config=config)
    assert result.seeds == [1, 2, 3] and len(result.ensembles) == 3
    assert result.report.labels == [1, 2, 3]
    # Each run is what generate gives for its seed.
    single = racerts.generate(mol, GroundState(), config=replace(config, seed=2))
    assert np.allclose(single.energies(), result.ensembles[1].energies())
    # The merged ensemble: no more conformers than the runs together, each from a run.
    assert max(map(len, result.ensembles)) <= len(result.merged)
    assert len(result.merged) <= sum(map(len, result.ensembles))
    assert {result.merged.provenance(i)["run"] for i in result.merged.conf_ids} <= {
        1,
        2,
        3,
    }
    assert result.report.merged[3] == pytest.approx((0.0, 0.0))

    with pytest.raises(ValueError, match="at least two"):
        racerts.generate_runs(mol, GroundState(), seeds=[1], config=config)
    with pytest.raises(ValueError, match="different seeds"):
        racerts.generate_runs(mol, GroundState(), seeds=[1, 1], config=config)

    # -- runs that ignore the seed are refused
    # A component with its own seed gives every run the same conformers.

    mol = Chem.AddHs(Chem.MolFromSmiles("CCCCCCO"))
    embed = racerts.Embed(default_embedder(GroundState(), 7), n_conformers=5)
    fixed = racerts.Pipeline([embed, racerts.Refine()])
    with pytest.raises(ValueError, match="same conformers.*seed"):
        racerts.generate_runs(mol, GroundState(), seeds=[1, 2], pipeline=fixed)

    # -- a fully frozen ts gives the same conformer for every seed
    # Nothing is free to sample: the runs agree because the task leaves no choice, not
    # because the pipeline ignores the seed.

    mol = build_mol(sn2_ts, -1, [0, 1, 2], input_smiles=["CCl", "[Cl-]"])
    task = racerts.TransitionState([0, 1, 2])
    assert set(task.frozen_atoms(mol).hard) == set(range(mol.GetNumAtoms()))
    runs = racerts.generate_runs(mol, task, seeds=[1, 2])
    assert [len(e) for e in runs.ensembles] == [1, 1] and len(runs.merged) == 1
