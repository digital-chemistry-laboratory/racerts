"""Byte-identity references for the restructuring (racer 2.0 step 0).

Each case runs the legacy API through one route (graph, embedder, optimizer, fallback) and
writes its ensemble with write_xyz. The files in data/baseline were made with Phase 1
(5b9216a) on RDKit 2025.03.2; embedding results differ between RDKit versions, so the
tests skip on others. The current API (generate_ts, pipelines, racerts ts) must write the
same files. To rewrite them (only for a deliberate change of results):

    python tests/test_baseline.py
"""

import filecmp
import os
import sys

import pytest
from rdkit import rdBase

import racerts
from racerts import ConformerGenerator
from racerts.embedder import BoundsMatrixEmbedder
from racerts.optimizer import UFFOptimizer

REFERENCE_RDKIT = "2025.03.2"
DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
BASELINE = os.path.join(DATA, "baseline")
EX = os.path.join(DATA, "ex.xyz")


def ex_cmap_mmff():
    # The default route: SMILES template, coordinate map, MMFF.
    cg = ConformerGenerator()
    cg.generate_conformers(
        EX, 0, [3, 4, 5], input_smiles=["CCCCCC=C"], number_of_conformers=50
    )
    return cg


def ex_dm_uff():
    cg = ConformerGenerator()
    cg.embedder = BoundsMatrixEmbedder()
    cg.optimizer = UFFOptimizer()
    cg.generate_conformers(
        EX, 0, [3, 4, 5], input_smiles=["CCCCCC=C"], number_of_conformers=50
    )
    return cg


def sn2_partial_maps():
    # Charge -1 in the extended XYZ line; only the nucleophile is mapped; the default
    # conformer count (no rotatable bonds: 30).
    cg = ConformerGenerator()
    cg.generate_conformers(
        os.path.join(BASELINE, "sn2_ts.xyz"),
        -1,
        [0, 1, 2],
        input_smiles=["CCl", "[Cl-:3]"],
    )
    return cg


def boronic_acid_uff_fallback():
    # No SMILES: bonds from DetermineBonds; MMFF has no boron, so UFF takes over.
    cg = ConformerGenerator()
    cg.generate_conformers(
        os.path.join(BASELINE, "boronic_acid.xyz"),
        0,
        [0, 1, 2],
        number_of_conformers=20,
    )
    return cg


def ex_ase_lj():
    from ase.calculators.lj import LennardJones

    from racerts.optimizer import ASEOptimizer

    cg = ConformerGenerator()
    cg.optimizer = ASEOptimizer(calculator=LennardJones(), fmax=0.1, max_steps=10)
    cg.generate_conformers(
        EX, 0, [3, 4, 5], input_smiles=["CCCCCC=C"], number_of_conformers=10
    )
    return cg


# file name -> (case, write_xyz arguments)
OUTPUTS = {
    "ex_cmap_mmff.xyz": (ex_cmap_mmff, {}),
    "ex_cmap_mmff_energies.xyz": (ex_cmap_mmff, {"use_energy": True}),
    "ex_dm_uff.xyz": (ex_dm_uff, {}),
    "sn2_partial_maps.xyz": (sn2_partial_maps, {}),
    "boronic_acid_uff_fallback.xyz": (boronic_acid_uff_fallback, {}),
    "ex_ase_lj.xyz": (ex_ase_lj, {}),
}


def ex_generate_ts():
    return racerts.generate_ts(
        EX,
        [3, 4, 5],
        smiles="CCCCCC=C",
        config=racerts.PipelineConfig(embed=racerts.EmbedConfig(n_conformers=50)),
    )


def ex_generate_ts_bounds_uff():
    config = racerts.PipelineConfig.from_dict(
        {"embed": {"mode": "bounds", "n_conformers": 50}, "refine": {"backend": "uff"}}
    )
    return racerts.generate_ts(EX, [3, 4, 5], smiles=["CCCCCC=C"], config=config)


def sn2_generate_ts():
    return racerts.generate_ts(
        os.path.join(BASELINE, "sn2_ts.xyz"),
        [0, 1, 2],
        charge=-1,
        smiles=["CCl", "[Cl-:3]"],
    )


def boronic_acid_generate_ts():
    config = racerts.PipelineConfig(embed=racerts.EmbedConfig(n_conformers=20))
    return racerts.generate_ts(
        os.path.join(BASELINE, "boronic_acid.xyz"), [0, 1, 2], config=config
    )


def ex_ase_pipeline():
    # Stages built by hand: their defaults are those of legacy racerts.
    from ase.calculators.lj import LennardJones

    from racerts.refine import ASEOptimizer

    pipeline = racerts.Pipeline(
        [
            racerts.Embed(n_conformers=10),
            racerts.Refine(
                ASEOptimizer(calculator=LennardJones(), fmax=0.1, max_steps=10)
            ),
            racerts.PruneEnergy(),
            racerts.PruneRMSD(),
        ]
    )
    return racerts.generate_ts(EX, [3, 4, 5], smiles="CCCCCC=C", pipeline=pipeline)


# file name -> (current API case, write_xyz arguments)
NEW_API = {
    "ex_cmap_mmff.xyz": (ex_generate_ts, {}),
    "ex_cmap_mmff_energies.xyz": (ex_generate_ts, {"use_energy": True}),
    "ex_dm_uff.xyz": (ex_generate_ts_bounds_uff, {}),
    "sn2_partial_maps.xyz": (sn2_generate_ts, {}),
    "boronic_acid_uff_fallback.xyz": (boronic_acid_generate_ts, {}),
    "ex_ase_lj.xyz": (ex_ase_pipeline, {}),
}

pytestmark = pytest.mark.skipif(
    rdBase.rdkitVersion != REFERENCE_RDKIT,
    reason=f"the baseline was made with RDKit {REFERENCE_RDKIT}",
)


def _write(name, directory, cases=OUTPUTS):
    case, kwargs = cases[name]
    path = os.path.join(directory, name)
    case().write_xyz(path, **kwargs)
    return path


def _assert_identical(name, tmp_path, cases=OUTPUTS):
    path = _write(name, str(tmp_path), cases)
    assert filecmp.cmp(path, os.path.join(BASELINE, name), shallow=False), (
        f"{name} differs from the baseline"
    )


@pytest.mark.parametrize("name", [name for name in OUTPUTS if name != "ex_ase_lj.xyz"])
def test_output_matches_the_baseline(name, tmp_path):
    _assert_identical(name, tmp_path)


@pytest.mark.ase
def test_ase_output_matches_the_baseline(tmp_path):
    pytest.importorskip("ase")
    _assert_identical("ex_ase_lj.xyz", tmp_path)


@pytest.mark.parametrize("name", [name for name in NEW_API if name != "ex_ase_lj.xyz"])
def test_new_api_output_matches_the_baseline(name, tmp_path):
    _assert_identical(name, tmp_path, NEW_API)


@pytest.mark.ase
def test_new_api_ase_output_matches_the_baseline(tmp_path):
    pytest.importorskip("ase")
    _assert_identical("ex_ase_lj.xyz", tmp_path, NEW_API)


@pytest.mark.parametrize(
    "command",
    [
        ["racerts", EX, "-atoms", "3", "4", "5", "-smiles", "CCCCCC=C", "-n", "50"],
        [
            "racerts",
            "run",
            EX,
            "-atoms",
            "3",
            "4",
            "5",
            "-smiles",
            "CCCCCC=C",
            "-n",
            "50",
        ],
        ["racerts", "ts", EX, "-r", "3", "4", "5", "-s", "CCCCCC=C", "-n", "50"],
    ],
    ids=["legacy", "run", "ts"],
)
def test_cli_output_matches_the_baseline(command, tmp_path, monkeypatch):
    from racerts.cli import main

    out = tmp_path / "cli.xyz"
    monkeypatch.setattr(sys, "argv", command + ["-o", str(out)])
    main()

    assert filecmp.cmp(out, os.path.join(BASELINE, "ex_cmap_mmff.xyz"), shallow=False)


if __name__ == "__main__":
    if rdBase.rdkitVersion != REFERENCE_RDKIT:
        sys.exit(f"Use RDKit {REFERENCE_RDKIT}, not {rdBase.rdkitVersion}.")
    for name in OUTPUTS:
        print("wrote", _write(name, BASELINE))
