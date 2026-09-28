"""Byte-identity references for the restructuring (racer 2.0 step 0).

Each case runs the 1.x API through one route (graph, embedder, optimizer, fallback) and
writes its ensemble with write_xyz. The files in data/baseline were made with Phase 1
(5b9216a) on RDKit 2025.03.2; embedding results differ between RDKit versions, so the
tests skip on others. To rewrite the files (only for a deliberate change of results):

    python test/test_baseline.py
"""

import filecmp
import os
import sys

import pytest
from rdkit import rdBase

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

pytestmark = pytest.mark.skipif(
    rdBase.rdkitVersion != REFERENCE_RDKIT,
    reason=f"the baseline was made with RDKit {REFERENCE_RDKIT}",
)


def _write(name, directory):
    case, kwargs = OUTPUTS[name]
    path = os.path.join(directory, name)
    case().write_xyz(path, **kwargs)
    return path


def _assert_identical(name, tmp_path):
    path = _write(name, str(tmp_path))
    assert filecmp.cmp(
        path, os.path.join(BASELINE, name), shallow=False
    ), f"{name} differs from the baseline"


@pytest.mark.parametrize("name", [name for name in OUTPUTS if name != "ex_ase_lj.xyz"])
def test_output_matches_the_baseline(name, tmp_path):
    _assert_identical(name, tmp_path)


@pytest.mark.ase
def test_ase_output_matches_the_baseline(tmp_path):
    pytest.importorskip("ase")
    _assert_identical("ex_ase_lj.xyz", tmp_path)


def test_cli_output_matches_the_baseline(tmp_path, monkeypatch):
    from racerts.cli import main

    out = tmp_path / "cli.xyz"
    argv = ["racerts", EX, "-atoms", "3", "4", "5", "-smiles", "CCCCCC=C"]
    monkeypatch.setattr(sys, "argv", argv + ["-n", "50", "-o", str(out)])
    main()

    assert filecmp.cmp(out, os.path.join(BASELINE, "ex_cmap_mmff.xyz"), shallow=False)


if __name__ == "__main__":
    if rdBase.rdkitVersion != REFERENCE_RDKIT:
        sys.exit(f"Use RDKit {REFERENCE_RDKIT}, not {rdBase.rdkitVersion}.")
    for name in OUTPUTS:
        print("wrote", _write(name, BASELINE))
