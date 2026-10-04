"""Byte-identity references for legacy racerts.

The routes of tests/routes.py must write the files in data/baseline, which were made with
the fixed legacy code (5b9216a) on RDKit 2025.03.2. Embedding results differ between RDKit versions, so
these tests skip on others (CI runs them in a job of their own); tests/test_compat.py
compares the routes with each other on any RDKit. To rewrite the files (only for a
deliberate change of results):

    python -m tests.test_baseline
"""

import filecmp
import os
import sys

import pytest
from rdkit import rdBase

from .routes import ASE_CASE, BASELINE, COMMANDS, CURRENT, LEGACY, run_command, write

REFERENCE_RDKIT = "2025.03.2"

pytestmark = pytest.mark.skipif(
    rdBase.rdkitVersion != REFERENCE_RDKIT,
    reason=f"the baseline was made with RDKit {REFERENCE_RDKIT}",
)


def _assert_identical(path, name):
    assert filecmp.cmp(path, os.path.join(BASELINE, name), shallow=False), (
        f"{name} differs from the baseline"
    )


@pytest.mark.parametrize("name", [name for name in LEGACY if name != ASE_CASE])
def test_output_matches_the_baseline(name, tmp_path):
    _assert_identical(write(LEGACY, name, tmp_path), name)


@pytest.mark.ase
def test_ase_output_matches_the_baseline(tmp_path):
    pytest.importorskip("ase")
    _assert_identical(write(LEGACY, ASE_CASE, tmp_path), ASE_CASE)


@pytest.mark.parametrize("name", [name for name in CURRENT if name != ASE_CASE])
def test_new_api_output_matches_the_baseline(name, tmp_path):
    _assert_identical(write(CURRENT, name, tmp_path), name)


@pytest.mark.ase
def test_new_api_ase_output_matches_the_baseline(tmp_path):
    pytest.importorskip("ase")
    _assert_identical(write(CURRENT, ASE_CASE, tmp_path), ASE_CASE)


@pytest.mark.parametrize("command", list(COMMANDS))
def test_cli_output_matches_the_baseline(command, tmp_path, monkeypatch):
    path = run_command(command, tmp_path / "cli.xyz", monkeypatch)
    _assert_identical(path, "ex_cmap_mmff.xyz")


if __name__ == "__main__":
    if rdBase.rdkitVersion != REFERENCE_RDKIT:
        sys.exit(f"Use RDKit {REFERENCE_RDKIT}, not {rdBase.rdkitVersion}.")
    for name in LEGACY:
        print("wrote", write(LEGACY, name, BASELINE))
