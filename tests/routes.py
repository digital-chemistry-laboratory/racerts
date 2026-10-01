"""
The routes that must write identical files: each case runs one route (graph, embedder,
optimizer, fallback) through the legacy API, the current API or the command line and
writes its ensemble with write_xyz. test_baseline compares them with the stored Phase 1
files (RDKit 2025.03.2 only); test_compat compares the routes with each other on any
RDKit.
"""

import os
import sys

import racerts
from racerts import ConformerGenerator
from racerts.embedder import BoundsMatrixEmbedder
from racerts.optimizer import UFFOptimizer

from .conftest import DATA, EX

BASELINE = os.path.join(DATA, "baseline")
ASE_CASE = "ex_ase_lj.xyz"


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


# file name -> (legacy API case, write_xyz arguments)
LEGACY = {
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
        config=racerts.PipelineConfig.legacy(embed={"n_conformers": 50}),
    )


def ex_generate_ts_bounds_uff():
    config = racerts.PipelineConfig.legacy(
        embed={"mode": "bounds", "n_conformers": 50}, refine={"backend": "uff"}
    )
    return racerts.generate_ts(EX, [3, 4, 5], smiles=["CCCCCC=C"], config=config)


def sn2_generate_ts():
    return racerts.generate_ts(
        os.path.join(BASELINE, "sn2_ts.xyz"),
        [0, 1, 2],
        charge=-1,
        smiles=["CCl", "[Cl-:3]"],
        config=racerts.PipelineConfig.legacy(),
    )


def boronic_acid_generate_ts():
    config = racerts.PipelineConfig.legacy(embed={"n_conformers": 20})
    return racerts.generate_ts(
        os.path.join(BASELINE, "boronic_acid.xyz"), [0, 1, 2], config=config
    )


def ex_ase_pipeline():
    # Stages built by hand, with the legacy embedding settings.
    from ase.calculators.lj import LennardJones

    from racerts.embed import CmapEmbedder
    from racerts.refine import ASEOptimizer

    embedder = CmapEmbedder(chirality_fallback="legacy", sequential_seeds=False)
    pipeline = racerts.Pipeline(
        [
            racerts.Embed(embedder, n_conformers=10),
            racerts.Refine(
                ASEOptimizer(calculator=LennardJones(), fmax=0.1, max_steps=10)
            ),
            racerts.PruneEnergy(),
            racerts.PruneRMSD(),
        ]
    )
    return racerts.generate_ts(EX, [3, 4, 5], smiles="CCCCCC=C", pipeline=pipeline)


# file name -> (current API case, write_xyz arguments): the same files
CURRENT = {
    "ex_cmap_mmff.xyz": (ex_generate_ts, {}),
    "ex_cmap_mmff_energies.xyz": (ex_generate_ts, {"use_energy": True}),
    "ex_dm_uff.xyz": (ex_generate_ts_bounds_uff, {}),
    "sn2_partial_maps.xyz": (sn2_generate_ts, {}),
    "boronic_acid_uff_fallback.xyz": (boronic_acid_generate_ts, {}),
    "ex_ase_lj.xyz": (ex_ase_pipeline, {}),
}

# Command lines that write the file of the default route (ex_cmap_mmff.xyz).
COMMANDS = {
    "legacy": [EX, "-atoms", "3", "4", "5", "-smiles", "CCCCCC=C", "-n", "50"],
    "run": ["run", EX, "-atoms", "3", "4", "5", "-smiles", "CCCCCC=C", "-n", "50"],
    "ts": ["ts", EX, "-r", "3", "4", "5", "-s", "CCCCCC=C", "-n", "50", "--legacy"],
}


def write(cases, name, directory):
    """Run the case for file name and write its ensemble to directory; its path."""
    case, kwargs = cases[name]
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, name)
    case().write_xyz(path, **kwargs)
    return path


def run_command(command, path, monkeypatch):
    """Run racerts with the arguments of COMMANDS[command], writing to path."""
    from racerts.cli import main

    monkeypatch.setattr(sys, "argv", ["racerts", *COMMANDS[command], "-o", str(path)])
    main()
    return path
