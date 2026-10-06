"""generate, generate_ts and generate_gs."""

import logging
import os

import numpy as np
import pytest
from rdkit.Chem import rdMolAlign

import racerts
from racerts import Constrained, EmbedConfig, PipelineConfig
from racerts.prune import RMSDPruner
from racerts.system import build_mol

from .conftest import EX

SMALL = PipelineConfig(embed=EmbedConfig(n_conformers=10))


def test_the_entry_points(hept_1_ene_ts):
    from racerts.system import MolGetterConnectivity

    # -- generate ts returns an ensemble with provenance
    ensemble = racerts.generate_ts(EX, [3, 4, 5], smiles="CCCCCC=C", config=SMALL)
    record = ensemble.record(ensemble.best())

    assert isinstance(ensemble, racerts.ConformerEnsemble)
    assert record.energy_method == "MMFFOptimizer"
    assert record.provenance == {
        "embedder": "CmapEmbedder",
        "seed": 12,
        "etkdg": False,
        "validation": {"stereo": "ok"},  # the stereo check after refinement
    }
    assert ensemble.mol.GetIntProp("charge") == 0

    # -- ground state conformers without a reference
    config = PipelineConfig(seed=3, embed={"n_conformers": 30})
    ensemble = racerts.generate_gs("CCCCCC=C", config=config)

    assert len(ensemble) > 1
    assert ensemble.record(ensemble.best()).energy_method == "MMFFOptimizer"
    # Pruned: pruning the result again removes nothing.
    mol = ensemble.mol
    before = mol.GetNumConformers()
    RMSDPruner().prune(mol)
    assert mol.GetNumConformers() == before

    # -- constrained keeps the hard atoms at the reference
    mol = hept_1_ene_ts
    ensemble = racerts.generate(mol, Constrained(hard=[3, 4, 5]), config=SMALL)
    reference = mol.GetConformer().GetPositions()[[3, 4, 5]]

    assert len(ensemble) > 1
    for conf in ensemble.mol.GetConformers():
        assert np.abs(conf.GetPositions()[[3, 4, 5]] - reference).max() < 1e-3
    # The rest of the molecule moves.
    first, second = ensemble.conf_ids[:2]
    assert rdMolAlign.GetBestRMS(ensemble.mol, ensemble.mol, first, second) > 0.1

    # -- random seed minus one
    config = PipelineConfig(seed=-1, embed=EmbedConfig(n_conformers=3))

    assert len(racerts.generate_ts(EX, [3, 4, 5], smiles="CCCCCC=C", config=config))

    # -- build mol carries the charge of connectivity graphs
    # No formal charges in a connectivity graph: the charge travels as a property.
    mol = build_mol(SN2_BASELINE, -1, [0, 1, 2], mol_getter=MolGetterConnectivity())
    ensemble = racerts.generate(mol, racerts.TransitionState([0, 1, 2]), config=SMALL)

    assert ensemble.mol.GetIntProp("charge") == -1
    assert ensemble.mol.GetIntProp("multiplicity") == 1


def test_what_the_entry_points_refuse(tmp_path):
    # -- invalid smiles for ground states raise
    with pytest.raises(ValueError, match="Invalid SMILES"):
        racerts.generate_gs("C1CC")

    # -- no fallback means no uff either
    # As ConformerGenerator().generate_conformers(..., auto_fallback=False): MMFF has
    # no boron parameters, and the error is not hidden by UFF.
    with pytest.raises(ValueError, match="MMFF parameters"):
        racerts.generate_ts(
            BORONIC_ACID,
            [0, 1, 2],
            smiles="C=CCB(O)O",
            config=SMALL,
            auto_fallback=False,
        )

    # -- an unreadable xyz file raises clearly
    path = tmp_path / "broken.xyz"
    path.write_text("3\n\nC 0 0 0\n")

    with pytest.raises(ValueError, match="No valid mol object"):
        racerts.generate_ts(str(path), [0], mol_getter=racerts.compat.MolGetterBonds())


def test_the_messages_of_the_entry_points(capsys, caplog):
    # -- nothing is printed
    racerts.generate_ts(EX, [3, 4, 5], smiles="CCCCCC=C", config=SMALL)
    racerts.ConformerGenerator().generate_conformers(
        EX, 0, [3, 4, 5], input_smiles=["CCCCCC=C"], number_of_conformers=10
    )

    assert capsys.readouterr().out == ""

    # -- verbose shows progress during the call
    caplog.clear()

    def run(verbose):
        racerts.ConformerGenerator(verbose=verbose).generate_conformers(
            EX, 0, [3, 4, 5], input_smiles=["CCCCCC=C"], number_of_conformers=5
        )

    # caplog's handler sits on the root logger, as in an application that configured
    # logging; the racerts logger keeps the default level (WARNING).
    run(verbose=True)
    assert "embed: 5 conformers" in caplog.text
    caplog.clear()
    run(verbose=False)
    assert caplog.text == ""

    # -- verbose for the new api
    caplog.clear()
    racerts.generate_ts(EX, [3, 4, 5], smiles="CCCCCC=C", config=SMALL, verbose=True)

    assert "embed: 10 conformers" in caplog.text

    # -- restraints do not settle the spin state again
    caplog.clear()
    # A radical with a restraint: the multiplicity that was passed is not questioned.
    config = PipelineConfig.from_dict(
        {"embed": {"n_conformers": 3}, "restraints": {"user": [[0, 2, 2.5]]}}
    )
    with caplog.at_level(logging.WARNING):
        ensemble = racerts.generate_gs("CC[CH2]", multiplicity=2, config=config)
    assert ensemble.mol.GetIntProp("multiplicity") == 2
    assert "odd number of electrons" not in caplog.text


def test_verbose_logs_to_stderr_without_a_configured_handler(capsys, monkeypatch):
    # As in a plain script: no handler anywhere (the pytest handlers are cut off).
    monkeypatch.setattr(logging.getLogger("racerts"), "propagate", False)
    racerts.ConformerGenerator(verbose=True).generate_conformers(
        EX, 0, [3, 4, 5], input_smiles=["CCCCCC=C"], number_of_conformers=5
    )
    captured = capsys.readouterr()

    assert captured.out == ""
    assert "INFO racerts.pipeline.runner: embed: 5 conformers" in captured.err
    assert not logging.getLogger("racerts").handlers  # removed after the call


SN2_BASELINE = os.path.join(os.path.dirname(EX), "baseline", "sn2_ts.xyz")
BORONIC_ACID = os.path.join(os.path.dirname(EX), "baseline", "boronic_acid.xyz")
