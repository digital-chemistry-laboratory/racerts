"""The legacy racerts API that stays for external code (racerts.compat)."""

import importlib
import json
import os
import subprocess
import sys

import pytest
from rdkit import Chem

import racerts
import racerts.compat.conformer_generator
from racerts import ConformerGenerator, Embed, Refine
from racerts.embedder import BaseEmbedder, CmapEmbedder
from racerts.optimizer import BaseOptimizer, MMFFOptimizer, UFFOptimizer

from .api_surface import incompatibilities
from .conftest import DATA, EX

TS_ARGS = dict(input_smiles=["CCCCCC=C"], number_of_conformers=10)

KEPT = {
    "racerts": [
        "ConformerGenerator",
        "embedders",
        "mol_getters",
        "optimizers",
        "pruners",
    ],
    "racerts.embedder": [
        "BaseEmbedder",
        "BoundsMatrixEmbedder",
        "CmapEmbedder",
        "embedders",
    ],
    "racerts.optimizer": [
        "ASEOptimizer",
        "BaseOptimizer",
        "MMFFOptimizer",
        "UFFOptimizer",
        "optimizers",
    ],
    "racerts.pruner": ["BasePruner", "EnergyPruner", "RMSDPruner", "pruners"],
    "racerts.mol_getter": [
        "BaseMolGetter",
        "MolGetterBonds",
        "MolGetterConnectivity",
        "MolGetterSMILES",
        "mol_getters",
    ],
    "racerts.compat": ["ConformerGenerator", "DEFAULT_CONF_FACTOR"],
}


@pytest.mark.parametrize("module", sorted(KEPT))
def test_legacy_import_paths_are_kept(module):
    imported = importlib.import_module(module)
    for name in KEPT[module]:
        assert getattr(imported, name) is getattr(racerts.compat, name)


# Deliberate differences to the API of racerts 0.1.7 (PyPI); see docs/migration.md.
CHANGED_SINCE_0_1_7 = {
    # Defaults of upstream main: conf_factor 80 as in the CLI; extended XYZ output.
    "ConformerGenerator.generate_conformers:conf_factor",
    "ConformerGenerator.write_xyz:comment",
    # An unused property whose getter raised AttributeError unless it was set first;
    # an attribute of that name behaves the same.
    "ConformerGenerator.bounds_generator",
    # Abstract constructors that raised NotImplementedError.
    "BaseEmbedder.__init__",
    "BaseOptimizer.__init__",
    # Names the command-line module imported (the command line itself is kept).
    "racerts.cli.ConformerGenerator",
    "racerts.cli.EnergyPruner",
    "racerts.cli.RMSDPruner",
    "racerts.cli.embedders",
    "racerts.cli.mol_getters",
    "racerts.cli.optimizers",
    # Slots of the lazy ASE imports, None until an optimization ran.
    "racerts.optimizer.ase.Atoms",
    "racerts.optimizer.ase.BFGS",
    "racerts.optimizer.ase.FixAtoms",
}


def test_the_api_of_racerts_0_1_7_is_kept():
    with open(os.path.join(DATA, "api_0.1.7.json")) as handle:
        snapshot = json.load(handle)
    assert incompatibilities(snapshot, CHANGED_SINCE_0_1_7) == []


@pytest.mark.ase
def test_patching_the_legacy_ase_atoms_builder(hept_1_ene_ts, monkeypatch):
    # catmlp replaces racerts.optimizer.ase.rdkit_conformer_to_ase_atoms.
    pytest.importorskip("ase")
    from ase.calculators.lj import LennardJones

    import racerts.optimizer.ase

    calls = []
    builder = racerts.optimizer.ase.rdkit_conformer_to_ase_atoms

    def counting(mol, conf_id=-1, multiplicity=None, charge=None):
        calls.append(conf_id)
        return builder(mol, conf_id, multiplicity=multiplicity, charge=charge)

    monkeypatch.setattr(racerts.optimizer.ase, "rdkit_conformer_to_ase_atoms", counting)
    mol = Chem.Mol(hept_1_ene_ts)
    racerts.optimizer.ASEOptimizer(
        calculator=LennardJones(), max_steps=1
    ).tune_ts_conformers(mol, hept_1_ene_ts, [3, 4, 5])

    assert calls == [conf.GetId() for conf in mol.GetConformers()]


def test_the_legacy_modules_live_in_racerts_compat():
    for name in ("embedder", "mol_getter", "optimizer", "pruner"):
        module = importlib.import_module(f"racerts.{name}")
        assert module is getattr(racerts, name)
        assert module is getattr(racerts.compat, name)
        assert module.__name__ == f"racerts.compat.{name}"


def test_the_legacy_classes_extend_the_current_ones():
    # The legacy methods (embed_TS, tune_ts_conformers) are only on the legacy classes.
    pairs = [
        (racerts.embedder.CmapEmbedder, racerts.embed.CmapEmbedder),
        (racerts.embedder.BoundsMatrixEmbedder, racerts.embed.BoundsMatrixEmbedder),
        (racerts.optimizer.MMFFOptimizer, racerts.refine.MMFFOptimizer),
        (racerts.optimizer.UFFOptimizer, racerts.refine.UFFOptimizer),
        (racerts.optimizer.ASEOptimizer, racerts.refine.ASEOptimizer),
    ]
    for legacy, new in pairs:
        assert issubclass(legacy, new) and legacy.__name__ == new.__name__
    assert not hasattr(racerts.embed.BaseEmbedder, "embed_TS")
    assert not hasattr(racerts.refine.BaseOptimizer, "tune_ts_conformers")
    assert racerts.pruner.RMSDPruner is racerts.prune.RMSDPruner
    assert racerts.mol_getter.MolGetterSMILES is racerts.system.MolGetterSMILES


def test_legacy_registries_are_complete():
    assert set(racerts.embedders) == {"dm", "cmap", "base"}
    assert set(racerts.mol_getters) == {"base", "bonds", "connect", "smiles"}
    assert set(racerts.optimizers) == {"mmff", "uff", "ase", "base"}
    assert set(racerts.pruners) == {"base", "energy", "rmsd"}
    assert racerts.compat.DEFAULT_CONF_FACTOR == 80


class RecordingEmbedder(BaseEmbedder):
    """A legacy-style embedder: it implements embed_TS only."""

    def __init__(self):
        self.calls = []

    def embed_TS(self, mol_ts, mol, reacting_atoms, frozen_atoms, n, verbose):
        self.calls.append((list(reacting_atoms), list(frozen_atoms), n))
        return CmapEmbedder(randomSeed=12).embed_TS(
            mol_ts, mol, reacting_atoms, frozen_atoms, n, verbose
        )


class ZeroEnergyOptimizer(BaseOptimizer):
    """A legacy-style optimizer: it implements tune_ts_conformers only."""

    def __init__(self):
        self.calls = 0

    def tune_ts_conformers(self, mol, reference, align_indices):
        self.calls += 1
        for conf in mol.GetConformers():
            conf.SetDoubleProp("energy", 0.0)


def test_legacy_style_components_run_in_the_generator():
    cg = ConformerGenerator()
    cg.embedder = RecordingEmbedder()
    cg.optimizer = ZeroEnergyOptimizer()
    mol = cg.generate_conformers(EX, 0, [3, 4, 5], **TS_ARGS)

    assert cg.embedder.calls == [([3, 4, 5], [2, 4, 14, 15, 3, 5, 16, 17, 6, 18], 10)]
    assert cg.optimizer.calls == 1
    assert mol.GetProp("energy_method") == "ZeroEnergyOptimizer"


def test_legacy_style_components_run_in_the_new_stages():
    embedder, optimizer = RecordingEmbedder(), ZeroEnergyOptimizer()
    pipeline = racerts.Pipeline([Embed(embedder, n_conformers=10), Refine(optimizer)])
    ensemble = racerts.generate_ts(EX, [3, 4, 5], smiles="CCCCCC=C", pipeline=pipeline)

    assert embedder.calls == [([3, 4, 5], [2, 4, 14, 15, 3, 5, 16, 17, 6, 18], 10)]
    assert optimizer.calls == 1
    assert len(ensemble) == 10 and set(ensemble.energies()) == {0.0}


def test_overridden_generator_steps_are_used():
    class KeepLowest(ConformerGenerator):
        def prune(self, mol):
            pruned = Chem.Mol(mol)
            energies = {
                c.GetId(): c.GetDoubleProp("energy") for c in mol.GetConformers()
            }
            lowest = min(energies, key=energies.get)
            for conf_id in energies:
                if conf_id != lowest:
                    pruned.RemoveConformer(conf_id)
            return pruned

    assert (
        KeepLowest().generate_conformers(EX, 0, [3, 4, 5], **TS_ARGS).GetNumConformers()
        == 1
    )


def test_legacy_components_without_an_implementation_fail_as_before():
    class Empty(BaseEmbedder):
        def __init__(self):
            pass

    class EmptyOptimizer(BaseOptimizer):
        def __init__(self):
            pass

    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    with pytest.raises(NotImplementedError):
        Empty().embed(mol, None, racerts.FrozenSet(), 1)
    with pytest.raises(TypeError):  # legacy racerts: tune_ts_conformers is abstract
        EmptyOptimizer()


def test_uff_fallback_keeps_the_generator_settings(monkeypatch):
    # MMFF has no boron: the generator's optimize falls back to UFF, with the settings
    # of the MMFF optimizer and the generator's threads.
    created = []

    class RecordingUFF(UFFOptimizer):
        def __init__(self, **kwargs):
            created.append(kwargs)
            super().__init__(**kwargs)

    monkeypatch.setattr(
        racerts.compat.conformer_generator, "UFFOptimizer", RecordingUFF
    )
    cg = ConformerGenerator(num_threads=2)
    cg.optimizer = MMFFOptimizer(force_constant=5e5, conf_id_ref=0)
    boronic_acid = os.path.join(os.path.dirname(EX), "baseline", "boronic_acid.xyz")
    mol = cg.generate_conformers(boronic_acid, 0, [0, 1, 2], number_of_conformers=3)

    assert mol.GetProp("energy_method") == "RecordingUFF"
    assert created == [
        dict(verbose=False, conf_id_ref=0, force_constant=5e5, num_threads=2)
    ]


@pytest.mark.parametrize(
    "code",
    [
        "import racerts; racerts.embedder.CmapEmbedder; racerts.optimizer.ASEOptimizer; "
        "racerts.pruner.RMSDPruner; racerts.mol_getter.MolGetterSMILES",
        "from racerts.optimizer import ASEOptimizer",
        "import racerts.embedder; racerts.embedder.CmapEmbedder",
        "from racerts import mol_getter; mol_getter.MolGetterSMILES",
    ],
)
def test_legacy_imports_in_a_fresh_interpreter(code):
    # Other tests import these modules themselves.
    env = dict(os.environ, PYTHONPATH=os.pathsep.join(sys.path))
    result = subprocess.run(
        [sys.executable, "-c", code], env=env, capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr


class CountingCmap(CmapEmbedder):
    """Legacy subclass of a built-in embedder that overrides embed_TS."""

    calls = 0

    def embed_TS(self, mol_ts, mol, reacting_atoms, frozen_atoms, n=10, verbose=False):
        CountingCmap.calls += 1
        return super().embed_TS(mol_ts, mol, reacting_atoms, frozen_atoms, n, verbose)


class ShiftedMMFF(MMFFOptimizer):
    """Legacy subclass of a built-in optimizer that overrides tune_ts_conformers."""

    def tune_ts_conformers(self, mol, reference, align_indices):
        super().tune_ts_conformers(mol, reference, align_indices)
        for conf in mol.GetConformers():
            conf.SetDoubleProp("energy", conf.GetDoubleProp("energy") + 1000.0)


def test_legacy_overrides_of_built_in_components_run_in_the_new_stages():
    def run(embedder, optimizer):
        pipeline = racerts.Pipeline(
            [Embed(embedder, n_conformers=5), Refine(optimizer, fallback=False)]
        )
        return racerts.generate_ts(EX, [3, 4, 5], smiles="CCCCCC=C", pipeline=pipeline)

    CountingCmap.calls = 0
    plain = run(CmapEmbedder(), MMFFOptimizer())
    shifted = run(CountingCmap(), ShiftedMMFF())

    assert CountingCmap.calls == 1
    assert shifted.energies() - plain.energies() == pytest.approx(1000.0)


class SuperCallingOptimizer(BaseOptimizer):
    """legacy racerts: tune_ts_conformers of BaseOptimizer was abstract and did nothing."""

    def __init__(self):
        pass

    def tune_ts_conformers(self, mol, reference, align_indices):
        super().tune_ts_conformers(mol, reference, align_indices)
        for conf in mol.GetConformers():
            conf.SetDoubleProp("energy", 0.0)


def test_super_calls_of_legacy_components_do_not_recurse():
    cg = ConformerGenerator()
    cg.optimizer = SuperCallingOptimizer()
    assert cg.generate_conformers(EX, 0, [3, 4, 5], **TS_ARGS).GetNumConformers() > 0

    pipeline = racerts.Pipeline(
        [Embed(n_conformers=3), Refine(SuperCallingOptimizer())]
    )
    ensemble = racerts.generate_ts(EX, [3, 4, 5], smiles="CCCCCC=C", pipeline=pipeline)
    assert set(ensemble.energies()) == {0.0}

    class SuperCallingEmbedder(BaseEmbedder):
        def __init__(self):
            pass

        def embed(self, mol, reference, frozen, n):
            return super().embed(mol, reference, frozen, n)

    with pytest.raises(NotImplementedError):
        SuperCallingEmbedder().embed(Chem.Mol(), None, racerts.FrozenSet(), 1)


def test_legacy_subclasses_with_their_own_init():
    class OwnInitCmap(CmapEmbedder):
        def __init__(self):  # the legacy attributes only, no super().__init__()
            self.verbose, self.randomSeed, self.pruneRmsThresh = False, 12, -1
            self.remove_all_conformers, self.ETversion = True, 2
            self.useRandomCoords, self.num_threads = True, 1

    class OwnInitGenerator(ConformerGenerator):
        def __init__(self):  # no randomSeed: legacy generate_conformers did not need it
            self.num_threads, self._verbose = 1, False
            self._mol_getter = racerts.compat.MolGetterSMILES()
            self._embedder, self._optimizer = OwnInitCmap(), MMFFOptimizer()
            self._energy_pruner = racerts.compat.EnergyPruner()
            self._rmsd_pruner = racerts.compat.RMSDPruner()

    def positions(mol):
        return [conf.GetPositions().tolist() for conf in mol.GetConformers()]

    expected = ConformerGenerator().generate_conformers(EX, 0, [3, 4, 5], **TS_ARGS)
    mol = OwnInitGenerator().generate_conformers(EX, 0, [3, 4, 5], **TS_ARGS)
    assert positions(mol) == positions(expected)


def test_embed_TS_keeps_its_legacy_defaults(hept_1_ene_ts):
    reference = hept_1_ene_ts
    mol = Chem.Mol(reference)
    mol.RemoveAllConformers()
    CmapEmbedder().embed_TS(reference, mol, [3, 4, 5], [3, 4, 5])

    assert mol.GetNumConformers() == 10


def test_random_seed_minus_one_as_in_legacy_racerts(tmp_path):
    # RDKit reads a negative seed as "random".
    mol = ConformerGenerator(randomSeed=-1).generate_conformers(
        EX, 0, [3, 4, 5], input_smiles=["CCCCCC=C"], number_of_conformers=3
    )
    assert mol.GetNumConformers() > 0

    from racerts.cli import main

    out = tmp_path / "random.xyz"
    main([EX, "-atoms", "3", "4", "5", "-n", "3", "--seed", "-1", "-o", str(out)])
    assert out.exists()
