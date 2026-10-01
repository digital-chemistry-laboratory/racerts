"""GFN2-xTB (tblite) in the pipeline: charge and spin reach the calculator."""

import numpy as np
import pytest

import racerts
from racerts import Rescore, TransitionState
from racerts.refine import ASEOptimizer
from racerts.utils.units import EV_TO_KCAL_MOL

pytestmark = [pytest.mark.ase, pytest.mark.xtb]
pytest.importorskip("ase")
TBLite = pytest.importorskip("tblite.ase").TBLite


def gfn2():
    """A calculator factory: one GFN2-xTB calculator per worker or conformer."""
    return TBLite(method="GFN2-xTB", verbosity=0)


@pytest.fixture
def sn2(sn2_ts):
    """Cl- + CH3Cl, charge -1: two refined conformers."""
    config = racerts.PipelineConfig(embed=racerts.EmbedConfig(n_conformers=4))
    pipeline = racerts.Pipeline([racerts.Embed(n_conformers=4), racerts.Refine()])
    ensemble = racerts.generate_ts(
        sn2_ts, [0, 1, 2], charge=-1, smiles="CCl.[Cl-]", config=config,
        pipeline=pipeline,
    )  # fmt: skip
    ctx = racerts.Context.create(ensemble.mol, TransitionState([0, 1, 2]))
    return ensemble, ctx


def _direct(ensemble, conf_id, charge, uhf=0):
    """GFN2-xTB energy (kcal/mol) with charge and unpaired electrons set directly."""
    from ase import Atoms

    mol = ensemble.mol
    atoms = Atoms(
        [a.GetSymbol() for a in mol.GetAtoms()],
        positions=mol.GetConformer(conf_id).GetPositions(),
    )
    atoms.calc = TBLite(method="GFN2-xTB", charge=charge, multiplicity=uhf + 1,
                        verbosity=0)  # fmt: skip
    return atoms.get_potential_energy() * EV_TO_KCAL_MOL


def test_gfn2_rescoring_of_an_anion(sn2):
    ensemble, ctx = sn2
    Rescore(gfn2, method="GFN2-xTB").run(ctx, ensemble)

    assert ensemble.energy_method == "GFN2-xTB"
    for conf_id in ensemble.conf_ids:
        assert ensemble.energy(conf_id) == pytest.approx(
            _direct(ensemble, conf_id, charge=-1), abs=1e-6
        )
    # The neutral system would be far off: the charge matters.
    conf_id = ensemble.conf_ids[0]
    assert abs(ensemble.energy(conf_id) - _direct(ensemble, conf_id, 0)) > 10


def test_gfn2_refinement_keeps_the_frozen_atoms(hept_1_ene_ts):
    ctx = racerts.Context.create(hept_1_ene_ts, TransitionState([3, 4, 5]))
    ensemble = racerts.Pipeline([racerts.Embed(n_conformers=2), racerts.Refine()]).run(
        ctx
    )
    before = ensemble.energies()
    ASEOptimizer(gfn2, fmax=0.1, max_steps=20, num_workers=2).refine(
        ensemble.mol, ctx.reference, ctx.frozen.hard
    )
    assert not np.allclose(ensemble.energies(), before)
    frozen = list(ctx.frozen.hard)
    reference = ctx.reference.GetConformer().GetPositions()[frozen]
    for conf in ensemble.mol.GetConformers():
        assert np.abs(conf.GetPositions()[frozen] - reference).max() < 1e-3
        assert conf.GetDoubleProp("energy") < -5000  # GFN2 total energy, kcal/mol
        assert ensemble.provenance(conf.GetId())["n_steps"] >= 1


def test_gfn2_of_a_radical_uses_the_multiplicity():
    # The methyl radical: a doublet from the electron count (9 electrons).
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.AddHs(Chem.MolFromSmiles("[CH3]"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    ensemble = racerts.ConformerEnsemble(mol)
    ctx = racerts.Context.create(mol, racerts.GroundState())
    ensemble = racerts.ConformerEnsemble(ctx.mol)
    ensemble.mol.AddConformer(mol.GetConformer(), assignId=True)
    Rescore(gfn2).run(ctx, ensemble)
    assert ensemble.energy(0) == pytest.approx(
        _direct(ensemble, 0, charge=0, uhf=1), abs=1e-6
    )


def test_saddle_search_and_validation_with_plug_ins(sn2_ts):
    # TS-like conformers -> Sella saddle search with GFN2-xTB (optimizer plug-in,
    # without anchors) -> one imaginary mode (validator plug-in).
    Sella = pytest.importorskip("sella").Sella
    from racerts.validate import ImaginaryModes

    saddle = ASEOptimizer(
        gfn2,
        optimizer_cls=Sella,
        optimizer_kwargs={"order": 1, "internal": True},
        fmax=0.005,
        max_steps=200,
    )
    pipeline = racerts.Pipeline(
        [
            racerts.Embed(n_conformers=2),
            racerts.Refine(),
            racerts.Refine(saddle, anchors=False, fallback=False),
            racerts.Validate(ImaginaryModes(gfn2, expected=1)),
        ]
    )
    ensemble = racerts.generate_ts(
        sn2_ts, [0, 1, 2], charge=-1, smiles="CCl.[Cl-]", pipeline=pipeline
    )
    assert len(ensemble) >= 1
    for conf_id in ensemble.conf_ids:
        provenance = ensemble.provenance(conf_id)
        assert provenance["converged"] is True
        assert provenance["validation"] == {"imaginary_modes": "ok"}
        # The GFN2 saddle is symmetric: both C-Cl bonds equally long.
        p = ensemble.mol.GetConformer(conf_id).GetPositions()
        d1, d2 = np.linalg.norm(p[0] - p[1]), np.linalg.norm(p[0] - p[2])
        assert d1 == pytest.approx(d2, abs=0.02) and 2.1 < d1 < 2.5
