"""Validation: the Validate stage and the built-in validators."""

import logging

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Geometry import Point3D

import racerts
from racerts import TransitionState, Validate
from racerts.validate import (
    Connectivity,
    FrozenCore,
    IdentityFilter,
    ImaginaryModes,
    validator,
)


@pytest.fixture
def ts_ensemble(hept_1_ene_ts):
    """Four MMFF conformers of the TS of ex.xyz, with their context."""
    ctx = racerts.Context.create(hept_1_ene_ts, TransitionState([3, 4, 5]))
    ensemble = racerts.Pipeline([racerts.Embed(n_conformers=4), racerts.Refine()]).run(
        ctx
    )
    return ensemble, ctx


def _gs(smiles, n=3, seed=7):
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    AllChem.EmbedMultipleConfs(mol, n, randomSeed=seed)
    ctx = racerts.Context.create(mol, racerts.GroundState())
    return racerts.ConformerEnsemble(Chem.Mol(mol)), ctx


def _move(ensemble, conf_id, atom, shift):
    conf = ensemble.mol.GetConformer(conf_id)
    p = conf.GetAtomPosition(atom)
    conf.SetAtomPosition(atom, Point3D(p.x + shift[0], p.y + shift[1], p.z + shift[2]))


def _mirror(ensemble, conf_id):
    conf = ensemble.mol.GetConformer(conf_id)
    for i, p in enumerate(conf.GetPositions()):
        conf.SetAtomPosition(i, Point3D(-p[0], p[1], p[2]))


def test_connectivity_passes_a_ts_ensemble(ts_ensemble):
    ensemble, ctx = ts_ensemble
    assert Connectivity().validate(ctx, ensemble) == {}


def test_bonds_between_reacting_atoms_are_exempt(ts_ensemble):
    # A graph without the C3-C4 bond, which the geometry has: 3 and 4 are reacting
    # atoms, whose bonds form or break, so the difference is not checked.
    ensemble, ctx = ts_ensemble
    graph = Chem.RWMol(ensemble.mol)
    graph.RemoveBond(3, 4)
    ensemble = racerts.ConformerEnsemble(graph.GetMol())
    assert Connectivity().validate(ctx, ensemble) == {}
    reasons = Connectivity(exempt=(), stereo=False).validate(ctx, ensemble)
    assert set(reasons.values()) == {"bonds [(3, 4)] extra"}


def test_connectivity_finds_a_broken_bond(ts_ensemble):
    ensemble, ctx = ts_ensemble
    conf_id = ensemble.conf_ids[1]
    _move(ensemble, conf_id, 0, (10.0, 0.0, 0.0))  # a terminal carbon pulled away
    reasons = Connectivity().validate(ctx, ensemble)
    assert list(reasons) == [conf_id]
    assert reasons[conf_id].startswith("bonds [(0, 1), ")
    assert reasons[conf_id].endswith("missing")


def test_connectivity_checks_the_specified_stereo():
    ensemble, ctx = _gs("C[C@H](O)CC")
    _mirror(ensemble, 1)
    assert Connectivity().validate(ctx, ensemble) == {1: "stereo of atom 1 inverted"}
    assert Connectivity(stereo=False).validate(ctx, ensemble) == {}

    # Unspecified stereo is a wildcard.
    ensemble, ctx = _gs("CC(O)CC")
    _mirror(ensemble, 1)
    assert Connectivity().validate(ctx, ensemble) == {}


def test_connectivity_checks_double_bonds():
    ensemble, ctx = _gs("C/C=C/C", n=1)
    z = Chem.AddHs(Chem.MolFromSmiles("C/C=C\\C"))  # the same atom order
    AllChem.EmbedMolecule(z, randomSeed=3)
    ensemble.mol.AddConformer(z.GetConformer(), assignId=True)
    reasons = Connectivity().validate(ctx, ensemble)
    assert list(reasons) == [1] and "bond 1" in reasons[1]


def test_frozen_core(ts_ensemble):
    ensemble, ctx = ts_ensemble
    assert FrozenCore().validate(ctx, ensemble) == {}
    conf_id = ensemble.conf_ids[2]
    _move(ensemble, conf_id, ctx.frozen.hard[0], (0.2, 0.0, 0.0))
    reasons = FrozenCore(tolerance=0.01).validate(ctx, ensemble)
    assert list(reasons) == [conf_id] and "moved by up to 0.1" in reasons[conf_id]


def test_validate_drops_or_flags(ts_ensemble, caplog):
    ensemble, ctx = ts_ensemble
    bad = ensemble.conf_ids[0]

    def not_the_first(mol, conf_id):
        return conf_id != bad  # False: fails

    flagged = Validate(validator(not_the_first), FrozenCore(), on_fail="flag").run(
        ctx, ensemble.copy()
    )
    assert flagged.conf_ids == ensemble.conf_ids
    assert flagged.provenance(bad)["validation"] == {
        "not_the_first": "failed",
        "frozen_core": "ok",
    }

    with caplog.at_level(logging.WARNING):
        dropped = Validate(validator(not_the_first)).run(ctx, ensemble.copy())
    assert bad not in dropped.conf_ids and len(dropped) == len(ensemble) - 1
    assert "1 of 4 conformers failed validation" in caplog.text


def test_validate_raises_if_none_passes(ts_ensemble):
    ensemble, ctx = ts_ensemble
    always = validator(lambda mol, conf_id: "no", name="never")
    with pytest.raises(RuntimeError, match="No conformer passed validation.*never: no"):
        Validate(always).run(ctx, ensemble.copy())
    assert len(Validate(always, require_any=False).run(ctx, ensemble.copy())) == 0

    with pytest.raises(TypeError, match="not a Validator"):
        Validate(lambda mol, conf_id: None)
    with pytest.raises(ValueError, match="at least one"):
        Validate()


def test_identity_filter_in_a_pipeline(hept_1_ene_ts):
    pipeline = racerts.Pipeline(
        [racerts.Embed(n_conformers=5), racerts.Refine(), IdentityFilter()]
    )
    ensemble = racerts.generate(
        hept_1_ene_ts, TransitionState([3, 4, 5]), pipeline=pipeline
    )
    assert len(ensemble) == 5
    assert all(
        ensemble.provenance(i)["validation"] == {"connectivity": "ok"}
        for i in ensemble.conf_ids
    )


class Spring:
    """A diatomic harmonic spring as an ASE calculator: E = k/2 (r - r0)^2."""

    def __init__(self, k=36.0, r0=0.74):
        from ase.calculators.calculator import Calculator, all_changes

        class _Spring(Calculator):
            implemented_properties = ["energy", "forces"]

            def calculate(inner, atoms=None, properties=None, changes=all_changes):
                super(_Spring, inner).calculate(atoms, properties, changes)
                d = atoms.positions[1] - atoms.positions[0]
                r = np.linalg.norm(d)
                inner.results["energy"] = 0.5 * k * (r - r0) ** 2
                f = -k * (r - r0) * d / r
                inner.results["forces"] = np.array([-f, f])

        self.calculator = _Spring()


@pytest.mark.ase
def test_frequencies_agree_with_ase_vibrations(tmp_path):
    ase = pytest.importorskip("ase")
    from ase.vibrations import Vibrations

    atoms = ase.Atoms("H2", positions=[[0, 0, 0], [0, 0, 0.74]])
    atoms.calc = Spring().calculator
    ours = ImaginaryModes(atoms.calc).frequencies(atoms)
    assert len(ours) == 1  # linear: 3N - 5 modes

    vibrations = Vibrations(atoms, name=str(tmp_path / "vib"), delta=0.005)
    vibrations.run()
    reference = np.real(vibrations.get_frequencies()).max()
    assert ours[0] == pytest.approx(reference, rel=1e-4)


def _ethane(eclipsed: bool):
    """Ethane along z; the H of the second carbon at the same azimuths if eclipsed."""
    from ase import Atoms

    positions = [[0, 0, -0.765], [0, 0, 0.765]]
    for carbon, z, offset in ((0, -1.16, 0.0), (1, 1.16, 0.0 if eclipsed else 60.0)):
        for k in range(3):
            angle = np.radians(offset + 120 * k)
            positions.append([1.02 * np.cos(angle), 1.02 * np.sin(angle), z])
    return Atoms("C2H6", positions=positions)


@pytest.mark.ase
@pytest.mark.xtb
def test_the_rotation_ts_of_ethane_has_one_imaginary_mode():
    pytest.importorskip("ase")
    TBLite = pytest.importorskip("tblite.ase").TBLite
    from ase.optimize import BFGS

    def optimized(eclipsed):
        atoms = _ethane(eclipsed)
        atoms.calc = TBLite(method="GFN2-xTB", verbosity=0)
        BFGS(atoms, logfile=None).run(fmax=0.001, steps=200)  # keeps the symmetry
        return atoms

    check = ImaginaryModes(lambda: TBLite(method="GFN2-xTB", verbosity=0))
    staggered = check.frequencies(optimized(False))
    eclipsed = check.frequencies(optimized(True))
    assert (staggered < -50).sum() == 0
    assert (eclipsed < -50).sum() == 1
    assert -400 < eclipsed.min() < -150  # the methyl torsion


def test_stereo_only_check():
    ensemble, ctx = _gs("C[C@H](O)CC")
    _move(ensemble, 1, 4, (10.0, 0.0, 0.0))  # a broken bond ...
    _mirror(ensemble, 2)  # ... and a mirror image
    reasons = Connectivity(bonds=False).validate(ctx, ensemble)
    assert reasons == {2: "stereo of atom 1 inverted"}
    assert Connectivity(bonds=False).name == "stereo"
    with pytest.raises(ValueError, match="bonds or stereo"):
        Connectivity(bonds=False, stereo=False)


def test_check_stereo_setting_adds_a_stereo_check(hept_1_ene_ts):
    from racerts import PipelineConfig

    names = [s.name for s in PipelineConfig().build(TransitionState([3])).stages]
    assert "validate" not in names
    config = PipelineConfig.from_dict(
        {"embed": {"n_conformers": 4}, "prune": {"check_stereo": True}}
    )
    stages = config.build(TransitionState([3, 4, 5])).stages
    assert [s.name for s in stages][:3] == ["embed", "refine", "validate"]
    check = stages[2].validators[0]
    assert (check.bonds, check.exempt) == (False, None)  # None: the task's core atoms
    ensemble = racerts.generate(
        hept_1_ene_ts, TransitionState([3, 4, 5]), config=config
    )
    assert {
        ensemble.provenance(i)["validation"]["stereo"] for i in ensemble.conf_ids
    } == {"ok"}


def test_ring_stereo_without_cip_labels():
    # cis/trans on a ring: chiral tags but no CIP labels.
    trans = Chem.AddHs(Chem.MolFromSmiles("C[C@H]1CC[C@@H](C)CC1"))
    AllChem.EmbedMultipleConfs(trans, 2, randomSeed=3)
    cis = Chem.AddHs(Chem.MolFromSmiles("C[C@H]1CC[C@H](C)CC1"))  # same atom order
    AllChem.EmbedMolecule(cis, randomSeed=3)
    trans.AddConformer(cis.GetConformer(), assignId=True)
    ctx = racerts.Context.create(trans, racerts.GroundState())
    reasons = Connectivity().validate(ctx, racerts.ConformerEnsemble(trans))
    assert list(reasons) == [2] and "inverted" in reasons[2]

    config = racerts.PipelineConfig.from_dict(
        {"embed": {"n_conformers": 4}, "prune": {"check_stereo": True}}
    )
    assert len(racerts.generate_gs("C[C@H]1CC[C@@H](C)CC1", config=config)) >= 1


def test_reaction_core_catches_another_saddle(ts_ensemble):
    # UMA benchmark: free saddle searches from windowed conformers
    # reached saddles of other steps (a proton on the other partner, 1.0 A off) that
    # pass ImaginaryModes and Connectivity, which exempts the reacting atoms.
    from racerts.validate import ReactionCore

    ensemble, ctx = ts_ensemble
    assert ReactionCore().validate(ctx, ensemble) == {}
    conf_id = ensemble.conf_ids[1]
    p = ensemble.mol.GetConformer(conf_id).GetPositions()
    direction = (p[5] - p[3]) / np.linalg.norm(p[5] - p[3])
    _move(ensemble, conf_id, 5, 0.8 * direction)
    reasons = ReactionCore().validate(ctx, ensemble)
    assert list(reasons) == [conf_id]
    assert reasons[conf_id].startswith("reacting-atom distances changed by up to 0.8")
    assert "3-5" in reasons[conf_id]
    assert ReactionCore(tolerance=0.9).validate(ctx, ensemble) == {}
    ground = racerts.Context.create(_gs("CCO")[1].mol, racerts.GroundState())
    assert ReactionCore().validate(ground, ensemble) == {}  # no core: nothing to check
