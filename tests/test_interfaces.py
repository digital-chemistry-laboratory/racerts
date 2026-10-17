"""The interfaces to CREST and ORCA's GOAT: their inputs, a run, and their ensembles read
back as conformers of the molecule (for benchmarks of the searches)."""

import os
import stat
from pathlib import Path

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import AllChem

import racerts
from racerts.interfaces import Crest, Goat, read_crest, read_goat
from racerts.utils.units import HARTREE_TO_KCAL_MOL

DATA = Path(__file__).parent / "data" / "external"


def _ethanol():
    mol = Chem.AddHs(Chem.MolFromSmiles("CCO"))
    AllChem.EmbedMolecule(mol, randomSeed=1)
    return mol


def _fake(tmp_path, name, script):
    """An executable that stands in for the program: it writes what the program would."""
    path = tmp_path / name
    path.write_text("#!/bin/sh\n" + script)
    path.chmod(path.stat().st_mode | stat.S_IEXEC)
    return str(path)


def test_external_searches_in_and_out(tmp_path):
    mol = _ethanol()

    # -- the ensembles as conformers: absolute energies (kcal/mol), the method, the source
    crest = read_crest(DATA / "ethanol.crest_conformers.xyz", mol)
    assert len(crest) == 4 and crest.energy_method == "CREST gfn2"
    relative = crest.energies() - crest.energies().min()
    np.testing.assert_allclose(relative, [0.0, 1.551, 2.487, 3.596], atol=1e-3)
    assert crest.energy(crest.conf_ids[0]) == pytest.approx(
        -11.39433932 * HARTREE_TO_KCAL_MOL
    )
    assert crest.provenance(crest.conf_ids[1]) == {"source": "crest", "frame": 1}
    goat = read_goat(DATA / "ethanol.goat.finalensemble.xyz", mol, method="GOAT XTB")
    assert len(goat) == 3 and goat.energy_method == "GOAT XTB"
    # the same minimum, the same GFN2 energy from both programs
    assert goat.energies().min() == pytest.approx(crest.energies().min(), abs=1e-3)
    # atoms in another order are not this molecule's conformers
    reordered = Chem.RenumberAtoms(mol, [2, 1, 0] + list(range(3, 9)))
    with pytest.raises(ValueError, match="order"):
        read_crest(DATA / "ethanol.crest_conformers.xyz", reordered)

    # -- the inputs: charge, spin, method, solvent; what a task holds is held
    assert Crest(method="gfnff", solvent="water", threads=4).arguments(
        "start.xyz", charge=-1, multiplicity=2
    ) == [
        "start.xyz",
        "--gfnff",
        "--chrg",
        "-1",
        "--uhf",
        "1",
        "--alpb",
        "water",
        "-T",
        "4",
    ]
    text = Goat(solvent="water", workers=4).input_text(
        "start.xyz", charge=0, multiplicity=1, constraints="%geom\nend\n"
    )
    assert text.splitlines()[0] == "! GOAT XTB ALPB(water)"
    assert "NWORKERS 4" in text and "%geom" in text
    assert "* xyzfile 0 1 start.xyz" in text

    # -- a run in a fresh folder, with stand-ins for the programs
    fake_crest = _fake(
        tmp_path,
        "crest",
        f'cp "{DATA / "ethanol.crest_conformers.xyz"}" crest_conformers.xyz\n',
    )
    found = Crest(executable=fake_crest).run(mol, workdir=tmp_path / "crest_run")
    assert len(found) == 4 and (tmp_path / "crest_run" / "start.xyz").exists()
    with pytest.raises(FileExistsError):  # CREST leaves files that a second run reads
        Crest(executable=fake_crest).run(mol, workdir=tmp_path / "crest_run")
    fake_orca = _fake(
        tmp_path,
        "orca",
        f'cp "{DATA / "ethanol.goat.finalensemble.xyz"}" goat.finalensemble.xyz\n',
    )
    assert len(Goat(executable=fake_orca).run(mol)) == 3  # a temporary folder
    failing = _fake(tmp_path, "broken", "echo 'no licence' >&2\nexit 3\n")
    with pytest.raises(racerts.errors.ExternalSearchError, match="no licence"):
        Crest(executable=failing).run(mol)

    # -- a transition state: what it holds goes in as the exported constraints
    # (racerts.restraints.export), with the reference geometry for CREST
    recording = _fake(
        tmp_path,
        "crest_args",
        'echo "$@" > args.txt\n'
        f'cp "{DATA / "ethanol.crest_conformers.xyz"}" crest_conformers.xyz\n',
    )
    run = tmp_path / "crest_ts"
    Crest(executable=recording).run(mol, racerts.TransitionState([0, 1]), workdir=run)
    assert "--cinp constraints.inp" in (run / "args.txt").read_text()
    assert "reference=reference.xyz" in (run / "constraints.inp").read_text()
    assert (run / "reference.xyz").exists()


@pytest.mark.skipif(
    not os.environ.get("RACERTS_CREST"), reason="RACERTS_CREST names no CREST binary"
)
def test_a_real_crest_run(tmp_path):
    # CREST 3 as installed (RACERTS_CREST=/path/to/crest), quick mode on ethanol
    found = Crest(executable=os.environ["RACERTS_CREST"], mode="quick", threads=2).run(
        _ethanol(), workdir=tmp_path / "run"
    )
    assert 2 <= len(found) <= 6 and found.energy_method == "CREST gfn2"


@pytest.mark.skipif(
    not os.environ.get("RACERTS_ORCA"), reason="RACERTS_ORCA names no ORCA binary"
)
def test_a_real_goat_run(tmp_path):
    # ORCA 6 as installed (RACERTS_ORCA=/full/path/to/orca), GOAT capped to one round
    goat = Goat(
        executable=os.environ["RACERTS_ORCA"],
        workers=4,
        min_global_iterations=1,
        max_global_iterations=1,
        max_iterations=5,
    )
    found = goat.run(_ethanol(), workdir=tmp_path / "run")
    assert 1 <= len(found) <= 6 and found.energy_method == "GOAT XTB"
