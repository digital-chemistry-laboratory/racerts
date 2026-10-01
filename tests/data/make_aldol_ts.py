"""Writes tests/data/aldol_ts.xyz: a TS-like geometry of the proline-catalysed aldol
reaction (Houk-List): the enamine of proline and acetone adds to benzaldehyde while the
carboxylic acid transfers its proton to the aldehyde O.

    python tests/data/make_aldol_ts.py   (needs racerts, ASE and tblite)

Embedded with racerts restraints on heavy atoms (forming C10-C12 at 2.2 A, O0...O11 at
2.5 A), with H19 placed on the O0...O11 line (O0-H19 1.15 A, H19...O11 1.35 A), then
GFN2-xTB (tblite) optimized with these three distances fixed; the lowest of the embedded
complexes is written.

Atoms (0-based): 0 O-H, 1 C(acid), 2 O=, 3 C-alpha, 7 N, 8 C(enamine), 9 CH3,
10 CH2 (attacking), 11 O (aldehyde), 12 C (aldehyde), 19 H (transferred).
Reacting atoms: 0, 10, 11, 12, 19.
"""

import os

import numpy as np
from ase import Atoms
from ase.constraints import FixBondLengths
from ase.optimize import BFGS
from tblite.ase import TBLite

import racerts

SMILES = "OC(=O)[C@@H]1CCCN1C(C)=C.O=Cc1ccccc1"
FIXED = [(10, 12, 2.2), (0, 19, 1.15), (19, 11, 1.35)]
EMBED = [(10, 12, 2.2), (0, 11, 2.5)]
REACTING = [0, 10, 11, 12, 19]
HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    config = racerts.PipelineConfig.from_dict(
        {
            "seed": 7,
            "embed": {"n_conformers": 12, "etkdg": False},
            "restraints": {"user": [list(t) for t in EMBED], "half_width": 0.1},
        }
    )
    ensemble = racerts.generate_gs(SMILES, config=config)
    symbols = [a.GetSymbol() for a in ensemble.mol.GetAtoms()]
    best = None
    for conf_id in ensemble.conf_ids:
        atoms = Atoms(symbols, positions=ensemble.mol.GetConformer(conf_id).GetPositions())
        pairs = [[i, j] for i, j, _ in FIXED]
        # C12 at 2.2 A from C10; O11 at 2.5 A from O0; H19 on the O0...O11 line.
        positions = atoms.get_positions()
        for i, j, d in ((10, 12, 2.2), (0, 11, 2.5)):
            v = positions[j] - positions[i]
            positions[j] = positions[i] + v / np.linalg.norm(v) * d
        v = positions[11] - positions[0]
        positions[19] = positions[0] + v / np.linalg.norm(v) * 1.15
        atoms.set_positions(positions)
        atoms.set_constraint(FixBondLengths(pairs))
        atoms.calc = TBLite(method="GFN2-xTB", verbosity=0)
        BFGS(atoms, logfile=None).run(fmax=0.02, steps=300)
        energy = atoms.get_potential_energy()
        if best is None or energy < best[0]:
            best = (energy, atoms.copy())
    energy, atoms = best
    lines = [str(len(atoms)), f"aldol TS-like (GFN2-xTB, C10-C12 2.2 A), E = {energy:.6f} eV"]
    lines += [
        f"{s:2s} {x:12.6f} {y:12.6f} {z:12.6f}"
        for s, (x, y, z) in zip(atoms.get_chemical_symbols(), atoms.get_positions())
    ]
    with open(os.path.join(HERE, "aldol_ts.xyz"), "w") as handle:
        handle.write("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
