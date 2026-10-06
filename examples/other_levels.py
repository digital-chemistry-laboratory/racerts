"""Another level of theory: any ASE calculator, here GFN2-xTB (pip install tblite)."""

from tblite.ase import TBLite

import racerts
from racerts.recipes import Level, staged
from racerts.refine import ASEOptimizer


def gfn2():
    return TBLite(method="GFN2-xTB", verbosity=0)


xtb = ASEOptimizer(
    gfn2, method="GFN2-xTB"
)  # a calculator, or a function that makes one
pipeline = staged(
    [
        Level(window=25),  # the force field first: conformers within 25 kcal/mol go on
        Level(xtb),  # refined with GFN2-xTB
    ],
    config=racerts.PipelineConfig(embed={"n_conformers": 10}),
)
ensemble = racerts.generate_gs("OCCO", pipeline=pipeline)  # ethane-1,2-diol

print(ensemble.summary())
ensemble.write_xyz("glycol_gfn2.xyz")
