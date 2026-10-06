"""A transition-state ensemble: the reacting atoms stay in place, the rest is sampled."""

from pathlib import Path

import racerts

ts = Path(__file__).parent / "ts.xyz"  # one TS geometry (a methyl transfer)

ensemble = racerts.generate_ts(
    str(ts),
    reacting_atoms=[7, 8, 22],  # 0-based: the atoms whose bonds form or break
    smiles="C/[NH+]=C(OC)/c1ccccc1.COS(=O)(=O)[O-]",  # the bonds, fragments joined by "."
    config=racerts.PipelineConfig(
        embed={"n_conformers": 30}
    ),  # default: by flexibility
)

print(ensemble.summary())
ensemble.write_xyz("ts_conformers.xyz")  # extended XYZ, lowest energy first
