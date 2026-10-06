"""Hints: hydrogen bonds that the graph makes possible are tried in a share of the conformers."""

import numpy as np

import racerts


def closed(ensemble) -> int:
    """The conformers with the two oxygens within 3.2 A of each other."""
    oxygens = [a.GetIdx() for a in ensemble.mol.GetAtoms() if a.GetSymbol() == "O"]
    positions = [ensemble.mol.GetConformer(i).GetPositions() for i in ensemble.conf_ids]
    return sum(np.linalg.norm(p[oxygens[0]] - p[oxygens[1]]) < 3.2 for p in positions)


for hints in (False, True):
    config = racerts.PipelineConfig(
        embed={"n_conformers": 40}, restraints={"hints": hints}
    )
    ensemble = racerts.generate_gs("OCCCCCO", config=config)  # pentane-1,5-diol
    print(f"hints {hints}: {closed(ensemble)} of {len(ensemble)} conformers closed;")
    print("  ", ensemble.summary())
