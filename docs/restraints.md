# Restraints

A restraint keeps the distance of two atoms in a window `[lower, upper]` (Å):

- in embedding, the window replaces the bounds of the pair (distance geometry places the
  atoms in it, in most conformers);
- in MMFF/UFF refinement, a flat-bottom term ½·k·(d − bound)² outside the window holds it
  (k in kcal/(mol Å²), as in RDKit; the default 20 with ±0.25 Å windows follows catmlp).

Reported energies never include the restraint terms (nor the terms that hold the frozen
atoms). Refinements with ASE calculators (xTB, MLIPs) run without restraints: the
restraints guide the starting geometries only.

```python
config = racerts.PipelineConfig.from_dict({
    "restraints": {"user": [[2, 7, 2.2], [2, 6, 3.16]], "hbonds": True},
})
ensemble = racerts.generate_ts("sn2_water.xyz", [0, 1, 2], charge=-1,
                               smiles=["CCl", "[Cl-]", "O"], config=config)
```

| Setting (`restraints.`) | Default | Restraints |
| --- | --- | --- |
| `user` | `[]` | `[i, j, d]`: `d` ± `half_width` |
| `hbonds` | false | the hydrogen bonds D–H···A of the input geometry: H···A and D···A at their distances |
| `contacts` | `[]` | `[i, j]`: the contact as in the input geometry, with i and the neighbours of j, j and the neighbours of i (for the orientation) |
| `keep_fragments` | false | each fragment without reacting atoms (solvent, counterions) at its closest contact to the core, as in the input geometry |
| `half_width`, `force_constant` | 0.25, 20.0 | of the `user`, `hbonds`, `contacts` and `keep_fragments` windows |

Rules:
- user restraints win over generated ones for the same pair; generated ones must agree;
- restraints between two frozen atoms are left out (their distance is fixed), with a
  warning;
- windows that need more triangle-smoothing tolerance than the bounds without them raise
  `ValueError` ("inconsistent"): smoothing would otherwise repair them silently, e.g. by
  stretching a bond.

On the command line: `--restraint I J D` (repeatable), `--keep-hbonds`, `--contact I J`,
`--keep-fragments`, `--restraint-half-width`,
`--restraint-force-constant`.

From Python, `racerts.restraints.RestraintSet` and `DistanceRestraint` (with `stage`
`"embed"`, `"refine"` or `"both"`) can be passed to `generate`, `generate_ts` or
`generate_gs` as `restraints=`, and `build_restraints(mol, frozen, ...)` builds them
from the sources above.
