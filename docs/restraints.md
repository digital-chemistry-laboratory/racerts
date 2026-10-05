# Restraints

A restraint keeps the distance of two atoms in a window `[lower, upper]` (Å):

- in embedding, the window replaces the bounds of the pair (distance geometry places the
  atoms in it, in most conformers);
- in refinement, a flat-bottom term ½·k·(d − bound)² outside the window holds it (k in
  kcal/(mol Å²), as in RDKit; default 20, with ±0.25 Å windows). MMFF/UFF add it to the
  force field; ASE calculators (xTB, MLIPs) get it through
  `racerts.refine.restrained.RestrainedCalculator`, which wraps the calculator.

Reported energies never include the restraint terms (nor the terms that hold the frozen
atoms).

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
| `hbonds` | false | the hydrogen bonds D–H···A of the reference geometry: H···A and D···A at their distances |
| `contacts` | `[]` | `[i, j]`: the contact as in the reference geometry, with i and the neighbours of j, j and the neighbours of i (for the orientation) |
| `keep_fragments` | false | each fragment without reacting atoms (solvent, counterions) at its closest contact to the core, as in the reference geometry |
| `fragment_links` | `[]` | `[i, j]` between two fragments: [1.0, 1.3] × the sum of the vdW radii |
| `link_fragments` | false | links chosen to join every fragment (user pairs between fragments, then charged pairs, then the least buried atoms) |
| `half_width` | 0.25 | of the `user`, `hbonds`, `contacts` and `keep_fragments` windows |
| `force_constant` | 20.0 | of all these windows and of the fragment links |
| `hints`, `max_hints`, `hint_share` | false, 8, 0.3 | see [Hints](#hints) |
| `contain`, `roles` | 0, none | see [Fragment roles and containment](#fragment-roles-and-containment) |

`hbonds`, `contacts` and `keep_fragments` need a reference geometry (not for
`generate_gs` from a SMILES).

Rules:

- User restraints win over generated ones for the same pair; two user restraints on one
  pair must be equal.
- Generated windows that the bounds cannot take with the user's are left out, with a
  warning.
- Restraints between two frozen atoms are left out (their distance is fixed), with a
  warning.
- User windows that need more triangle-smoothing tolerance than the bounds without them
  raise `ValueError` ("inconsistent"): smoothing would otherwise repair them silently,
  e.g. by stretching a bond. Fragment links are widened once (lower factor 0.8) before
  that; user windows never are.

On the command line: `--restraint I J D` (repeatable), `--hints`,
`--restraint-half-width` and `--restraint-force-constant` for `racerts ts` and
`racerts gs`; `--keep-hbonds`, `--contact I J` and `--keep-fragments` for `racerts ts`;
`--link-fragments` for `racerts gs`.

From Python, `racerts.restraints.RestraintSet` and `DistanceRestraint` (with `stage`
`"embed"`, `"refine"` or `"both"`) can be passed to `generate`, `generate_ts` or
`generate_gs` as `restraints=`, and `build_restraints(mol, frozen, ...)` builds them
from the sources above.

## Hints

`restraints.hints` adds candidate hydrogen bonds from the graph (at most `max_hints`)
as embedding-only windows H···A 1.7–2.3 Å:

- donors N–H and O–H; acceptors N, O and F, but not amide, aniline or pyrrole-type N and
  not the alkoxy O of esters;
- pairs that close a pseudo-ring of at least 6 atoms, or lie in different fragments;
- no charged partners (MMFF's Coulomb term pulls them together anyway);
- ranked by how close the pseudo-ring is to 7 atoms.

A share `hint_share` of the conformers is embedded with a hint: one batch per hint and
one with all of them, if they fit together; the others without. Refinement is free, so
a hint only chooses starting points. The provenance of each conformer lists its hints
(`active_restraints`).

## Fragment roles and containment

In a system of several molecules (solvent, counterions, a separate substrate), every
fragment has a role that says how it may move as a whole (`racerts.system.roles`):

| Role | Fragment | Moves in `Exploit` |
| --- | --- | --- |
| reactive | holds core atoms of the task (for a TS the reacting atoms); without any, the largest fragment | none |
| anchored | a restraint of the refinement ties it to the core, directly or through other anchored fragments (hints do not count) | turns about its anchor |
| contained | only a containment restraint holds it near the core | turns about its central atom and shifts |
| free | nothing holds it | turns and shifts |

- **Where roles come from:** racerts infers the roles from the task and the restraints,
  and a pipeline logs them when it starts (logger `racerts.pipeline.runner`, level
  INFO).
- **Containment:** `restraints.contain` puts a containment restraint on every fragment
  that no restraint anchors. It is a window `[0, contain]` (Å) between the fragment's
  central atom and the central atom of the core.
- **Overrides:** `restraints.roles` changes single fragments, keyed by any atom of the
  fragment (not the reactive ones).
- **Conformer count:** the count policy `"fragments"` counts 3 rotational degrees of
  freedom for an anchored fragment and 6 for the others.

```python
config = racerts.PipelineConfig.from_dict({
    "restraints": {"hbonds": True, "contain": 6.0, "roles": {"9": "free"}},
})
```

## Restraints for other programs

`--export-restraints` writes the frozen atoms and the restraints of a system for another
program and stops before embedding. It is the exit for sampling with another program,
e.g. CREST with racerts' restraints:

```bash
racerts ts ts.xyz -r 3 4 5 --keep-hbonds --export-restraints crest   # restraints.xcontrol
crest ts.xyz --cinp restraints.xcontrol ...
```

| Format | Written | For |
| --- | --- | --- |
| `xtb` | `$fix` for the frozen atoms, `$constrain` for the windows | `xtb --input` |
| `crest` | `$constrain` for the frozen atoms (with `reference=`, written next to the file) and the windows, `$metadyn` on the other atoms | `crest --cinp` |
| `orca` | a `%geom` block with the frozen atoms and the windows as constraints | ORCA optimizations |
| `json` | the frozen atoms and the restraints | anything else |

- **Windows:** none of these programs has flat-bottom distances, so every window becomes a
  harmonic constraint at its centre (ORCA: an exact one), and a warning says so. Windows
  without a lower bound (containment) are left out.
- **Force constants:** xtb and CREST take one force constant for all constraints. racerts
  converts it (xtb: E = fc·(d − d0)² in Eh/bohr², checked with xtb 6.6.1). With frozen
  atoms, CREST gets xtb's default 0.5 Eh/bohr², so that the frozen atoms hold; the
  windows are then held as stiffly.
- **Atom numbering:** 1-based for xtb and CREST, 0-based for ORCA and JSON.

In Python, `racerts.restraints.export.export_restraints(ctx, "crest")` returns the text,
and the stage `ExportRestraints(format, path)` writes it.
