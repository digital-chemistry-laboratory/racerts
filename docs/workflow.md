# Staged workflow

The staged workflow refines conformers in levels, from cheap to expensive, and searches
around the best ones at the expensive level before the final ranking.
`racerts.recipes.staged` builds it as a pipeline:

```python
import racerts
from racerts.recipes import staged
from racerts.refine import ASEOptimizer

uma = ASEOptimizer(uma_calculator, method="UMA-s-1p2", fmax=0.02)
pipeline = staged(uma, config=racerts.PipelineConfig(embed={"n_conformers": 100}))
ensemble = racerts.generate_ts("ts.xyz", [3, 4, 5], smiles="...", pipeline=pipeline)
```

| Step | Stages | Purpose |
| --- | --- | --- |
| 1 | `Embed`, `Validate(Clash(0.5))` | one embedding: its batches mix biased (restraints, hints, active-bond targets) and unbiased settings, and every conformer records its batch; conformers with overlapping heavy atoms are dropped |
| 2 | `Refine(cheap)`, the gate, `AttackFace` (windowed TSs), `PruneEnergy` (25 kcal/mol), `PruneRMSD` | remove embedding artifacts and duplicates; the window is loose, since the cheap energies should not decide populations |
| 3 | `Refine(expensive)`, the gate, `PruneEnergy` (8 kcal/mol), `PruneRMSD` | rank at the expensive level: the first point where "low energy" means anything |
| 4 | `Exploit` | Monte Carlo around the best conformers with the same refinement ([Exploit](pipeline.md#exploit)); its acceptance already applies an energy window and the RMSD check |
| 5 | `PruneRMSD`, `PruneEnergy` (6 kcal/mol), optionally `Rescore` | the final ensemble; rescoring (a higher level, the target solvent) restores the populations after the biasing upstream |

Arguments:

- **`expensive`:** the refinement of steps 3 and 4, an optimizer or a `Refine` stage. It
  should be an `ASEOptimizer` (xTB, an MLIP), since `Exploit` needs one.
- **`cheap`:** default: the refinement of the default pipeline (MMFF, not converged:
  it only has to remove the artifacts of the embedding).
- **`windows`:** the three energy windows in kcal/mol.
- **`embed`:** the `Embed` stage; default: that of the default pipeline.
- **`config`:** the [settings](pipeline.md#settings) of the default embedding and cheap
  refinement, e.g. `PipelineConfig(embed={"n_conformers": 100})`.
- **`exploit`:** the settings of `Exploit` as a dict, or `None` to leave it out.
- **`rescore`:** a `Rescore` stage.
- **`clash_filter`:** step 1 drops conformers with heavy atoms closer than this times
  their vdW sum (default 0.5; `None`: no filter). Distance geometry may place atoms four
  bonds apart at their lower bound, 0.7 times the vdW sum (0.555 for some), and
  refinement relaxes such contacts, so only overlaps are dropped here; the gate after each
  refinement checks clashes at 0.7. A filter at 0.7 here can drop the conformer that is
  the lowest after refinement.

## The gate

After each refinement, `racerts.validate.gate()` drops the conformers that are no longer
valid:

- frozen atoms that moved by more than 0.1 Å (`FrozenCore`; MMFF/UFF leave them up to
  0.09 Å from the reference);
- changed bonds or stereo (`Connectivity`);
- clashing heavy atoms (`Clash`);
- restraints broken by more than 0.5 Å (`RestraintViolation`).

A conformer that fails is dropped, not reset to its input geometry. Its input is no
minimum at the new level, and a structure that breaks at the expensive level often has
no basin there. The log counts the failures per check, and warns when more than 30 %
fail: that points to a wrong charge, restraint or hypothesis rather than to single bad
conformers.

## Restraints at the expensive level

ASE calculators take the restraints like MMFF/UFF do: `ASEOptimizer` adds flat-bottom
terms for distance windows and soft atoms to its calculator, and the energies it stores
leave them out.

## Transition states

The active-bond windows of a TS act in embedding only:

- **Refinement:** every refinement holds each conformer's active bonds at its embedded
  length or target (± 0.02 Å). `Exploit` keeps each child's targets.
- **Comparing energies:** energies at different held lengths are not comparable, so the
  energy windows and the RMSD pruning work per window bin.
- **Output:** the result is a set of TS-like structures for a TS optimization, e.g.
  `Refine(ASEOptimizer(..., optimizer_cls=Sella), anchors=False)` followed by
  `Validate(ImaginaryModes(...), ReactionCore())`.

## Cost

- **Calculators on a GPU:** pass a calculator instance and `num_workers=1`. A calculator
  factory makes a new calculator for every structure when it runs without worker
  processes.
- **Budget:** `Exploit` stops when it rarely finds new minima, and at the latest after
  `max_optimizations`. `exploit.stats` reports its numbers.
