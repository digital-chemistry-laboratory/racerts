# Pipelines and tasks

racerts splits a conformer search into a **task** (what stays fixed), a **pipeline**
of stages (what is done), and a **context** that carries the molecule, the task, the
seed and the restraints through the stages. `PipelineConfig.legacy()` (`racerts ts
--legacy`) reproduces legacy racerts exactly. The defaults differ from it in these
settings: one seed per conformer, the `frozen_first` chirality fallback with a stereo
check after refinement, energies without the anchor terms, conformer counts that
include the freedom of separate fragments, and duplicates decided by their RMSD alone,
without the two prefilters (see [Settings](settings.md)).

See also: [settings](settings.md), [plug-in points](plugins.md), [restraints](restraints.md),
[active-bond windows](active-bonds.md) and [swaps](swap.md).

```python
import racerts

ensemble = racerts.generate_ts("ts.xyz", [3, 4, 5], smiles="CCCCCC=C")
```

is the same as

```python
from racerts.system import build_mol

mol = build_mol("ts.xyz", charge=0, reacting_atoms=[3, 4, 5], input_smiles=["CCCCCC=C"])
ensemble = racerts.generate(mol, racerts.TransitionState([3, 4, 5]))
```

## Tasks

| Task | Frozen atoms | Needs a reference geometry |
| --- | --- | --- |
| `TransitionState(reacting_atoms, frozen_atoms=None)` | reacting atoms and their neighbours (or `frozen_atoms`) | yes |
| `TransitionState.from_endpoints(reactant, product)` | the same, with the reacting atoms from the bonds that form or break between two atom-aligned endpoints | yes |
| `GroundState()` | none | no |
| `Constrained(hard, soft=(), core=None)` | the `hard` atoms; `soft` atoms start at the reference and are held near it | yes |

A task returns a `FrozenSet`: `hard` atoms are placed at the reference positions during
embedding and held there during refinement; `soft` atoms are placed there too and held
within 0.3 Å by position restraints in MMFF/UFF refinement (e.g. the kept atoms of a
[swap](swap.md)). In the bounds-matrix embedder (`embed.mode = "bounds"`), the distances
between the `core` atoms (for a TS the reacting atoms) and all hard atoms are fixed.

The distance bounds of the graph assume its equilibrium geometry, which a TS core need
not have: with the product SMILES of an oxidative addition, for example, the graph puts
the metal in the plane of the aryl ring it sits over in the TS. When the bounds cannot
be reconciled with the frozen atoms, RDKit embeds nothing. racerts then widens the bounds
that exclude a distance of the reference to that distance (`embed.reference_bounds =
"fallback"`, with a warning that names the pairs; the conformers record
`widened_bounds` in their provenance). The SMILES of the other side of the reaction may
fit the core better.

```python
# Ground states (ETKDGv3 embedding, nothing frozen, stereocentres kept)
ensemble = racerts.generate_gs("OC(=O)[C@@H]1CCCN1C(C)=C")

# Keep a motif of a structure fixed, sample the rest
mol = build_mol("complex.xyz", charge=0, reacting_atoms=[])
ensemble = racerts.generate(mol, racerts.Constrained(hard=[0, 1, 2, 3]))
```

## Pipelines and stages

The default pipeline is `Embed → Refine → Validate (stereo) → PruneEnergy → PruneRMSD`
(the legacy one has no stereo check). Each stage takes the
context and the ensemble so far and returns an ensemble; stages built without arguments
use the legacy racerts settings, except that they embed with a seed per conformer and
that `PruneRMSD()` decides duplicates by their RMSD alone, with the polar hydrogens. A
pipeline can be put together by hand, e.g. to refine
with an ASE calculator (here Lennard-Jones, which stands in for a real one such as
GFN2-xTB or a machine-learned potential):

```python
from ase.calculators.lj import LennardJones
from racerts.refine import ASEOptimizer

pipeline = racerts.Pipeline([
    racerts.Embed(n_conformers=10),
    racerts.Refine(ASEOptimizer(calculator=LennardJones(), fmax=0.1)),
    racerts.PruneEnergy(),
    racerts.PruneRMSD(),
])
ensemble = racerts.generate_ts("ts.xyz", [3, 4, 5], pipeline=pipeline)
```

With `pipeline=`, `generate` uses only the seed and the restraints of `config`.

Any object with a `name` and a `run(ctx, ensemble)` method is a stage:

```python
class KeepLowest:
    name = "keep_lowest"

    def __init__(self, n):
        self.n = n

    def run(self, ctx, ensemble):
        order = sorted(ensemble.conf_ids, key=ensemble.energy)
        return ensemble.filter(order[: self.n])
```

The context gives a stage the molecule with its reference geometry (`ctx.mol`,
`ctx.reference`), the task and its frozen atoms (`ctx.task`, `ctx.frozen`), the seed
and the restraints. `generate` makes it; to run a pipeline directly:

```python
ctx = racerts.Context.create(mol, racerts.TransitionState([3, 4, 5]), seed=12)
ensemble = pipeline.run(ctx)
```

Components are the objects that stages call. A new embedder subclasses
`racerts.embed.BaseEmbedder` and implements `embed(mol, reference, frozen, n)`; a new
optimizer subclasses `racerts.refine.BaseOptimizer` and implements
`_refine(mol, reference, anchors)`, which `refine` calls after checking its arguments.
Components written for legacy racerts (`embed_TS`, `tune_ts_conformers`) run in the stages
too: their base classes in `racerts.compat` route `embed` and `refine` to the legacy
methods.

More stages:

| Stage | Does |
| --- | --- |
| `Embed(references="all" or ids)` | embeds from each of several reference geometries (conformers of the context's molecule, e.g. TSs found by a TS search); the provenance records `reference`, and `Refine` refines each conformer against its own reference |
| `Refine(optimizer, anchors=False)` | refines all atoms freely, e.g. a saddle-point search from TS-like conformers; also releases [active-bond windows](active-bonds.md) |
| `Rescore(calculator)` or `Rescore(batch=fn)` | replaces the energies by single points of an ASE calculator (xTB, MLIPs), or of a function that evaluates a list of Atoms at once; the replaced energy goes into the provenance. `add=True` adds them instead (a correction, e.g. a solvation term) |
| `PruneCount(n_max, renumber=False)` | keeps the `n_max` lowest conformers (with [active-bond windows](active-bonds.md): the lowest of each target in turn) |
| `PruneCluster(ClusterPruner(...))` | keeps one conformer per cluster (`ClusterPruner` from `racerts.prune`): Butina, hierarchical (scipy) or leader clustering, on the RMSD after superposition with fixed atoms (`kernel="aligned"`, e.g. for TS graphs without bonds), the symmetry-aware RMSD (`"symmetric"`; with `graph=` if the conformers are stored without bonds), or any `metric(mol, a, b)`; the lowest or the central member |
| `racerts.prune.FamilySelector(n_max, clusterer)` | up to `n_max` conformers spread over the clusters: the best of each, then the second best, ... (families ordered by their best member) |
| `Validate(*validators, on_fail="drop" or "flag")` | checks the conformers (see below): drops the failing ones (an error if none passes), or keeps them with the reasons in their provenance (a warning if none passes) |
| `Exploit(Refine(ASEOptimizer(...)))` | usage-directed Monte Carlo around the pruned conformers: new minima nearby, from torsions, rigid-body moves of fragments and ring flips, refined by an ASE calculator (xTB, MLIPs); stops when new minima become rare (see below) |

`Pipeline.run(ctx, ensemble)` continues an ensemble (e.g. to refine it with another
method) and leaves the ensemble passed in unchanged; stages may change the ensemble they
get in place. `Embed` starts an ensemble: to combine two embedding runs (with different
seeds), merge their ensembles. Components passed to a stage (an embedder with its seed,
an optimizer) keep their own settings; stages create default ones for the task, e.g.
`Embed()` uses ETKDGv3 for ground states, with RDKit's macrocycle and small-ring torsion
terms. The small-ring terms are left out with restraints, so that a window can take a
ring out of its chair. RDKit fails with them on cyclopentane rings: the embedding then
goes on without them, with a warning.

The pipeline logs every stage with its number of conformers and run time (logger
`racerts.pipeline.runner`, level INFO); `generate`, `generate_ts` and `generate_gs` take
`verbose=True` to show these messages.

### Exploit

`Exploit` searches around the conformers it gets, after the pruners: it never leaves
their neighbourhood, so its value depends on them (`Embed → Refine → PruneEnergy →
PruneRMSD → Exploit → PruneRMSD`). It is usage-directed multiple-minimum Monte Carlo
(Chang, Guida, Still 1989):

- each iteration takes the `batch` least-used conformers within `energy_window` of the
  minimum and applies one to three random moves:
  - torsions: never the side with frozen atoms; not amide or ester bonds, not methyl
    groups;
  - rotations of fragments by their role ([fragment roles](restraints.md#fragment-roles-and-containment)):
    anchored fragments turn about their anchor, contained and free ones also shift;
  - ring flips of non-fused five- and six-membered rings: chair flips, twist-boats,
    envelopes and twists. The substituents turn with their ring atom, so a chair flip
    swaps axial and equatorial;
- candidates whose heavy atoms clash are drawn again; the others are refined in one
  call by the stage's refiner, which must be an `ASEOptimizer` (xTB, MLIPs: force-field
  minima would be the wrong ones; with MMFF/UFF, Exploit warns and changes nothing);
- a refined candidate is kept if it passes the gate checks, lies within `energy_window`
  of the minimum and is further than `rmsd_threshold` from every conformer; a duplicate
  counts as a re-find. Structures that stop on a torsional barrier (turning a moved
  torsion by ±10° lowers the energy) are optimized again;
- it stops when the estimated chance of a new minimum within `stop_window` of the
  minimum drops below `saturation` (Good–Turing: the conformers found once, divided by
  the optimizations), or after `max_optimizations`.

```python
from racerts.refine import ASEOptimizer

uma = ASEOptimizer(uma_calculator, method="UMA-s-1p2", fmax=0.02)
pipeline = racerts.Pipeline([
    racerts.Embed(), racerts.Refine(), racerts.PruneEnergy(), racerts.PruneRMSD(),
    racerts.Refine(uma), racerts.PruneEnergy(), racerts.PruneRMSD(),
    racerts.Exploit(uma, max_optimizations=200),
    racerts.PruneRMSD(),
])
```

The conformers it gets must carry the energies of Exploit's refiner (`energy_method`,
e.g. the `method` label of the `ASEOptimizer`). `Exploit(uma, rank=Rescore(...))` ranks
by another energy than the refiner's ([the ranking energy](workflow.md#the-ranking-energy)).
New conformers record `route="mc"`, `parent`,
`moves` and `mc_iteration`; every conformer records `mc_usage` and `mc_hits`, and the
stage keeps the numbers of its last run in `stats` (of a pipeline:
`next(s for s in pipeline.stages if s.name == "exploit").stats`). For a windowed TS,
children keep their parent's active-bond targets, and energies are compared per window
bin.

## Independent runs

Is the search converged? For a flexible system one run does not tell: its own statistics
see only the region it reached. `generate_runs` repeats the search with several seeds,
merges the runs and compares them:

```python
result = racerts.generate_runs(mol, task, seeds=[1, 2, 3], config=config)
print(result.report)
ensemble = result.merged      # every conformer once; provenance "run" and "found_by"
result.ensembles              # the ensemble of each seed
if not result.report.converged(tolerance=0.3):
    ...                       # more seeds, more conformers, or a local search (Exploit)
```

```
3 runs, energies MMFFOptimizer (kcal/mol, 298.15 K)
run         conformers   lowest   free energy   (above the union of all)
1                  177     0.43          0.51
2                  173     0.00          0.65
3                  190     0.44          0.46
spread of the free energy between runs: 0.19
found again: the population of a run within 2 of its minimum that another run has too
  1 -> 2:  18% (12 of 77 conformers)
  ...
  mean 21%, lowest 14%
merged runs: the free energy above that of all runs (mean, worst)
  1 run: 0.54, 0.65
  2 runs: 0.19, 0.22
  3 runs: 0.00, 0.00
leaving one run out changes the free energy by up to 0.22
```

- **free energy**: the ensemble free energy −RT ln Σ exp(−E/RT) of a run, above that of
  the union. It is what enters a barrier or a reaction energy.
- **found again**: the share of a run's low-energy population that another run has too.
  The free energy converges before the list of conformers does: many conformers of
  similar energy give the same free energy whichever of them a run finds.
- **leaving one run out**: what the last run still changed. `report.converged(tolerance)`
  asks whether that is at most `tolerance` kcal/mol.

Two conformers are the same if their energies agree (0.1 kcal/mol for MMFF and UFF, 1.0
for other methods; `energy_tolerance`) and their symmetry-aware heavy-atom RMSD is at
most `rmsd` (0.25 Å). `racerts.compare_runs(ensembles)` and `racerts.merge_runs(ensembles)`
do the same for ensembles from elsewhere (one method, an energy for every conformer).
Active bonds that are still held at their targets do not compare: search the saddle
points freely first.

## Conformer ensembles

A `ConformerEnsemble` wraps the RDKit molecule with its conformers (`ensemble.mol`).
Everything known about the conformers is stored in RDKit properties, so RDKit code and
the legacy classes see the same data:

| Property | Of | Content |
| --- | --- | --- |
| `energy` | each conformer | energy in kcal/mol |
| `provenance` | each conformer | JSON, e.g. `{"embedder": "CmapEmbedder", "seed": 12, "etkdg": false}` |
| `energy_method` | the molecule | the optimizer or calculator that gave the energies |
| `charge`, `multiplicity` | the molecule | total charge and spin multiplicity |

Conformer ids are those of the embedding; pruning leaves gaps, so use `conf_ids`:

```python
ensemble.conf_ids              # e.g. [0, 3, 4, 9]
ensemble.energies()            # numpy array in the order of conf_ids, NaN where missing
ensemble.energies(unit="eV")   # also "kJ/mol", "hartree"
best = ensemble.best()         # id of the lowest conformer
ensemble.record(best)          # ConformerRecord(conf_id, energy, energy_method, provenance)
ensemble.filter([9, 0])        # a copy with these conformers, in this order
ensemble.filter([9, 0], renumber=True)  # ... with the ids 0, 1
ensemble.merge(other)          # conformers of both (same graph and energy method)
ensemble.merge(other, identity="elements")  # same elements only, e.g. other bond orders
ensemble.to_ase(best)          # ASE Atoms with charge and multiplicity
ensemble.write_xyz("out.xyz")  # extended XYZ; use_energy=True for CREST-style energies
ensemble.write_sdf("out.sdf")  # one record per conformer, with energy and provenance
racerts.ConformerEnsemble.from_frames(mol, frames)  # external geometries (Atoms or arrays)
```

Graphs for geometries from elsewhere: `racerts.system.mol_from_geometry(atoms, smiles)`
takes the bond orders, charges and radicals of an explicit-hydrogen SMILES onto the
connectivity of a geometry and raises if they do not fit; `mol_from_explicit_h_smiles`
and `bondless_mol` are the pieces. A ground state (`generate_gs`, or `generate` with
`GroundState()`) whose graph has radical electrons (e.g. `[CH2]`) gets the multiplicity
1 + their number, unless one is given.

Ensembles can be pickled (e.g. to return them from worker processes) with all their
data; plain RDKit pickling of `ensemble.mol` drops the properties.
