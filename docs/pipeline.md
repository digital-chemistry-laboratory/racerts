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

`generate_ts` builds the molecule and a `TransitionState` from its arguments. Any other
task goes through `generate` with a molecule from `build_mol`, as above:

```python
task = racerts.TransitionState.from_endpoints(reactant, product)  # two RDKit molecules
ensemble = racerts.generate(mol, task)
```

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
| `racerts.prune.FamilySelector(n_max, clusterer)` | up to `n_max` conformers spread over the clusters: the best of each, then the second best, ... (families ordered by their best member). The same order for clusters of your own: `round_robin(ClusterPruner(...).clusters(mol))` |
| `Validate(*validators, on_fail="drop" or "flag")` | checks the conformers (see below): drops the failing ones (an error if none passes), or keeps them with the reasons in their provenance (a warning if none passes) |
| `Exploit(Refine(ASEOptimizer(...)))` | usage-directed Monte Carlo around the pruned conformers: new minima nearby, from torsions, rigid-body moves of fragments and ring flips, refined by an ASE calculator (xTB, MLIPs); stops when new minima become rare (see below) |
| `Metadynamics(ASEOptimizer(...), md_calculator=...)` | molecular dynamics from the conformers it gets, pushed away from the structures it has visited by a bias on the RMSD (as in CREST); the snapshots are refined by an ASE calculator and added where they are new minima (see below) |
| `Recombine(optimizer)` | new conformers from the torsions of the low conformers it gets: a child is one of them with some torsions of another (the crossing of CREST, in torsions); the children are refined by any optimizer and added where they are new minima (see below) |
| `Resample(optimizer)` | new conformers whose torsions are each drawn from all the conformers it gets: a child is one of them in which every torsion takes the value of a member drawn for it alone (the distribution of each torsion over the ensemble); refined and added as by `Recombine` (see below) |

`Pipeline.run(ctx, ensemble)` continues an ensemble (e.g. to refine it with another
method) and leaves the ensemble passed in unchanged; stages may change the ensemble they
get in place. `Embed` starts an ensemble: to combine two embedding runs (with different
seeds), merge their ensembles. Components passed to a stage (an embedder with its seed,
an optimizer) keep their own settings; stages create default ones for the task, e.g.
`Embed()` uses ETKDGv3 for ground states, with RDKit's macrocycle and small-ring torsion
terms. ETKDG's torsion preferences are statistics of small-molecule crystal structures;
`embed.torsion_preferences = false` leaves them out (RDKit's KDG), for molecules whose
conformers in solution need not follow them, and keeps the knowledge terms (planar sp2
atoms, linear sp atoms). With restraints, ETKDG weights the distance bounds 100 times higher against its
torsion terms (`racerts.embed.RESTRAINT_BOUNDS_WEIGHT`, RDKit's `boundsMatForceScaling`),
which override about 60 % of the windows otherwise, and leaves the small-ring terms out,
so that a window can take a ring out of its chair. Conformers that are embedded with
restraints therefore start from weaker torsion preferences than those embedded without;
the refinement is the same for both. RDKit fails with the small-ring terms on
cyclopentane rings: the embedding then goes on without them, with a warning.

The pipeline logs every stage with its number of conformers and run time (logger
`racerts.pipeline.runner`, level INFO); `generate`, `generate_ts` and `generate_gs` take
`verbose=True` to show these messages.

Four stages search on from the conformers they get. `Exploit` makes local moves around
them, `Metadynamics` explores away from them, and `Recombine` combines what they
already have between them: the stage for a large flexible molecule whose low conformers
each have a part of a lower one. `Resample` draws each torsion from all of them.

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

### Metadynamics

`Metadynamics` explores where `Exploit` exploits: it leaves the neighbourhood of the
conformers it gets, and it needs no moves, so it also folds a chain and changes a
macrocycle. It is the metadynamics of CREST (Grimme 2019), with ASE:

- each run is molecular dynamics from one conformer (Langevin, `temperature` 300 K,
  steps of 2 fs with hydrogen masses of 4 u). Every `deposit` (1 ps) the structure it
  is at becomes a reference, and every reference adds `k exp(-alpha RMSD²)` to the
  potential, which pushes the run away from where it has been. The RMSD is that of
  the heavy atoms after superposition (`atoms=` for others), and `k` is `k_per_atom`
  times the number of atoms;
- the runs differ in strength: by default the twelve combinations of `k_per_atom` =
  3.0, 1.5, 0.75 mEh and `alpha` = 1.3, 0.78, 0.47, 0.28 bohr⁻² (CREST's). `runs=4`
  takes the first four of them, `runs=[(3.0, 0.28)]` the strengths given. The runs
  take the conformers in turn, the lowest in energy first; with one conformer all
  start from it;
- a run is `length` ps long. Unless it is given, it follows the size and the
  flexibility of the molecule: `0.1 (n + 0.1 n²)` ps, at least 5 and at most 200,
  where `n` is the number of atoms that the task does not hold times the share of the
  bonds between heavy atoms that turn (the rotatable bonds of `Exploit` and the single
  bonds of rings of eight or more atoms; not amide bonds). It is modelled on the rule
  of CREST, with a flexibility from the graph: 5 ps for butane, 9 ps for
  heptane-1,7-diol, 36 ps for icosane (`racerts.metadynamics.default_length(mol)`);
- the atoms that the task holds (the frozen atoms of a TS; those that `Exploit` does
  not move) are fixed in the dynamics and take no part in the bias. Where they fix the
  frame (three or more, not on a line), the bias is on where the other atoms are, so
  that a solvent molecule or a counter-ion moves around them; otherwise it is on the
  shape of the other atoms. The dynamics hold the restraints that the refinement holds
  for the conformer a run starts from, and a fragment that nothing holds is kept near
  the core by a containment restraint (`racerts.system.roles.containment`) at its
  distance in the start plus `containment` (2 A; `None`: none);
- a snapshot every `snapshot` (0.1 ps) is refined by the stage's refiner, an
  `ASEOptimizer` as for `Exploit`, and kept if it passes the gate checks and is
  further than `rmsd_threshold` from every conformer of similar energy; a duplicate
  counts as a re-find. There is no energy window: prune after the stage.

A round of the defaults is twelve runs and a refinement for every 0.1 ps of them (for
heptane-1,7-diol 52,500 steps of dynamics and about 1,000 refinements), so the dynamics
usually run on a cheaper potential than the level of the pipeline: `md_calculator`, an
ASE calculator or a function that returns one (a new one for every run). Without it
they run on the calculator of the refinement.

```python
from racerts.refine import ASEOptimizer

uma = ASEOptimizer(uma_calculator, method="UMA-s-1p2", fmax=0.02, max_steps=300)
# the dynamics on a cheaper ASE calculator, e.g. GFN2-xTB or a force field
search = racerts.Metadynamics(uma, md_calculator=cheap_calculator, rounds=3)
pipeline = racerts.Pipeline([
    racerts.Embed(), racerts.Refine(), racerts.PruneEnergy(), racerts.PruneRMSD(),
    racerts.Refine(uma), racerts.PruneEnergy(), racerts.PruneRMSD(),
    search,
    racerts.PruneEnergy(), racerts.PruneRMSD(),
])
ensemble = racerts.generate(mol, task, pipeline=pipeline)
print(search.stats)
```

- **Rounds:** one round does not find the low conformers of a large flexible
  molecule. `rounds=5` runs up to five: after a round that found a conformer more than
  0.1 kcal/mol below the lowest before it, the runs start again from the lowest
  conformers found so far, with other random numbers, until a round finds nothing
  lower (the iteration of CREST). The default is one round; `stats["stop"]` then says
  whether it still found a lower conformer (`"rounds"`: more rounds may find more) or
  not (`"nothing lower"`). A second `Metadynamics` stage in a pipeline also starts
  from the lowest conformers it gets, but with the random numbers of the first: use
  `rounds`.
- **The refinement** starts from structures at temperature, further from a minimum
  than an embedded conformer: give it the steps (`max_steps`), or drop what did not
  converge (`drop_unconverged=True`). The conformers must carry its energies, as for
  `Exploit`. An optimizer that relaxes with calculators of its own (`_relax`) has
  none for the dynamics: give `md_calculator`.
- **Worker processes:** the runs go to the worker processes of the refinement
  (`num_workers` of its `ASEOptimizer`), one run each; the calculator of the dynamics
  then is one that can be copied, or a function of a module.
- **Seeds:** every run draws its velocities and its thermostat from the seed of the
  context and its number, so the same seed gives the same conformers. Dynamics amplify
  the last digit: on another machine or with other libraries the same seed can give
  other trajectories.

New conformers record `route="mtd"`, `parent` (the conformer their run started from),
`mtd_round`, `mtd_run`, `mtd_k`, `mtd_alpha` and `mtd_time` (ps); every conformer
records `mtd_hits`, the snapshots that ended in it. `stats` has the numbers of the last
call: `rounds`, `runs`, `steps`, `snapshots`, `optimizations`, `observations`,
`accepted`, `refound`, `rejected`, `missing_mass` and `completeness` (as for `Exploit`,
of the minima within 3 kcal/mol), the `length` of a run, `containment` and `stop`.

### Recombine

The low conformers of a large flexible molecule can have every torsion of a lower
conformer and never all of them in one: an embedding draws the torsions almost
independently, and neither local moves nor dynamics from one conformer bring together
what different conformers have. `Recombine` does, as the crossing of CREST in torsions:

- the parents are the conformers within `window` (6 kcal/mol) of the lowest, at most
  `parents` (120) of them;
- a child is a first parent, drawn with Boltzmann weights at `temperature` (2 kcal/mol),
  in which every torsion that differs by more than `different` (40°) from that of a
  second parent, drawn evenly, takes the second one's value with the probability
  `share` (0.5); one at least does. One side of the bond turns rigidly: the side without
  atoms that the task holds, or the smaller one. The frozen atoms of a TS do not move;
- the torsions are the single bonds outside rings between two heavy atoms that each
  carry another heavy atom. Unlike `Exploit`, this includes the planar bonds of amides
  and esters: a child only gets a value that a parent has, cis or trans, so no isomer is
  made that the ensemble does not have;
- children with overlapping atoms (closer than `overlap_factor` = 0.5 times their vdW
  sum) are dropped. The others are refined in one call by the stage's refiner, which
  can be any optimizer, a force field too, checked by the gate, and kept if they are
  further than `rmsd_threshold` from every conformer of similar energy;
- a round makes `children`: by default 50 for every torsion, 2400 at most, and fewer
  if the parents allow no more (no child is made twice). `rounds=5` runs up to five
  rounds, each with the new conformers among its parents, as long as a round finds a
  conformer more than 0.1 kcal/mol below the lowest before it.

```python
pipeline = racerts.Pipeline([
    racerts.Embed(), racerts.Refine(), racerts.PruneEnergy(), racerts.PruneRMSD(),
    racerts.Refine(uma), racerts.PruneEnergy(), racerts.PruneRMSD(),
    racerts.Recombine(uma, children=500, rounds=3),
    racerts.PruneEnergy(), racerts.PruneRMSD(),
])
```

The conformers must carry the energies of the stage's refiner, as for `Exploit`. New
conformers record `route="cross"`, `parents` (the first and the second), `cross_torsions`
(how many were turned) and `cross_round`; every conformer records `cross_hits`, the
children that ended in it. `stats` has the numbers of the last call: `rounds`, `parents`,
`torsions`, `children`, `overlapping`, `optimizations`, `observations`, `accepted`,
`refound`, `rejected`, `missing_mass`, `completeness` (as for `Exploit`) and `stop` (`"rounds"` if the last round still found a lower conformer, `"nothing lower"`,
or why nothing was made). The same seed gives the same children.

### Resample

`Recombine` passes on what two parents have; `Resample` what the whole ensemble has. A
child is a conformer of the ensemble, drawn evenly, in which every torsion takes the
value of a member drawn for that torsion alone: the torsions are drawn independently
from their distributions over the ensemble, and the child keeps the rest of its parent
(ring forms, the reference of a held core, its active-bond targets).

- The members drawn from: all, or those within `window` of the lowest, at most `parents`
  of them. Evenly by default, so a torsion state is drawn as often as the minima found
  have it; with `temperature` (kcal/mol) by the weights exp(-E / `temperature`).
- `explore`: the probability that a torsion takes a random value instead, so that states
  that no member has can appear (never the planar bonds of amides and esters).
- The torsions are those of `Recombine` and, with `polar_hydrogens` (by default), the
  rotors of OH, NH2 and SH groups.
- A child differs from its parent by more than `different` (40°) in one torsion at least,
  and none is made twice. The torsions, the atoms held, the overlap filter, the
  refinement, the `children` of a round and the `rounds` are those of `Recombine`; each
  round draws from the members after the last.

New conformers record `route="resample"`, `parents` (the one parent) and
`resample_round`; every conformer records `resample_hits`; `stats` has the keys of
`Recombine`, with `parents` the members drawn from. The defaults are provisional, until a
benchmark sets them.

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
G_conf: the conformational free energy on these energies, -RT ln sum exp(-E/RT), above the union
(no vibrational, rotational or solvation terms); S_conf: -R sum p ln p (cal/(mol K))
run         conformers   lowest   G_conf   S_conf
1                  541     0.00     0.35     7.43
2                  548     0.00     0.26     7.82
3                  546     0.00     0.25     7.77
all runs                            0.00     9.59
spread of G_conf between runs: 0.11
found again: the population of a run within 2 of its minimum that another run has too
  1 -> 2:  87% (6 of 12 conformers)
  ...
  mean 76%, lowest 68%
merged runs: G_conf above that of all runs (mean, worst)
  1 run: 0.29, 0.35
  2 runs: 0.10, 0.12
  3 runs: 0.00, 0.00
leaving one run out changes G_conf by up to 0.12
```

(Heptane-1,7-diol, `GroundState()`, the default settings, seeds 1 to 3.)

- **G_conf**: the conformational free energy −RT ln Σ exp(−E/RT) of a run on the given
  energies, above that of the union. Only the method's energies enter (no vibrational,
  rotational or solvation terms): it is not a Gibbs free energy, and a search converged
  on MMFF energies says nothing about convergence on another method's. It is the
  criterion: it never increases when a conformer is added, and a missed global minimum
  shows in it.
- **S_conf**: the conformational entropy −R Σ p ln p on the same energies, as CREST
  reports it; it shows how much the populations spread, but neither of the two
  properties above.
- **found again**: the share of a run's low-energy population that another run has too.
  G_conf converges before the list of conformers does: many conformers of similar
  energy give the same G_conf whichever of them a run finds.
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

## Errors

What racerts cannot do with a molecule, for a reason it names, is a
`racerts.RacerTSError`. A caller that goes through many molecules catches it and goes on:

| error | also a | when |
|---|---|---|
| `NoConformersError` | `RuntimeError` | a step left no conformer: nothing embedded, every calculation failed, none converged, none passed a check |
| `MoleculeError` | `ValueError` | the molecule cannot be used as it was given: a SMILES that is not valid, a file that cannot be read, a geometry, a charge or a multiplicity that is not that of the SMILES, two endpoints that are not one reaction |
| `InconsistentRestraints` | `ValueError` | the restraints cannot be met together: windows that contradict each other, the fixed distances or the frozen atoms (see [Restraints](restraints.md)) |
| `SwapError` | `ValueError` | a swap that cannot be done as given (see [Swaps](swap.md)) |

```python
for name, smiles in molecules.items():
    try:
        ensembles[name] = racerts.generate_gs(smiles)
    except racerts.RacerTSError as error:
        failed[name] = error
```

A call that is wrong whatever the molecule (an unknown setting, a value out of range, a
stage before the one it needs) raises a `ValueError` or `TypeError` that is none of these.
What a calculator or RDKit raises comes as it is; `ASEOptimizer(expected_errors=...)`
names the errors of a calculation that cost one conformer.
