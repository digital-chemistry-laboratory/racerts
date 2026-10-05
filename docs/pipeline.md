# Pipelines and tasks

racerts splits a conformer search into a **task** (what stays fixed), a **pipeline**
of stages (what is done), and a **context** that carries the molecule, the task, the
seed and the restraints through the stages. `PipelineConfig.legacy()` (`racerts ts
--legacy`) reproduces legacy racerts exactly. The defaults differ from it in five
settings: one seed per conformer, the `frozen_first` chirality fallback with a stereo
check after refinement, energies without the anchor terms, and conformer counts that
include the freedom of separate fragments (see [Settings](#settings)).

See also: [restraints](restraints.md), [active-bond windows](active-bonds.md) and
[swaps](swap.md).

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

## Settings

`PipelineConfig` holds the settings of the default pipeline as plain data; it can be
written to and read from JSON or YAML files. Values are checked when the config is made:
unknown keys and values of the wrong type raise `ValueError`. In YAML files, numbers such
as `1e6` are read as numbers (as in YAML 1.2).

```python
config = racerts.PipelineConfig.from_dict(
    {"seed": 7, "embed": {"mode": "bounds", "n_conformers": 200}, "refine": {"backend": "uff"}}
)
config.to_file("settings.yaml")
ensemble = racerts.generate_ts("ts.xyz", [3, 4, 5], config=config)
```

| Section | Setting | Default | Meaning |
| --- | --- | --- | --- |
| | `seed` | 12 | the seed of a run: the seeds of its conformers and batches and the random draws of its stages are all derived from it by a hash (`racerts.utils.seeds`), the same with every version of Python and NumPy; -1: not reproducible |
| | `num_threads` | 1 | threads for embedding and force fields |
| `embed` | `mode` | `cmap` | `cmap` (coordinate map) or `bounds` (bounds matrix) |
| | `n_conformers` | -1 | conformers to embed; -1: rotatable bonds × `conf_factor` + 30 |
| | `conf_factor` | 80 | |
| | `etkdg` | None | ETKDGv3 instead of plain distance geometry; None: only without frozen atoms |
| | `use_random_coords` | true | |
| | `count_policy` | `fragments` | how -1 is counted: `legacy`; `fragments` (adds 3 or 6 rigid-body degrees of freedom per fragment without frozen atoms, e.g. a solvent molecule); `per_bond` (max(7, 10 × rotatable bonds)) |
| | `sequential_seeds` | true | one seed per conformer, in a stream that starts at a value derived from `seed` (different seeds do not overlap); legacy racerts embeds its first 3 conformers twice |
| | `reference_bounds` | `fallback` | bounds of the graph that exclude a distance of the reference: widened to it when embedding fails without (`fallback`), `always`, or `never` (as legacy racerts: no conformers then) |
| | `chirality_fallback` | `frozen_first` | what happens when the frozen atoms make the chirality checks fail: `legacy` or `frozen_first` (see below) |
| `refine` | `backend` | `mmff` | `mmff` or `uff` |
| | `fallback` | true | UFF if MMFF has no parameters |
| | `force_constant` | 1e6 | kcal/mol/Å² on the frozen atoms |
| | `converge` | false | minimize until the energy stops dropping; legacy racerts stops early next to the frozen atoms |
| | `energies_without_anchors` | true | energies without the terms that hold the frozen atoms (always left out with restraints, soft atoms or active-bond windows) |
| | `dielectric_model`, `dielectric_constant` | `constant`, 1.0 | MMFF electrostatics; e.g. `distance`, 4.0 damps salt bridges in vacuum |
| | `num_workers` | 1 | worker processes of the force-field refinement, with the results of one process; threads (`num_threads`) gain nothing there, since RDKit's minimizer holds Python's lock |
| `prune` | `energy_threshold` | 20.0 | kcal/mol above the lowest conformer |
| | `rmsd_threshold` | 0.125 | Å, heavy atoms |
| | `check_stereo` | true | after refinement, drop conformers whose specified stereo differs from the graph (see below) |
| | `method` | `rmsd` | `rmsd` (duplicates, as legacy racerts) or `cluster` (one conformer per cluster) |
| | `cluster_method`, `cluster_threshold` | `butina`, 1.5 | `butina`, `hierarchical` or `leader`; Å, heavy-atom RMSD after superposition |
| | `hydrogens` | none | the hydrogens in the duplicate RMSD: `none` (heavy atoms), `polar` (also those on N, O, P, S, so that the rotamers of a hydrogen bond stay apart; needs both prefilters off) or `all` |
| | `filter_energies`, `filter_rotations`, `rmsd_energy_threshold`, `rot_fraction_threshold`, `max_matches` | true, true, 0.1, 0.03, 10000 | of `method = rmsd`, see [pruner](modules/pruner.md) (there: `energy_threshold`, `maxMatches`) |
| `restraints` | | | see [restraints](restraints.md) |

**Chirality fallback.** `legacy` drops all chiral tags, or stops enforcing chirality, so
free stereocentres can come out inverted. `frozen_first` first drops only the tags of
the frozen atoms whose configuration the reference fixes (those with at most one free
neighbour); afterwards these atoms take the configuration of the reference, and
conformers with inverted stereo are removed. A free substituent that alone sets the
configuration of a frozen stereocentre is held at the reference in embedding and
refinement.

**Stereo check.** `check_stereo` compares the stereo outside the core atoms of the task
(e.g. the reacting atoms) with the graph. After the `legacy` fallback there is nothing
left to compare with (it drops the tags): use `frozen_first`.

`PipelineConfig.legacy(**settings)` gives the settings of legacy racerts whatever the
defaults, updated by the settings given (`racerts ts --legacy`, and
`PipelineConfig.from_file(path, legacy=True)` for files). `ConformerGenerator` and the
legacy command line always use them.

## Pipelines and stages

The default pipeline is `Embed → Refine → Validate (stereo) → PruneEnergy → PruneRMSD`
(the legacy one has no stereo check). Each stage takes the
context and the ensemble so far and returns an ensemble; stages built without arguments
use the legacy racerts settings, except that they embed with a seed per conformer. A
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
| `PruneCluster(ClusterPruner(...))` | keeps one conformer per cluster (`ClusterPruner` from `racerts.prune`): Butina, hierarchical (scipy) or leader clustering, on the RMSD after superposition with fixed atoms (`kernel="aligned"`, e.g. for TS graphs without bonds), the symmetry-aware RMSD (`"symmetric"`), or any `metric(mol, a, b)`; the lowest or the central member |
| `racerts.prune.FamilySelector(n_max, clusterer)` | up to `n_max` conformers spread over the clusters: the best of each, then the second best, ... (families ordered by their best member) |
| `Validate(*validators, on_fail="drop" or "flag")` | checks the conformers (see below): drops the failing ones (an error if none passes), or keeps them with the reasons in their provenance (a warning if none passes) |
| `Exploit(Refine(ASEOptimizer(...)))` | usage-directed Monte Carlo around the pruned conformers: new minima nearby, from torsions, rigid-body moves of fragments and ring flips, refined by an ASE calculator (xTB, MLIPs); stops when new minima become rare (see below) |

`Pipeline.run(ctx, ensemble)` continues an ensemble (e.g. to refine it with another
method) and leaves the ensemble passed in unchanged; stages may change the ensemble they
get in place. `Embed` starts an ensemble: to combine two embedding runs (with different
seeds), merge their ensembles. Components passed to a stage (an embedder with its seed,
an optimizer) keep their own settings; stages create default ones for the task, e.g.
`Embed()` uses ETKDGv3 for ground states.

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
`moves` and `mc_iteration`; every conformer records `mc_usage` and `mc_hits`, and
`exploit.stats` the numbers of the run. For a windowed TS, children keep their parent's
active-bond targets, and energies are compared per window bin.

## Plug-in points

Calculators, optimizers and checks from other packages go into the stages unchanged:

- **Calculators:** `ASEOptimizer(calculator=...)` and `Rescore(calculator)` take an ASE
  calculator or a callable that returns one (a factory: one calculator per worker
  process, or per conformer without workers). Wrapping calculators work as they are,
  e.g. a bias potential such as AFIR.
- **Optimizers:** `ASEOptimizer(optimizer_cls=..., optimizer_kwargs=...)` takes any class
  with the ASE optimizer interface, e.g. `sella.Sella` with `{"order": 1}` for saddle
  points. Every conformer records `converged`, `n_steps` and `wall_time` in its
  provenance; `drop_unconverged=True` removes the unconverged ones.
- **Warm-up:** `prepare(calculator, reference_atoms)` is called once for every
  calculator before its first conformer, with the reference geometry (e.g. for
  calculators that take their topology from the first geometry they see).
- **Batch energies:** `Rescore(batch=fn)`, with `fn(list_of_atoms)` returning energies
  in eV.
- **Validators:** any object with a `name` and `validate(ctx, ensemble)`, returning the
  reason for each conformer that fails; `racerts.validate.validator(fn)` turns a
  function `fn(mol, conf_id)` into one. The validators of one `Validate` stage need
  distinct names.

Built-in validators (`racerts.validate`):

- `Connectivity()`: the bonds perceived from the geometry and the specified stereo are
  those of the graph (bonds between reacting atoms exempt); `IdentityFilter()` drops
  the conformers that fail it.
- `FrozenCore(tolerance)`: the frozen atoms are at the reference.
- `ImaginaryModes(calculator, expected=1)`: finite-difference frequencies, for
  stationary points.
- `ReactionMode(calculator)`: one imaginary mode that moves an active bond of the TS
  (a rotor of a loose complex is a first-order saddle point too), and `Converged()`:
  see [the saddle search](workflow.md#from-ts-like-conformers-to-transition-states).
- `ReactionCore(tolerance=0.5)`: the distances between the reacting atoms are those of
  the reference TS within tolerance (Å). After a free saddle search it tells a TS of the
  reaction from other saddles of the same atoms, which pass the two checks above.
- `AttackFace()`: for [active-bond windows](active-bonds.md).
- `Clash(factor=0.7)`: no heavy atoms more than three bonds apart (or in different
  fragments) closer than factor × their vdW sum. Pairs of hard and core atoms keep the
  reference geometry (e.g. a forming bond) and are not checked; pairs that are close in
  the reference count only if they come 0.2 Å closer.
- `RestraintViolation(tolerance=0.5)`: no restraint of the refinement (distance windows,
  soft atoms) violated by more than tolerance (Å): a contact that broke, not the small
  excess that flat-bottom terms allow.

`racerts.validate.gate()` combines `FrozenCore`, `Connectivity`, `Clash` and
`RestraintViolation` into the validity gate after a refinement: it drops the conformers
that fail and warns when more than 30 % fail (`Validate(..., warn_above=0.3)`), which
points to a wrong charge, restraint or hypothesis rather than to single bad conformers.
Its `FrozenCore` allows 0.1 Å: MMFF and UFF hold the frozen atoms with stiff springs,
which leave them a few hundredths of an Å from the reference.

From TS-like conformers to transition states with GFN2-xTB (tblite) and Sella:

```python
from sella import Sella
from tblite.ase import TBLite
from racerts.refine import ASEOptimizer
from racerts.validate import ImaginaryModes, ReactionCore

def gfn2():
    return TBLite(method="GFN2-xTB", verbosity=0)

saddle = ASEOptimizer(gfn2, optimizer_cls=Sella, optimizer_kwargs={"order": 1}, fmax=0.005)
pipeline = racerts.Pipeline([
    racerts.Embed(),
    racerts.Refine(),
    racerts.PruneEnergy(),
    racerts.PruneRMSD(),
    racerts.Rescore(gfn2, method="GFN2-xTB"),
    racerts.PruneCount(5),
    racerts.Refine(saddle, anchors=False, fallback=False),
    racerts.Validate(ImaginaryModes(gfn2, expected=1), ReactionCore()),
])
ensemble = racerts.generate_ts("sn2.xyz", [0, 1, 2], charge=-1, smiles="CCl.[Cl-]",
                               pipeline=pipeline)
```

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
and `bondless_mol` are the pieces. `generate_gs` gives SMILES with radical electrons
(e.g. `[CH2]`) the multiplicity 1 + their number.

Ensembles can be pickled (e.g. to return them from worker processes) with all their
data; plain RDKit pickling of `ensemble.mol` drops the properties.
