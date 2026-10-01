# Pipelines and tasks

racerts splits a conformer search into a **task** (what stays fixed), a **pipeline**
of stages (what is done), and a **context** that carries the molecule, the task and the
settings through the stages. The defaults reproduce legacy racerts exactly;
`PipelineConfig.legacy()` does so whatever the defaults.

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
| `Constrained(hard)` | the `hard` atoms | yes |

A task returns a `FrozenSet`: `hard` atoms are placed at the reference positions during
embedding and held there during refinement. In the bounds-matrix embedder
(`embed.mode = "bounds"`), the distances between the `core` atoms (for a TS the reacting
atoms) and all hard atoms are fixed.

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
| | `seed` | 12 | RDKit embedding seed |
| | `num_threads` | 1 | threads for embedding and force fields |
| `embed` | `mode` | `cmap` | `cmap` (coordinate map) or `bounds` (bounds matrix) |
| | `n_conformers` | -1 | conformers to embed; -1: rotatable bonds × `conf_factor` + 30 |
| | `conf_factor` | 80 | |
| | `etkdg` | None | ETKDGv3 instead of plain distance geometry; None: only without frozen atoms |
| | `use_random_coords` | true | |
| | `count_policy` | `legacy` | how -1 is counted: `legacy`; `fragments` (adds 3 or 6 rigid-body degrees of freedom per fragment without frozen atoms, e.g. a solvent molecule); `catmlp` (max(7, 10 × rotatable bonds)) |
| | `sequential_seeds` | false | one seed per conformer, in a stream that starts at a value derived from `seed` (different seeds do not overlap); legacy racerts embeds its first 3 conformers twice |
| | `chirality_fallback` | `legacy` | when the frozen atoms make the chirality checks fail: `legacy` (drop all chiral tags, or stop enforcing chirality; free stereocentres can invert) or `frozen_first` (drop the tags of the frozen atoms first; afterwards the frozen atoms take the configuration of the reference, and conformers with inverted stereo are removed) |
| `refine` | `backend` | `mmff` | `mmff` or `uff` |
| | `fallback` | true | UFF if MMFF has no parameters |
| | `force_constant` | 1e6 | kcal/mol/Å² on the frozen atoms |
| | `converge` | false | minimize until the energy stops dropping; legacy racerts stops early next to the frozen atoms |
| | `anchor_free_energies` | false | energies without the terms that hold the frozen atoms (they add 0.02–0.18 kcal/mol) |
| | `dielectric_model`, `dielectric_constant` | `constant`, 1.0 | MMFF electrostatics; e.g. `distance`, 4.0 damps salt bridges in vacuum |
| `prune` | `energy_threshold` | 20.0 | kcal/mol above the lowest conformer |
| | `eht_energies` | false | rank by extended Hückel energies (deprecated: use `Rescore`) |
| | `rmsd_threshold` | 0.125 | Å, heavy atoms |
| | `check_stereo` | false | after refinement, drop conformers whose specified stereo (outside the core atoms of the task, e.g. the reacting atoms) differs from the graph; needs `frozen_first` when the embedding falls back (the legacy fallback drops the tags) |
| | `method` | `rmsd` | `rmsd` (duplicates, as legacy racerts) or `cluster` (one conformer per cluster) |
| | `cluster_method`, `cluster_threshold` | `butina`, 1.5 | `butina`, `hierarchical` or `leader`; Å, heavy-atom RMSD after superposition |
| | `include_hs`, `filter_energies`, `filter_rotations`, `rmsd_energy_threshold`, `rot_fraction_threshold`, `max_matches` | | see [pruner](modules/pruner.md) |

`PipelineConfig.legacy(**settings)` gives the settings of legacy racerts whatever the
defaults, updated by the settings given (`racerts ts --legacy`, and
`PipelineConfig.from_file(path, legacy=True)` for files). `ConformerGenerator` and the
legacy command line always use them.

## Pipelines and stages

The default pipeline is `Embed → Refine → PruneEnergy → PruneRMSD` (with
`prune.check_stereo`, a stereo check follows `Refine`). Each stage takes the
context and the ensemble so far and returns an ensemble; stages built without arguments
use the legacy racerts defaults. A pipeline can be put together by hand, e.g. to refine
with an ASE calculator:

```python
from ase.calculators.lj import LennardJones
from racerts.refine import ASEOptimizer

pipeline = racerts.Pipeline([
    racerts.Embed(n_conformers=50),
    racerts.Refine(ASEOptimizer(calculator=LennardJones(), fmax=0.1)),
    racerts.PruneEnergy(),
    racerts.PruneRMSD(),
])
ensemble = racerts.generate_ts("ts.xyz", [3, 4, 5], pipeline=pipeline)
```

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
| `Refine(optimizer, anchors=False)` | refines all atoms freely, e.g. a saddle-point search from TS-like conformers |
| `Rescore(calculator)` or `Rescore(batch=fn)` | replaces the energies by single points of an ASE calculator (xTB, MLIPs), or of a function that evaluates a list of Atoms at once; the replaced energy goes into the provenance |
| `PruneCount(n_max, renumber=False)` | keeps the `n_max` lowest conformers |
| `PruneCluster(ClusterPruner(...))` | keeps one conformer per cluster: Butina, hierarchical (scipy) or leader clustering, on the RMSD after superposition with fixed atoms (`kernel="aligned"`, e.g. for TS graphs without bonds), the symmetry-aware RMSD (`"symmetric"`), or any `metric(mol, a, b)`; the lowest or the central member |
| `FamilySelector(n_max, clusterer)` | up to `n_max` conformers spread over the clusters: the best of each, then the second best, ... (families ordered by their best member) |
| `Validate(*validators, on_fail="drop" or "flag")` | checks the conformers (see below) |

`Pipeline.run(ctx, ensemble)` continues an ensemble (e.g. to refine it with another
method) and leaves the ensemble passed in unchanged; stages may change the ensemble they
get in place. `Embed` starts an ensemble: to combine two embedding runs (with different
seeds), merge their ensembles. Components passed to a stage (an embedder with its seed,
an optimizer) keep their own settings; stages create default ones for the task, e.g.
`Embed()` uses ETKDGv3 for ground states.

The pipeline logs every stage with its number of conformers and run time (logger
`racerts.pipeline.runner`, level INFO); `generate`, `generate_ts` and `generate_gs` take
`verbose=True` to show these messages.

## Plug-in points

Calculators, optimizers and checks from other packages go into the stages unchanged,
e.g. those of catmlp and catmlptools (which racerts does not depend on):

- **Calculators:** `ASEOptimizer(calculator=...)` and `Rescore(calculator)` take an ASE
  calculator or a callable that returns one (a factory: one calculator per worker
  process, or per conformer without workers). Wrapping calculators work as they are,
  e.g. a bias potential such as catmlptools' `AFIRCalculator`.
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
  function `fn(mol, conf_id)` into one. Built in:
  - `Connectivity()`: the bonds perceived from the geometry and the specified stereo are
    those of the graph (bonds between reacting atoms exempt); `IdentityFilter()` drops
    the conformers that fail it;
  - `FrozenCore(tolerance)`: the frozen atoms are at the reference;
  - `ImaginaryModes(calculator, expected=1)`: finite-difference frequencies, for
    stationary points.

From TS-like conformers to transition states with GFN2-xTB (tblite) and Sella:

```python
from sella import Sella
from tblite.ase import TBLite
from racerts.refine import ASEOptimizer
from racerts.validate import ImaginaryModes

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
    racerts.Validate(ImaginaryModes(gfn2, expected=1)),
])
ensemble = racerts.generate_ts("sn2.xyz", [0, 1, 2], charge=-1, smiles="CCl.[Cl-]",
                               pipeline=pipeline)
```

## Conformer ensembles

A `ConformerEnsemble` wraps the RDKit molecule with its conformers (`ensemble.mol`).
Everything known about the conformers is stored in RDKit properties, so RDKit code and
the legacy classes see the same data:

| Property | Of | Content |
| --- | --- | --- |
| `energy` | each conformer | energy in kcal/mol |
| `provenance` | each conformer | JSON, e.g. `{"embedder": "CmapEmbedder", "seed": 12, "etkdg": false}` |
| `energy_method` | the molecule | the optimizer that gave the energies |
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
