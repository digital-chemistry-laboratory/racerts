# Pipelines and tasks

racerts splits a conformer search into a **task** (what stays fixed), a **pipeline**
of stages (what is done), and a **context** that carries the molecule, the task and the
settings through the stages. The defaults reproduce legacy racerts exactly.

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
| `refine` | `backend` | `mmff` | `mmff` or `uff` |
| | `fallback` | true | UFF if MMFF has no parameters |
| | `force_constant` | 1e6 | kcal/mol/Å² on the frozen atoms |
| `prune` | `energy_threshold` | 20.0 | kcal/mol above the lowest conformer |
| | `eht_energies` | false | rank by extended Hückel energies (deprecated: use `Rescore`) |
| | `rmsd_threshold` | 0.125 | Å, heavy atoms |
| | `include_hs`, `filter_energies`, `filter_rotations`, `rmsd_energy_threshold`, `rot_fraction_threshold`, `max_matches` | | see [pruner](modules/pruner.md) |

## Pipelines and stages

The default pipeline is `Embed → Refine → PruneEnergy → PruneRMSD`. Each stage takes the
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
| `Refine(optimizer, anchors=False)` | refines all atoms freely, e.g. a saddle-point search from TS-like conformers |
| `Rescore(calculator)` or `Rescore(batch=fn)` | replaces the energies by single points of an ASE calculator (xTB, MLIPs), or of a function that evaluates a list of Atoms at once; the replaced energy goes into the provenance |
| `PruneCount(n_max, renumber=False)` | keeps the `n_max` lowest conformers |

`Pipeline.run(ctx, ensemble)` continues an ensemble (e.g. to refine it with another
method) and leaves the ensemble passed in unchanged; stages may change the ensemble they
get in place. `Embed` starts an ensemble: to combine two embedding runs (with different
seeds), merge their ensembles. Components passed to a stage (an embedder with its seed,
an optimizer) keep their own settings; stages create default ones for the task, e.g.
`Embed()` uses ETKDGv3 for ground states.

The pipeline logs every stage with its number of conformers and run time (logger
`racerts.pipeline.runner`, level INFO); `generate`, `generate_ts` and `generate_gs` take
`verbose=True` to show these messages.

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
