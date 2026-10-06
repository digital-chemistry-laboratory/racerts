# Settings

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
| | `hydrogens` | polar | the hydrogens in the duplicate RMSD: `none` (heavy atoms, as legacy racerts), `polar` (also those on N, O, P, S, so that the rotamers of a hydrogen bond stay apart; needs both prefilters off) or `all` |
| | `filter_energies`, `filter_rotations` | false, false | the prefilters of legacy racerts: compute the RMSD only for pairs within `rmsd_energy_threshold` (0.1 kcal/mol) and with principal moments within `rot_fraction_threshold` (0.03). Off: every pair is decided by its RMSD (a pair is skipped only where a lower bound of the RMSD is above the threshold). On, duplicates whose energies differ by more stay in the ensemble |
| | `max_matches` | 100 | the most equivalent atom mappings that are listed for the RMSD (legacy racerts: 10000); above it, local symmetry is assigned without a list (prefilters off), or the list is cut (prefilters on); see [pruner](modules/pruner.md) (there: `maxMatches`) |
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
