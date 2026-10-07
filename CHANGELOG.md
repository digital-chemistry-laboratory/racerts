# Changelog

## 0.2.0 (unreleased)

racerts becomes a pipeline of stages with tasks, restraints, validation, other levels of
theory, a conformer search and swaps, while everything legacy racerts (0.1.7) did keeps
working and gives the same files. 
`docs/migration.md` has the changes related to existing code.

### Summary

**Compatibility first.** Everything legacy racerts did still works and gives the same files:
`ConformerGenerator().generate_conformers(...)`, the command line `racerts file.xyz ...`, the legacy module paths and custom subclasses of the legacy base classes. The legacy API lives in `racerts.compat` and runs on the new pipeline; `PipelineConfig.legacy()` and `racerts ts --legacy` give the legacy settings through the new entry points. A test compares the public names with the 0.1.7 wheel, and 15 output files are compared byte for byte.

**What is new, in one line each:**

| area | now possible |
|---|---|
| Entry points | `generate_ts`, `generate_gs`, `generate` (any task), `generate_runs` (independent seeds, merged and compared), `swap`; `racerts ts`, `gs`, `swap` on the command line |
| Tasks | transition states, ground states, and `Constrained`: any motif held fixed (hard), kept near the reference (soft), or free |
| Pipeline | conformer generation as stages (`Embed`, `Refine`, `Rescore`, `PruneEnergy`, `PruneRMSD`, `PruneCluster`, `PruneCount`, `Validate`, `Exploit`) that can be reordered, replaced or extended; all settings as one checked `PipelineConfig` (JSON/YAML) |
| Restraints | distance windows from the user, from the hydrogen bonds and contacts of the reference, between the fragments of a complex, and as hints from the graph; held in embedding and refinement; exported for xtb, CREST and ORCA |
| Active bonds | the forming and breaking bonds of a TS sampled over a window instead of frozen, at stratified targets |
| Validation | checks after any stage: connectivity, stereo, frozen core, clashes, restraints, attack face, reaction core, number and character of imaginary modes, convergence; `gate()` bundles the cheap ones |
| Other levels | any ASE calculator (extras `ase`, and `xtb` for GFN2-xTB with tblite) as optimizer or for re-ranking, with worker processes; a staged recipe (cheap levels first, the expensive one on what survives) and a saddle-point recipe |
| Search | `Exploit`: a Monte Carlo search around the conformers found (torsions, ring flips, rigid moves of fragments) that stops when it rarely finds new minima |
| Swaps | replace a group, a ligand or ring atoms of a reference ensemble and sample only the new part, with stereo taken from the reference or given explicitly |
| Pruning | duplicates decided by a symmetry-aware RMSD that does not list every atom mapping (so hydrogens can count), cluster pruning, family selection |

**Defaults that differ from legacy racerts** (the new entry points; `PipelineConfig.legacy()`
or `--legacy` restores each):

| setting | legacy | now | why |
|---|---|---|---|
| seeds | the first three conformers embedded twice | one seed per conformer, derived from the run's seed by a hash | no duplicates, reproducible across Python and NumPy versions |
| chirality fallback | all chiral tags dropped | `frozen_first`, with a stereo check after refinement | free stereocentres no longer come out inverted |
| reported energies | include the terms that hold the frozen atoms | without them | energies of conformers are comparable |
| conformer count | rotatable bonds only | also the rigid-body freedom of separate fragments | complexes with solvent or counter-ions get enough conformers |
| duplicates | RMSD only for pairs with nearly equal energy and moments of inertia | every pair by its RMSD | the prefilters kept duplicates apart (hexan-1-ol: 149 against 86 conformers) |
| hydrogens in the RMSD | none | polar ones (on N, O, P, S) | rotamers of a hydrogen bond stay apart |
| listed atom mappings | up to 10,000 | up to 100, local symmetry assigned above that | large symmetric molecules no longer stall |
| window targets | – | five per active-bond window | – |

`RMSDPruner()` and `PruneRMSD()` built by hand have the same three pruning defaults; the
pruner with the legacy ones is `racerts.pruner.RMSDPruner`.

### In detail

#### Structure

- **Layout.** `task/` (what is held), `embed/`, `refine/`, `prune/`, `validate/`, `exploit/`
  (the stages), `pipeline/` (`Context`, `ConformerEnsemble`, `Pipeline`, runs), `restraints/`,
  `system/` (graphs, charge and spin, fragments, stereo, swaps), `recipes/`, `io/`, `utils/`,
  `compat/` (legacy racerts), and `config.py`, `api.py`, `cli.py`, `geometry.py`,
  `symmetry.py`, `swaps.py` (the swap entry point).
- **`Context`**: the molecule with its reference geometry, the task, the frozen atoms, the
  restraints, the seed, charge and multiplicity; made once by `Context.create` and passed to
  every stage.
- **`ConformerEnsemble`**: the conformers with energy, energy method and provenance stored
  on the conformers themselves (picklable). `filter`, `merge`, `best`, `record`, `summary`,
  `from_frames` (geometries from elsewhere), `write_xyz` (extended XYZ), `write_sdf`,
  `to_ase`.
- **`Pipeline`**: a list of stages; a stage is anything with `run(ctx, ensemble)`. Plug-in
  points are documented: custom embedders, optimizers, pruners, validators, tasks.
- **`PipelineConfig`** with the sections `embed`, `refine`, `prune`, `restraints`: plain
  data, checked when made (unknown keys, wrong types and values raise and name the setting),
  read from and written to JSON and YAML, NumPy numbers accepted. `build(task)` gives the
  default pipeline.
- **Command line.** `racerts ts`, `racerts gs`, `racerts swap`, each with only its options;
  `--config` for a settings file that the other options override; wrong input ends with a
  message and exit status 2. `racerts file.xyz ...` (also `racerts run`) is the legacy
  command line, unchanged.
- **Seeds.** Every seed and random draw comes from `racerts.utils.seeds`: derived from the
  run's seed and a label by a hash, so stages do not share streams and results do not depend
  on the Python or NumPy version.
- **Logging** instead of printing (logger `racerts`; `verbose=`, `-v`, `-vv`).

#### Tasks

- `TransitionState(reacting_atoms, frozen_atoms=None, ...)`: the reacting atoms and their
  neighbours frozen, as legacy racerts; `TransitionState.from_endpoints(reactant, product)`
  finds the reacting atoms from the bonds that change; `remap` for new atom indices.
- `GroundState()`: nothing frozen, ETKDGv3, stereocentres kept; a graph with radical
  electrons gets the multiplicity 1 + their number.
- `Constrained(hard=, soft=, core=)` and `FrozenSet`: hard atoms stay at the reference, soft
  atoms start there and are held by position restraints (0.3 Å tolerance) in refinement,
  core atoms are exempt from the stereo check.
- A task may define `restraints(mol)`, `multiplicity(mol)` and `remap(index_map)`.

#### Embedding

- Coordinate map (`cmap`) or bounds matrix (`bounds`), one implementation for both and for
  the legacy embedders.
- One seed per conformer (`sequential_seeds`); conformer count policies `legacy`,
  `fragments`, `per_bond`.
- Chirality fallback `frozen_first`: only the tags of frozen atoms whose configuration the
  reference fixes are dropped; a free substituent that alone sets a frozen stereocentre is
  held at the reference; conformers with inverted stereo are removed afterwards.
- `reference_bounds`: when RDKit's bounds exclude the reference geometry (a metal over a
  bond), they are widened to it, with a warning.
- Several reference geometries in one run (`references="all"` or ids), each with seeds of
  its own.
- Distance windows in the bounds; a window that cannot be embedded is reported, and a hint
  that does not embed within its budget is dropped while the conformer count is kept.
- A warning when a window lies between two atoms that the coordinate map places.
- With restraints, ETKDG weights the distance bounds 100 times higher against its torsion
  terms (`racerts.embed.RESTRAINT_BOUNDS_WEIGHT`), which override about 60 % of the windows
  otherwise.
- ETKDGv3 (ground states) runs with RDKit's macrocycle and small-ring torsion terms. With
  restraints the small-ring terms are left out: from RDKit 2026.09 on they override a
  window that needs another ring conformation. RDKit fails with them on cyclopentane rings
  ("bad direction in linearSearch"): the embedding goes on without them, with a warning.

#### Restraints

- `DistanceRestraint` (a window between two atoms, with force constant, stage and source)
  and `RestraintSet` (one per pair, merge rules, JSON round trip, sampling); `PositionRestraint`
  for soft atoms.
- Sources, combined by `build_restraints` with the user's window winning: the user
  (`--restraint a b d`), hydrogen bonds of the reference (`--keep-hbonds`), declared contacts
  with the neighbours that fix their orientation (`--contact`), the closest contact between
  fragments (`--keep-fragments`), links between the fragments of a ground-state complex
  (`--link-fragments`), and hints: candidate hydrogen bonds from the graph, each tried in its
  own share of the conformers (`--hints`).
- `restraints.fraction`: each embedding batch takes the restraints of the reference with
  this probability, so biased and unbiased conformers come in one ensemble; the summary
  of the ensemble counts both.
- Fragment roles (reactive, anchored, contained, free) and containment restraints that keep
  a loose fragment near the core.
- Held in MMFF/UFF refinement as flat-bottom terms and in ASE refinement as constraints;
  reported energies leave the terms out. `Refine(restraints=False)` releases them for one
  refinement, e.g. at the level whose energies decide; the gate knows.
- Windows that contradict each other or the fixed distances raise
  `racerts.InconsistentRestraints` (a `ValueError`). `racerts.embed.bounds_matrix` runs the
  same test before a run, and `racerts.restraints.link_window` gives the contact window of
  a fragment link for any pair.
- `export_restraints` and `--export-restraints {xtb,crest,orca,json}`: the frozen atoms and
  restraints as input for other programs.

#### Active-bond windows

- `TransitionState(..., active_window=0.2 | (lo, hi), active_bonds=..., stratify=n)`: the
  forming and breaking bonds are sampled within a window around the reference (or between
  absolute limits) instead of being frozen; the other distances of the reacting atoms keep a
  narrow window (`neighbor_window`).
- Stratified targets: each conformer is held at one of `stratify` target lengths (the
  midpoints of equal parts of the window; five by default); several bonds combine their
  targets as a Latin hypercube. Pruning then works per target, so no length is lost to an
  energy window.
- `AttackFace` removes conformers in which a reacting atom is approached from the other
  face; `ReactionCore` checks the reacting atoms against the reference.
- The lengths are written next to the output (`<output>.active_bonds.csv`) and summarised.

#### Refinement and other levels

- Force fields: one implementation for MMFF and UFF; `converge` (minimise until the energy
  stops dropping), `energies_without_anchors`, MMFF dielectric settings, worker processes
  (`num_workers`), the MMFF → UFF fallback recorded in `energy_method`.
- `ASEOptimizer`: any ASE calculator and optimizer; a `prepare` hook per structure;
  convergence, steps and wall time in the provenance; `drop_unconverged`; restraints and
  soft atoms as ASE constraints; worker processes; charge and spin passed to the calculator.
  `expected_errors` names the errors of a calculation that cost one conformer (default:
  any), every other error ends the run; the error of a failed conformer is in its
  provenance; `serial_below` starts no worker processes for few conformers. The step
  that relaxes the structures is one method, `_relax`: a subclass replaces it to hand
  them to the relaxation of another package (e.g. one that takes several structures at
  once on a GPU) and keeps the rest; `racerts.refine.optimize_one` is the relaxation of
  one structure, for an override that only runs the structures its own way.
- `racerts.NoConformersError` (a `RuntimeError`): a step left no conformer (nothing
  embedded, every calculation failed, none converged, none passed a check).
- `Refine(anchors=False)` refines all atoms; `Refine` removes conformers whose optimization
  failed, with a warning.
- `Rescore`: single-point energies at another level, or any function of the structure (a
  free-energy correction), as the ranking energy.

#### Pruning

- Thresholds are checked; conformers with a missing or non-finite energy are dropped with a
  warning; a threshold of 0 no longer hangs; copies of a linear molecule are duplicates.
- The RMSD over a list of symmetry maps (`racerts.geometry`), with the maps computed once per
  pruning: the kernel of legacy racerts and the reference that the tests compare against.
- `racerts.symmetry`: the RMSD minimised over the symmetry of the graph without listing
  every atom mapping. Up to `max_maps` mappings are listed (exact); above that, the local
  symmetry of terminal groups is assigned per parent atom, and identical molecules (solvent
  molecules, the molecules of a cluster) are exchanged as wholes, molecule to molecule: the
  number of their exchanges no longer matters. The value is never below the true
  RMSD. Two lower bounds of the RMSD skip 76 to 99.8 % of the pairs without computing it.
- `racerts.prune.RMSDPruner(threshold, hydrogens="none" | "polar" | "all", align, max_maps)`
  decides every pair by its RMSD. The pruner of legacy racerts, with its prefilters, keywords
  and overridable methods, is its subclass `racerts.pruner.RMSDPruner` (in `racerts.compat`).
  Clusters, Exploit and the comparison of runs use the same kernel. For conformers that
  are stored without bonds, `graph=` gives the bonded molecule (also in `ClusterPruner`
  with the symmetric kernel and in the pruner of legacy racerts); without it all atoms of
  an element count as equivalent, and a warning says so.
- `PruneCluster` (Butina, hierarchical or leader clustering; one conformer per cluster) and
  `FamilySelector` (families of similar conformers, for pools); `PruneCount` (the n lowest).
  `ClusterPruner.clusters(mol)` and `round_robin(clusters)` give the clusters and the
  order that takes them in turns, for a selection of one's own.
- `prune.check_stereo`: conformers whose stereo differs from the graph are removed after
  refinement, centres with a lone pair and ring double bonds included.

#### Validation

- `Validate(*validators, on_fail="drop" | "flag", warn_above=share)` and the `Validator`
  protocol (`validator(function)` wraps a function); every conformer records what it passed.
- Validators: `Connectivity` (the bonds of the graph in the geometry, reacting atoms
  exempt), `IdentityFilter`, `FrozenCore`, `Clash`, `RestraintViolation`, `AttackFace`,
  `ReactionCore`, `ImaginaryModes` (the number of imaginary modes), `ReactionMode` (the
  imaginary mode stretches an active bond), `Converged`.
- `gate()`: frozen core, connectivity, clashes and restraints in one stage, for use between
  levels.
- The mode checks compute the Hessian by finite differences of the calculator's forces, or
  take it from a function (`hessian=`).

#### Exploit

- A Monte Carlo search around the pruned conformers: parents are picked least used first,
  moved (torsions, ring flips, rigid moves of free fragments), optimized, checked as the gate
  checks, and kept if they are new minima within the energy window.
- It stops on a budget or when the estimated chance of a new low-energy minimum falls below
  `saturation`; `stats` reports the missing mass and a Chao1 completeness.
- A ranking energy (`rank=Rescore(...)`) can differ from the energy of the optimizer; for a
  windowed TS children keep their parent's targets.

#### Recipes

- `staged(levels, final_window=...)`: a list of `Level(optimizer, window, exploit, rank,
  pool, pool_by)`, cheap first; each level refines what the window of the level before kept,
  with the gate in between. `staged(optimizer)` is the usual two levels.
- `saddles(search, calculator=..., hessian=..., checks=...)`: from TS-like conformers to
  saddle points with a search function, then `Converged`, `ReactionMode`, `ReactionCore`,
  `Connectivity` and any further checks.

#### Swaps

- `Swap(fragment, site= | remove_atoms= | center=, substructure= | old_fragment=, ...)`: a
  SMILES fragment with dummies replaces a labelled site, listed atoms, a group at an atom or
  a matched pattern; several attachments (ligands, rings) with `attach_map` and `bond_types`.
- `apply_swap`: the new graph with every reference conformer, kept atoms where they were
  and a single-attachment fragment grafted along the removed bond; charge and multiplicity
  carried over; restraints remapped and lost ones reported.
- `racerts.swap(reference, swap, conserve="hard" | "soft" | "free", ...)`: samples only the
  new atoms around the kept geometry; optional rigid poses of the fragment; clash warnings;
  tasks are remapped, so a swap on a TS keeps its reacting atoms.
- Stereo, in three rules: what the graphs specify is carried over; what they leave open
  takes the reference geometry where that defines it (which hydrogen leaves chooses the
  configuration); `stereo={...}` gives it explicitly (R/S, E/Z, cis/trans).
- Rings: ring atoms replaced by a chain of another length, fused rings opened, a ring closed
  on a double bond cis or trans for any ring size (a configuration that the force field
  cannot hold raises).
- Ligand swaps keep the coordination geometry when the metal and donors are held.
- `label_hydrogen` and `substitute_groups` for several groups at once; `racerts swap` on the
  command line.

#### Independent runs

- `generate_runs(mol, task, seeds=...)`: the same search from several seeds, the merged
  ensemble and a `ConvergenceReport` (how much of each run the others found again);
  `merge_runs`, `compare_runs`.

#### Graphs, charge and spin

- Graph methods as before (SMILES, bond perception, connectivity) with atom maps: complete
  maps are used directly, partial ones pin the substructure search; fragment-wise matching.
- `mol_from_geometry(atoms, smiles)` and `mol_from_explicit_h_smiles`: graphs for geometries
  from elsewhere, radicals included; `bondless_mol`.
- Charge and multiplicity are settled once, stored on the molecule and written to the
  output; the default multiplicity is the lowest for the electron count; a multiplicity
  below 1 is refused.

#### Legacy API and behaviour changes

As `docs/migration.md` lists them: the legacy names and paths (checked against the 0.1.7
wheel), custom subclasses, the legacy command line. Behaviour changes that also reach legacy
calls: extended-XYZ comment lines, messages through `logging`, `conf_factor=80` in the
Python API as on the command line, stricter input (an invalid SMILES or one that disagrees
with the charge is an error of the SMILES getter), an embedding without conformers raises,
optimizer errors other than missing MMFF parameters are raised, the default multiplicity,
widened bounds when RDKit embeds nothing, copies of an atom or a linear molecule pruned,
a `provenance` conformer property from the ASE optimizer.

#### Fixes to legacy behaviour

Input (hydrogens of `.sdf`/`.mol` files kept; map-number lookup), extended-Hückel energies
per conformer and in kcal/mol, a stale minimal-energy cache, a `NameError` in verbose mode,
CLI flags that did not reach the default embedder, `racerts_energy` instead of `energy` in
the output (which ASE read as the structure's energy), identical solvent molecules no longer
permuted in the atom matching.

#### Tests and tooling

- `tests/`: the tests of the current API, `tests/legacy` (the tests of legacy racerts),
  `test_baseline.py` (15 output files of legacy racerts byte for byte, on RDKit 2025.03.2),
  `test_compat.py` (the public names against the 0.1.7 wheel), `test_docs.py` (every setting
  and command-line option has its row in the docs), `test_examples.py` (the scripts of
  `examples/` run). Markers: `ase` (needs ASE) and `xtb` (needs tblite).
- CI: ruff, the default and ASE suites on the latest RDKit, a baseline job on RDKit
  2025.03.2.
