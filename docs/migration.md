# Migrating from legacy racerts

racerts gives the same ensembles as legacy racerts (0.1.7 and its fixes) with the same inputs:
`ConformerGenerator().generate_conformers(...)`, `racerts.generate_ts(...)`,
`racerts file.xyz ...` and `racerts ts file.xyz ...` write identical files.

## What stays

The legacy API keeps working; all of it lives in `racerts.compat`:

- Every module path and name of racerts 0.1.7: `from racerts import ConformerGenerator,
  embedders, mol_getters, optimizers, pruners`, `racerts.embedder`, `racerts.optimizer`,
  `racerts.pruner`, `racerts.mol_getter`, `racerts.conformer_generator`,
  `racerts.embedder.utils`, `racerts.optimizer.ase`, `racerts.utils.get_frozen_atoms`,
  `racerts.visualizer`, and so on (`tests/test_compat.py` checks them against the 0.1.7
  release).
- Custom subclasses of the legacy base classes. The legacy embedders and optimizers are
  subclasses of the current ones in `racerts.embed` and `racerts.refine` that add the
  legacy methods `embed_TS` and `tune_ts_conformers`; the current classes do not have
  them. Pruners and mol getters are the same classes in both.
- The command line `racerts file.xyz [options]` with all its options (also as
  `racerts run ...`), in `racerts.compat.cli`.

## Behaviour changes since 0.1.7

- Output files: the comment lines are extended XYZ (energy in eV, charge, spin,
  `energy_method`) instead of charge and multiplicity (`0 1`); `write_xyz(path,
  comment="0 1")` writes such a line as given. CREST-style energies (`use_energy=True`, `--out_energies`) leave out conformers
  without an energy and use the CODATA 2022 Hartree (627.5094740629 kcal/mol instead of
  627.509), so the sixth decimal can differ.
- Messages go through Python `logging` (logger `racerts`) instead of stdout: warnings to
  stderr, progress with `verbose=True` or `-v`/`-vv`.
- `generate_conformers` uses `conf_factor=80` by default, as the command line always did
  (it was 30 in the Python API).
- Stricter input: invalid SMILES, SMILES whose formal charges disagree with `charge`, and an
  embedding without conformers raise errors; errors of optimizers other than MMFF are raised
  instead of being retried silently with UFF.
- The default multiplicity is the lowest one for the number of electrons (not the radical
  count of the TS graph), which reaches ASE calculators as their spin.
- The chirality fallback of the embedders (for TS geometries that contradict a chiral tag)
  now logs a warning.

## For new code

| legacy racerts | now |
| --- | --- |
| `ConformerGenerator().generate_conformers(file, charge, reacting_atoms, input_smiles=...)` returns a `Mol` | `racerts.generate_ts(file, reacting_atoms, charge=..., smiles=...)` returns a `ConformerEnsemble` (`.mol` is the `Mol`) |
| components set on the generator | stages of a `Pipeline`, or settings of a `PipelineConfig` |
| `cg.embedder = BoundsMatrixEmbedder()` | `PipelineConfig(embed=EmbedConfig(mode="bounds"))` |
| `cg.optimizer = UFFOptimizer()` | `PipelineConfig(refine=RefineConfig(backend="uff"))` |
| `cg.optimizer = ASEOptimizer(calculator=...)` | `Refine(ASEOptimizer(calculator=...))` in a pipeline |
| `cg.write_xyz(path)` | `ensemble.write_xyz(path)` |
| TS only | also `generate_gs(smiles)` and `Constrained(hard=[...])` |

Where the legacy code lives now:

| legacy racerts | now |
| --- | --- |
| `racerts.conformer_generator` | `racerts.compat` (`DEFAULT_CONF_FACTOR` also in `racerts.embed`; `KCAL_TO_HARTREE` keeps 627.509, `racerts.utils.units.HARTREE_TO_KCAL_MOL` is the CODATA value) |
| `racerts.embedder.embedder` | `racerts.embed` (`CmapEmbedder`, `BoundsMatrixEmbedder`, without `embed_TS`) |
| `racerts.embedder.utils` | `racerts.embed.bounds`: `get_bounds_matrix(mol_ts, new_mol, frozen_atoms, reacting_atoms)` is `bounds_matrix(new_mol, mol_ts, fixed_distance_pairs(FrozenSet(tuple(frozen_atoms), tuple(reacting_atoms))))`; `tol_function`; `print_bounds_matrix_errors` is `log_inconsistent_bounds` |
| `racerts.optimizer.ff_optimizer`, `racerts.optimizer.ase` | `racerts.refine` (without `tune_ts_conformers`), `racerts.io.ase` (`rdkit_conformer_to_ase_atoms`, `write_ase_positions_to_rdkit`) |
| `racerts.optimizer.parallel` | `racerts.refine.parallel` |
| `racerts.pruner.pruner` | `racerts.prune` (also `drop_conformers_without_energy`) |
| `racerts.mol_getter.mol_getter` | `racerts.system` (the mol getters, and `build_mol`, the legacy `get_mol` logic) |
| `racerts.utils` | `get_frozen_atoms`: `racerts.task.transition_state` (or `TransitionState(...).frozen_atoms(mol)`); `infer_charge_and_multiplicity`, `count_electrons`: `racerts.system`; `EV_TO_KCAL_MOL`: `racerts.utils.units`; `suppress_std`: `racerts.system.build`. `racerts.utils` itself is now the package of helpers (logging, optional imports, units). |
| `racerts.visualizer` | `racerts.io.viz` |

To replace how conformers are converted to ASE Atoms (e.g. for other charge or spin
conventions), override `ASEOptimizer._to_atoms`, or patch
`racerts.io.ase.rdkit_conformer_to_ase_atoms`. The legacy `ASEOptimizer` also honours a
patched `racerts.optimizer.ase.rdkit_conformer_to_ase_atoms`.
