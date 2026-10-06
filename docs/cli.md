# Command-line interface

The `racerts` CLI mirrors the core functionality of the Python API. It has the
subcommands `racerts ts`, `racerts gs` and `racerts swap` ([Swaps](swap.md)), and keeps
the legacy racerts form `racerts filename.xyz [options]` (also
`racerts run filename.xyz [options]`), described [below](#racerts-filenamexyz-legacy).
The legacy form always uses the settings of legacy racerts; `racerts ts --legacy` writes
the same ensemble.

## racerts ts / racerts gs

```bash
$ racerts ts ts.xyz --reacting-atoms 3 4 5 --smiles "CCCCCC=C" [options]
$ racerts gs "OC(=O)[C@@H]1CCCN1C(C)=C" [options]
```

| Option | Explanation |
| --- | --- |
| `-r`, `--reacting-atoms` (ts) | 0-based indices of the atoms whose bonds form or break |
| `-s`, `--smiles` (ts) | SMILES of the TS topology, one per fragment |
| `--frozen-atoms` (ts) | frozen atoms instead of the reacting atoms and their neighbours |
| `--graph` (ts) | first method for the graph: `smiles`, `bonds` or `connect` |
| `-c`, `--charge` | total charge (ts: default 0; gs: from the SMILES) |
| `--multiplicity` | spin multiplicity (default: the lowest for the electrons) |
| `--config` | [PipelineConfig](settings.md) as JSON or YAML; the other options override it |
| `--legacy` | the settings of legacy racerts, whatever the defaults (the config file and the other options override them) |
| `-n`, `--n-conformers`, `--conf-factor` | conformers to embed |
| `--count-policy` | how the default number is counted: `legacy`, `fragments`, `per_bond` |
| `--embed` | `cmap` or `bounds` (`embed.mode`) |
| `--etkdg`, `--no-etkdg` | ETKDGv3 instead of plain distance geometry (default: only for gs) |
| `--sequential-seeds`, `--no-sequential-seeds` | one seed per conformer |
| `--chirality-fallback` (ts) | `legacy` or `frozen_first` |
| `--reference-bounds` (ts, swap) | `fallback`, `always` or `never`: bounds of the graph that exclude a distance of the reference are widened to it when embedding fails without, always, or never |
| `--active-window`, `--active-bond`, `--stratify`, `--neighbor-window` (ts) | [active-bond windows](active-bonds.md) |
| `--restraint`, `--hints`, `--restraint-half-width`, `--restraint-force-constant`; `--keep-hbonds`, `--contact`, `--keep-fragments` (ts); `--link-fragments` (gs) | [restraints](restraints.md) |
| `--refine` | `mmff` or `uff` (`refine.backend`) |
| `--converge`, `--no-converge` | minimize until the energy stops dropping |
| `--energies-without-anchors`, `--no-energies-without-anchors` | energies without the terms that hold the frozen atoms |
| `--dielectric MODEL CONSTANT` | MMFF dielectric, e.g. `distance 4` |
| `--no-fallback` | no fallback for the graph (bonds, connectivity) or MMFF (UFF) |
| `--seed`, `--num-threads` | |
| `--energy-threshold`, `--rmsd-threshold`, `--rmsd-hydrogens [none, polar, all]` | pruning (`prune.energy_threshold`, `prune.rmsd_threshold`, `prune.hydrogens`; the option without a value means `all`) |
| `--check-stereo`, `--no-check-stereo` | drop conformers whose stereo differs from the graph after refinement |
| `-o`, `--output` | output file (default `conformer_ensemble.xyz`) |
| `--crest-energies` | only the energy (Hartree) on each comment line, as CREST does |
| `--restraint-fraction F` (ts) | the probability with which an embedding batch takes each restraint of the reference geometry (`--keep-hbonds`, `--contact`, `--keep-fragments`) ([restraints](restraints.md#biased-and-unbiased-conformers-in-one-ensemble)) |
| `--export-restraints {xtb,crest,orca,json}` | write the frozen atoms and restraints for another program and stop before embedding ([restraints](restraints.md#restraints-for-other-programs)) |
| `--export-to PATH` | the file of `--export-restraints` (default `restraints.xcontrol`, `.inp` or `.json` next to the output) |
| `-v`, `-vv` | progress (INFO) or details (DEBUG) on stderr |

Wrong input (an invalid setting, a missing file, an invalid SMILES) ends with a message
and exit status 2; `-vv` shows the traceback.

## racerts filename.xyz (legacy)

The command is:
```bash
$ racerts filename.xyz [options]
```
where the `filename.xyz` is the path to an .xyz file (or .sdf/.mol with bonds) containing a single TS conformer.

### General inputs

| Command | Options | Explanation |
| --- | --- | --- |
| `-c`<br>`--charge` | &lt;INT&gt; [0] | Overall molecular charge |
| `-mult`<br>`--multiplicity` | &lt;INT&gt; | Spin multiplicity; the lowest for the number of electrons (1 or 2) if omitted |
| `-smiles`<br>`--input_smiles` | &lt;STR...&gt; | One (or multiple) SMILES for either product<br>or starting material to define topology |
| `-atoms`<br>`--reacting_atoms` | &lt;INT...&gt; | List of (0-based) indices of reacting atoms |
| `-frozen`<br>`--frozen_atoms` | &lt;INT...&gt; | Optional: overwrite of atoms to freeze;<br>if omitted, neighbors of reacting atoms are inferred |

### Ensemble size

| Command | Options | Explanation |
| --- | --- | --- |
| `-cf`<br>`--conf_factor` | &lt;INT&gt; [80] | Number of initially generated conformers: <br>NumRotatableBonds × `conf_factor` + 30  |
| `-n`<br>`--number_of_conformers` | &lt;INT&gt; [-1] | Specify number of initially generated conformers <br>(and overwrite `conf_factor`) |


### Conformer Generation Configuration

| Command | Options | Explanation |
| --- | --- | --- |
| `-m`<br>`--mol` | &lt;smiles\|bonds\|connect&gt; | Method to infer molecular graph from .xyz:<br>`smiles` use [`MolGetterSMILES`](./modules/mol_getter.md#molgettersmiles);<br>`bonds` use [`MolGetterBonds`](./modules/mol_getter.md#molgetterbonds);<br>`connect` use [`MolGetterConnectivity`](./modules/mol_getter.md#molgetterconnectivity) |
| `-e`<br>`--embed` | &lt;dm\|cmap&gt; [cmap] | Embedding method:<br>`dm` = bounds-matrix; uses [`BoundsMatrixEmbedder`](./modules/embedder.md#boundsmatrixembedder);<br>`cmap` = coordinate map; uses [`CmapEmbedder`](./modules/embedder.md#cmapembedder) |
| `-ff`<br>`--ff` | &lt;mmff\|uff&gt; [mmff] | Force-field refinement: ;<br>`mmff` uses [`MMFFOptimizer`](./modules/ff_optimizer.md#mmff_optimizer);<br>`uff` uses [`UFFOptimizer`](./modules/ff_optimizer.md#uff_optimizer) |


### Output

| Command | Options | Explanation |
| --- | --- | --- |
| `-o`<br>`--output` | &lt;STR&gt; [conformer_ensemble.xyz] | Output .xyz filename; the comment lines are in extended XYZ format<br>(energy in eV, charge, spin; readable with `ase.io.read`) |
| `--out_energies` | - | Instead, write only the energy in E<sub>h</sub> on each comment line (as in CREST ensembles);<br>conformer property remains in kcal/mol |
| `-v`<br>`--verbose` | - | Progress on stderr (`-vv`: details) |

### Performance and reproducibility

| Command | Options | Explanation |
| --- | --- | --- |
| `--num_threads` | &lt;INT&gt; [1] | Threads for embedding/optimization |
| `--seed` | &lt;INT&gt; [12] | Random seed |
| `--no_fallback` | - | Disable automatic fallback (see below) |

!!! note "Fallback"
    In the default setup, an automatic fallback substitutes modules upon failure:
    
    **MolGetter:**<br>
    1. [`MolGetterSMILES`](./modules/mol_getter.md#molgettersmiles)<br>
    2. [`MolGetterBonds`](./modules/mol_getter.md#molgetterbonds)<br>
    3. [`MolGetterConnectivity`](./modules/mol_getter.md#molgetterconnectivity)<br>
    
    **Optimizer**<br>
    1. [`MMFFOptimizer`](./modules/ff_optimizer.md#mmff_optimizer)<br>
    2. [`UFFOptimizer`](./modules/ff_optimizer.md#uff_optimizer)<br>

    The next module in the chain is used only if the previous one fails.

### molgetter options

| Command  | Explanation |
| ---  | --- |
| `--no_assignbonds` | For `bonds` getter: don’t assign bonds [default: True] |
| `--disallow_charged_fragments` | For `bonds` getter: disallow charged fragments [default: True] |

### embedder options

| Command | Explanation |
| --- | --- |
| `--no_random_coords` | Disable random coordinates |

### optimizer options

| Command | Options | Explanation |
| --- | --- | --- |
| `--force_constant` | &lt;FLOAT&gt; [1e6] | Distance-constraint force constant used during<br>FF optimization |

### RMSDPruner options

| Command | Options | Explanation |
| --- | --- | --- |
| `-rmsd`<br>`--rmsd_thres` | &lt;FLOAT&gt; [0.125] | RMSD threshold in Å |
| `-rmsd_hs`<br>`--rmsd_include_hs` | - | Include hydrogens when computing RMSDs <br>[default: False] |
| `--no_filter_energies` | - | Disable energy-based dissimilarity prefilter <br>[default: True] |
| `--no_filter_rotations` | - | Disable rotational-profile prefilter <br>[default: True] |
| `--rmsd_energy` | &lt;FLOAT&gt; [0.1] | Energy difference threshold (kcal/mol) for the prefilter |
| `--rmsd_rot_fraction` | &lt;FLOAT&gt; [0.03] | Rotational-difference fraction threshold |
| `-rmsd_match`<br>`--rmsd_max_matches` | &lt;INT&gt; [10000] | Max substructure matches for RMSD calculation |

### EnergyPruner options

| Command | Options | Explanation |
| --- | --- | --- |
| `-energy`<br>`--energy_thres` | &lt;FLOAT&gt; [20.0] | Energy window for pruning (kcal/mol) |

### Common recipes
Basic run
```bash
$ racerts ts.xyz --charge 0 --reacting_atoms 7 8 22
```

Use SMILES-defined topology (one SMILES per fragment of the TS, here benzene and water)
and coordinate-map embedding
```bash
$ racerts ts.xyz --charge 0 \
  --reacting_atoms 2 3 4 \
  --input_smiles "C1=CC=CC=C1" "O" \
  --mol smiles --embed cmap --ff mmff --out_energies
```

Increase diversity by allowing more initial conformers
```bash
$ racerts ts.xyz --charge 0 --reacting_atoms 7 8 22 --conf_factor 120
```
More threads for the embedding
```bash
$ racerts ts.xyz --charge 0 --reacting_atoms 7 8 22 --num_threads 4
```
