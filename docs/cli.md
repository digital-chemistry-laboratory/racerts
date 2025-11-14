# Command-line interface

The `racerts` CLI mirrors the core functionality of the Python API.
The command is always:
```bash
$ racerts filename.xyz [options]
```
where the `filename.xyz` is the path to an .xyz file containing a single TS conformer.

### General inputs

| Command | Options | Explanation |
| --- | --- | --- |
| `-c`<br>`--charge` | &lt;INT&gt; [0] | Overall molecular charge |
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
| `-m`<br>`--mol` | &lt;smiles\|bonds\|connect&gt; | Method to infer molecular graph from .xyz:<br>`smiles` use [`MolGetterSMILES`](./modules/conformer_generator.md#get_mol);<br>`bonds` use [`MolGetterBonds`](./modules/conformer_generator.md#get_mol);<br>`connect` use [`MolGetterConnectivity`](./modules/conformer_generator.md#get_mol) |
| `-e`<br>`--embed` | &lt;dm\|cmap&gt; | Embedding method:<br>`dm` = bounds-matrix; uses [`BoundsMatrixEmbedder`](./modules/conformer_generator.md#embed_TS);<br>`cmap` = coordinate map; uses [`CmapEmbedder`](./modules/conformer_generator.md#embed_TS) |
| `-ff`<br>`--ff` | &lt;mmff\|uff&gt; [mmff] | Force-field refinement: ;<br>`mmff` uses [`MMFFOptimizer`](./modules/ff_optimizer.md#mmff_optimizer);<br>`uff` uses [`UFFOptimizer`](./modules/ff_optimizer.md#uff_optimizer) |


### Output

| Command | Options | Explanation |
| --- | --- | --- |
| `-o`<br>`--output` | &lt;STR&gt; [conformer_ensemble.xyz] | Output .xyz filename |
| `--out_energies` | - | Write energies in E<sub>h</sub> in the comment line;<br>conformer property remains in kcal/mol |
| `-v`<br>`--verbose` | - | Verbose output |

### Performance and reproducibility

| Command | Options | Explanation |
| --- | --- | --- |
| `--num_threads` | &lt;INT&gt; [1] | Threads for embedding/optimization |
| `--seed` | &lt;INT&gt; [12] | Random seed |
| `--no_fallback` | - | Disable automatic fallback (see below) |

!!! note "Fallback"
    In the default setup, an automatic fallback substitutes modules upon failure:
    
    **MolGetter:**<br>
    1. [`MolGetterSMILES`](./modules/conformer_generator.md#get_mol)<br>
    2. [`MolGetterBonds`](./modules/conformer_generator.md#get_mol)<br>
    3. [`MolGetterConnectivity`](./modules/conformer_generator.md#get_mol)<br>
    
    **Optimizer**<br>
    1. [`MMFFOptimizer`](./modules/ff_optimizer.md#mmff_optimizer)<br>
    2. [`UFFOptimizer`](./modules/ff_optimizer.md#uff_optimizer)<br>

    The next module in the chain is used only if the previous one fails.

### molgetter options

| Command  | Explanation |
| ---  | --- |
| `--no_assignbonds` | For `bonds` getter: don’t assign bonds [default: True] |
| `--disallow-charged` | For `bonds` getter: disallow charged fragments [default: True] |

### embedder options

| Command | Explanation |
| --- | --- |
| `--no-random_coords` | Disable random coordinates |

### optimizer options

| Command | Explanation |
| --- | --- |
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
$ racerts example.xyz --charge 0 --reacting_atoms 2 3 4
```

Use SMILES-defined topology and coordinate-map embedding
```bash
$ racerts example.xyz --charge 0 \
  --reacting_atoms 2 3 4 \
  --input_smiles "C1=CC=CC=C1" "O" \
  --mol smiles --embed cmap --ff mmff --out_energies
```

Increase diversity by allowing more initial conformers
```bash
$ racerts example.xyz --charge 0 --reacting_atoms 2 3 4 --conf_factor 120
```
Faster refinement on many conformers with threads (if using multithreaded optimizers)
```bash
$ racerts example.xyz --charge 0 --reacting_atoms 2 3 4 --num_threads 4
```
