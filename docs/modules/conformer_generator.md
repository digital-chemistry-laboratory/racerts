# **conformer_generator**

The central orchestrator for generating TS-constrained conformer ensembles.

```python
from racerts import ConformerGenerator
cg = ConformerGenerator(verbose=False, randomSeed=12, num_threads=1)
```

### Settings

- `verbose` : print additional information on the progress and diagnostics
- `randomSeed` : seed for randomization
- `num_threads` : multithreading

The conformer generator object is initialized with this default configuration with default parameters:
```python
cg.mol_getter = MolGetterSMILES()
cg.embedder = CmapEmbedder()
cg.optimizer = MMFFOptimizer()
cg.energy_pruner = EnergyPruner()
cg.rmsd_pruner = RMSDPruner()
```

You can swap components via properties, for example:
```python
cg.mol_getter = MolGetterBonds(assignBonds=True, allowChargedFragments=True)
cg.embedder = BoundsMatrixEmbedder(...)
cg.optimizer = UFFOptimizer(...)
cg.energy_pruner = EnergyPruner(threshold=20.0, ...)
cg.rmsd_pruner = RMSDPruner(threshold=0.125, ...)
```
### Functions

<a id="generate_conformers"></a>
###### generate_conformers(file_name, charge=0, reacting_atoms=[], frozen_atoms=[], input_smiles=[], number_of_conformers=-1, conf_factor=30, auto_fallback=True)
End-to-end ensemble generation:
>1. Build mol ([`get_mol`](#get_mol))
>2. Embed conformers ([`embed_TS`](#embed_TS))
>3. Refine conformers with a force field ([`optimize`](#optimize))
>4. Prune by force field energy and RMSD ([`prune`](#prune))

Main parameters:

- `reacting_atoms` : (0-based) indices of atoms whose connectivity changes; drives constraints
- `file_name` : .xyz file path with a single TS geometry (3D)
- `charge` : total molecular charge (important for `MolGetterBonds`)
- `input_smiles` : optional SMILES fragments to define topology (used by `MolGetterSMILES`)

Additional options to configure the workflow:

- `number_of_conformers` : number of generated conformers (see [embed_TS(...)](#embed_TS))
- `conf_factor` : tunes the number of generated conformers (see [embed_TS(...)](#embed_TS))
- `auto_fallback` : activate fallback (see [get_mol(...)](#get_mol))
- `frozen_atoms` : optional way to override the set to constrained atoms <br>(if omitted, `reacting_atoms` and neighboring atoms are used)

<a id="get_mol"></a>
###### get_mol(file_name, charge, reacting_atoms, input_smiles=[], auto_fallback=True)
Builds the starting molecule using the configured getter. If that fails (and `auto_fallback=True`) a fallback mechanism automatically tries alternative options in with priority:
>1. [`MolGetterSMILES`](./conformer_generator.md#get_mol)<br>
>2. [`MolGetterBonds`](./conformer_generator.md#get_mol)<br>
>3. [`MolGetterConnectivity`](./conformer_generator.md#get_mol)<br>
    
<a id="embed_TS"></a>
###### embed_TS(mol_ts, new_mol, reacting_atoms, frozen_atoms, number_of_conformers=-1, conf_factor=30)
Embeds conformers with the configured embedder variant.  
The number of conformers can be set with the `number_of_conformers` parameter.
Otherwise, it is calculated using the `conf_factor`, proportional to the number of rotatable bonds of the molecule, `#rotatable_bonds`:
> `number_of_conformers =  #rotatable_bonds * conf_factor + 30`.

<a id="optimize"></a>
###### force_field(new_mol, mol_ts, frozen_atoms, auto_fallback=True)
Refines conformers with constraints on `frozen_atoms`. If the `MMFFOptimizer` fails, it automatically falls back to `UFFOptimizer`.

<a id="prune"></a>
###### prune(mol)
Runs `energy_pruner` then `rmsd_pruner` to reduce the number of conformers.

<a id="write_xyz"></a>
###### write_xyz(file_name, use_energy=False, comment="0 1")
Writes the ensemble as a combined .xyz file. If `use_energy=True`, the per-conformer energy is written in Hartree (converted from the internal kcal/mol `energy` property).
The `comment` is added to the second line of the .xyz file. As default, the charge and spin multiplicity of 1 is used.
