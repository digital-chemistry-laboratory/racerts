# **conformer_generator**

The central orchestrator of legacy racerts for generating TS-constrained conformer
ensembles, kept in `racerts.compat` (and importable as `from racerts import
ConformerGenerator`). It runs on the pipeline and gives the same ensembles as
`racerts.generate_ts` with `config=PipelineConfig.legacy()`; see [Pipelines and
tasks](../pipeline.md) and [Migrating from legacy racerts](../migration.md).

```python
from racerts import ConformerGenerator
cg = ConformerGenerator(verbose=False, randomSeed=12, num_threads=1)
```

### Settings

- `verbose` : log progress and diagnostics (INFO level of the `racerts` logger) during `generate_conformers`
- `randomSeed` : seed for randomization
- `num_threads` : multithreading

Warnings (e.g. fallbacks, dropped conformers, a multiplicity that does not fit the number of electrons) are logged with Python's `logging` under the `racerts` logger. Unless your program configures logging, they are printed to stderr; `logging.getLogger("racerts").setLevel(logging.ERROR)` silences them.

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
# optional dependency:
# pip install racerts[ase]
cg.optimizer = ASEOptimizer(...)
cg.energy_pruner = EnergyPruner(threshold=20.0, ...)
cg.rmsd_pruner = RMSDPruner(threshold=0.125, ...)
```

You can manually swap in ASE optimization as well:

```python
from ase.calculators.lj import LennardJones
from racerts.optimizer import ASEOptimizer

cg.optimizer = ASEOptimizer(calculator=LennardJones())
```
### Functions

<a id="generate_conformers"></a>
###### generate_conformers(file_name, charge=0, reacting_atoms=[], frozen_atoms=[], input_smiles=None, number_of_conformers=-1, conf_factor=80, auto_fallback=True, multiplicity=None)
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
- `multiplicity` : spin multiplicity; by default the lowest one for the number of electrons (1, or 2 for an odd number). Radical electrons of the TS graph are not used, as they mostly stand for bonds that form or break. Set it e.g. for a triplet.

Charge and multiplicity are stored on the molecule for later steps (e.g. [`ASEOptimizer`](./ff_optimizer.md#ase_optimizer), `write_xyz`), also when the molecular graph has no formal charges (`MolGetterConnectivity`). A warning is logged once if the multiplicity does not fit the number of electrons, or if an odd number of electrons makes it 2 by default (often a forgotten charge).

Additional options to configure the workflow:

- `number_of_conformers` : number of generated conformers (see [embed_TS(...)](#embed_TS))
- `conf_factor` : tunes the number of generated conformers (see [embed_TS(...)](#embed_TS))
- `auto_fallback` : activate fallback (see [get_mol(...)](#get_mol))
- `frozen_atoms` : optional way to override the set to constrained atoms <br>(if omitted, `reacting_atoms` and neighboring atoms are used)

<a id="get_mol"></a>
###### get_mol(file_name, charge, reacting_atoms, input_smiles=None, auto_fallback=True)
Builds the starting molecule using the configured getter. If that fails (and `auto_fallback=True`), the alternatives are tried in this order:
>1. [`MolGetterSMILES`](./conformer_generator.md#get_mol)<br>
>2. [`MolGetterBonds`](./conformer_generator.md#get_mol)<br>
>3. [`MolGetterConnectivity`](./conformer_generator.md#get_mol)<br>
    
<a id="embed_TS"></a>
###### embed_TS(mol_ts, new_mol, reacting_atoms, frozen_atoms, number_of_conformers=-1, conf_factor=80)
Embeds conformers with the configured embedder variant.  
The number of conformers can be set with the `number_of_conformers` parameter.
Otherwise, it is calculated using the `conf_factor`, proportional to the number of rotatable bonds of the molecule, `#rotatable_bonds`:
> `number_of_conformers =  #rotatable_bonds * conf_factor + 30`.

<a id="optimize"></a>
###### optimize(new_mol, mol_ts, frozen_atoms, auto_fallback=True)
Refines conformers with constraints on `frozen_atoms`. If the `MMFFOptimizer` fails (e.g. missing MMFF parameters), it automatically falls back to `UFFOptimizer` and logs a warning; errors of other optimizers are raised. The optimizer used is stored in the `energy_method` property of the molecule.

<a id="prune"></a>
###### prune(mol)
Runs `energy_pruner` then `rmsd_pruner` to reduce the number of conformers.

<a id="write_xyz"></a>
###### write_xyz(file_name, use_energy=False, comment=None)
Writes the ensemble as a combined .xyz file. By default, the comment lines are in [extended XYZ](https://wiki.fysik.dtu.dk/ase/ase/io/formatoptions.html#extxyz) format, e.g.
```
Properties=species:S:1:pos:R:3 racerts_energy=2.16817320 charge=-1 spin=1 multiplicity=1 energy_method=MMFFOptimizer pbc="F F F"
```
with the energy in eV as `racerts_energy` (converted from the internal kcal/mol `energy` property; left out for conformers without an energy; not called `energy`, which ASE would take as the potential energy of the structure, although it is usually a force-field energy, see `energy_method`) and the spin multiplicity as `spin` (as used by e.g. fairchem) and `multiplicity`. `ase.io.read(file_name, index=":")` reads the conformers with these values.
If `use_energy=True`, only the energy in Hartree is written instead, as in CREST ensembles; conformers without an energy are left out with a warning. A given `comment` is written as is.
