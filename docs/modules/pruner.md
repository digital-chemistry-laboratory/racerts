# pruner

Reduce conformer ensembles based on relative conformer energy and structural similarity.

Conformers without an `energy` property (e.g. from failed calculations) cannot be ranked; both pruners drop them and log a warning.

### EnergyPruner
Removes conformers with high energy (`energy - minimal_energy > threshold`):
Options:

- `threshold: float = 20.0` : window above minimum energy in kcal/mol

### RMSDPruner
Removes duplicates: conformers within `threshold` of a lower one by the RMSD over the symmetry of the graph, after superposition. Every pair is decided by its RMSD (`racerts.symmetry.SymmetricRMSD`); a pair is skipped only where a lower bound of the RMSD is above the threshold.

Options of `racerts.prune.RMSDPruner`:

- `threshold: float = 0.125` : RMSD in Å below which two conformers are duplicates
- `hydrogens: str = "polar"` : the hydrogens that count: `"none"` (heavy atoms), `"polar"` (also those on N, O, P and S, so that the rotamers of a hydrogen bond stay apart) or `"all"`
- `align: bool = True` : superpose before comparing; `False` compares the conformers in the frame they share (e.g. of a frozen core)
- `max_maps: int = 100` : the most equivalent atom mappings that are listed; above it, local symmetry (the hydrogens of a methyl, the methyls of a tert-butyl) is assigned without a list

#### The RMSD pruner of legacy racerts
`racerts.pruner.RMSDPruner` (`racerts.compat.pruner.RMSDPruner`, a subclass of the one above) is what `ConformerGenerator`, the legacy command line, `PipelineConfig.legacy()` and a config with a prefilter switched on use. It computes the RMSD of `racerts.geometry` (the smallest over a list of symmetry maps, with the same values as `rdMolAlign.GetBestRMS`) only for pairs that pass its two prefilters, so duplicates whose energies differ by more than `energy_threshold` both stay.

- `threshold: float = 0.125`, `align: bool = True` : as above
- `include_hs: bool = False` : include all hydrogens (`hydrogens="all"`; the default is the heavy atoms)
- `filter_energies: bool = True` : use prefilter by energy difference
- `energy_threshold: float = 0.1` : minimum energy in kcal/mol deviation to identify different conformers
- `filter_rotations: bool = True` : use prefilter by rotational constants
- `rot_fraction_threshold: float = 0.03` : minimum rotational constant deviation to identify different conformers
- `maxMatches: int = 10000` : maximum number of symmetry maps; beyond it the others are left out, with a warning

With both prefilters off it decides every pair by its RMSD, as the pruner above.

The same RMSD for other uses:

```python
from racerts.geometry import rmsd, symmetry_maps

symmetry = symmetry_maps(mol)  # once per graph: the heavy atoms and their maps
a = mol.GetConformer(0).GetPositions()
b = mol.GetConformer(1).GetPositions()
rmsd(a, b, symmetry.atoms, symmetry.maps)  # align=False: without superposition
rmsd(a, b)  # all atoms, fixed correspondence
```