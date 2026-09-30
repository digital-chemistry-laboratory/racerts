# pruner

Reduce conformer ensembles based on relative conformer energy and structural similarity.

Conformers without an `energy` property (e.g. from failed calculations) cannot be ranked; both pruners drop them and log a warning.

### EnergyPruner
Removes conformers with high energy (`energy - minimal_energy > threshold`):
Options:

- `threshold: float = 20.0` : window above minimum energy in kcal/mol

### RMSDPruner
Removes conformers that are structurally identical by the RMSD of `racerts.geometry`: after superposition, the smallest over all symmetry maps (automorphisms) of the graph, with the same values as `rdMolAlign.GetBestRMS`.
The maps are computed once per pruning, and the identity map is compared first, which settles most duplicates.
Energy and rotational prefilters decrease the computational cost by efficiently identifying dissimilar conformers as deviations of energy and rotational constants.

Options:

- `threshold: float = 0.125` : RMSD in Å cutoff for considering conformers similar
- `include_hs: bool = False` : include hydrogens in RMSD calculations
- `align: bool = True` : superpose before comparing; `False` compares the conformers in the frame they share (e.g. of a frozen core)
- `filter_energies: bool = True` : use prefilter by energy difference
- `energy_threshold: float = 0.1` : minimum energy in kcal/mol deviation to identify different conformers
- `filter_rotations: bool = True` : use prefilter by rotational constants
- `rot_fraction_threshold: float = 0.03` : minimum rotational constant deviation to identify different conformers
- `maxMatches: int = 10000` : maximum number of symmetry maps; beyond it the others are left out, with a warning

The same RMSD for other uses:

```python
from racerts.geometry import rmsd, symmetry_maps

symmetry = symmetry_maps(mol)  # once per graph: the heavy atoms and their maps
a = mol.GetConformer(0).GetPositions()
b = mol.GetConformer(1).GetPositions()
rmsd(a, b, symmetry.atoms, symmetry.maps)  # align=False: without superposition
rmsd(a, b)  # all atoms, fixed correspondence
```