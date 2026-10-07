# Pruners (`racerts.prune`)

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
- `max_maps: int = 100` : the most equivalent atom mappings that are listed; above it, local symmetry (the hydrogens of a methyl, the methyls of a tert-butyl) and the exchange of identical molecules (solvent molecules, the molecules of a cluster) are assigned without a list
- `graph: Mol = None` : for conformers that are stored without bonds (e.g. read from coordinates alone): a molecule of the same atoms in the same order with the bonds
- `energy_tolerance: float = None` : conformers whose energies differ by more than this (kcal/mol) are not duplicates, however close they are. For ensembles refined at one level, where two structures with different energies are two stationary points; all conformers need an energy. Default: the RMSD alone decides

At the default threshold the RMSD alone is enough for refined structures. A wider one merges stationary points: of 1433 pairs of refined minima and saddle points of two peptide catalysts (100 to 150 atoms) that lie within 0.25 Å, 455 differ by more than 0.5 kcal/mol in energy; of the 781 within 0.125 Å, one does. With a threshold above the default, give `energy_tolerance` (0.5 kcal/mol is far above the noise of an optimization and below what tells two conformers apart).

Identical molecules of the system are equivalent as wholes: a conformer in which two solvent molecules have changed places is a duplicate. Their exchanges are not listed (eight methanols have 40320, three benzenes 10368) but assigned, molecule to molecule, so their number does not matter.

The equivalent atoms come from the bonds. In a molecule without bonds all atoms of an element count as equivalent and no hydrogen can be left out: two structures that differ in which hydrogen sits on the oxygen are taken for one. The pruner warns in that case; with `graph=` it reads the equivalent atoms and the hydrogens from the bonded molecule:

```python
from racerts.prune import RMSDPruner

RMSDPruner(graph=bonded).prune(mol)  # mol: the same atoms, stored without bonds
```

#### The RMSD pruner of legacy racerts
`racerts.pruner.RMSDPruner` (`racerts.compat.pruner.RMSDPruner`, a subclass of the one above) is what `ConformerGenerator`, the legacy command line, `PipelineConfig.legacy()` and a config with a prefilter switched on use. It computes the RMSD of `racerts.geometry` (the smallest over a list of symmetry maps, with the same values as `rdMolAlign.GetBestRMS`) only for pairs that pass its two prefilters, so duplicates whose energies differ by more than `energy_threshold` both stay.

- `threshold: float = 0.125`, `align: bool = True` : as above
- `include_hs: bool = False` : include all hydrogens (`hydrogens="all"`; the default is the heavy atoms)
- `filter_energies: bool = True` : use prefilter by energy difference
- `energy_threshold: float = 0.1` : minimum energy in kcal/mol deviation to identify different conformers
- `filter_rotations: bool = True` : use prefilter by rotational constants
- `rot_fraction_threshold: float = 0.03` : minimum rotational constant deviation to identify different conformers
- `maxMatches: int = 10000` : maximum number of symmetry maps; beyond it the others are left out, with a warning (also the exchanges of identical molecules: it can then miss a copy)
- `graph: Mol = None` : as above, for conformers that are stored without bonds

With both prefilters off it decides every pair by its RMSD, as the pruner above.

The RMSD of the pruner for other uses:

```python
from racerts.symmetry import SymmetricRMSD

kernel = SymmetricRMSD(mol, "polar")  # once per graph; "heavy", "polar", "all" or atom indices
a = mol.GetConformer(0).GetPositions()
b = mol.GetConformer(1).GetPositions()
kernel.rmsd(a, b)
kernel.within(a, b, 0.125)  # with early exits
```

The RMSD over a list of symmetry maps, as legacy racerts computes it:

```python
from racerts.geometry import rmsd, symmetry_maps

symmetry = symmetry_maps(mol)  # once per graph: the heavy atoms and their maps
a = mol.GetConformer(0).GetPositions()
b = mol.GetConformer(1).GetPositions()
rmsd(a, b, symmetry.atoms, symmetry.maps)  # align=False: without superposition
rmsd(a, b)  # all atoms, fixed correspondence
```