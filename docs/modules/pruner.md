# pruner

Reduce conformer ensembles based on relative conformer energy and structural similarity.

### EnergyPruner
Removes conformers with high energy (`energy - minimal_energy > threshold`):
Options:

- `threshold: float = 20.0` : window above minimum energy in kcal/mol

### RMSDPruner
Removes conformers that are structurally identical using `rdMolAlign.GetBestRMS`, accounting for symmetry by using all substructure matches.
Energy and rotational prefilters decrease the computational cost by efficiently identifying dissimilar conformers as deviations of energy and rotational constants.

Options:

- `threshold: float = 0.125` : RMSD in Å cutoff for considering conformers similar
- `include_hs: bool = False` : include hydrogens in RMSD calculations
- `filter_energies: bool = True` : use prefilter by energy difference
- `energy_threshold: float = 0.1` : minimum energy in kcal/mol deviation to identify different conformers
- `filter_rotations: bool = True` : use prefilter by rotational constants
- `rot_fraction_threshold: float = 0.03` : minimum rotational constant deviation to identify different conformers
- `maxMatches: int = 10000` : maximum number of substructure matches for symmetry handling in `rdMolAlign.GetBestRMS`