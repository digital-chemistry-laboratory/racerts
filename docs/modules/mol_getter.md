# mol_getter

Build RDKit molecules from a TS .xyz file (and optional SMILES), providing different strategies for topology inference.

### MolGetterSMILES
Combines one or more SMILES (as separate fragments) and maps atoms to the .xyz structure to transfer bond orders where applicable.

!!! tip "Choosing quickly"
    - Have SMILES? Use `smiles`.
    - No SMILES but you know the charge? Pick `bonds`.

### MolGetterBonds
Reads the .xyz, then uses `rdDetermineBonds.DetermineBonds` to infer bonding (needs `charge`).
These arguments are passed on to the DetermineBonds algorithm:
- `assignBonds: bool = True`
- `allowChargedFragments: bool = True`
  
### MolGetterConnectivity
Reads the .xyz and uses RDKit's `rdDetermineBonds.DetermineConnectivity` (sets connectivity without bond orders), sanitizes (hybridization, aromaticity, conjugation, symmetric rings), and assigns stereochemistry from 3D.