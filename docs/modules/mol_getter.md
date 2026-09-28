# mol_getter

Build RDKit molecules from a TS .xyz file (and optional SMILES), providing different strategies for topology inference.

### MolGetterSMILES
Combines one or more SMILES (as separate fragments) and maps atoms to the .xyz structure to transfer bond orders where applicable.
The SMILES must contain the same atoms as the .xyz file (including hydrogens), and their formal charges must add up to `charge`, otherwise an error is raised. Atom map number *n* stands for atom *n* of the .xyz file (1-based). The atoms are matched in one of three ways:

- **Complete atom maps:** if every heavy atom has a map number, e.g. `[CH3:1][Cl:2].[Cl-:3]`, the numbers are used directly. Hydrogens without a number are matched to the nearest hydrogens of their atom.
- **Partial atom maps:** if only some atoms have a map number, e.g. the reacting atoms, the SMILES is searched as a substructure of the connectivity of the .xyz file (bond orders aside) with the mapped atoms held in place. Bonds between reacting atoms, which may form or break, are allowed where the connectivity lacks them. Mapping the reacting atoms settles the reaction centre, where a TS geometry alone can be ambiguous (e.g. which chlorine of a symmetric SN2 TS is the chloride).
- **Maximum common substructure (MCS):** without atom maps, the atoms are matched by an iterative MCS search, which can be slow for large TSs.

If atom maps do not fit the .xyz file (numbers out of range or repeated, other elements, or bonds of the SMILES that the geometry does not have), a warning is logged and the MCS search is used instead. A warning also flags MCS matches with bonds that the geometry does not have; bonds between reacting atoms may form or break and are not checked.

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