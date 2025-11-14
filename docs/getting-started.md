# Getting started

## What is racer<sup>TS</sup>?

racer<sup>TS</sup> generates conformer ensembles for transition states (TS) using RDKit functions.

This documentation covers how to install and use racer<sup>TS</sup>, the command-line interface, and how to configure the individual components.

- [Command-line interface](cli.md)
- [Modules overview](modules/README.md)


<img src="../img/TOC.png" alt="TOC" width="400">

## Installation

- Python 3.9+
- RDKit >= 2023.9.5 (installed automatically via pip dependency)

```bash
pip install racerts
```

## Input format

- Input geometry must be an .xyz file of a TS structure
- Reacting atoms are the atoms whose connectivity changes in the reaction
- Overall molecular charge
- Optionally, a SMILES for either product or starting material can be provided to define the topology around the TS


## Quickstart (Python)

```python
from racerts import ConformerGenerator

cg = ConformerGenerator()
# example.xyz is a single-geometry TS structure; reacting atoms are 0-based indices
mol = cg.generate_conformers(file_name="example.xyz", charge=0, reacting_atoms=[2,3,4])
cg.write_xyz("ensemble.xyz")
```

## Quickstart (CLI)

```bash
racerts example.xyz --charge 0 --reacting_atoms 2 3 4
```

This will generate a pruned ensemble and write `conformer_ensemble.xyz` in the current directory by default.