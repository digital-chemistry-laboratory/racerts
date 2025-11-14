# Modules overview

racer<sup>TS</sup> is made up of independently configurable components:

- [ConformerGenerator](conformer_generator.md) — Orchestrates the full pipeline
- [MolGetter](mol_getter.md) — Builds an RDKit molecule from the TS .xyz (and optional SMILES)
- [Embedder](embedder.md) — Generates TS-constrained 3D conformers
- [Optimizer](ff_optimizer.md) — Optimizes conformers with constraints (MMFF/UFF)
- [Pruner](pruner.md) — Reduces ensembles by energy and RMSD