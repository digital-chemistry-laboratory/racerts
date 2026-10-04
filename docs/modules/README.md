# Modules overview

racer<sup>TS</sup> is organized in layers:

- Components that do the work, usable on their own:
  - [MolGetter](mol_getter.md) (`racerts.system`) — Builds an RDKit molecule from the TS .xyz (and optional SMILES)
  - [Embedder](embedder.md) (`racerts.embed`) — Generates 3D conformers with the frozen atoms in place
  - [Optimizer](ff_optimizer.md) (`racerts.refine`) — Optimizes conformers with constraints (MMFF/UFF, optional ASE)
  - [Pruner](pruner.md) (`racerts.prune`) — Reduces ensembles by energy and RMSD
  - [Restraints](../restraints.md) (`racerts.restraints`), validators
    (`racerts.validate`), RMSDs (`racerts.geometry`), [swaps](../swap.md)
    (`racerts.system.swap`) and file output (`racerts.io`)
- [Tasks, pipelines, stages and ensembles](../pipeline.md) (`racerts.task`, `racerts.pipeline`, `racerts.config`, `racerts.api`)
- The legacy racerts API in `racerts.compat`: [ConformerGenerator](conformer_generator.md), the components with their legacy methods (`racerts.embedder`, `racerts.optimizer`, `racerts.pruner`, `racerts.mol_getter`) and the legacy command line; see [Migrating from legacy racerts](../migration.md)
