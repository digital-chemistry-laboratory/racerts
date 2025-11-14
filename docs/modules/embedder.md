# **embedder**

Embedding strategies for TS-constrained conformer generation.

Constructor parameters:

- `remove_all_conformers: bool = True` : Removes all prior conformers, *i.e.*, the .xyz structure.
- `useRandomCoords: bool = True` : Use random coordinates for the embedding.
- `ETversion: int = 2` : Used for the RDKit embedding parameters.
- `pruneRmsThresh: float = -1` : RMSD threshold for pruning directly after embedding. (default: remove identical conformers with RMSD = 0)


### CmapEmbedder
Uses a coordinate map for the distance geometry-based embedding. 
The atomic positions of `frozen_atoms` are fixed to the TS coordinates.

### BoundsMatrixEmbedder
Uses a distance-bounds matrix derived from the TS geometry:
>1. Fixes distances among reacting atoms and between reacting atoms and their neighbors (via `frozen_atoms`), matching TS distances.
>2. Performs triangle smoothing (`DoTriangleSmoothing`) with gradually increased tolerance until success
