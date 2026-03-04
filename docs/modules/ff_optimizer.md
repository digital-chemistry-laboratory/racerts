# optimizer

Optimizers for refining TS-constrained conformers.

Constructor parameters:

- `conf_id_ref: int = -1` : reference conformer ID on the TS molecule
- `force_constant: float = 1e6` : force constant of distance constraints to TS reference points

<a id="mmff_optimizer"></a>
### MMFFOptimizer
Optimizes each conformer with MMFF94:
>1. Align to the TS reference conformer (on `align_indices`)
>2. Create RDKit force-field extra points at the TS coordinates for each `align_indices` atom and add distance constraints with `force_constant`
>3. Minimize and set the conformer `energy` property from `ff.CalcEnergy()` in kcal/mol
>4. Re-align to the TS reference

<a id="uff_optimizer"></a>
### UFFOptimizer
Same flow as `MMFF_optimizer` using the RDKit implementation of UFF.


!!! note MMFF vs UFF
    - MMFF: Good general-purpose performance, if MMFF parameters are available.
    - UFF: Use if no MMFF parameters available or as fallback.

<a id="ase_optimizer"></a>
### ASEOptimizer (optional dependency)

`ASEOptimizer` uses the [Atomic Simulation Environment (ASE)](https://wiki.fysik.dtu.dk/ase/):
>1. Align to the TS reference conformer on `align_indices`
>2. Convert each RDKit conformer to ASE `Atoms`
>3. Apply `FixAtoms(indices=align_indices)` to freeze these atoms
>4. Optimize with an ASE optimizer (`BFGS` by default) and the provided ASE calculator (`calculator=...`, instance or callable)
>5. Write positions back to RDKit and set conformer `energy` in kcal/mol (`eV * 23.06054783061903`)
>6. Re-align to the TS reference

!!! note Index convention
    `align_indices` follow standard Python indexing (0-based), consistent with RDKit and ASE atom indices.

Install ASE support with:

```bash
pip install racerts[ase]
```

Example with a calculator instance:

```python
from ase.calculators.lj import LennardJones
from racerts.optimizer import ASEOptimizer

optimizer = ASEOptimizer(
    calculator=LennardJones(),
    fmax=0.05,
    max_steps=100,
)
```

Example with a calculator callable:

```python
from ase.calculators.lj import LennardJones
from racerts.optimizer import ASEOptimizer

optimizer = ASEOptimizer(
    calculator=LennardJones,
    num_threads=4,
)
```
